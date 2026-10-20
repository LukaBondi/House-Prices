#!/usr/bin/env python3
"""Probe an already running House Prices API and optional trusted trainer CLI.

Examples:
  python tools/smoke_stack.py --base-url http://127.0.0.1:8000
  python tools/smoke_stack.py --base-url http://127.0.0.1:3000/api --frontend-url http://127.0.0.1:3000
  python tools/smoke_stack.py --base-url http://127.0.0.1:3000/api --frontend-url http://127.0.0.1:3000 --trainer-service trainer --exercise-adaptation --require-timed-retraining --exercise-restart
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT_PYTHON = Path(r"C:\Users\lukab\.conda\envs\sys-304\python.exe")


class SmokeFailure(RuntimeError):
    """A live stack did not satisfy a requested smoke check."""


def _json_request(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 10.0,
) -> tuple[int, Any, dict[str, str], str]:
    data = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
    return _request_bytes(base_url, path, method=method, data=data, timeout=timeout)


def _request_bytes(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    data: bytes | None = None,
    timeout: float = 10.0,
) -> tuple[int, Any, dict[str, str], str]:
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers=headers,
        method=method,
    )
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        body = response.read(1_000_000).decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body) if body else None
        except json.JSONDecodeError:
            parsed = None
        return int(response.status), parsed, {key.lower(): value for key, value in response.headers.items()}, body


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def _parse_cli_json(output: str, *, context: str) -> Any:
    """Parse the complete stdout document; CLI JSON may be pretty-printed."""
    try:
        return json.loads(output.strip())
    except json.JSONDecodeError as error:
        raise SmokeFailure(f"{context} returned invalid JSON.") from error


def _summary_total(summary: Any, *, context: str) -> int:
    _require(isinstance(summary, dict), f"{context} monitoring response must be a JSON object.")
    requests = summary.get("requests")
    _require(isinstance(requests, dict), f"{context} monitoring response is missing the requests object.")
    total = requests.get("total")
    _require(
        isinstance(total, int) and not isinstance(total, bool) and total >= 0,
        f"{context} monitoring response has no nonnegative integer requests.total.",
    )
    return total


def _is_uuid(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError):
        return False
    return str(parsed) == value.lower()


def _prediction_attributions(
    responses: list[tuple[int, Any, dict[str, str], str]],
    *,
    context: str,
) -> list[dict[str, str]]:
    """Require every successful response to identify its own captured request/version."""
    attributions: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for status, body, headers, _ in responses:
        _require(status == 200 and isinstance(body, dict), f"{context} prediction did not return HTTP 200 JSON.")
        header_id = headers.get("x-request-id")
        body_id = body.get("request_id")
        _require(_is_uuid(header_id) and _is_uuid(body_id), f"{context} prediction is missing a valid request UUID.")
        _require(header_id == body_id, f"{context} prediction header/body request UUIDs differ.")
        _require(header_id not in seen_ids, f"{context} requests did not receive independent UUIDs.")
        version = body.get("model_version")
        _require(isinstance(version, str) and version, f"{context} prediction is missing its captured model version.")
        seen_ids.add(header_id)
        attributions.append({"request_id": header_id, "model_version": version})
    return attributions


def _validate_cache_activity(
    *,
    miss_delta: int,
    hit_delta: int,
    coalesced_delta: int,
) -> dict[str, int]:
    """A concurrent group must show a miss followed by a hit or in-flight reuse."""
    deltas = {"misses": miss_delta, "hits": hit_delta, "coalesced": coalesced_delta}
    _require(
        all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in deltas.values()),
        "Monitoring response has malformed cache counter deltas.",
    )
    _require(miss_delta > 0, "Concurrent identical requests did not produce an initial cache miss.")
    _require(
        hit_delta + coalesced_delta > 0,
        "Concurrent identical requests produced neither a cache hit nor an in-flight coalesced event after the miss.",
    )
    return {**deltas, "reused_after_miss": hit_delta + coalesced_delta}


def _read_request_events(
    *,
    request_ids: list[str],
    database: str | None,
    database_path: str,
    trainer_service: str | None,
    python: str,
) -> list[dict[str, Any]] | None:
    if trainer_service:
        code = (
            "import json,sqlite3,sys;"
            "db,raw_ids=sys.argv[1:3];ids=json.loads(raw_ids);"
            "c=sqlite3.connect('file:'+db+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;"
            "marks=','.join('?' for _ in ids);"
            "rows=[dict(row) for row in c.execute('SELECT request_id,path,lifecycle,status_code,model_version FROM requests WHERE request_id IN ('+marks+')',ids).fetchall()];"
            "c.close();print(json.dumps(rows))"
        )
        command = _trainer_command(
            code,
            [database_path, json.dumps(request_ids)],
            root=ROOT,
            service=trainer_service,
            python=python,
            raw_python=True,
        )
        output = _run_cli(command, root=ROOT)
        rows = _parse_cli_json(output, context="Request-event DB inspection")
        _require(isinstance(rows, list), "Request-event DB inspection did not return a row list.")
        return rows
    if database:
        import sqlite3

        connection = sqlite3.connect(f"file:{Path(database).as_posix()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            placeholders = ",".join("?" for _ in request_ids)
            rows = connection.execute(
                "SELECT request_id,path,lifecycle,status_code,model_version FROM requests "
                f"WHERE request_id IN ({placeholders})",
                request_ids,
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            connection.close()
    return None


def _assert_request_event_rows(
    rows: list[dict[str, Any]],
    *,
    attributions: list[dict[str, str]],
    malformed_id: str,
    malformed_status: int,
) -> None:
    expected: dict[str, tuple[int, str | None]] = {
        item["request_id"]: (200, item["model_version"]) for item in attributions
    }
    expected[malformed_id] = (malformed_status, None)
    by_id = {row.get("request_id"): row for row in rows if isinstance(row, dict)}
    _require(len(by_id) == len(rows), "Persisted request-event rows contain duplicate or invalid UUID keys.")
    _require(set(by_id) == set(expected), "One or more HTTP request UUIDs are missing from the persisted audit rows.")
    for request_id, (status_code, model_version) in expected.items():
        row = by_id[request_id]
        _require(row.get("path") == "/predict", f"Persisted request {request_id} has the wrong route.")
        _require(row.get("lifecycle") == "completed", f"Persisted request {request_id} is not completed.")
        _require(row.get("status_code") == status_code, f"Persisted request {request_id} has the wrong prepared status.")
        if model_version is not None:
            _require(row.get("model_version") == model_version, f"Persisted request {request_id} lost its serving-version attribution.")


def _source_id_set(values: Any, *, label: str) -> set[int]:
    _require(isinstance(values, list), f"Timed run is missing its {label} source ID list.")
    _require(
        all(isinstance(value, int) and not isinstance(value, bool) for value in values),
        f"Timed run has malformed {label} source IDs.",
    )
    return {int(value) for value in values}


def _validate_timed_run(
    candidate_run: dict[str, Any],
    *,
    imported_at: float,
    imported_source_ids: set[int],
    minimum_labels: int,
) -> dict[str, Any]:
    started_at = candidate_run.get("started_at")
    _require(
        isinstance(started_at, (int, float)) and not isinstance(started_at, bool)
        and math.isfinite(float(started_at)) and started_at >= imported_at,
        "Automatic retraining did not start after delayed truth import began.",
    )
    run_status = candidate_run.get("status")
    _require(
        run_status in {"promoted", "rejected"},
        f"Automatic retraining ended with unsupported status {run_status!r}; expected promoted or rejected.",
    )
    completed_at = candidate_run.get("completed_at")
    _require(
        isinstance(completed_at, (int, float)) and not isinstance(completed_at, bool)
        and math.isfinite(float(completed_at)) and completed_at >= started_at,
        "Automatic retraining verdict is missing a valid completion timestamp.",
    )
    details = candidate_run.get("details")
    _require(isinstance(details, dict), "Automatic retraining run is missing its persisted details object.")
    label_count = details.get("label_count")
    _require(
        isinstance(label_count, int) and not isinstance(label_count, bool) and label_count >= minimum_labels,
        "Automatic retraining snapshot did not record enough eligible labels.",
    )
    eligible_ids = _source_id_set(details.get("eligible_ids"), label="eligible snapshot")
    new_ids = _source_id_set(details.get("new_ids"), label="new snapshot")
    _require(imported_source_ids <= eligible_ids, "Automatic retraining snapshot omitted imported workload source IDs.")
    _require(imported_source_ids <= new_ids, "Automatic retraining run did not treat imported workload IDs as new eligible labels.")
    return {
        "status": run_status,
        "run_id": candidate_run.get("run_id"),
        "started_at": started_at,
        "completed_at": completed_at,
        "label_count": label_count,
        "eligible_source_ids": details["eligible_ids"],
        "new_source_ids": details["new_ids"],
    }


def _trainer_command(
    module_or_code: str,
    arguments: list[str],
    *,
    root: Path,
    service: str | None,
    python: str,
    raw_python: bool = False,
) -> list[str]:
    if raw_python:
        command = ["python", "-c", module_or_code, *arguments]
    else:
        command = ["python", "-m", module_or_code, *arguments]
    if service:
        return ["docker", "compose", "exec", "-T", service, *command]
    return [python, *command[1:]]


def _run_cli(command: list[str], *, root: Path, timeout: float = 180.0) -> str:
    completed = subprocess.run(
        command,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=os.environ.copy(),
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout)[-3000:]
        token = os.environ.get("ADMIN_TOKEN")
        if token:
            detail = detail.replace(token, "[redacted]")
        raise SmokeFailure(f"Trainer command failed with exit {completed.returncode}: {detail}")
    return completed.stdout.strip()


def _choose_python(value: str | None) -> str:
    if value:
        return value
    configured = os.environ.get("SYS304_PYTHON")
    if configured:
        return configured
    if DEFAULT_PROJECT_PYTHON.is_file():
        return str(DEFAULT_PROJECT_PYTHON)
    return sys.executable


def _feature_sample(schema: Any) -> dict[str, Any]:
    sample = schema.get("sample") if isinstance(schema, dict) else None
    if isinstance(sample, dict) and sample:
        return sample
    return {
        "OverallQual": 7,
        "GrLivArea": 1710,
        "YearBuilt": 2003,
        "Neighborhood": "CollgCr",
    }


def _last_run(summary: Any) -> dict[str, Any] | None:
    _require(isinstance(summary, dict), "Monitoring response must be an object while waiting for retraining.")
    retraining = summary.get("retraining")
    _require(isinstance(retraining, dict) and "last_run" in retraining, "Monitoring response is missing retraining.last_run.")
    run = retraining["last_run"]
    _require(run is None or isinstance(run, dict), "Monitoring response contains an invalid retraining.last_run value.")
    return run


def _request_event_workload(
    base_url: str,
    *,
    run_id: str,
    schema: Any,
    timeout: float,
    database: str | None,
    database_path: str,
    trainer_service: str | None,
    python: str,
) -> dict[str, Any]:
    """Exercise malformed, cached, and concurrent duplicate paths over real HTTP."""
    status, before, _, _ = _json_request(base_url, "/monitoring/summary?window_minutes=15", timeout=timeout)
    _require(status == 200, "Monitoring summary failed before the request-event checks.")
    before_total = _summary_total(before, context="Pre-event")
    before_cache = before.get("cache") if isinstance(before, dict) else None
    _require(isinstance(before_cache, dict), "Monitoring response is missing cache totals before request-event checks.")
    before_hits = before_cache.get("hits")
    before_coalesced = before_cache.get("coalesced")
    _require(
        isinstance(before_hits, int) and not isinstance(before_hits, bool) and before_hits >= 0
        and isinstance(before_coalesced, int) and not isinstance(before_coalesced, bool) and before_coalesced >= 0,
        "Monitoring response has malformed cache counters before request-event checks.",
    )

    malformed_status, _, malformed_headers, _ = _request_bytes(
        base_url,
        "/predict",
        method="POST",
        data=b"{",
        timeout=timeout,
    )
    _require(400 <= malformed_status < 500, f"Malformed JSON should receive a client error, got HTTP {malformed_status}.")
    malformed_id = malformed_headers.get("x-request-id")
    _require(_is_uuid(malformed_id), "Malformed prediction request was missing its audited UUID header.")

    repeated_features = dict(_feature_sample(schema))
    repeated_features.update({
        "OverallQual": 5,
        "GrLivArea": 2037,
        "YearBuilt": 1997,
        "Neighborhood": "Edwards",
    })

    def post_prediction(features: dict[str, Any], request_run_id: str) -> tuple[int, Any, dict[str, str], str]:
        return _json_request(
            base_url,
            "/predict",
            method="POST",
            payload={"features": features, "run_id": request_run_id},
            timeout=timeout,
        )

    first_status, first_body, first_headers, _ = post_prediction(repeated_features, run_id + "-cache")
    second_status, second_body, second_headers, _ = post_prediction(repeated_features, run_id + "-cache")
    _require(first_status == 200 and second_status == 200, "Repeated cache-path predictions did not both succeed.")
    _require(isinstance(first_body, dict) and isinstance(second_body, dict), "Repeated cache-path predictions returned invalid JSON.")
    _require(_is_uuid(first_headers.get("x-request-id")) and _is_uuid(second_headers.get("x-request-id")), "Repeated predictions did not receive audited UUIDs.")

    status, repeated_summary, _, _ = _json_request(base_url, "/monitoring/summary?window_minutes=15", timeout=timeout)
    _require(status == 200 and isinstance(repeated_summary, dict), "Monitoring summary failed after repeated predictions.")
    repeated_cache = repeated_summary.get("cache")
    _require(isinstance(repeated_cache, dict), "Monitoring response is missing cache totals before concurrent duplicates.")
    concurrent_before_hits = repeated_cache.get("hits")
    concurrent_before_misses = repeated_cache.get("misses")
    concurrent_before_coalesced = repeated_cache.get("coalesced")
    _require(
        all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in (concurrent_before_hits, concurrent_before_misses, concurrent_before_coalesced)),
        "Monitoring response has malformed cache counters before concurrent duplicates.",
    )

    coalesced_delta = 0
    concurrent_hit_delta = 0
    concurrent_miss_delta = 0
    concurrent_group = 0
    concurrent_attributions: list[dict[str, str]] = []
    for group in range(3):
        features = dict(repeated_features)
        features["LotArea"] = 7000 + group
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            futures = [
                executor.submit(post_prediction, features, f"{run_id}-coalesce-{group}")
                for _ in range(32)
            ]
            responses = [future.result() for future in futures]
        attributions = _prediction_attributions(responses, context="Concurrent duplicate")
        concurrent_attributions.extend(attributions)
        status, current, _, _ = _json_request(base_url, "/monitoring/summary?window_minutes=15", timeout=timeout)
        _require(status == 200 and isinstance(current, dict), "Monitoring summary failed after a concurrent duplicate workload.")
        cache = current.get("cache")
        _require(isinstance(cache, dict), "Monitoring response is missing cache counters after concurrent duplicates.")
        hits, misses, coalesced = (cache.get(name) for name in ("hits", "misses", "coalesced"))
        _require(
            all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in (hits, misses, coalesced)),
            "Monitoring response has malformed cache counters after concurrent duplicates.",
        )
        concurrent_hit_delta = hits - concurrent_before_hits
        concurrent_miss_delta = misses - concurrent_before_misses
        coalesced_delta = coalesced - concurrent_before_coalesced
        if concurrent_hit_delta + coalesced_delta > 0:
            concurrent_group = group + 1
            break
    activity = _validate_cache_activity(
        miss_delta=concurrent_miss_delta,
        hit_delta=concurrent_hit_delta,
        coalesced_delta=coalesced_delta,
    )

    status, after, _, _ = _json_request(base_url, "/monitoring/summary?window_minutes=15", timeout=timeout)
    _require(status == 200, "Monitoring summary failed after the request-event checks.")
    after_total = _summary_total(after, context="Post-event")
    cache = after.get("cache") if isinstance(after, dict) else None
    _require(isinstance(cache, dict), "Monitoring response is missing cache totals after request-event checks.")
    after_hits = cache.get("hits")
    _require(isinstance(after_hits, int) and not isinstance(after_hits, bool), "Monitoring response has malformed cache hit total.")
    total_successful_attributions = _prediction_attributions(
        [
            (first_status, first_body, first_headers, ""),
            (second_status, second_body, second_headers, ""),
        ],
        context="Repeated cache-path",
    ) + concurrent_attributions
    all_success_ids = [item["request_id"] for item in total_successful_attributions]
    _require(len(all_success_ids) == len(set(all_success_ids)), "Repeated and concurrent predictions reused a request UUID.")
    _require(malformed_id not in set(all_success_ids), "Malformed and successful requests reused a request UUID.")
    expected_event_count = 1 + len(total_successful_attributions)
    _require(after_total >= before_total + expected_event_count, "One or more malformed, repeated, or concurrent predictions are missing from persisted monitoring totals.")
    _require(after_hits > before_hits, "Repeated predictions did not produce a cache hit.")
    audit_rows = _read_request_events(
        request_ids=[malformed_id, *all_success_ids],
        database=database,
        database_path=database_path,
        trainer_service=trainer_service,
        python=python,
    )
    if audit_rows is not None:
        _assert_request_event_rows(
            audit_rows,
            attributions=total_successful_attributions,
            malformed_id=malformed_id,
            malformed_status=malformed_status,
        )
    return {
        "malformed_status": malformed_status,
        "malformed_request_id": malformed_id,
        "cache_hit_delta": after_hits - before_hits,
        "concurrent_cache_miss_delta": activity["misses"],
        "concurrent_cache_hit_delta": activity["hits"],
        "concurrent_cache_reused_after_miss": activity["reused_after_miss"],
        "coalesced_delta": coalesced_delta,
        "concurrent_group": concurrent_group,
        "successful_request_attributions": total_successful_attributions,
        "persisted_request_rows_verified": audit_rows is not None,
        "audit_total_before": before_total,
        "audit_total_after": after_total,
    }


def _database_snapshot(
    args: argparse.Namespace,
    *,
    smoke_request_id: str,
    run_id: str,
) -> dict[str, Any] | None:
    if args.trainer_service:
        code = (
            "import json,pathlib,sqlite3,sys;"
            "db,state,smoke_id,run_id=sys.argv[1:5];"
            "c=sqlite3.connect('file:'+db+'?mode=ro',uri=True);"
            "c.row_factory=sqlite3.Row;"
            "smoke=c.execute('SELECT request_id,lifecycle,status_code,model_version FROM requests WHERE request_id=?',(smoke_id,)).fetchone();"
            "run_count=c.execute(\"SELECT COUNT(*) FROM requests WHERE run_id=? AND path='/predict'\",(run_id,)).fetchone()[0];"
            "labels=[dict(row) for row in c.execute(\"SELECT l.observation_id,l.source_id FROM labels l JOIN requests r ON r.request_id=l.request_id WHERE r.run_id=? ORDER BY l.source_id\",(run_id,)).fetchall()];"
            "latest=c.execute(\"SELECT run_id,status,started_at,completed_at,details_json,metrics_json FROM runs ORDER BY started_at DESC LIMIT 1\").fetchone();"
            "pointer_path=pathlib.Path(state);"
            "pointer=json.loads(pointer_path.read_text()) if pointer_path.is_file() else None;"
            "print(json.dumps({'smoke_request':dict(smoke) if smoke else None,'workload_requests':run_count,'workload_labels':len(labels),'workload_label_rows':labels,'latest_run':dict(latest) if latest else None,'pointer':pointer}))"
        )
        command = _trainer_command(
            code,
            [args.database_path, args.state_pointer, smoke_request_id, run_id],
            root=ROOT,
            # Restart checks deliberately pause the scheduled trainer to avoid
            # a concurrent activation. The healthy backend has read access to
            # both /data and /state, so it is the stable inspection process.
            service="backend",
            python=args.python,
            raw_python=True,
        )
        output = _run_cli(command, root=ROOT)
        result = _parse_cli_json(output, context="Compose DB/state inspection")
        _require(isinstance(result, dict), "Compose DB/state inspection did not return a JSON object.")
        return result

    if not args.database:
        return None
    import sqlite3

    database = Path(args.database)
    if not database.is_file():
        raise SmokeFailure(f"Configured smoke database does not exist: {database}")
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        smoke = connection.execute(
            "SELECT request_id, lifecycle, status_code, model_version FROM requests WHERE request_id = ?",
            (smoke_request_id,),
        ).fetchone()
        run_count = connection.execute(
            "SELECT COUNT(*) FROM requests WHERE run_id = ? AND path = '/predict'",
            (run_id,),
        ).fetchone()[0]
        label_rows = connection.execute(
            "SELECT l.observation_id, l.source_id FROM labels l JOIN requests r ON r.request_id = l.request_id WHERE r.run_id = ? ORDER BY l.source_id",
            (run_id,),
        ).fetchall()
        latest = connection.execute(
            "SELECT run_id, status, started_at, completed_at, details_json, metrics_json FROM runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        pointer_path = Path(args.state_pointer)
        pointer = json.loads(pointer_path.read_text(encoding="utf-8")) if pointer_path.is_file() else None
        return {
            "smoke_request": dict(smoke) if smoke else None,
            "workload_requests": run_count,
            "workload_labels": len(label_rows),
            "workload_label_rows": [dict(row) for row in label_rows],
            "latest_run": dict(latest) if latest else None,
            "pointer": pointer,
        }
    finally:
        connection.close()


def _assert_restart_persistence(
    before: Any,
    after: Any,
    *,
    request_id: str,
    active_version: str,
) -> dict[str, Any]:
    _require(isinstance(before, dict) and isinstance(after, dict), "Restart persistence snapshots must be JSON objects.")
    before_request = before.get("smoke_request")
    after_request = after.get("smoke_request")
    _require(
        isinstance(before_request, dict)
        and before_request.get("request_id") == request_id
        and before_request.get("lifecycle") == "completed"
        and before_request.get("status_code") == 200,
        "Saved smoke request UUID is missing or was not a successful completed audit row.",
    )
    _require(after_request == before_request, "Saved request UUID/audit row changed across restart.")
    _require(before.get("workload_labels") == after.get("workload_labels"), "Delayed label count changed across restart.")
    _require(before.get("workload_label_rows") == after.get("workload_label_rows"), "Delayed label UUID/source rows changed across restart.")
    pointer_before = before.get("pointer")
    pointer_after = after.get("pointer")
    _require(isinstance(pointer_before, dict), "Serving pointer is missing before restart.")
    _require(isinstance(pointer_after, dict), "Serving pointer is missing after restart.")
    _require(pointer_before.get("version") == active_version, "Serving pointer does not match the active model before restart.")
    _require(pointer_after == pointer_before, "Persisted serving pointer changed across the backend restart.")
    _require(pointer_after.get("version") == active_version, "Persisted serving pointer does not match the active model after restart.")
    return pointer_after


def _controlled_promotion_fixture(
    args: argparse.Namespace,
    parent_version: str,
    eligible_source_ids: set[int],
) -> dict[str, Any]:
    code = (
        "import hashlib,json,os,sys,urllib.request,uuid;"
        "from pathlib import Path;"
        "from model_training.registry import schema_sha256,write_immutable_version;"
        "from model_training.training_config import MODEL_REGISTRY_PATH,SCHEMA_PATH;"
        "parent,seed,eligible_json=sys.argv[1:4]; eligible_ids=json.loads(eligible_json);"
        "archive=Path('/app/models/house_prices_native.zip').read_bytes();"
        "candidate='ci-fixture-'+hashlib.sha256((parent+seed).encode()).hexdigest()[:16];"
        "gate={'policy':'Controlled CI activation fixture only; no model-performance claim','results':{'fixture':True},'accepted':True};"
        "training={'parameters':{'mode':'integration-activation-fixture'},'cutoff':None,'snapshot_hash':hashlib.sha256(('activation-fixture:'+parent+json.dumps(eligible_ids,sort_keys=True)).encode()).hexdigest(),'eligible_ids':eligible_ids,'new_ids':[],'environment':{'purpose':'isolated CI activation path check'}};"
        "version,_,_=write_immutable_version(MODEL_REGISTRY_PATH,archive,parent_version=parent,gate=gate,training=training,schema_sha256=schema_sha256(SCHEMA_PATH),version_override=candidate);"
        "token=os.environ['ADMIN_TOKEN']; activation=str(uuid.uuid4());"
        "payload=json.dumps({'candidate_version':version,'expected_parent':parent,'activation_id':activation}).encode();"
        "request=urllib.request.Request(os.getenv('API_BASE_URL','http://backend:8000').rstrip('/')+'/admin/activate',data=payload,headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'},method='POST');"
        "response=urllib.request.urlopen(request,timeout=30); body=json.loads(response.read());"
        "assert response.status==200 and body.get('status')=='committed' and body.get('active_version')==version,body;"
        "print(json.dumps({'status':'committed','active_version':version,'activation_id':activation,'fixture':'same frozen model bytes; activation path only; no performance claim'}))"
    )
    command = _trainer_command(
        code,
        [parent_version, args.run_id or "smoke", json.dumps(sorted(eligible_source_ids))],
        root=ROOT,
        service=args.trainer_service,
        python=args.python,
        raw_python=True,
    )
    output = _run_cli(command, root=ROOT, timeout=args.trainer_timeout)
    fixture = _parse_cli_json(output, context="Controlled promotion fixture")
    _require(isinstance(fixture, dict), "Controlled promotion fixture did not return a JSON object.")
    _require(fixture.get("status") == "committed", "Controlled promotion fixture was not committed.")
    _require(fixture.get("active_version") != parent_version, "Controlled promotion fixture did not publish a new version.")
    return fixture

def _restart_backend(args: argparse.Namespace) -> None:
    if not args.trainer_service:
        raise SmokeFailure("--exercise-restart requires --trainer-service so the shared DB and state can be checked.")
    command = ["docker", "compose", "restart", "backend"]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120, check=False)
    if completed.returncode:
        detail = (completed.stderr or completed.stdout)[-2000:]
        token = os.environ.get("ADMIN_TOKEN")
        if token:
            detail = detail.replace(token, "[redacted]")
        raise SmokeFailure(f"Backend restart failed: {detail}")


def _set_trainer_running(args: argparse.Namespace, *, running: bool) -> None:
    if not args.trainer_service:
        raise SmokeFailure("A trainer service is required to coordinate the restart check.")
    command = ["docker", "compose"]
    if running:
        command.extend(["start", args.trainer_service])
    else:
        command.extend(["stop", "--timeout", "10", args.trainer_service])
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60, check=False)
    if completed.returncode:
        detail = (completed.stderr or completed.stdout)[-2000:]
        token = os.environ.get("ADMIN_TOKEN")
        if token:
            detail = detail.replace(token, "[redacted]")
        action = "resume" if running else "pause"
        raise SmokeFailure(f"Could not {action} the scheduled trainer for restart verification: {detail}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_id = args.run_id or "smoke-" + uuid.uuid4().hex
    status, health, _, _ = _json_request(args.base_url, "/health")
    _require(status == 200 and isinstance(health, dict), f"API health check failed (HTTP {status}).")

    status, schema, _, _ = _json_request(args.base_url, "/schema")
    _require(status == 200 and isinstance(schema, dict), f"API schema check failed (HTTP {status}).")

    status, before, _, _ = _json_request(
        args.base_url,
        "/monitoring/summary?window_minutes=" + str(args.window_minutes),
    )
    _require(status == 200 and isinstance(before, dict), f"Monitoring summary failed (HTTP {status}).")
    before_requests = _summary_total(before, context="Before smoke prediction")

    payload = {"features": _feature_sample(schema), "run_id": run_id}
    status, prediction, headers, _ = _json_request(
        args.base_url,
        "/predict",
        method="POST",
        payload=payload,
        timeout=args.timeout,
    )
    _require(status == 200 and isinstance(prediction, dict), f"Prediction failed (HTTP {status}).")
    price = prediction.get("sale_price")
    _require(isinstance(price, (int, float)) and math.isfinite(float(price)), "Prediction response has no finite sale_price.")
    request_id = headers.get("x-request-id")
    _require(_is_uuid(request_id), "Prediction response is missing a valid audited request UUID header.")
    _require(_is_uuid(prediction.get("request_id")), "Prediction body is missing a valid audited request UUID.")
    _require(prediction["request_id"] == request_id, "X-Request-ID does not match the prediction response UUID.")
    _require(isinstance(prediction.get("model_version"), str) and prediction["model_version"], "Prediction response is missing its captured model version.")

    status, after, _, _ = _json_request(
        args.base_url,
        "/monitoring/summary?window_minutes=" + str(args.window_minutes),
    )
    _require(status == 200 and isinstance(after, dict), f"Monitoring summary after prediction failed (HTTP {status}).")
    after_requests = _summary_total(after, context="After smoke prediction")
    if args.require_audit:
        _require(after_requests > before_requests, "A successful prediction did not appear in prediction request totals.")

    result: dict[str, Any] = {
        "health": "ok",
        "schema": "ok",
        "prediction": "ok",
        "request_id": request_id,
        "model_version": prediction.get("model_version") or after.get("model", {}).get("version"),
        "audit_total_before": before_requests,
        "audit_total_after": after_requests,
    }

    if args.frontend_url:
        for route, expected in (
            ("/dashboard.html", "Operations dashboard"),
            ("/dashboard.js", "monitoring/summary"),
            ("/", "House Price Predictor"),
        ):
            route_status, _, _, body = _json_request(args.frontend_url, route)
            _require(route_status == 200 and expected in body, f"Frontend route {route} did not serve expected content (HTTP {route_status}).")
        admin_status, _, _, _ = _json_request(args.frontend_url, "/api/admin/activate")
        _require(admin_status == 404, f"Public nginx admin route should return 404, got HTTP {admin_status}.")
        result["frontend_nginx"] = "ok"
        result["public_admin_route"] = "denied"

    if args.exercise_request_events:
        result["request_events"] = _request_event_workload(
            args.base_url,
            run_id=run_id,
            schema=schema,
            timeout=args.timeout,
            database=args.database,
            database_path=args.database_path,
            trainer_service=args.trainer_service,
            python=args.python,
        )

    workload_run_id = run_id + "-adapt"
    labels_imported = None
    import_started_at = None
    imported_source_ids: set[int] = set()
    imported_observation_ids: set[str] = set()
    if args.exercise_adaptation or args.require_timed_retraining:
        _require(args.trainer_service is not None or args.python, "Adaptation smoke requires a trainer CLI or Python executable.")
        _require(args.request_count >= args.minimum_labels, "Request count must meet the configured minimum-label count.")
        output_path = args.container_run_manifest if args.trainer_service else str(ROOT / "tmp" / (workload_run_id + ".json"))
        truth_path = args.container_truth_manifest if args.trainer_service else str(ROOT / "tmp" / (workload_run_id + "-truth.json"))
        workload_api = args.trainer_api_base_url if args.trainer_service else args.base_url
        command = _trainer_command(
            "model_training.generate_workload",
            [
                "--base-url", workload_api,
                "--scenario", "adaptation",
                "--requests", str(args.request_count),
                "--seed", str(args.seed),
                "--run-id", workload_run_id,
                "--concurrency", str(args.workload_concurrency),
                "--label-delay-seconds", str(args.label_delay_seconds),
                "--output", output_path,
                "--truth-output", truth_path,
                "--manifest", args.container_split_manifest if args.trainer_service else str(ROOT / "dataset" / "split_manifest.json"),
            ],
            root=ROOT,
            service=args.trainer_service,
            python=args.python,
        )
        _run_cli(command, root=ROOT, timeout=args.trainer_timeout)
        time.sleep(max(0.0, args.label_delay_seconds) + 0.1)
        import_started_at = time.time()
        command = _trainer_command(
            "model_training.simulate_truth",
            ["--run-manifest", truth_path, "--split-manifest", args.container_split_manifest if args.trainer_service else str(ROOT / "dataset" / "split_manifest.json")],
            root=ROOT,
            service=args.trainer_service,
            python=args.python,
        )
        truth_output = _run_cli(command, root=ROOT, timeout=args.trainer_timeout)
        truth_result = _parse_cli_json(truth_output, context="Delayed truth importer")
        _require(isinstance(truth_result, dict), "Delayed truth importer did not return a JSON object.")
        imported_count = truth_result.get("imported", 0)
        already_imported_count = truth_result.get("already_imported", 0)
        _require(
            isinstance(imported_count, int) and not isinstance(imported_count, bool) and imported_count >= 0
            and isinstance(already_imported_count, int) and not isinstance(already_imported_count, bool) and already_imported_count >= 0,
            "Delayed truth importer did not return numeric nonnegative imported counts.",
        )
        labels_imported = imported_count + already_imported_count
        _require(labels_imported >= args.minimum_labels, f"Delayed truth imported only {labels_imported} labels; need at least {args.minimum_labels}.")
        result["delayed_labels_imported"] = labels_imported
        result["workload_run_id"] = workload_run_id
        if args.require_timed_retraining:
            _require(
                args.trainer_service is not None or args.database is not None,
                "Timed retraining smoke needs --trainer-service or an explicit --database for imported-ID verification.",
            )
            label_snapshot = _database_snapshot(
                args,
                smoke_request_id=request_id,
                run_id=workload_run_id,
            )
            _require(label_snapshot is not None, "Could not inspect persisted delayed labels before waiting for retraining.")
            label_rows = label_snapshot.get("workload_label_rows")
            _require(isinstance(label_rows, list), "Persisted delayed-label snapshot is missing its observation IDs.")
            imported_source_ids = _source_id_set([row.get("source_id") for row in label_rows if isinstance(row, dict)], label="imported")
            imported_observation_ids = {
                row["observation_id"]
                for row in label_rows
                if isinstance(row, dict) and isinstance(row.get("observation_id"), str)
            }
            _require(len(label_rows) == len(imported_source_ids) == len(imported_observation_ids), "Persisted delayed labels contain missing or duplicate source/observation IDs.")
            _require(len(imported_source_ids) >= args.minimum_labels, "Persisted delayed labels do not meet the minimum unique source-ID count.")
            _require(len(imported_source_ids) == labels_imported, "Truth import counts do not match the persisted workload labels.")
            result["imported_source_ids"] = sorted(imported_source_ids)

    if args.require_timed_retraining:
        _require(import_started_at is not None, "Timed retraining smoke did not capture the pre-import timestamp.")
        start_wait = time.monotonic()
        terminal_run = None
        while time.monotonic() - start_wait < args.retrain_timeout:
            current_status, current_summary, _, _ = _json_request(
                args.base_url,
                "/monitoring/summary?window_minutes=" + str(args.window_minutes),
            )
            _require(current_status == 200 and isinstance(current_summary, dict), "Monitoring failed while waiting for timed retraining.")
            candidate_run = _last_run(current_summary)
            started_at = candidate_run.get("started_at") if candidate_run else None
            if isinstance(started_at, (int, float)) and not isinstance(started_at, bool) and started_at >= import_started_at:
                run_status = candidate_run.get("status")
                if run_status == "running":
                    time.sleep(args.retrain_poll_seconds)
                    continue
                terminal_run = _validate_timed_run(
                    candidate_run,
                    imported_at=import_started_at,
                    imported_source_ids=imported_source_ids,
                    minimum_labels=args.minimum_labels,
                )
                break
            time.sleep(args.retrain_poll_seconds)
        _require(terminal_run is not None, f"No eligible automatic retraining verdict appeared within {args.retrain_timeout:.0f} seconds after delayed labels were imported.")
        result["timed_retraining"] = terminal_run

    if args.exercise_promotion_fixture:
        _require(args.trainer_service is not None, "--exercise-promotion-fixture requires --trainer-service.")
        timed_result = result.get("timed_retraining")
        _require(timed_result is not None, "Promotion fixture requires a completed scheduled retraining run.")
        if timed_result.get("status") == "promoted":
            result["promotion_fixture"] = {
                "status": "skipped",
                "reason": "The real scheduled candidate already exercised a successful activation.",
            }
        else:
            current_status, current_summary, _, _ = _json_request(
                args.base_url,
                "/monitoring/summary?window_minutes=" + str(args.window_minutes),
            )
            _require(current_status == 200 and isinstance(current_summary, dict), "Monitoring failed before controlled activation fixture.")
            _summary_total(current_summary, context="Before controlled activation fixture")
            active_model = current_summary.get("model", {})
            parent_version = active_model.get("version") if isinstance(active_model, dict) else None
            _require(isinstance(parent_version, str) and parent_version, "Active model version is unavailable for controlled activation fixture.")
            _require(imported_source_ids, "Controlled promotion fixture requires the delayed workload's imported source IDs.")
            result["promotion_fixture"] = _controlled_promotion_fixture(args, parent_version, imported_source_ids)
    if args.exercise_restart:
        _require(args.trainer_service is not None, "--exercise-restart requires --trainer-service for shared state and scheduler coordination.")
        _set_trainer_running(args, running=False)
        try:
            pre_status, pre_restart_summary, _, _ = _json_request(
                args.base_url,
                "/monitoring/summary?window_minutes=" + str(args.window_minutes),
            )
            _require(pre_status == 200 and isinstance(pre_restart_summary, dict), "Could not read active model immediately before restart.")
            pre_total = _summary_total(pre_restart_summary, context="Immediately before restart")
            pre_model = pre_restart_summary.get("model")
            pre_version = pre_model.get("version") if isinstance(pre_model, dict) else None
            _require(isinstance(pre_version, str) and pre_version, "Active model version is unavailable immediately before restart.")

            persisted_before = _database_snapshot(
                args,
                smoke_request_id=request_id,
                run_id=workload_run_id if labels_imported is not None else run_id,
            )
            _require(persisted_before is not None, "Could not read shared DB/state immediately before restart.")
            smoke_row = persisted_before.get("smoke_request")
            _require(
                isinstance(smoke_row, dict)
                and smoke_row.get("request_id") == request_id
                and smoke_row.get("lifecycle") == "completed"
                and smoke_row.get("status_code") == 200
                and smoke_row.get("model_version") == prediction.get("model_version"),
                "Smoke prediction UUID, success state, or captured model version was not persisted.",
            )
            if labels_imported is not None:
                _require(persisted_before.get("workload_requests", 0) >= args.request_count, "Workload request audit rows are missing before restart.")
                _require(persisted_before.get("workload_labels", 0) == labels_imported, "Delayed labels are missing or duplicated before restart.")
                restart_label_rows = persisted_before.get("workload_label_rows")
                _require(isinstance(restart_label_rows, list), "Restart snapshot is missing persisted label IDs.")
                restart_source_ids = _source_id_set([row.get("source_id") for row in restart_label_rows if isinstance(row, dict)], label="restart")
                restart_observation_ids = {row.get("observation_id") for row in restart_label_rows if isinstance(row, dict)}
                _require(restart_source_ids == imported_source_ids, "Restart snapshot does not contain the exact imported workload source IDs.")
                _require(restart_observation_ids == imported_observation_ids, "Restart snapshot does not contain the exact imported observation UUIDs.")

            pointer_before = persisted_before.get("pointer")
            _require(isinstance(pointer_before, dict), "Serving pointer is missing immediately before restart.")
            _require(pointer_before.get("version") == pre_version, "Serving pointer does not match the fresh active model before restart.")

            _restart_backend(args)
            deadline = time.monotonic() + args.restart_timeout
            restarted_summary = None
            while time.monotonic() < deadline:
                try:
                    restarted_status, candidate, _, _ = _json_request(
                        args.base_url,
                        "/monitoring/summary?window_minutes=" + str(args.window_minutes),
                        timeout=3,
                    )
                    if restarted_status == 200 and isinstance(candidate, dict):
                        restarted_summary = candidate
                        break
                except (OSError, TimeoutError, urllib.error.URLError):
                    pass
                time.sleep(1)
            _require(restarted_summary is not None, "Backend did not become healthy after restart.")
            post_total = _summary_total(restarted_summary, context="After restart")
            post_model = restarted_summary.get("model")
            post_version = post_model.get("version") if isinstance(post_model, dict) else None
            _require(pre_version == post_version, f"Serving model version changed across backend restart ({pre_version!r} -> {post_version!r}).")
            _require(post_total >= pre_total, "Prediction audit count did not persist across backend restart.")

            persisted_after = _database_snapshot(
                args,
                smoke_request_id=request_id,
                run_id=workload_run_id if labels_imported is not None else run_id,
            )
            _require(persisted_after is not None, "Could not inspect DB/state after restart.")
            pointer_after = _assert_restart_persistence(
                persisted_before,
                persisted_after,
                request_id=request_id,
                active_version=post_version,
            )
            result["persisted_request_uuid"] = request_id
            result["persisted_labels"] = persisted_after.get("workload_labels")
            result["state_pointer"] = pointer_after
            result["restart_persistence"] = "ok"
        finally:
            _set_trainer_running(args, running=True)

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Backend base URL, or nginx proxy base ending in /api.")
    parser.add_argument("--frontend-url", default=None, help="Optional nginx frontend origin for static asset/admin-denial probes.")
    parser.add_argument("--window-minutes", type=int, choices=(15, 60, 1440), default=15)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--no-audit-check", dest="require_audit", action="store_false", help="Do not require the audit request total to increase.")
    parser.add_argument("--exercise-request-events", action="store_true", help="Send malformed, repeated, and concurrent prediction requests and verify audit/cache counters.")
    parser.add_argument("--exercise-adaptation", action="store_true", help="Generate real adaptation HTTP traffic and import its delayed source labels.")
    parser.add_argument("--require-timed-retraining", action="store_true", help="Require a scheduler-triggered completed run after delayed labels are available.")
    parser.add_argument("--exercise-restart", action="store_true", help="Restart the Compose backend and verify saved request, labels, counts, and model version.")
    parser.add_argument("--exercise-promotion-fixture", action="store_true", help="If real timed training rejects, activate an explicitly marked same-model artifact fixture to test the real promotion API and pointer.")
    parser.add_argument("--trainer-service", default=None, help="Docker Compose service that carries the trainer image and shared DB/artifact/state volumes.")
    parser.add_argument("--trainer-api-base-url", default="http://backend:8000", help="API address visible from the trainer service.")
    parser.add_argument("--database-path", default="/data/requests.sqlite3")
    parser.add_argument("--state-pointer", default="/state/current.json")
    parser.add_argument("--database", default=None, help="Optional host database path when not using Compose.")
    parser.add_argument("--container-run-manifest", default="/data/smoke-run.json")
    parser.add_argument("--container-truth-manifest", default="/data/smoke-truth.json")
    parser.add_argument("--container-split-manifest", default="/app/dataset/split_manifest.json")
    parser.add_argument("--python", default=None, help="Trainer Python path; defaults to SYS304_PYTHON, the project env, or the current interpreter.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--request-count", type=int, default=40)
    parser.add_argument("--minimum-labels", type=int, default=32)
    parser.add_argument("--workload-concurrency", type=int, default=8)
    parser.add_argument("--label-delay-seconds", type=float, default=1.0)
    parser.add_argument("--trainer-timeout", type=float, default=300.0)
    parser.add_argument("--retrain-timeout", type=float, default=180.0)
    parser.add_argument("--retrain-poll-seconds", type=float, default=2.0)
    parser.add_argument("--restart-timeout", type=float, default=60.0)
    args = parser.parse_args()
    if args.request_count < 1 or args.request_count > 146 or args.minimum_labels < 1 or args.workload_concurrency < 1:
        raise SystemExit("request count must be 1..146, minimum labels positive, and workload concurrency positive")
    if args.label_delay_seconds < 0 or args.trainer_timeout <= 0 or args.retrain_timeout <= 0:
        raise SystemExit("delay must be nonnegative and timeouts positive")
    args.python = _choose_python(args.python)
    if args.require_timed_retraining and not args.exercise_adaptation:
        args.exercise_adaptation = True
    if args.exercise_promotion_fixture:
        args.exercise_adaptation = True
        args.require_timed_retraining = True
    if args.exercise_restart and not args.trainer_service:
        raise SystemExit("--exercise-restart requires --trainer-service")
    if args.require_timed_retraining and args.trainer_service is None and args.database is None:
        raise SystemExit("--require-timed-retraining requires --trainer-service or --database for persisted-label verification")
    return args


def main() -> None:
    try:
        result = run(parse_args())
    except SmokeFailure as error:
        print(json.dumps({"status": "failed", "error": str(error)}, indent=2), file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps({"status": "passed", **result}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


