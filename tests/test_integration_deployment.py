from __future__ import annotations

import json
import re
from argparse import Namespace
from html.parser import HTMLParser
from pathlib import Path

import pytest

import tools.smoke_stack as smoke_stack
from tools.benchmark_logging import aggregate_results, write_report
from tools.smoke_stack import (
    SmokeFailure,
    _assert_request_event_rows,
    _assert_restart_persistence,
    _database_snapshot,
    _is_uuid,
    _parse_cli_json,
    _prediction_attributions,
    _summary_total,
    _validate_cache_activity,
    _validate_timed_run,
)

ROOT = Path(__file__).resolve().parents[1]


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.ids: set[str] = set()
        self.label_targets: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "a" and values.get("href"):
            self.links.append(values["href"])
        if values.get("id"):
            self.ids.add(str(values["id"]))
        if tag == "label" and values.get("for"):
            self.label_targets.add(str(values["for"]))


def parse_page(path: str) -> PageParser:
    parser = PageParser()
    parser.feed((ROOT / path).read_text(encoding="utf-8"))
    return parser


def test_prediction_form_and_dashboard_link_to_each_other() -> None:
    form = parse_page("frontend/index.html")
    dashboard = parse_page("frontend/dashboard.html")

    assert "dashboard.html" in form.links
    assert "index.html" in dashboard.links
    assert "dashboard.html" in dashboard.links
    assert "Operations dashboard" in (ROOT / "frontend/dashboard.html").read_text(encoding="utf-8")


def test_dashboard_controls_and_live_states_are_accessible() -> None:
    parser = parse_page("frontend/dashboard.html")
    page = (ROOT / "frontend/dashboard.html").read_text(encoding="utf-8")
    script = (ROOT / "frontend/dashboard.js").read_text(encoding="utf-8")

    assert "window-minutes" in parser.label_targets
    assert {"refresh-status", "dashboard-error", "api-health", "summary-window-start", "summary-window-end"} <= parser.ids
    assert 'role="status"' in page
    assert 'role="alert"' in page
    assert all(f'value="{minutes}"' in page for minutes in ("15", "60", "1440"))
    assert "const REFRESH_MS = 5000;" in script
    assert '"/monitoring/summary?window_minutes="' in script
    assert "insufficient_data" in script
    assert "No prediction requests in this window." in script
    assert "textContent" in script
    assert "innerHTML" not in script
    assert "handler_p50_ms" in script and "handler_p95_ms" in script
    assert "unknown_count" in script and "missing_count" in script


def test_dashboard_is_in_the_nginx_image_and_admin_routes_are_denied() -> None:
    dockerfile = (ROOT / "frontend/Dockerfile").read_text(encoding="utf-8")
    nginx = (ROOT / "frontend/nginx.conf").read_text(encoding="utf-8")

    assert "frontend/dashboard.html /usr/share/nginx/html/dashboard.html" in dockerfile
    assert "frontend/dashboard.js /usr/share/nginx/html/dashboard.js" in dockerfile
    assert "location = /api/admin" in nginx
    assert "location ^~ /api/admin/" in nginx
    assert nginx.count("return 404;") >= 2
    assert "proxy_pass http://backend:8000/;" in nginx


def test_compose_has_persistent_volume_permissions_and_safe_defaults() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "dockerfile: model_training/Dockerfile" in compose
    assert "condition: service_completed_successfully" in compose
    assert "condition: service_healthy" in compose
    assert "model_artifacts:/artifacts:ro" in compose
    assert "serving_state:/state:ro" in compose
    assert "request_data:/data" in compose
    assert "local-demo-only-change-this-value" in compose
    assert 'PREDICTION_CACHE_ENABLED: "1"' in compose
    assert 'DYNAMIC_BATCHING_ENABLED: "0"' in compose
    assert 'WEB_CONCURRENCY: "1"' in compose
    assert re.search(r"127\.0\.0\.1:3000:80", compose)
    assert re.search(r"127\.0\.0\.1:8000:8000", compose)


def test_linux_ci_runs_container_smoke_uploads_logs_and_cleans_up() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "docker compose config -q" in workflow
    assert "docker compose up --build" in workflow
    assert "tools/smoke_stack.py" in workflow
    assert "upload-artifact" in workflow
    assert "if: always()" in workflow
    assert "docker compose down --volumes --remove-orphans" in workflow


def test_benchmarks_use_isolated_temporary_servers_and_real_http() -> None:
    smoke = (ROOT / "tools/smoke_stack.py").read_text(encoding="utf-8")
    benchmark = (ROOT / "tools/benchmark_logging.py").read_text(encoding="utf-8")

    assert "urllib.request.urlopen" in smoke
    assert "model_training.generate_workload" in smoke
    assert "model_training.simulate_truth" in smoke
    assert "httpx.AsyncClient" in benchmark
    assert "TemporaryDirectory" in benchmark
    assert '"unique", "repeated"' in benchmark
    assert '"client_p50_ms"' in benchmark
    assert '"client_p95_ms"' in benchmark
    assert '"successful_requests_per_second"' in benchmark
def test_docker_context_allows_trainer_image_inputs() -> None:
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")

    for allowed in (
        "!dataset/train.csv",
        "!dataset/test.csv",
        "!dataset/split_manifest.json",
        "!model_training/*.py",
        "!model_training/requirements-trainer.txt",
        "!models/baseline_xgb_pipeline.pkl",
        "!models/drift_reference.json",
    ):
        assert allowed in dockerignore


def test_dashboard_behavior_regressions_are_part_of_ci() -> None:
    test_file = ROOT / "tests/dashboard_behavior.test.cjs"
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert test_file.is_file()
    assert "node --test tests/dashboard_behavior.test.cjs" in workflow


def test_smoke_requires_numeric_audit_totals_and_real_request_uuids() -> None:
    assert _summary_total({"requests": {"total": 4}}, context="test") == 4
    for malformed_total in (None, True, -1, 4.5, "4"):
        with pytest.raises(SmokeFailure, match="requests.total"):
            _summary_total({"requests": {"total": malformed_total}}, context="test")
    assert _is_uuid("cb6c6540-9f92-4d4c-b342-9700fbd3c93a")
    assert not _is_uuid("request-123")


def test_cli_json_parser_accepts_formatted_document_and_rejects_trailing_or_truncated_text() -> None:
    formatted = '{\n  "imported": 40,\n  "already_imported": 0\n}\n'
    assert _parse_cli_json(formatted, context="truth importer") == {
        "imported": 40,
        "already_imported": 0,
    }
    for invalid in ('{\n  "imported": 40,\n}\n', '{"imported":40}\nextra output\n'):
        with pytest.raises(SmokeFailure, match="truth importer returned invalid JSON"):
            _parse_cli_json(invalid, context="truth importer")


def test_request_event_smoke_accepts_cache_hit_or_coalescing_and_checks_each_identity() -> None:
    request_a = "cb6c6540-9f92-4d4c-b342-9700fbd3c93a"
    request_b = "ebf0a5c0-6407-4f89-8f61-13d68c91f41b"
    responses = [
        (200, {"request_id": request_a, "model_version": "bootstrap-a"}, {"x-request-id": request_a}, "{}"),
        (200, {"request_id": request_b, "model_version": "candidate-b"}, {"x-request-id": request_b}, "{}"),
    ]

    attributions = _prediction_attributions(responses, context="test")
    assert attributions == [
        {"request_id": request_a, "model_version": "bootstrap-a"},
        {"request_id": request_b, "model_version": "candidate-b"},
    ]
    assert _validate_cache_activity(miss_delta=1, hit_delta=1, coalesced_delta=0)["reused_after_miss"] == 1
    assert _validate_cache_activity(miss_delta=1, hit_delta=0, coalesced_delta=1)["reused_after_miss"] == 1

    with pytest.raises(SmokeFailure, match="neither a cache hit nor an in-flight coalesced"):
        _validate_cache_activity(miss_delta=1, hit_delta=0, coalesced_delta=0)
    duplicate = [responses[0], (200, {"request_id": request_a, "model_version": "bootstrap-a"}, {"x-request-id": request_a}, "{}")]
    with pytest.raises(SmokeFailure, match="independent UUIDs"):
        _prediction_attributions(duplicate, context="test")


def test_request_event_smoke_matches_http_attributions_to_completed_audit_rows() -> None:
    success_id = "cb6c6540-9f92-4d4c-b342-9700fbd3c93a"
    malformed_id = "ebf0a5c0-6407-4f89-8f61-13d68c91f41b"
    attributions = [{"request_id": success_id, "model_version": "bootstrap-a"}]
    rows = [
        {"request_id": success_id, "path": "/predict", "lifecycle": "completed", "status_code": 200, "model_version": "bootstrap-a"},
        {"request_id": malformed_id, "path": "/predict", "lifecycle": "completed", "status_code": 400, "model_version": None},
    ]

    _assert_request_event_rows(rows, attributions=attributions, malformed_id=malformed_id, malformed_status=400)
    with pytest.raises(SmokeFailure, match="missing from the persisted audit rows"):
        _assert_request_event_rows(rows[:1], attributions=attributions, malformed_id=malformed_id, malformed_status=400)
    wrong_version = [{**rows[0], "model_version": "another-version"}, rows[1]]
    with pytest.raises(SmokeFailure, match="lost its serving-version attribution"):
        _assert_request_event_rows(wrong_version, attributions=attributions, malformed_id=malformed_id, malformed_status=400)


def test_restart_database_inspection_uses_backend_while_trainer_is_paused(monkeypatch: pytest.MonkeyPatch) -> None:
    args = Namespace(
        trainer_service="trainer",
        database_path="/data/requests.sqlite3",
        state_pointer="/state/current.json",
        python="python",
    )
    captured: dict[str, list[str]] = {}

    def fake_run_cli(command: list[str], *, root: Path, timeout: float = 180.0) -> str:
        captured["command"] = command
        if command[4] != "backend":
            raise SmokeFailure("trainer service is paused")
        return json.dumps({"smoke_request": None, "pointer": {"version": "bootstrap-a"}})

    monkeypatch.setattr(smoke_stack, "_run_cli", fake_run_cli)
    snapshot = _database_snapshot(
        args,
        smoke_request_id="cb6c6540-9f92-4d4c-b342-9700fbd3c93a",
        run_id="run-1",
    )

    assert snapshot == {"smoke_request": None, "pointer": {"version": "bootstrap-a"}}
    assert captured["command"][:6] == ["docker", "compose", "exec", "-T", "backend", "python"]


def test_timed_smoke_accepts_policy_verdict_only_when_imported_ids_are_in_snapshot() -> None:
    run = {
        "run_id": "run-1",
        "status": "rejected",
        "started_at": 20.0,
        "completed_at": 22.0,
        "details": {"label_count": 3, "eligible_ids": [10, 11, 12], "new_ids": [10, 11, 12]},
    }
    verdict = _validate_timed_run(run, imported_at=19.0, imported_source_ids={10, 12}, minimum_labels=2)
    assert verdict["status"] == "rejected"
    assert verdict["eligible_source_ids"] == [10, 11, 12]

    for status in ("failed", "skipped", "mystery"):
        failed = {**run, "status": status}
        with pytest.raises(SmokeFailure, match="unsupported status"):
            _validate_timed_run(failed, imported_at=19.0, imported_source_ids={10, 12}, minimum_labels=2)

    incomplete = {**run, "details": {**run["details"], "new_ids": [10, 11]}}
    with pytest.raises(SmokeFailure, match="new eligible labels"):
        _validate_timed_run(incomplete, imported_at=19.0, imported_source_ids={10, 12}, minimum_labels=2)


def test_restart_smoke_requires_exact_pointer_uuid_and_label_persistence() -> None:
    pointer = {"version": "candidate-a", "activation_id": "activation-a", "previous_version": "bootstrap-a"}
    before = {
        "smoke_request": {"request_id": "cb6c6540-9f92-4d4c-b342-9700fbd3c93a", "lifecycle": "completed", "status_code": 200},
        "workload_labels": 2,
        "workload_label_rows": [{"observation_id": "dataset:1", "source_id": 1}, {"observation_id": "dataset:2", "source_id": 2}],
        "pointer": pointer,
    }
    after = {**before, "pointer": dict(pointer)}
    assert _assert_restart_persistence(
        before,
        after,
        request_id=before["smoke_request"]["request_id"],
        active_version="candidate-a",
    ) == pointer

    changed_pointer = {**pointer, "activation_id": "activation-b"}
    with pytest.raises(SmokeFailure, match="pointer changed"):
        _assert_restart_persistence(
            before,
            {**before, "pointer": changed_pointer},
            request_id=before["smoke_request"]["request_id"],
            active_version="candidate-a",
        )
    with pytest.raises(SmokeFailure, match="missing after restart"):
        _assert_restart_persistence(
            before,
            {**before, "pointer": None},
            request_id=before["smoke_request"]["request_id"],
            active_version="candidate-a",
        )


def test_all_error_benchmark_group_writes_null_latency_and_raw_errors(tmp_path: Path) -> None:
    trial = {
        "workload": "unique",
        "concurrency": 8,
        "trial": 1,
        "request_logging_enabled": True,
        "client_p50_ms": None,
        "client_p95_ms": None,
        "successful_requests_per_second": 0.0,
        "errors": 3,
        "error_samples": ["HTTP 503", "HTTP 503", "HTTP 500"],
    }
    document = {"summary": aggregate_results([trial]), "trials": [trial]}
    output = tmp_path / "benchmark.json"
    write_report(output, document)
    stored = json.loads(output.read_text(encoding="utf-8"))
    group = stored["summary"]["unique|c8|logging-1"]
    assert group["median_client_p50_ms"] is None
    assert group["median_client_p95_ms"] is None
    assert group["errors"] == 3
    assert stored["trials"][0]["error_samples"] == ["HTTP 503", "HTTP 503", "HTTP 500"]

