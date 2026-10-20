"""Send reproducible workload scenarios to an already-running prediction API."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import random
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from model_training.data_partitions import (
    DEFAULT_DATASET,
    DEFAULT_MANIFEST,
    IDENTITY,
    TARGET,
    load_manifest,
    sha256_file,
)
from model_training.training_config import DATA_DIR

SCENARIOS = ("baseline", "repeated", "adaptation", "drift", "malformed", "unlabeled")


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, np.generic):
        value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            return None
    return value


def _feature_record(row: pd.Series, feature_names: list[str]) -> dict[str, Any]:
    return {name: _json_value(row[name]) for name in feature_names}


def _dataset_records(
    dataset_path: str | Path,
    manifest_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any], str]:
    manifest = load_manifest(manifest_path, dataset_path=dataset_path)
    if sha256_file(dataset_path) != manifest["source"]["sha256"]:
        raise ValueError("Source dataset hash does not match split manifest")
    return pd.read_csv(dataset_path), manifest, manifest["source"]["sha256"]


def build_scenario_calls(
    scenario: str,
    count: int,
    seed: int,
    *,
    dataset_path: str | Path = DEFAULT_DATASET,
    test_path: str | Path | None = None,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    drift_kind: str = "mixed",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    """Return request specs, delayed target rows, and source data hash."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unsupported scenario {scenario!r}")
    if count < 1:
        raise ValueError("Request count must be positive")
    if drift_kind not in {"numeric", "categorical", "missing", "mixed"}:
        raise ValueError("drift_kind must be numeric, categorical, missing, or mixed")
    rng = random.Random(seed)
    test_path = Path(test_path) if test_path else Path(dataset_path).with_name("test.csv")
    source_frame, manifest, source_hash = _dataset_records(dataset_path, manifest_path)
    feature_names = manifest["source"]["feature_names"]
    calls: list[dict[str, Any]] = []
    truth: dict[int, dict[str, Any]] = {}

    if scenario == "malformed":
        malformed = [
            (b"{", "application/json"),
            (b"null", "application/json"),
            (b'{"features":[]}', "application/json"),
            (b'{"features":{}}', "application/json"),
            (b'{"features":{"OverallQual":"bad"}}', "application/json"),
            (b"", "application/json"),
        ]
        for index in range(count):
            body, content_type = malformed[index % len(malformed)]
            calls.append({"body": body, "content_type": content_type, "source_id": None, "observation": None})
        return calls, [], source_hash

    if scenario == "unlabeled":
        if not test_path.exists():
            raise FileNotFoundError(f"Unlabeled Kaggle test file not found: {test_path}")
        test_frame = pd.read_csv(test_path)
        if TARGET in test_frame.columns or IDENTITY not in test_frame.columns:
            raise ValueError("Unlabeled test input must have Id and must not contain SalePrice")
        positions = list(range(len(test_frame)))
        rng.shuffle(positions)
        if count > len(positions):
            positions.extend(rng.choices(range(len(test_frame)), k=count - len(positions)))
        for position in positions[:count]:
            row = test_frame.iloc[position]
            record = {name: _json_value(row[name]) for name in feature_names}
            calls.append({"features": record, "source_id": int(row[IDENTITY]), "observation": None})
        return calls, [], sha256_file(test_path)

    partition_name = {
        "baseline": "original_fit_ids",
        "repeated": "adaptation_ids",
        "adaptation": "adaptation_ids",
        "drift": "adaptation_ids",
    }[scenario]
    eligible_ids = manifest["partitions"][partition_name]
    by_id = source_frame.set_index(IDENTITY, drop=False)
    if scenario == "adaptation" and count > len(eligible_ids):
        raise ValueError(f"Adaptation scenario supports at most {len(eligible_ids)} unique rows")
    if scenario == "repeated":
        chosen_ids = [rng.choice(eligible_ids[: min(16, len(eligible_ids))]) for _ in range(count)]
    elif scenario == "drift":
        chosen_ids = [rng.choice(eligible_ids) for _ in range(count)]
    elif scenario == "baseline" and count > len(eligible_ids):
        chosen_ids = []
        while len(chosen_ids) < count:
            cycle = list(eligible_ids)
            rng.shuffle(cycle)
            chosen_ids.extend(cycle[: count - len(chosen_ids)])
    else:
        chosen_ids = list(eligible_ids)
        rng.shuffle(chosen_ids)
        chosen_ids = chosen_ids[:count]

    numeric_features = manifest["source"]["numeric_features"]
    categorical_features = manifest["source"]["categorical_features"]
    for source_id in chosen_ids:
        row = by_id.loc[int(source_id)]
        record = _feature_record(row, feature_names)
        if scenario == "drift":
            mode = drift_kind if drift_kind != "mixed" else ("numeric", "categorical", "missing")[len(calls) % 3]
            if mode in {"numeric", "mixed"} and numeric_features:
                numeric_name = "GrLivArea" if "GrLivArea" in numeric_features else numeric_features[0]
                base = record.get(numeric_name)
                record[numeric_name] = float(base or 0) * 3.0 + 1000.0
            if mode in {"categorical", "mixed"} and categorical_features:
                category_name = "Neighborhood" if "Neighborhood" in categorical_features else categorical_features[0]
                record[category_name] = f"unseen_category_{seed}"
            if mode in {"missing", "mixed"}:
                for name in feature_names:
                    if name not in {"OverallQual", "GrLivArea"}:
                        record[name] = None
        has_observation = scenario in {"baseline", "repeated", "adaptation"}
        observation = {"dataset_sha256": source_hash, "source_id": int(source_id)} if has_observation else None
        calls.append({"features": record, "source_id": int(source_id), "observation": observation})
        if scenario in {"adaptation", "repeated"}:
            truth.setdefault(int(source_id), {
                "source_id": int(source_id),
                "sale_price": float(row[TARGET]),
                "request_ids": [],
                "available_at": None,
            })
    return calls, list(truth.values()), source_hash


def _post(base_url: str, call: dict[str, Any], run_id: str, timeout: float) -> dict[str, Any]:
    start = time.perf_counter()
    completed_at = time.time()
    body = call.get("body")
    if body is None:
        payload = {"features": call["features"]}
        if call.get("observation") is not None:
            payload["observation"] = call["observation"]
        payload["run_id"] = run_id
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/predict",
        data=body,
        headers={"Content-Type": call.get("content_type", "application/json"), "Accept": "application/json"},
        method="POST",
    )
    status: int | None = None
    response_text = ""
    response_headers: Any = {}
    transport_error = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = int(response.status)
            response_headers = response.headers
            response_text = response.read(4096).decode("utf-8", errors="replace")
            completed_at = time.time()
    except urllib.error.HTTPError as error:
        status = int(error.code)
        response_headers = error.headers
        response_text = error.read(4096).decode("utf-8", errors="replace")
        completed_at = time.time()
    except (OSError, TimeoutError, urllib.error.URLError) as error:
        transport_error = f"{type(error).__name__}: {error}"
        completed_at = time.time()
    try:
        response_json = json.loads(response_text) if response_text else None
    except json.JSONDecodeError:
        response_json = None
    request_id = response_headers.get("X-Request-ID") if response_headers else None
    if not request_id and isinstance(response_json, dict):
        request_id = response_json.get("request_id")
    return {
        "request_id": request_id,
        "source_id": call.get("source_id"),
        "status_code": status,
        "model_version": response_json.get("model_version") if isinstance(response_json, dict) else None,
        "response": response_json,
        "response_text": response_text if response_json is None else None,
        "transport_error": transport_error,
        "client_latency_ms": (time.perf_counter() - start) * 1000.0,
        "completed_at": completed_at,
    }


def _latency_summary(results: list[dict[str, Any]]) -> dict[str, float | None]:
    values = [float(item["client_latency_ms"]) for item in results if item["status_code"] is not None]
    if not values:
        return {"p50_ms": None, "p95_ms": None, "mean_ms": None}
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "mean_ms": float(np.mean(values)),
    }


def run_workload(
    *,
    base_url: str,
    scenario: str,
    request_count: int | None = 100,
    duration_seconds: float | None = None,
    seed: int = 42,
    run_id: str | None = None,
    concurrency: int = 1,
    rate: float = 0.0,
    timeout: float = 10.0,
    label_delay_seconds: float = 300.0,
    output_path: str | Path | None = None,
    truth_output_path: str | Path | None = None,
    drift_kind: str = "mixed",
    dataset_path: str | Path = DEFAULT_DATASET,
    test_path: str | Path | None = None,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    if (
        concurrency < 1
        or not math.isfinite(rate)
        or rate < 0
        or not math.isfinite(label_delay_seconds)
        or label_delay_seconds < 0
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("Concurrency must be positive and rates/delays nonnegative")
    if duration_seconds is not None:
        if request_count is not None:
            raise ValueError("Choose request_count or duration_seconds, not both")
        if not math.isfinite(duration_seconds) or duration_seconds <= 0:
            raise ValueError("duration_seconds must be finite and positive")
        request_count = max(1, math.floor(duration_seconds * rate)) if rate > 0 else max(1, math.ceil(duration_seconds * concurrency))
        if rate == 0:
            rate = request_count / duration_seconds
    if request_count is None:
        request_count = 100
    run_id = run_id or str(uuid.uuid4())
    calls, truth_rows, dataset_hash = build_scenario_calls(
        scenario,
        request_count,
        seed,
        dataset_path=dataset_path,
        test_path=test_path,
        manifest_path=manifest_path,
        drift_kind=drift_kind,
    )
    start_clock = time.perf_counter()

    def send(index_call: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        index, call = index_call
        if rate > 0:
            wait = index / rate - (time.perf_counter() - start_clock)
            if wait > 0:
                time.sleep(wait)
        result = _post(base_url, call, run_id, timeout)
        result["scenario"] = scenario
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(send, enumerate(calls)))
    for item in results:
        source_id = item.get("source_id")
        if source_id is None or source_id not in {truth["source_id"] for truth in truth_rows}:
            continue
        for truth in truth_rows:
            if truth["source_id"] == source_id:
                if item.get("request_id"):
                    truth["request_ids"].append(item["request_id"])
                if item["status_code"] is not None and 200 <= item["status_code"] < 300:
                    completion = float(item["completed_at"]) + label_delay_seconds
                    truth["available_at"] = completion if truth["available_at"] is None else min(truth["available_at"], completion)
                break
    attempted = len(calls)
    response_count = sum(item["status_code"] is not None for item in results)
    run_manifest = {
        "format": "house-prices-workload-v1",
        "run_id": run_id,
        "scenario": scenario,
        "base_url": base_url,
        "seed": seed,
        "duration_seconds": duration_seconds,
        "scheduled_rate_per_second": rate,
        "requested_calls": attempted,
        "responses_received": response_count,
        "request_ids": [item["request_id"] for item in results if item["request_id"]],
        "client_latency": _latency_summary(results),
        "serving_versions": sorted({item["model_version"] for item in results if item["model_version"]}),
        "results": results,
        "dataset_sha256": dataset_hash,
        "created_at": time.time(),
    }
    if truth_rows:
        private_path = Path(truth_output_path) if truth_output_path else DATA_DIR / "private_truth" / f"{run_id}.json"
        private_path.parent.mkdir(parents=True, exist_ok=True)
        truth_manifest = {
            "format": "house-prices-delayed-truth-v1",
            "run_id": run_id,
            "dataset_sha256": dataset_hash,
            "delay_seconds": label_delay_seconds,
            "labels": truth_rows,
        }
        temporary = private_path.with_suffix(private_path.suffix + ".tmp")
        temporary.write_text(json.dumps(truth_manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, private_path)
        run_manifest["private_truth_manifest"] = str(private_path)
        run_manifest["private_truth_labels"] = len(truth_rows)
    if output_path is None:
        output_path = DATA_DIR / "workloads" / f"{run_id}.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(run_manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    return run_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--requests", type=int, default=None)
    parser.add_argument("--duration-seconds", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--rate", type=float, default=0.0, help="Requests per second; 0 sends as quickly as workers allow")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--label-delay-seconds", type=float, default=300.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--truth-output", type=Path, default=None)
    parser.add_argument("--drift-kind", choices=("numeric", "categorical", "missing", "mixed"), default="mixed")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--test-dataset", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    if args.requests is not None and args.duration_seconds is not None:
        parser.error("use either --requests or --duration-seconds")
    if args.requests is None and args.duration_seconds is None:
        args.requests = 100
    manifest = run_workload(
        base_url=args.base_url,
        scenario=args.scenario,
        request_count=args.requests,
        duration_seconds=args.duration_seconds,
        seed=args.seed,
        run_id=args.run_id,
        concurrency=args.concurrency,
        rate=args.rate,
        timeout=args.timeout,
        label_delay_seconds=args.label_delay_seconds,
        output_path=args.output,
        truth_output_path=args.truth_output,
        drift_kind=args.drift_kind,
        dataset_path=args.dataset,
        test_path=args.test_dataset,
        manifest_path=args.manifest,
    )
    print(json.dumps({key: manifest[key] for key in ("run_id", "scenario", "requested_calls", "responses_received", "serving_versions", "private_truth_labels") if key in manifest}, indent=2))


if __name__ == "__main__":
    main()
