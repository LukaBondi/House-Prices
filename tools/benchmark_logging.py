#!/usr/bin/env python3
"""Measure matched client-visible prediction latency with request logging on/off.

Every trial starts a fresh local Uvicorn process with a temporary SQLite DB,
artifact directory, and serving state directory. It uses identical real
dataset records for the logging-enabled and logging-disabled cases.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT_PYTHON = Path(r"C:\Users\lukab\.conda\envs\sys-304\python.exe")
NATIVE = ROOT / "models" / "house_prices_native.zip"
SCHEMA = ROOT / "models" / "feature_schema.json"
DATASET = ROOT / "dataset" / "train.csv"


def choose_python(value: str | None) -> str:
    if value:
        return value
    configured = os.environ.get("SYS304_PYTHON")
    if configured:
        return configured
    if DEFAULT_PROJECT_PYTHON.is_file():
        return str(DEFAULT_PROJECT_PYTHON)
    return sys.executable


def choose_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    ratio = position - lower
    return ordered[lower] * (1 - ratio) + ordered[upper] * ratio


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_records() -> list[dict[str, Any]]:
    import pandas as pd

    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    frame = pd.read_csv(DATASET)
    frame = frame.drop(columns=["Id", "SalePrice"], errors="ignore")
    feature_names = schema.get("feature_names")
    if isinstance(feature_names, list):
        frame = frame.reindex(columns=feature_names)
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        normalized: dict[str, Any] = {}
        for key, value in row.items():
            if value is None or bool(pd.isna(value)):
                normalized[key] = None
            elif hasattr(value, "item"):
                normalized[key] = value.item()
            else:
                normalized[key] = value
        records.append(normalized)
    if len(records) < 10:
        raise RuntimeError("The source dataset must provide at least ten real feature rows.")
    return records


def make_payloads(records: list[dict[str, Any]], workload: str, count: int) -> list[dict[str, Any]]:
    if workload == "unique" and count > len(records):
        raise ValueError(f"unique workload has {len(records)} rows; request count cannot exceed that")
    source = records[:10] if workload == "repeated" else records
    payloads = []
    for index in range(count):
        record = dict(source[index % len(source)])
        if workload == "unique":
            base = record.get("LotArea")
            record["LotArea"] = float(base or 0) + (index + 1) / 1_000_000
        payloads.append(record)
    return payloads

def ensure_drift_reference(output_path: Path) -> None:
    if (ROOT / "models" / "drift_reference.json").is_file():
        output_path.write_bytes((ROOT / "models" / "drift_reference.json").read_bytes())
        return
    from model_training.build_drift_reference import write_reference
    from model_training.data_partitions import write_manifest

    manifest_path = output_path.parent / "split_manifest.json"
    write_manifest(manifest_path)
    write_reference(output_path, manifest_path=manifest_path)


async def get_json(client: Any, url: str) -> dict[str, Any]:
    response = await client.get(url)
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("Health endpoint returned a non-object response.")
    return body


async def measure_requests(
    base_url: str,
    payloads: list[dict[str, Any]],
    concurrency: int,
    timeout: float,
) -> dict[str, Any]:
    import httpx

    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    client_timeout = httpx.Timeout(timeout, connect=min(timeout, 10.0))
    latencies: list[float] = []
    errors = 0
    error_samples: list[str] = []

    async with httpx.AsyncClient(limits=limits, timeout=client_timeout) as client:
        health = await get_json(client, base_url + "/health")
        if health.get("model_loaded") is False:
            raise RuntimeError("Fresh local API did not report a loaded model.")
        await client.get(base_url + "/schema")

        async def worker(worker_index: int) -> None:
            nonlocal errors
            for request_index in range(worker_index, len(payloads), concurrency):
                started = time.perf_counter()
                try:
                    response = await client.post(
                        base_url + "/predict",
                        json={"features": payloads[request_index]},
                    )
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    if response.status_code != 200:
                        errors += 1
                        if len(error_samples) < 5:
                            error_samples.append("HTTP " + str(response.status_code))
                        continue
                    try:
                        body = response.json()
                    except ValueError:
                        body = None
                    price = body.get("sale_price") if isinstance(body, dict) else None
                    if not isinstance(price, (int, float)) or not math.isfinite(float(price)):
                        errors += 1
                        if len(error_samples) < 5:
                            error_samples.append("invalid prediction response")
                        continue
                    latencies.append(elapsed_ms)
                except Exception as error:
                    errors += 1
                    if len(error_samples) < 5:
                        error_samples.append(type(error).__name__)

        started = time.perf_counter()
        await asyncio.gather(*(worker(index) for index in range(concurrency)))
        elapsed = time.perf_counter() - started

    return {
        "requested": len(payloads),
        "succeeded": len(latencies),
        "errors": errors,
        "error_samples": error_samples,
        "elapsed_seconds": elapsed,
        "successful_requests_per_second": len(latencies) / elapsed if elapsed else 0.0,
        "client_p50_ms": percentile(latencies, 0.50),
        "client_p95_ms": percentile(latencies, 0.95),
    }


def start_server(
    *,
    python: str,
    port: int,
    logging_enabled: bool,
    temporary_root: Path,
    reference_path: Path,
    timeout: float,
) -> tuple[subprocess.Popen[Any], Any, float]:
    base_url = f"http://127.0.0.1:{port}"
    state_root = temporary_root / "state"
    artifact_root = temporary_root / "artifacts"
    data_root = temporary_root / "data"
    state_root.mkdir(parents=True)
    artifact_root.mkdir(parents=True)
    data_root.mkdir(parents=True)
    environment = os.environ.copy()
    environment.update(
        {
            "MODEL_PATH": str(NATIVE),
            "SCHEMA_PATH": str(SCHEMA),
            "DATA_DIR": str(data_root),
            "DB_PATH": str(data_root / "requests.sqlite3"),
            "MODEL_REGISTRY_PATH": str(artifact_root),
            "MODEL_STATE_PATH": str(state_root),
            "DRIFT_REFERENCE_PATH": str(reference_path),
            "REQUEST_LOGGING_ENABLED": "1" if logging_enabled else "0",
            "ADMIN_TOKEN": "temporary-" + uuid.uuid4().hex,
            "PREDICTION_CACHE_ENABLED": "1",
            "DYNAMIC_BATCHING_ENABLED": "0",
            "WEB_CONCURRENCY": "1",
            "INFERENCE_WORKERS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONPATH": os.pathsep.join(
                [str(ROOT), environment.get("PYTHONPATH", "")]
            ),
        }
    )
    log = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
    command = [
        python,
        "-m",
        "uvicorn",
        "backend.app:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--workers",
        "1",
        "--no-access-log",
        "--log-level",
        "warning",
    ]
    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=log)
    started = time.perf_counter()
    try:
        import httpx

        deadline = time.monotonic() + timeout
        with httpx.Client(timeout=1.0) as client:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    log.seek(0)
                    raise RuntimeError(f"Uvicorn exited with {process.returncode}: {log.read()[-3000:]}")
                try:
                    response = client.get(base_url + "/health")
                    if response.status_code == 200:
                        return process, log, time.perf_counter() - started
                except httpx.HTTPError:
                    pass
                time.sleep(0.1)
        raise TimeoutError("Fresh local Uvicorn process did not become healthy.")
    except BaseException:
        stop_server(process, log)
        raise


def stop_server(process: subprocess.Popen[Any], log: Any) -> None:
    try:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    finally:
        log.close()


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize matched trials while retaining null latency for all-error groups."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        key = f"{item['workload']}|c{item['concurrency']}|logging-{int(item['request_logging_enabled'])}"
        grouped.setdefault(key, []).append(item)

    def median_or_none(values: list[float | None]) -> float | None:
        available = [value for value in values if value is not None]
        return statistics.median(available) if available else None

    return {
        key: {
            "trials": len(cases),
            "median_client_p50_ms": median_or_none([item["client_p50_ms"] for item in cases]),
            "median_client_p95_ms": median_or_none([item["client_p95_ms"] for item in cases]),
            "median_successful_requests_per_second": median_or_none(
                [item["successful_requests_per_second"] for item in cases]
            ),
            "errors": sum(item["errors"] for item in cases),
        }
        for key, cases in grouped.items()
    }


def write_report(output_path: Path, document: dict[str, Any]) -> None:
    """Persist raw trials and summary before the caller reports request failures."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_case(
    *,
    python: str,
    workload: str,
    concurrency: int,
    trial: int,
    logging_enabled: bool,
    payloads: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="house-prices-logging-") as temp_name:
        temporary_root = Path(temp_name)
        reference = temporary_root / "drift_reference.json"
        ensure_drift_reference(reference)
        port = choose_port()
        process, log, startup_seconds = start_server(
            python=python,
            port=port,
            logging_enabled=logging_enabled,
            temporary_root=temporary_root,
            reference_path=reference,
            timeout=args.startup_timeout,
        )
        try:
            measured = asyncio.run(
                measure_requests(
                    f"http://127.0.0.1:{port}",
                    payloads,
                    concurrency,
                    args.request_timeout,
                )
            )
            return {
                "workload": workload,
                "concurrency": concurrency,
                "trial": trial,
                "request_logging_enabled": logging_enabled,
                "startup_seconds": startup_seconds,
                **measured,
            }
        finally:
            stop_server(process, log)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=None, help="Defaults to SYS304_PYTHON, the project env, or the current interpreter.")
    parser.add_argument("--workloads", nargs="+", choices=("unique", "repeated"), default=["unique", "repeated"])
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--requests", type=int, default=160)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--startup-timeout", type=float, default=45.0)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=ROOT / "tmp" / "benchmark_logging.json")
    args = parser.parse_args()
    if args.requests < 1 or args.trials < 1 or not args.concurrency or min(args.concurrency) < 1:
        raise SystemExit("requests, trials, and concurrency values must be positive")
    if args.startup_timeout <= 0 or args.request_timeout <= 0:
        raise SystemExit("timeouts must be positive")
    args.python = choose_python(args.python)
    return args


def main() -> None:
    args = parse_args()
    if not NATIVE.is_file() or not SCHEMA.is_file() or not DATASET.is_file():
        raise SystemExit("Native model archive, feature schema, and source dataset are required.")
    records = read_records()
    results: list[dict[str, Any]] = []
    total = len(args.workloads) * len(args.concurrency) * args.trials * 2
    index = 0
    for workload in args.workloads:
        payloads = make_payloads(records, workload, args.requests)
        for concurrency in args.concurrency:
            for trial in range(1, args.trials + 1):
                for logging_enabled in (True, False):
                    index += 1
                    label = "on" if logging_enabled else "off"
                    print(
                        f"[{index}/{total}] workload={workload} concurrency={concurrency} "
                        f"trial={trial} logging={label}",
                        flush=True,
                    )
                    results.append(
                        run_case(
                            python=args.python,
                            workload=workload,
                            concurrency=concurrency,
                            trial=trial,
                            logging_enabled=logging_enabled,
                            payloads=payloads,
                            args=args,
                        )
                    )

    document = {
        "benchmark": "house_prices_logging_overhead",
        "schema_version": 1,
        "method": "Fresh local one-worker Uvicorn process and isolated temporary DB/artifact/state for each matched case; identical dataset payloads and cache settings; client timings include full HTTP round trip.",
        "settings": {
            "workloads": args.workloads,
            "concurrency": args.concurrency,
            "requests_per_trial": args.requests,
            "trials": args.trials,
            "logging_states": ["on", "off"],
            "cache_enabled": True,
            "dynamic_batching_enabled": False,
            "workers": 1,
        },
        "environment": {
            "python": sys.version,
            "executable": args.python,
            "platform": platform.platform(),
            "dataset_sha256": sha256(DATASET),
            "native_model_sha256": sha256(NATIVE),
        },
        "summary": aggregate_results(results),
        "trials": results,
    }
    write_report(args.output, document)
    print(f"Wrote matched logging benchmark to {args.output}")
    errors = sum(item["errors"] for item in results)
    if errors:
        raise SystemExit(f"Benchmark recorded {errors} request errors; inspect {args.output}")


if __name__ == "__main__":
    main()




