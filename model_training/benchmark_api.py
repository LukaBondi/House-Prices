"""Benchmark real HTTP predictions against fresh local Uvicorn processes.

Examples:
  python model_training/benchmark_api.py --requests 240 --trials 3
  python model_training/benchmark_api.py --configs native_full --workloads repeated --concurrency 1 8

Each configuration/workload/concurrency/trial starts a clean service process.
Clients reuse HTTP connections. Cache and batching counters are sampled around
the measured request phase, after health checks and connection warmup.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "models" / "baseline_xgb_pipeline.pkl"
NATIVE = ROOT / "models" / "house_prices_native.zip"
CONFIGS = {
    "phase2_uncached": {"model": BASELINE, "cache": "0", "batching": "0"},
    "native_plain": {"model": NATIVE, "cache": "0", "batching": "0"},
    "native_batch": {"model": NATIVE, "cache": "0", "batching": "1"},
    "native_cache": {"model": NATIVE, "cache": "1", "batching": "0"},
    "native_full": {"model": NATIVE, "cache": "1", "batching": "1"},
}
COUNTERS = (
    "cache_hits",
    "cache_misses",
    "cache_size",
    "batch_requests",
    "batch_count",
    "batch_rows",
    "queue_overloads",
    "in_flight_coalesced",
)


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent / 100
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] * (1 - (position - lo)) + ordered[hi] * (position - lo)


def choose_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def make_records() -> list[dict[str, Any]]:
    import pandas as pd

    frame = pd.read_csv(ROOT / "dataset" / "train.csv")
    frame = frame.drop(columns=["Id", "SalePrice"])
    return [
        {key: (None if isinstance(value, float) and math.isnan(value) else value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def workload_records(records: list[dict[str, Any]], workload: str, count: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index in range(count):
        if workload == "repeated":
            result.append(records[index % min(10, len(records))])
            continue
        unique = workload == "unique" or index % 5 == 0
        if not unique:
            result.append(records[index % min(10, len(records))])
            continue
        row = dict(records[index % len(records)])
        # Use a tiny deterministic numeric change to guarantee a distinct cache
        # key without moving the request outside a plausible house-price range.
        row["LotArea"] = float(row.get("LotArea") or 0) + (index + 1) * 0.125
        result.append(row)
    return result


def owned_process_tree_rss(process: Any) -> int:
    dependency_path = ROOT / "tmp" / "benchmark-deps"
    if dependency_path.is_dir() and str(dependency_path) not in sys.path:
        sys.path.insert(0, str(dependency_path))
    try:
        import psutil
    except ImportError as error:
        raise RuntimeError("psutil is required for API RSS measurement; install backend/requirements.txt") from error
    try:
        root_process = psutil.Process(process.pid)
        owned = [root_process, *root_process.children(recursive=True)]
        total = 0
        for item in owned:
            try:
                total += item.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total
    except Exception as error:
        raise RuntimeError(f"Could not read server process-tree RSS: {error}") from error


def process_tree_peak_sampler(process: Any) -> tuple[threading.Event, dict[str, int], threading.Thread]:
    done = threading.Event()
    peak = {"bytes": owned_process_tree_rss(process)}

    def sample() -> None:
        while not done.wait(0.02):
            peak["bytes"] = max(peak["bytes"], owned_process_tree_rss(process))

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    return done, peak, thread


async def get_json(client: Any, url: str) -> dict[str, Any]:
    response = await client.get(url)
    response.raise_for_status()
    return response.json()


async def _send_phase(client: Any, base_url: str, payloads: list[dict[str, Any]], concurrency: int) -> tuple[list[float], int, list[str], float]:
    latencies: list[float] = []
    errors = 0
    error_samples: list[str] = []

    async def worker(worker_index: int) -> None:
        nonlocal errors
        for request_index in range(worker_index, len(payloads), concurrency):
            started = time.perf_counter()
            try:
                response = await client.post(f"{base_url}/predict", json={"features": payloads[request_index]})
                elapsed = (time.perf_counter() - started) * 1000
                if response.status_code != 200:
                    errors += 1
                    if len(error_samples) < 5:
                        error_samples.append(f"HTTP {response.status_code}: {response.text[:400]}")
                    continue
                body = response.json()
                if not isinstance(body.get("sale_price"), (int, float)):
                    errors += 1
                    if len(error_samples) < 5:
                        error_samples.append(f"Invalid response: {body!r}")
                else:
                    latencies.append(elapsed)
            except Exception as error:
                errors += 1
                if len(error_samples) < 5:
                    error_samples.append(f"{type(error).__name__}: {error}")

    started = time.perf_counter()
    await asyncio.gather(*(worker(index) for index in range(concurrency)))
    return latencies, errors, error_samples, time.perf_counter() - started


async def warmup_and_measure(base_url: str, warmup: list[dict[str, Any]], payloads: list[dict[str, Any]], concurrency: int) -> tuple[list[float], int, list[str], float, dict[str, Any], dict[str, Any]]:
    import httpx

    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    timeout = httpx.Timeout(60.0, connect=10.0)
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        _, warm_errors, warm_samples, _ = await _send_phase(client, base_url, warmup, min(concurrency, len(warmup)))
        if warm_errors:
            raise RuntimeError(f"HTTP warmup failed: {warm_samples}")
        before = await get_json(client, f"{base_url}/metrics")
        latencies, errors, samples, elapsed = await _send_phase(client, base_url, payloads, concurrency)
        after = await get_json(client, f"{base_url}/metrics")
        return latencies, errors, samples, elapsed, before, after


def terminate_owned_process_tree(process: Any) -> None:
    dependency_path = ROOT / "tmp" / "benchmark-deps"
    if dependency_path.is_dir() and str(dependency_path) not in sys.path:
        sys.path.insert(0, str(dependency_path))
    try:
        import psutil

        root_process = psutil.Process(process.pid)
        children = root_process.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        try:
            root_process.terminate()
        except psutil.NoSuchProcess:
            pass
        _, alive = psutil.wait_procs([*children, root_process], timeout=5)
        for child in alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
    except ImportError:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def run_case(args: argparse.Namespace, config_name: str, workload: str, concurrency: int, trial: int) -> dict[str, Any]:
    import httpx

    config = CONFIGS[config_name]
    model_path = Path(config["model"])
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing artifact for {config_name}: {model_path}")
    port = choose_port()
    base_url = f"http://127.0.0.1:{port}"
    environment = os.environ.copy()
    environment.update(
        {
            "MODEL_PATH": str(model_path),
            "PREDICTION_CACHE_ENABLED": config["cache"],
            "DYNAMIC_BATCHING_ENABLED": config["batching"],
            "BATCH_MAX_SIZE": str(args.batch_max_size),
            "BATCH_MAX_DELAY_MS": str(args.batch_max_delay_ms),
            "BATCH_QUEUE_MAX_SIZE": "256",
            "INFERENCE_WORKERS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONPATH": os.pathsep.join(
                [str(ROOT), str(ROOT / "tmp" / "benchmark-deps"), environment.get("PYTHONPATH", "")]
            ),
        }
    )
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.app:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--workers",
        str(args.workers),
        "--no-access-log",
        "--log-level",
        "warning",
    ]
    log = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
    started = time.perf_counter()
    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=log)
    startup_seconds: float | None = None
    try:
        with httpx.Client(timeout=1.0) as client:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    log.seek(0)
                    raise RuntimeError(f"Uvicorn exited with {process.returncode}: {log.read()[-4000:]}")
                try:
                    response = client.get(f"{base_url}/health")
                    if response.status_code == 200 and response.json().get("model_loaded"):
                        startup_seconds = time.perf_counter() - started
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.1)
            if startup_seconds is None:
                raise TimeoutError(f"Uvicorn health check timed out for {config_name}")
            # Confirm the metrics endpoint before starting measured traffic.
            with httpx.Client(timeout=10.0) as warmup_client:
                warmup_client.get(f"{base_url}/health").raise_for_status()
                warmup_client.get(f"{base_url}/metrics").raise_for_status()

        records = make_records()
        # Each warmup request has a unique, tiny LotArea perturbation so it
        # cannot pre-populate either the repeated hot set or the unique stream.
        warm_payloads = []
        for index in range(max(5, concurrency)):
            row = dict(records[index % len(records)])
            row["LotArea"] = float(row.get("LotArea") or 0) + 100000 + index
            warm_payloads.append(row)
        payloads = workload_records(records, workload, args.requests)
        loaded_rss = owned_process_tree_rss(process)
        done, peak, sampler = process_tree_peak_sampler(process)
        latencies, errors, error_samples, elapsed, before, after = asyncio.run(
            warmup_and_measure(base_url, warm_payloads, payloads, concurrency)
        )
        done.set()
        sampler.join()
        counter_delta = None if args.workers > 1 else {
            key: int(after.get(key, 0) or 0) - int(before.get(key, 0) or 0)
            for key in COUNTERS
        }
        cache_requests = (counter_delta["cache_hits"] + counter_delta["cache_misses"]) if counter_delta else 0
        return {
            "config": config_name,
            "workload": workload,
            "concurrency": concurrency,
            "trial": trial,
            "workers": args.workers,
            "requests_requested": len(payloads),
            "requests_succeeded": len(latencies),
            "errors": errors,
            "error_samples": error_samples,
            "elapsed_seconds": elapsed,
            "successful_requests_per_second": len(latencies) / elapsed if elapsed else 0.0,
            "latency_ms_samples": latencies,
            "latency_p50_ms": percentile(latencies, 50) if latencies else None,
            "latency_p95_ms": percentile(latencies, 95) if latencies else None,
            "latency_p99_ms": percentile(latencies, 99) if latencies else None,
            "startup_seconds": startup_seconds,
            "server_process_tree_loaded_rss_bytes": loaded_rss,
            "server_process_tree_peak_rss_bytes": peak["bytes"],
            "cache_and_batch_counter_delta": counter_delta,
            "cache_hit_ratio": counter_delta["cache_hits"] / cache_requests if cache_requests else None,
        }
    finally:
        terminate_owned_process_tree(process)
        log.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", choices=tuple(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--workloads", nargs="+", choices=("unique", "repeated", "mixed"), default=["unique", "repeated", "mixed"])
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--requests", type=int, default=240)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-max-size", type=int, default=16)
    parser.add_argument("--batch-max-delay-ms", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks" / "api_benchmark.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.requests, args.trials, args.workers, args.batch_max_size) < 1 or args.batch_max_delay_ms < 0 or not args.concurrency or min(args.concurrency) < 1:
        raise ValueError("requests, trials, workers, and concurrency values must be positive")
    results = []
    total = len(args.configs) * len(args.workloads) * len(args.concurrency) * args.trials
    index = 0
    for config in args.configs:
        for workload in args.workloads:
            for concurrency in args.concurrency:
                for trial in range(1, args.trials + 1):
                    index += 1
                    print(f"[{index}/{total}] {config} workload={workload} concurrency={concurrency} trial={trial}", flush=True)
                    results.append(run_case(args, config, workload, concurrency, trial))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        key = f"{item['config']}|{item['workload']}|c{item['concurrency']}"
        grouped.setdefault(key, []).append(item)
    summary = {}
    for key, trials in grouped.items():
        throughputs = [float(item["successful_requests_per_second"]) for item in trials]
        p50 = [float(item["latency_p50_ms"]) for item in trials if item["latency_p50_ms"] is not None]
        p95 = [float(item["latency_p95_ms"]) for item in trials if item["latency_p95_ms"] is not None]
        summary[key] = {
            "median_successful_requests_per_second": statistics.median(throughputs),
            "median_request_p50_ms": statistics.median(p50) if p50 else None,
            "median_request_p95_ms": statistics.median(p95) if p95 else None,
            "trials": len(trials),
            "any_errors": sum(int(item["errors"]) for item in trials),
        }
    document = {
        "benchmark": "house_prices_real_http_api",
        "schema_version": 1,
        "method": "Fresh local Uvicorn process for every trial; persistent httpx connections; warmup counters excluded by before/after metrics snapshots.",
        "workloads": {
            "unique": "Every payload receives a distinct LotArea value.",
            "repeated": "Cycles through the first ten cleaned training rows.",
            "mixed": "Four repeated hot-set requests per one distinct request (80/20).",
        },
        "settings": {"configs": args.configs, "workloads": args.workloads, "concurrency": args.concurrency, "requests_per_trial": args.requests, "trials": args.trials, "workers": args.workers, "batch_max_size": args.batch_max_size, "batch_max_delay_ms": args.batch_max_delay_ms, "inference_workers": 1},
        "environment": _environment(),
        "summary": summary,
        "trials": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote API benchmark results to {args.output}")
    failures = sum(int(result["errors"]) for result in results)
    if failures:
        raise SystemExit(f"API benchmark recorded {failures} errors; inspect {args.output}")


def _environment() -> dict[str, Any]:
    import httpx
    import numpy
    import pandas
    import sklearn
    import xgboost

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "httpx": httpx.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "dataset_sha256": _sha256(ROOT / "dataset" / "train.csv"),
        "baseline_sha256": _sha256(BASELINE),
        "native_sha256": _sha256(NATIVE) if NATIVE.exists() else None,
    }


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
