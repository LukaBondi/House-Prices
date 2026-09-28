"""Compare Phase 2, historical Phase 3, and native-runtime model inference.

Each model runs in a fresh child process. The timed boundary accepts the same
prebuilt raw feature-record dictionaries: the pickle path creates its DataFrame
inside prediction while the native path accepts those records directly. RSS is
measured in a separate untimed pass, with absolute loaded and active values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "dataset" / "train.csv"
BASELINE_PATH = ROOT / "models" / "baseline_xgb_pipeline.pkl"
HISTORICAL_PATH = ROOT / "models" / "optimized_xgb_pipeline.pkl"
NATIVE_PATH = ROOT / "models" / "house_prices_native.zip"
CONFIGS = {
    "baseline": ("pickle", BASELINE_PATH),
    "phase3_pickle": ("pickle", HISTORICAL_PATH),
    "native": ("native", NATIVE_PATH),
}
EXPECTED_BASELINE_SHA256 = "0EED717DD49AA5B5E307577D14B8E11BCA813C131F35DC39FB9E41A7C9C26D42"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_scalar(value: str) -> Any:
    if value.strip().lower() in {"", "na", "n/a", "nan", "null", "none", "<na>"}:
        return None
    try:
        number = float(value)
    except ValueError:
        return value
    # Keep numeric inputs as floats, matching the API's canonical feature
    # representation and sklearn's fitted float64 imputer inputs.
    return number


def _percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent / 100
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] * (1 - (position - low)) + ordered[high] * (position - low)


def _read_dataset() -> tuple[list[str], list[dict[str, Any]], list[float]]:
    with DATA_PATH.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        feature_columns = [column for column in columns if column not in {"Id", "SalePrice"}]
        records: list[dict[str, Any]] = []
        targets: list[float] = []
        for row in reader:
            targets.append(float(row.pop("SalePrice")))
            row.pop("Id", None)
            records.append({key: _csv_scalar(value) for key, value in row.items()})
    return feature_columns, records, targets


def _child(args: argparse.Namespace) -> None:
    os.environ.update(
        OMP_NUM_THREADS=str(args.threads),
        OPENBLAS_NUM_THREADS=str(args.threads),
        MKL_NUM_THREADS=str(args.threads),
        NUMEXPR_NUM_THREADS=str(args.threads),
    )
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    kind, artifact = CONFIGS[args.config]
    if not artifact.is_file():
        raise FileNotFoundError(f"Missing model artifact: {artifact}")
    try:
        import psutil
    except ImportError as error:
        raise RuntimeError("psutil is required for memory measurement; install backend/requirements.txt") from error

    feature_columns, records, targets = _read_dataset()
    import numpy as np

    import_started = time.perf_counter()
    if kind == "native":
        from backend.native_model import NativeModel

        model: Any = NativeModel(artifact)
        predict = model.predict_records
    else:
        import joblib
        import pandas as pd

        model = joblib.load(artifact)
        try:
            model.regressor_.named_steps["regressor"].set_params(n_jobs=args.threads)
        except (AttributeError, KeyError, TypeError, ValueError):
            try:
                model.named_steps["regressor"].set_params(n_jobs=args.threads)
            except (AttributeError, KeyError, TypeError, ValueError):
                pass

        def predict(raw_records: list[dict[str, Any]]) -> Any:
            frame = pd.DataFrame.from_records(raw_records, columns=feature_columns).replace({None: np.nan})
            return model.predict(frame)

    cold_load_seconds = time.perf_counter() - import_started
    process = psutil.Process(os.getpid())
    loaded_rss_bytes = process.memory_info().rss

    # Match sklearn's existing train_test_split(random_state=42) holdout for
    # descriptive comparison, without importing sklearn on the native path.
    permutation = np.random.RandomState(42).permutation(len(records))
    test_count = math.ceil(len(records) * 0.2)
    holdout_indices = [int(index) for index in permutation[:test_count]]
    y_true = [targets[index] for index in holdout_indices]
    y_pred = [float(value) for value in predict([records[index] for index in holdout_indices])]
    if len(y_true) != len(y_pred):
        raise RuntimeError(f"Prediction count mismatch: got {len(y_pred)}, expected {len(y_true)}")
    errors = [actual - predicted for actual, predicted in zip(y_true, y_pred, strict=True)]
    mean_actual = sum(y_true) / len(y_true)
    ss_total = sum((actual - mean_actual) ** 2 for actual in y_true)
    descriptive = {
        "holdout_mae": sum(abs(error) for error in errors) / len(errors),
        "holdout_rmse": math.sqrt(sum(error * error for error in errors) / len(errors)),
        "holdout_r2": 1 - sum(error * error for error in errors) / ss_total,
        "holdout_rmsle": math.sqrt(
            sum((math.log1p(actual) - math.log1p(max(predicted, 0))) ** 2 for actual, predicted in zip(y_true, y_pred, strict=True))
            / len(y_true)
        ),
    }
    full_predictions = [float(value) for value in predict(records)]

    batches: list[dict[str, Any]] = []
    for batch_size in (1, 8, 64):
        batch = records[:batch_size]
        predict(batch)  # Warmup is outside the timed trials.
        latency_samples: list[float] = []
        trial_elapsed: list[float] = []
        for _ in range(args.trials):
            started = time.perf_counter()
            for _ in range(args.repeats):
                call_started = time.perf_counter()
                predict(batch)
                latency_samples.append((time.perf_counter() - call_started) * 1000)
            trial_elapsed.append(time.perf_counter() - started)

        # A separate untimed pass measures active RSS without changing latency.
        rss_peak = {"bytes": loaded_rss_bytes}
        stop_sampler = threading.Event()

        def sample_rss() -> None:
            while not stop_sampler.wait(0.005):
                try:
                    rss_peak["bytes"] = max(rss_peak["bytes"], process.memory_info().rss)
                except Exception:
                    return

        sampler = threading.Thread(target=sample_rss, daemon=True)
        sampler.start()
        for _ in range(args.memory_samples):
            predict(batch)
        stop_sampler.set()
        sampler.join()
        active_peak_rss = rss_peak["bytes"]
        batches.append(
            {
                "batch_size": batch_size,
                "repeats_per_trial": args.repeats,
                "trial_elapsed_seconds": trial_elapsed,
                "batch_latency_ms_samples": latency_samples,
                "p50_latency_ms": _percentile(latency_samples, 50),
                "p95_latency_ms": _percentile(latency_samples, 95),
                "rows_per_second": batch_size * args.repeats / statistics.median(trial_elapsed),
                "loaded_rss_bytes": loaded_rss_bytes,
                "active_peak_rss_bytes": active_peak_rss,
            }
        )

    print(
        json.dumps(
            {
                "config": args.config,
                "artifact_path": str(artifact.relative_to(ROOT)),
                "artifact_bytes": artifact.stat().st_size,
                "artifact_sha256": sha256(artifact),
                "cold_import_and_load_seconds": cold_load_seconds,
                "loaded_rss_bytes": loaded_rss_bytes,
                "active_peak_rss_bytes": max(batch["active_peak_rss_bytes"] for batch in batches),
                "metrics_existing_random_state_42_holdout_descriptive_only": descriptive,
                "_full_predictions_for_parent_parity_check": full_predictions,
                "batches": batches,
            },
            allow_nan=False,
        )
    )


def _compressed_baseline_bytes() -> int:
    compressed = io.BytesIO()
    with zipfile.ZipFile(compressed, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as output:
        output.writestr(BASELINE_PATH.name, BASELINE_PATH.read_bytes())
    return len(compressed.getvalue())


def _environment(threads: int) -> dict[str, Any]:
    import numpy
    import pandas
    import sklearn
    import xgboost

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "threads": threads,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "dataset_sha256": sha256(DATA_PATH),
        "baseline_sha256": sha256(BASELINE_PATH),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", choices=tuple(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--memory-samples", type=int, default=100)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks" / "model_benchmark.json")
    parser.add_argument("--child-config", choices=tuple(CONFIGS), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child_config:
        args.config = args.child_config
        _child(args)
        return
    if min(args.repeats, args.trials, args.threads, args.memory_samples) < 1:
        raise ValueError("repeats, trials, threads, and memory-samples must be positive")
    if sha256(BASELINE_PATH).upper() != EXPECTED_BASELINE_SHA256:
        raise RuntimeError(f"Frozen Phase 2 artifact hash changed: {sha256(BASELINE_PATH)}")

    results = []
    for config in args.configs:
        command = [sys.executable, str(Path(__file__).resolve()), "--child-config", config,
                   "--repeats", str(args.repeats), "--trials", str(args.trials),
                   "--threads", str(args.threads), "--memory-samples", str(args.memory_samples)]
        child_env = os.environ.copy()
        child_env.update({key: str(args.threads) for key in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"
        )})
        child_env["PYTHONPATH"] = os.pathsep.join(
            [str(ROOT), str(ROOT / "tmp" / "benchmark-deps"), child_env.get("PYTHONPATH", "")]
        )
        completed = subprocess.run(command, cwd=ROOT, env=child_env, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(f"{config} benchmark failed:\n{completed.stderr}\n{completed.stdout}")
        try:
            results.append(json.loads(completed.stdout))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"{config} child did not emit clean JSON: {completed.stdout[-1000:]}") from error

    by_name = {result["config"]: result for result in results}
    parity = None
    if "baseline" in by_name and "native" in by_name:
        baseline_predictions = by_name["baseline"].pop("_full_predictions_for_parent_parity_check")
        native_predictions = by_name["native"].pop("_full_predictions_for_parent_parity_check")
        differences = [abs(left - right) for left, right in zip(baseline_predictions, native_predictions, strict=True)]
        cent_mismatches = sum(
            round(left, 2) != round(right, 2)
            for left, right in zip(baseline_predictions, native_predictions, strict=True)
        )
        parity = {
            "rows_compared": len(differences),
            "max_absolute_prediction_difference": max(differences, default=0.0),
            "max_absolute_prediction_difference_cents": max(differences, default=0.0) * 100,
            "rows_with_different_rounded_cent_response": cent_mismatches,
        }
    for result in results:
        result.pop("_full_predictions_for_parent_parity_check", None)
    document = {
        "benchmark": "house_prices_model_inference",
        "schema_version": 2,
        "input_boundary": "The same prebuilt raw feature records enter each predict call; pickle paths construct a DataFrame inside the call.",
        "memory_measurement": "Absolute process RSS after model load and during a separate untimed repeated-prediction pass.",
        "settings": {"threads": args.threads, "repeats_per_trial": args.repeats, "trials": args.trials,
                     "memory_samples_per_batch": args.memory_samples, "batch_sizes": [1, 8, 64]},
        "environment": _environment(args.threads),
        "compressed_baseline_comparison_bytes": _compressed_baseline_bytes(),
        "baseline_to_native_full_rows_parity": parity,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(document, indent=2, allow_nan=False))
    print(f"Wrote model benchmark results to {args.output}")


if __name__ == "__main__":
    main()
