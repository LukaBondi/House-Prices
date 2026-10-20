"""Describe frozen-baseline and selected-model performance on the reserved audit rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.artifacts import schema_sha256 as serving_schema_sha256
from backend.native_model import NativeModel
from model_training.candidate_export import gate_metrics
from model_training.data_partitions import (
    DEFAULT_DATASET,
    DEFAULT_MANIFEST,
    FROZEN_BASELINE_SHA256,
    IDENTITY,
    TARGET,
    canonical_json,
    load_manifest,
    sha256_file,
)
from model_training.registry import load_manifest as load_registry_manifest
from model_training.retrain import _format_records, _load_baseline_pipeline
from model_training.training_config import MODEL_REGISTRY_PATH, MODEL_STATE_PATH, SCHEMA_PATH


def _audit_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    metrics = gate_metrics(actual, prediction)
    for name in ("rmsle", "mae", "rmse"):
        value = metrics.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"Audit {name} must be finite and nonnegative")
    return metrics


def run_audit(
    *,
    registry_path: str | Path,
    output_path: str | Path,
    state_path: str | Path = MODEL_STATE_PATH,
    version: str | None = None,
    dataset_path: str | Path = DEFAULT_DATASET,
    split_manifest_path: str | Path = DEFAULT_MANIFEST,
    schema_path: str | Path = SCHEMA_PATH,
    baseline_pipeline_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate exactly the fixed final-audit IDs; never fit or publish anything."""
    dataset_path = Path(dataset_path)
    split = load_manifest(split_manifest_path, dataset_path=dataset_path)
    source_hash = sha256_file(dataset_path)
    if source_hash != split["source"]["sha256"]:
        raise ValueError("Audit dataset does not match the frozen split manifest")

    registry_path = Path(registry_path)
    pointer_path = Path(state_path) / "current.json"
    pointer = None
    if version is None:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        version = pointer.get("version")
        if not isinstance(version, str):
            raise ValueError("Active serving pointer has no model version")
        selection_source = "active_pointer"
    else:
        selection_source = "explicit_version"
    registry_manifest = load_registry_manifest(registry_path, version)
    if registry_manifest.get("schema_sha256") != serving_schema_sha256(schema_path):
        raise ValueError("Selected model schema hash does not match the frozen feature schema")
    if registry_manifest.get("gate", {}).get("accepted") is not True:
        raise ValueError("Selected registry version is not an accepted serving model")
    artifact_path = registry_path / version / "model.zip"
    model = NativeModel(artifact_path)

    baseline_path = Path(baseline_pipeline_path) if baseline_pipeline_path else Path(__file__).resolve().parents[1] / "models" / "baseline_xgb_pipeline.pkl"
    if sha256_file(baseline_path) != FROZEN_BASELINE_SHA256:
        raise ValueError("Frozen baseline pipeline checksum changed")
    baseline = _load_baseline_pipeline(baseline_path)

    audit_ids = [int(value) for value in split["partitions"]["final_audit_ids"]]
    frame = pd.read_csv(dataset_path).set_index(IDENTITY, drop=False)
    audit_rows = frame.loc[audit_ids]
    feature_names = split["source"]["feature_names"]
    if list(model.feature_names) != feature_names:
        raise ValueError("Selected model feature names do not match the frozen split manifest")
    actual = audit_rows[TARGET].to_numpy(dtype=np.float64)
    records = _format_records(audit_rows, feature_names)
    baseline_predictions = np.asarray(baseline.predict(audit_rows.loc[:, feature_names]), dtype=np.float64)
    selected_predictions = np.asarray(model.predict_records(records), dtype=np.float64)
    baseline_metrics = _audit_metrics(actual, baseline_predictions)
    selected_metrics = _audit_metrics(actual, selected_predictions)

    output = {
        "format": "house-prices-final-audit-v1",
        "evaluated_at": time.time(),
        "source_dataset_sha256": source_hash,
        "audit_ids": audit_ids,
        "audit_ids_sha256": hashlib.sha256(canonical_json(audit_ids)).hexdigest(),
        "sample_count": len(audit_ids),
        "selection_source": selection_source,
        "active_pointer": pointer,
        "model_version": version,
        "model_artifact_sha256": registry_manifest["artifact_sha256"],
        "baseline_pipeline_sha256": sha256_file(baseline_path),
        "baseline_native_sha256": split["frozen"]["native_archive_sha256"],
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xgboost": __import__("xgboost").__version__,
        },
        "metrics": {"frozen_baseline": baseline_metrics, "selected_model": selected_metrics},
        "selection_used": False,
        "limitations": [
            "This is descriptive evaluation after model selection; these audit metrics were not used to choose a model or tune a threshold.",
            "The fixed 73-row audit partition was historically evaluated as part of the original 292-row holdout, so it is not statistically untouched external data.",
        ],
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=MODEL_REGISTRY_PATH)
    parser.add_argument("--state", type=Path, default=MODEL_STATE_PATH)
    parser.add_argument("--version", default=None, help="Evaluate this version; otherwise read the active current.json pointer")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--schema", type=Path, default=SCHEMA_PATH)
    parser.add_argument("--baseline-pipeline", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit(
        registry_path=args.registry,
        state_path=args.state,
        version=args.version,
        dataset_path=args.dataset,
        split_manifest_path=args.split_manifest,
        schema_path=args.schema,
        baseline_pipeline_path=args.baseline_pipeline,
        output_path=args.output,
    )
    print(json.dumps({"model_version": result["model_version"], "sample_count": result["sample_count"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
