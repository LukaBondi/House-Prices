"""Create the frozen bootstrap version and its split/drift references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import xgboost

from backend.native_model import NativeModel
from model_training.build_drift_reference import write_reference
from model_training.data_partitions import sha256_file, write_manifest
from model_training.registry import (
    archive_metadata,
    load_manifest,
    preprocessing_sha256,
    schema_sha256,
    sha256_bytes,
    write_immutable_version,
)
from model_training.training_config import (
    BASELINE_ARCHIVE,
    DRIFT_REFERENCE_PATH,
    MODEL_REGISTRY_PATH,
    SCHEMA_PATH,
    SPLIT_MANIFEST_PATH,
)


def bootstrap(
    registry_path: str | Path = MODEL_REGISTRY_PATH,
    *,
    reference_path: str | Path = BASELINE_ARCHIVE,
    split_manifest_path: str | Path = SPLIT_MANIFEST_PATH,
    drift_reference_path: str | Path = DRIFT_REFERENCE_PATH,
    schema_path: str | Path = SCHEMA_PATH,
) -> tuple[str, Path, dict[str, Any]]:
    reference_path = Path(reference_path)
    split_manifest = write_manifest(split_manifest_path)
    drift_reference = write_reference(
        drift_reference_path, manifest_path=split_manifest_path
    )
    archive_bytes = reference_path.read_bytes()
    native = NativeModel(reference_path)
    sample = {name: None for name in native.feature_names}
    prediction = native.predict_records([sample])
    if len(prediction) != 1 or not np.isfinite(prediction).all() or prediction[0] < 0:
        raise ValueError("Frozen bootstrap model failed finite nonnegative load smoke")
    metadata = native.metadata
    if metadata.get("feature_schema_sha256") != sha256_file(schema_path):
        raise ValueError("Frozen archive feature schema hash does not match schema file")
    gate = {
        "policy": "frozen bootstrap version; candidate promotion gate not applicable",
        "results": None,
        "accepted": True,
    }
    training = {
        "parameters": {"mode": "frozen-bootstrap", "source_model_sha256": metadata.get("source_sha256")},
        "cutoff": None,
        "snapshot_hash": split_manifest["source"]["sha256"],
        "eligible_ids": [],
        "new_ids": [],
        "environment": {"xgboost": xgboost.__version__},
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "drift_reference_id": drift_reference["reference_id"],
    }
    version = f"bootstrap-{sha256_bytes(archive_bytes)[:12]}"
    existing_directory = Path(registry_path) / version
    if existing_directory.exists():
        existing = load_manifest(registry_path, version)
        existing_archive = (existing_directory / "model.zip").read_bytes()
        existing_metadata, _ = archive_metadata(existing_archive)
        existing_training = existing.get("training", {})
        expected_provenance = {
            "source_model_sha256": metadata.get("source_sha256"),
            "split_manifest_sha256": sha256_file(split_manifest_path),
            "drift_reference_id": drift_reference["reference_id"],
        }
        actual_provenance = {
            "source_model_sha256": existing_training.get("parameters", {}).get("source_model_sha256"),
            "split_manifest_sha256": existing_training.get("split_manifest_sha256"),
            "drift_reference_id": existing_training.get("drift_reference_id"),
        }
        if (
            existing_archive != archive_bytes
            or existing.get("parent_version") is not None
            or existing.get("schema_sha256") != schema_sha256(schema_path)
            or existing.get("preprocessing_sha256") != preprocessing_sha256(metadata)
            or preprocessing_sha256(existing_metadata) != preprocessing_sha256(metadata)
            or existing_training.get("snapshot_hash") != split_manifest["source"]["sha256"]
            or existing_training.get("eligible_ids") != []
            or existing_training.get("new_ids") != []
            or existing_training.get("parameters", {}).get("mode") != "frozen-bootstrap"
            or actual_provenance != expected_provenance
        ):
            raise FileExistsError(
                f"Existing bootstrap {version} has incompatible artifact or provenance; preserving immutable version"
            )
        return version, existing_directory, existing
    return write_immutable_version(
        registry_path,
        archive_bytes,
        parent_version=None,
        gate=gate,
        training=training,
        schema_sha256=schema_sha256(schema_path),
        version_override=version,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=MODEL_REGISTRY_PATH)
    parser.add_argument("--reference", type=Path, default=BASELINE_ARCHIVE)
    parser.add_argument("--manifest", type=Path, default=SPLIT_MANIFEST_PATH)
    parser.add_argument("--drift-reference", type=Path, default=DRIFT_REFERENCE_PATH)
    parser.add_argument("--schema", type=Path, default=SCHEMA_PATH)
    args = parser.parse_args()
    version, directory, manifest = bootstrap(
        args.registry,
        reference_path=args.reference,
        split_manifest_path=args.manifest,
        drift_reference_path=args.drift_reference,
        schema_path=args.schema,
    )
    print(json.dumps({"version": version, "directory": str(directory), "artifact_sha256": manifest["artifact_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
