"""Reconstruct and validate the frozen training/holdout provenance contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "dataset" / "train.csv"
DEFAULT_MANIFEST = ROOT / "dataset" / "split_manifest.json"
BASELINE_PATH = ROOT / "models" / "baseline_xgb_pipeline.pkl"
NATIVE_PATH = ROOT / "models" / "house_prices_native.zip"
SCHEMA_PATH = ROOT / "models" / "feature_schema.json"
FROZEN_BASELINE_SHA256 = "0eed717dd49aa5b5e307577d14b8e11bca813c131f35dc39fb9e41a7c9c26d42"
FROZEN_DATASET_SHA256 = "1e18addf81e5e4d347cc17ee6075bbe4a42b7fa26b9e5b063e8f692a5f929d41"
FROZEN_NATIVE_SHA256 = "4e605ea450b01531142398e273729f326df625ae21b8f0a5235f030382721db4"
FROZEN_SCHEMA_SHA256 = "15c7bd2ce117f4d539781b1b439ef9626ead290f4df23cf3a3e68d2e9758818a"
TARGET = "SalePrice"
IDENTITY = "Id"
SEED = 42
FORMAT = "house-prices-split-v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, content: bytes) -> None:
    """Durably publish immutable generated data without changing its bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _feature_types(frame: pd.DataFrame, names: list[str]) -> tuple[list[str], list[str]]:
    numeric = frame.loc[:, names].select_dtypes(include=[np.number]).columns.tolist()
    categorical = [name for name in names if name not in set(numeric)]
    return numeric, categorical


def build_split_manifest(
    dataset_path: str | Path = DEFAULT_DATASET,
    *,
    baseline_path: str | Path = BASELINE_PATH,
    native_path: str | Path = NATIVE_PATH,
    schema_path: str | Path = SCHEMA_PATH,
) -> dict[str, Any]:
    """Use the original row ordering and sklearn split to reproduce all partitions."""
    dataset_path = Path(dataset_path)
    frame = pd.read_csv(dataset_path)
    if frame.shape != (1460, 81) or IDENTITY not in frame or TARGET not in frame:
        raise ValueError(f"Expected original Ames train.csv with 1,460 rows and 81 columns; got {frame.shape}")
    if frame[IDENTITY].isna().any() or frame[IDENTITY].duplicated().any():
        raise ValueError("Source Id values must be unique and nonmissing")
    if frame[TARGET].isna().any() or not np.isfinite(frame[TARGET].to_numpy(dtype=np.float64)).all():
        raise ValueError("Source SalePrice values must be finite and nonmissing")
    feature_names = [name for name in frame.columns if name not in {IDENTITY, TARGET}]
    numeric, categorical = _feature_types(frame, feature_names)
    fit_positions, holdout_positions = train_test_split(
        np.arange(len(frame)), test_size=0.2, random_state=SEED
    )
    holdout_ids = frame.iloc[holdout_positions][IDENTITY].astype(int).tolist()
    fit_ids = frame.iloc[fit_positions][IDENTITY].astype(int).tolist()
    manifest = {
        "format": FORMAT,
        "source": {
            "path": "dataset/train.csv",
            "sha256": sha256_file(dataset_path),
            "rows": int(len(frame)),
            "feature_names": feature_names,
            "numeric_features": numeric,
            "categorical_features": categorical,
            "identity_column": IDENTITY,
            "target_column": TARGET,
        },
        "split": {
            "algorithm": "sklearn.model_selection.train_test_split",
            "test_size": 0.2,
            "random_state": SEED,
            "holdout_order": "test indices returned by train_test_split, unchanged",
        },
        "partitions": {
            "original_fit_ids": fit_ids,
            "adaptation_ids": holdout_ids[:146],
            "promotion_gate_ids": holdout_ids[146:219],
            "final_audit_ids": holdout_ids[219:],
        },
        "frozen": {
            "baseline_pipeline_sha256": sha256_file(baseline_path),
            "native_archive_sha256": sha256_file(native_path),
            "feature_schema_sha256": sha256_file(schema_path),
        },
        "eligibility": {
            "observation_identity": ["source.sha256", "source.identity_column"],
            "features_exclude": [IDENTITY, TARGET],
            "training_labels": "unchanged successful adaptation observations only",
            "original_fit_ids_are_new_labels": False,
            "promotion_gate_is_fit_or_workload_data": False,
            "final_audit_is_fit_or_workload_data": False,
        },
    }
    if manifest["frozen"]["baseline_pipeline_sha256"] != FROZEN_BASELINE_SHA256:
        raise RuntimeError("Frozen baseline pipeline checksum changed; refusing to trust split provenance")
    if manifest["source"]["sha256"] != FROZEN_DATASET_SHA256:
        raise RuntimeError("Original train.csv checksum changed; refusing to trust split provenance")
    if manifest["frozen"]["native_archive_sha256"] != FROZEN_NATIVE_SHA256:
        raise RuntimeError("Frozen native archive checksum changed; refusing to trust split provenance")
    if manifest["frozen"]["feature_schema_sha256"] != FROZEN_SCHEMA_SHA256:
        raise RuntimeError("Frozen feature schema checksum changed; refusing to trust split provenance")
    validate_split_manifest(manifest, dataset_path=dataset_path, verify_frozen=True)
    return manifest


def validate_split_manifest(
    manifest: dict[str, Any],
    *,
    dataset_path: str | Path = DEFAULT_DATASET,
    verify_frozen: bool = True,
) -> None:
    if manifest.get("format") != FORMAT:
        raise ValueError("Unsupported split manifest format")
    source = manifest.get("source", {})
    observed_dataset_hash = sha256_file(dataset_path)
    if observed_dataset_hash != source.get("sha256") or observed_dataset_hash != FROZEN_DATASET_SHA256:
        raise ValueError("Source dataset SHA-256 does not match split manifest")
    frame = pd.read_csv(dataset_path)
    if frame.shape != (1460, 81) or IDENTITY not in frame or TARGET not in frame:
        raise ValueError("Source dataset is not the original 1,460-row train.csv")
    all_ids = set(frame[IDENTITY].astype(int).tolist())
    expected_features = [name for name in frame.columns if name not in {IDENTITY, TARGET}]
    if source.get("feature_names") != expected_features:
        raise ValueError("Feature names differ from original train.csv column order")
    expected_numeric, expected_categorical = _feature_types(frame, expected_features)
    if source.get("numeric_features") != expected_numeric or source.get("categorical_features") != expected_categorical:
        raise ValueError("Feature type manifest does not match the source CSV")
    if source.get("rows") != len(frame) or source.get("identity_column") != IDENTITY or source.get("target_column") != TARGET:
        raise ValueError("Source row count or identity/target contract is invalid")
    expected_split = {
        "algorithm": "sklearn.model_selection.train_test_split",
        "test_size": 0.2,
        "random_state": SEED,
        "holdout_order": "test indices returned by train_test_split, unchanged",
    }
    if manifest.get("split") != expected_split:
        raise ValueError("Split algorithm or seed does not match the frozen data contract")
    fit_positions, holdout_positions = train_test_split(
        np.arange(len(frame)), test_size=0.2, random_state=SEED
    )
    holdout_ids = frame.iloc[holdout_positions][IDENTITY].astype(int).tolist()
    exact_partitions = {
        "original_fit_ids": frame.iloc[fit_positions][IDENTITY].astype(int).tolist(),
        "adaptation_ids": holdout_ids[:146],
        "promotion_gate_ids": holdout_ids[146:219],
        "final_audit_ids": holdout_ids[219:],
    }
    partitions = manifest.get("partitions", {})
    keys = ("original_fit_ids", "adaptation_ids", "promotion_gate_ids", "final_audit_ids")
    id_sets: list[set[int]] = []
    expected_counts = (1168, 146, 73, 73)
    for key, expected_count in zip(keys, expected_counts):
        values = partitions.get(key)
        if not isinstance(values, list) or len(values) != expected_count:
            raise ValueError(f"Partition {key} must contain exactly {expected_count} IDs")
        current = [int(value) for value in values]
        if current != exact_partitions[key]:
            raise ValueError(f"Partition {key} is not the deterministic original split")
        current_set = set(current)
        if len(current_set) != len(current):
            raise ValueError(f"Partition {key} contains duplicate IDs")
        id_sets.append(current_set)
    for index, left in enumerate(id_sets):
        if any(left & right for right in id_sets[index + 1 :]):
            raise ValueError("Data partitions overlap")
    if set.union(*id_sets) != all_ids:
        raise ValueError("Partitions do not cover the source dataset exactly")
    if verify_frozen:
        expected = {
            "baseline_pipeline_sha256": (BASELINE_PATH, FROZEN_BASELINE_SHA256),
            "native_archive_sha256": (NATIVE_PATH, FROZEN_NATIVE_SHA256),
            "feature_schema_sha256": (SCHEMA_PATH, FROZEN_SCHEMA_SHA256),
        }
        frozen = manifest.get("frozen", {})
        for key, (path, pinned) in expected.items():
            observed = sha256_file(path)
            if frozen.get(key) != observed or observed != pinned:
                raise ValueError(f"Frozen artifact checksum mismatch: {key}")


def load_manifest(
    path: str | Path = DEFAULT_MANIFEST,
    *,
    verify_frozen: bool = True,
    dataset_path: str | Path = DEFAULT_DATASET,
) -> dict[str, Any]:
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    validate_split_manifest(manifest, dataset_path=dataset_path, verify_frozen=verify_frozen)
    return manifest


def write_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    path = Path(path)
    manifest = build_split_manifest()
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8-sig"))
        if canonical_json(current) != canonical_json(manifest):
            raise FileExistsError(f"Refusing to replace immutable split manifest at {path}")
        return current
    _atomic_write(path, json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", nargs="?", const=str(DEFAULT_MANIFEST), default=None)
    parser.add_argument("--check", type=Path, default=None, help="Validate an existing manifest")
    args = parser.parse_args()
    if args.write_manifest:
        manifest = write_manifest(args.write_manifest)
        print(json.dumps({"manifest": str(Path(args.write_manifest)), "sha256": hashlib.sha256(canonical_json(manifest)).hexdigest()}))
    elif args.check:
        manifest = json.loads(args.check.read_text(encoding="utf-8-sig"))
        validate_split_manifest(manifest)
        print(f"Validated {args.check}")
    else:
        parser.error("choose --write-manifest or --check")


if __name__ == "__main__":
    main()
