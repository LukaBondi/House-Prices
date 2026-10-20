"""Validated registry content and durable pointer tests."""

import json
import shutil
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from backend.artifacts import (
    ArtifactError,
    atomic_write_pointer,
    load_bootstrap_snapshot,
    load_registry_snapshot,
    preprocessing_sha256,
    read_archive_metadata,
    read_pointer,
    schema_sha256,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]


def _candidate(registry, version, *, accepted=True):
    bootstrap_path = ROOT / "models" / "house_prices_native.zip"
    schema_path = ROOT / "models" / "feature_schema.json"
    baseline = load_bootstrap_snapshot(bootstrap_path, schema_path)
    target = registry / version
    target.mkdir(parents=True)
    archive = target / "model.zip"
    shutil.copyfile(bootstrap_path, archive)
    metadata = read_archive_metadata(archive)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    manifest = {
        "format": "house-prices-registry-v1",
        "version": version,
        "artifact_sha256": sha256_file(archive),
        "preprocessing_sha256": preprocessing_sha256(metadata),
        "schema_sha256": schema_sha256(schema),
        "parent_version": baseline.version,
        "gate": {"policy": "test", "accepted": accepted},
        "training": {"parameters": {}, "snapshot": "test"},
    }
    (target / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return baseline, target


def test_registry_candidate_is_verified_and_snapshot_schema_is_immutable(tmp_path):
    registry = tmp_path / "registry"
    registry.mkdir()
    baseline, _ = _candidate(registry, "candidate-one")
    schema_path = ROOT / "models" / "feature_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    snapshot = load_registry_snapshot(
        registry,
        "candidate-one",
        schema=schema,
        expected_preprocessing_sha256=baseline.preprocessing_sha256,
    )
    assert snapshot.version == "candidate-one"
    assert snapshot.schema_sha256 == baseline.schema_sha256
    assert snapshot.model.predict_records([{"OverallQual": 7}])[0] > 0
    returned_schema = snapshot.schema
    returned_schema["features"].clear()
    assert snapshot.feature_names
    with pytest.raises(FrozenInstanceError):
        snapshot.version = "changed"


def test_registry_rejects_unaccepted_tampered_and_unsafe_candidates(tmp_path):
    registry = tmp_path / "registry"
    registry.mkdir()
    _, rejected = _candidate(registry, "candidate-rejected", accepted=False)
    schema = json.loads((ROOT / "models" / "feature_schema.json").read_text(encoding="utf-8"))
    with pytest.raises(ArtifactError, match="promotion gate"):
        load_registry_snapshot(registry, "candidate-rejected", schema=schema)

    _, tampered = _candidate(registry, "candidate-tampered")
    archive = tampered / "model.zip"
    archive.write_bytes(archive.read_bytes() + b"changed")
    with pytest.raises(ArtifactError, match="checksum"):
        load_registry_snapshot(registry, "candidate-tampered", schema=schema)

    with pytest.raises(ArtifactError, match="path-safe"):
        load_registry_snapshot(registry, "../outside", schema=schema)
    assert rejected.is_dir()


def test_serving_pointer_is_replaced_atomically_and_validated(tmp_path):
    pointer_path = tmp_path / "state" / "current.json"
    assert read_pointer(pointer_path) is None
    atomic_write_pointer(
        pointer_path,
        {"version": "candidate-one", "activation_id": "activation-1", "previous_version": "baseline-a"},
    )
    assert read_pointer(pointer_path) == {
        "version": "candidate-one",
        "activation_id": "activation-1",
        "previous_version": "baseline-a",
    }
    pointer_path.write_text('{"version":"../bad"}', encoding="utf-8")
    with pytest.raises(ArtifactError, match="path-safe"):
        read_pointer(pointer_path)
