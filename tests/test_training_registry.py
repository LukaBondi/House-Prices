from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pytest
import xgboost

from backend.artifacts import preprocessing_sha256 as backend_preprocessing_sha256
from backend.artifacts import schema_sha256 as backend_schema_sha256
from backend.native_model import NativeModel
from model_training.bootstrap_registry import bootstrap
from model_training.candidate_export import (
    build_candidate_archive,
    evaluate_gate,
    verify_candidate_parity,
)
from model_training.data_partitions import sha256_file
from model_training.export_native import export
from model_training.registry import (
    archive_metadata,
    preprocessing_sha256,
    schema_sha256,
    write_immutable_version,
)


def _with_archive_comment(archive_bytes: bytes, comment: bytes) -> bytes:
    source = zipfile.ZipFile(io.BytesIO(archive_bytes))
    output = io.BytesIO()
    with source, zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for name in source.namelist():
            target.writestr(name, source.read(name))
        target.comment = comment
    return output.getvalue()


def test_positive_gate_fixture_accepts_real_rmsle_gain_and_rmse_nonregression():
    actual = np.asarray([100_000.0, 180_000.0, 250_000.0, 400_000.0, 525_000.0])
    champion = np.asarray([80_000.0, 215_000.0, 275_000.0, 360_000.0, 570_000.0])
    challenger = actual.copy()
    result = evaluate_gate(actual, champion, challenger)
    assert result["accepted"] is True
    assert result["relative_rmsle_gain"] >= 0.01
    assert result["challenger"]["rmse"] <= result["champion"]["rmse"]


def test_gate_rejects_tie_and_invalid_candidate_predictions():
    actual = np.asarray([100.0, 200.0, 300.0])
    incumbent = np.asarray([90.0, 230.0, 275.0])
    tied = evaluate_gate(actual, incumbent, incumbent)
    invalid = evaluate_gate(actual, incumbent, np.asarray([100.0, -1.0, 300.0]))
    assert tied["accepted"] is False
    assert invalid["accepted"] is False
    assert invalid["challenger"] is None


def test_gate_rejects_rmsle_gain_when_rmse_regresses():
    actual = np.asarray([100.0, 100.0, 100.0, 1_000_000.0])
    champion = np.asarray([200.0, 200.0, 200.0, 1_000_000.0])
    challenger = np.asarray([100.0, 100.0, 100.0, 1_000_500.0])
    result = evaluate_gate(actual, champion, challenger)
    assert result["relative_rmsle_gain"] > 0.01
    assert result["challenger"]["rmse"] > result["champion"]["rmse"]
    assert result["accepted"] is False


def test_candidate_archive_matches_frozen_preprocessor_and_native_inference():
    pipeline = joblib.load("models/baseline_xgb_pipeline.pkl")
    booster = pipeline.named_steps["regressor"].get_booster()
    archive = build_candidate_archive("models/house_prices_native.zip", booster)
    metadata, _ = archive_metadata(archive)
    assert preprocessing_sha256(metadata) == preprocessing_sha256(NativeModel("models/house_prices_native.zip").metadata)
    assert preprocessing_sha256(metadata) == backend_preprocessing_sha256(metadata)
    assert schema_sha256("models/feature_schema.json") == backend_schema_sha256("models/feature_schema.json")
    feature_names = list(pipeline.feature_names_in_)
    sample = {name: None for name in feature_names}
    sample.update({"OverallQual": 0, "GrLivArea": None, "Neighborhood": "unknown"})
    records = [sample, {"OverallQual": 7, "GrLivArea": 1710, "Neighborhood": "CollgCr"}]
    parity = verify_candidate_parity(archive, booster, pipeline.named_steps["preprocessor"], records)
    assert parity["rows"] == 2
    assert parity["max_absolute_difference"] <= 1e-4


def test_registry_version_is_immutable_and_manifest_fingerprint_validates(tmp_path):
    archive = Path("models/house_prices_native.zip").read_bytes()
    metadata, _ = archive_metadata(archive)
    training = {"parameters": {"mode": "fixture"}, "cutoff": 10.0, "snapshot_hash": "fixture", "eligible_ids": [], "new_ids": [], "environment": {}}
    gate = {"policy": "fixture", "results": None, "accepted": True}
    version, directory, manifest = write_immutable_version(
        tmp_path,
        archive,
        parent_version=None,
        gate=gate,
        training=training,
        schema_sha256="schema-fixture",
        version_override="fixture-bootstrap",
    )
    assert version == "fixture-bootstrap"
    assert (directory / "model.zip").read_bytes() == archive
    assert manifest["preprocessing_sha256"] == preprocessing_sha256(metadata)
    with pytest.raises(FileExistsError, match="collision"):
        write_immutable_version(
            tmp_path,
            _with_archive_comment(archive, b"different archive hash"),
            parent_version=None,
            gate=gate,
            training=training,
            schema_sha256="schema-fixture",
            version_override="fixture-bootstrap",
        )


def test_bootstrap_reuses_immutable_artifact_when_environment_version_changes(tmp_path, monkeypatch):
    registry = tmp_path / "registry"
    split = tmp_path / "split.json"
    drift = tmp_path / "drift.json"
    first = bootstrap(registry, split_manifest_path=split, drift_reference_path=drift)
    first_environment = first[2]["training"]["environment"]
    monkeypatch.setattr(xgboost, "__version__", "different-runtime-version")
    second = bootstrap(registry, split_manifest_path=split, drift_reference_path=drift)
    assert second[0] == first[0]
    assert second[1] == first[1]
    assert second[2] == first[2]
    assert second[2]["training"]["environment"] == first_environment


def test_bootstrap_rejects_old_or_tampered_canonical_schema_hash(tmp_path):
    registry = tmp_path / "registry"
    split = tmp_path / "split.json"
    drift = tmp_path / "drift.json"
    version, directory, _ = bootstrap(registry, split_manifest_path=split, drift_reference_path=drift)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileExistsError, match="incompatible artifact or provenance"):
        bootstrap(registry, split_manifest_path=split, drift_reference_path=drift)
    assert (registry / version / "model.zip").is_file()


def test_exporter_output_option_keeps_committed_native_archive_unchanged(tmp_path):
    protected = Path("models/house_prices_native.zip")
    original_hash = sha256_file(protected)
    output = tmp_path / "temporary-parity.zip"
    result = export(output)
    assert result["artifact"] == str(output)
    assert output.is_file()
    assert NativeModel(output).feature_names == NativeModel(protected).feature_names
    assert sha256_file(protected) == original_hash


def test_exporter_refuses_explicit_output_to_protected_archive():
    protected = Path("models/house_prices_native.zip")
    original_hash = sha256_file(protected)
    with pytest.raises(ValueError, match="protected checked-in native archive"):
        export(protected)
    assert sha256_file(protected) == original_hash
