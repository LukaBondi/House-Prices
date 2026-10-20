from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import jensenshannon

from backend.monitoring import _feature_histogram
from model_training.build_drift_reference import build_reference
from model_training.data_partitions import (
    FROZEN_DATASET_SHA256,
    load_manifest,
    sha256_file,
    validate_split_manifest,
    write_manifest,
)


def test_split_manifest_reconstructs_exact_disjoint_partitions(tmp_path):
    path = tmp_path / "split_manifest.json"
    manifest = write_manifest(path)
    loaded = load_manifest(path)

    partitions = loaded["partitions"]
    ids = [
        set(partitions["original_fit_ids"]),
        set(partitions["adaptation_ids"]),
        set(partitions["promotion_gate_ids"]),
        set(partitions["final_audit_ids"]),
    ]
    assert [len(values) for values in ids] == [1168, 146, 73, 73]
    assert sum(len(values) for values in ids) == 1460
    assert all(not left.intersection(right) for index, left in enumerate(ids) for right in ids[index + 1 :])
    assert manifest["source"]["sha256"] == FROZEN_DATASET_SHA256
    assert "SalePrice" not in manifest["source"]["feature_names"]
    assert "Id" not in manifest["source"]["feature_names"]
    assert sha256_file(path) == "ee20e8982d27f4953035f627e3a8edb75078586cda919b742443a804e9c78b64"


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("original_fit_ids", "promotion_gate_ids"),
        ("adaptation_ids", "promotion_gate_ids"),
    ],
)
def test_split_manifest_rejects_swapped_ids_even_when_counts_and_coverage_survive(tmp_path, left, right):
    manifest = write_manifest(tmp_path / "split_manifest.json")
    manifest["partitions"][left][0], manifest["partitions"][right][0] = (
        manifest["partitions"][right][0],
        manifest["partitions"][left][0],
    )
    with pytest.raises(ValueError, match="deterministic original split"):
        validate_split_manifest(manifest)


def test_drift_reference_matches_backend_base2_bucketing(tmp_path):
    manifest_path = tmp_path / "split_manifest.json"
    write_manifest(manifest_path)
    reference = build_reference(manifest_path=manifest_path)
    assert len(reference["fit_ids"]) == 1168
    assert "Id" not in reference["features"]
    assert "SalePrice" not in reference["features"]
    frame = pd.read_csv("dataset/train.csv").set_index("Id", drop=False)
    fitted = frame.loc[reference["fit_ids"], list(reference["features"])]
    records = [
        {name: None if pd.isna(value) else value for name, value in row.items()}
        for row in fitted.to_dict(orient="records")
    ]

    for feature, spec in reference["features"].items():
        if spec["kind"] == "numeric":
            counts = spec["bins"]["counts"]
            assert len(counts) == len(spec["bins"]["edges"]) + 3
            assert len(spec["bins"]["probabilities"]) == len(counts)
            assert np.isclose(sum(spec["bins"]["probabilities"]), 1.0)
        else:
            counts = spec["counts"]
            assert len(counts) == len(spec["categories"]) + 2
            assert np.isclose(sum(spec["probabilities"]), 1.0)
        result = _feature_histogram(records, feature, spec)
        assert result["distance"] == 0.0

    same_distribution = jensenshannon([0.25, 0.75], [0.25, 0.75], base=2)
    assert same_distribution == 0.0
    assert json.loads(json.dumps(reference, allow_nan=False))["reference_id"] == reference["reference_id"]


def test_drift_reference_exposes_underflow_overflow_missing_and_unknown(tmp_path):
    manifest_path = tmp_path / "split_manifest.json"
    write_manifest(manifest_path)
    reference = build_reference(manifest_path=manifest_path)
    numeric_name = "OverallQual"
    categorical_name = "Neighborhood"
    numeric = reference["features"][numeric_name]
    categorical = reference["features"][categorical_name]
    records = [
        {numeric_name: None, categorical_name: None},
        {numeric_name: numeric["minimum"] - 1, categorical_name: "unknown-neighborhood"},
        {numeric_name: numeric["maximum"] + 1, categorical_name: categorical["categories"][0]},
    ]
    numeric_result = _feature_histogram(records, numeric_name, numeric)
    categorical_result = _feature_histogram(records, categorical_name, categorical)
    assert numeric_result["missing_count"] == 1
    assert numeric_result["underflow_count"] == 1
    assert numeric_result["overflow_count"] == 1
    assert categorical_result["missing_count"] == 1
    assert categorical_result["unknown_count"] == 1


def test_drift_upper_edges_are_included_in_their_reference_buckets():
    spec = {
        "kind": "numeric",
        "bins": {"minimum": 1, "edges": [2, 4, 6], "counts": [0, 0, 0, 1, 0, 0]},
    }
    at_interior_upper_edge = _feature_histogram([{"x": 4}], "x", spec)
    assert at_interior_upper_edge["distance"] == 0.0

    maximum_spec = {
        "kind": "numeric",
        "bins": {"minimum": 1, "edges": [2, 4, 6], "counts": [0, 0, 0, 0, 1, 0]},
    }
    at_maximum_upper_edge = _feature_histogram([{"x": 6}], "x", maximum_spec)
    assert at_maximum_upper_edge["distance"] == 0.0


def test_split_and_drift_references_refuse_content_replacement(tmp_path):
    from model_training.build_drift_reference import write_reference

    manifest_path = tmp_path / "split_manifest.json"
    write_manifest(manifest_path)
    reference_path = tmp_path / "drift_reference.json"
    write_reference(reference_path, manifest_path=manifest_path)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    reference["reference_id"] = "edited"
    reference_path.write_text(json.dumps(reference), encoding="utf-8")
    with pytest.raises(FileExistsError, match="immutable drift reference"):
        write_reference(reference_path, manifest_path=manifest_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["split"]["random_state"] = 7
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileExistsError, match="immutable split manifest"):
        write_manifest(manifest_path)
