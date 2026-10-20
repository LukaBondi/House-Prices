"""Deterministic monitoring windows and fixed-reference drift tests."""

import hashlib
import json
from pathlib import Path

import pytest

from backend.event_store import canonical_json, initialize_database, open_database
from backend.monitoring import (
    MonitoringError,
    _feature_histogram,
    _load_reference,
    build_monitoring_summary,
)


def _write_reference(path):
    source = Path(__file__).resolve().parents[1] / "models" / "drift_reference.json"
    reference = json.loads(source.read_text(encoding="utf-8"))
    path.write_text(json.dumps(reference), encoding="utf-8")


def _insert_request(connection, index, *, path="/predict", status=200, record=None):
    encoded = canonical_json(record) if record is not None else None
    fingerprint = f"fingerprint-{index}" if record is not None else None
    connection.execute(
        "INSERT INTO requests "
        "(request_id, received_at, completed_at, method, path, status_code, lifecycle, "
        "handler_duration_ms, features_json, feature_fingerprint, cache_outcome, model_version) "
        "VALUES (?, ?, ?, 'POST', ?, ?, 'completed', ?, ?, ?, 'miss', 'version-a')",
        (
            f"request-{index}",
            1900.0,
            1900.1,
            path,
            status,
            float(index + 1),
            encoded,
            fingerprint,
        ),
    )


def test_prediction_window_excludes_polling_and_reports_shifted_inputs(tmp_path):
    database = tmp_path / "monitoring.sqlite3"
    reference_path = tmp_path / "drift-reference.json"
    initialize_database(database)
    _write_reference(reference_path)
    connection = open_database(database)
    try:
        for index in range(110):
            _insert_request(
                connection,
                index,
                record={
                    "OverallQual": 12.0 + index / 100.0,
                    "Neighborhood": "unseen",
                },
            )
        _insert_request(connection, 1000, path="/health", record=None)
        connection.execute(
            "INSERT INTO labels "
            "(observation_id, request_id, dataset_sha256, source_id, features_json, "
            "feature_fingerprint, sale_price, available_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "observation-a",
                "request-0",
                "a" * 64,
                2000,
                '{"Neighborhood":"unseen","OverallQual":12.0}',
                "fingerprint-0",
                250000.0,
                1900.0,
            ),
        )
        connection.execute(
            "INSERT INTO runs (run_id, status, started_at, completed_at) "
            "VALUES (?, ?, ?, ?)",
            ("run-a", "rejected", 1899.0, 1900.0),
        )
        connection.commit()
    finally:
        connection.close()

    summary = build_monitoring_summary(
        database,
        reference_path,
        window_minutes=15,
        current_model_version="version-a",
        logging_status={"enabled": True, "healthy": True, "failures": 0},
        as_of=1901.0,
    )
    assert summary["requests"]["total"] == 110
    assert summary["requests"]["completed"] == 110
    assert summary["requests"]["distinct_inputs"] == 110
    assert summary["drift"]["sample_count"] == 110
    assert summary["drift"]["reference_id"] == json.loads(
        reference_path.read_text(encoding="utf-8")
    )["reference_id"]
    assert summary["drift"]["status"] == "advisory"
    assert summary["drift"]["features"]["OverallQual"]["overflow_count"] == 110
    assert summary["drift"]["features"]["Neighborhood"]["unknown_count"] == 110
    assert summary["labels"] == {"count": 1, "distinct_observations": 1}
    assert summary["retraining"]["last_run"]["status"] == "rejected"
    assert summary["model"]["version"] == "version-a"


def test_empty_and_low_sample_windows_do_not_invent_percentiles_or_flags(tmp_path):
    database = tmp_path / "monitoring.sqlite3"
    reference_path = tmp_path / "drift-reference.json"
    initialize_database(database)
    _write_reference(reference_path)
    connection = open_database(database)
    try:
        for index in range(3):
            _insert_request(
                connection,
                index,
                record={"OverallQual": 12.0, "Neighborhood": "unseen"},
            )
        connection.commit()
    finally:
        connection.close()

    low_sample = build_monitoring_summary(
        database, reference_path, as_of=1901.0
    )
    assert low_sample["drift"]["status"] == "insufficient_data"
    assert low_sample["latency"]["handler_p50_ms"] == 2.0
    assert low_sample["errors"]["4xx"]["rate"] == 0.0

    empty = build_monitoring_summary(
        database, reference_path, window_minutes=60, as_of=999999.0
    )
    assert empty["requests"]["total"] == 0
    assert empty["latency"]["handler_p50_ms"] is None
    assert empty["errors"]["4xx"]["rate"] is None
    assert empty["drift"]["status"] == "insufficient_data"


def test_reference_rejects_rehashed_untrusted_provenance_or_bucket_counts(tmp_path):
    reference_path = tmp_path / "drift-reference.json"
    _write_reference(reference_path)
    original = json.loads(reference_path.read_text(encoding="utf-8"))

    for mutate in (
        lambda value: value.update(source_dataset_sha256="a" * 64),
        lambda value: value["fit_ids"].__setitem__(0, value["fit_ids"][1]),
        lambda value: value["features"]["OverallQual"]["bins"]["counts"].__setitem__(2, 999),
    ):
        damaged = json.loads(json.dumps(original))
        mutate(damaged)
        damaged.pop("reference_id")
        if damaged.get("fit_ids") != original["fit_ids"]:
            damaged["fit_ids_sha256"] = hashlib.sha256(
                canonical_json(damaged["fit_ids"]).encode("utf-8")
            ).hexdigest()
        damaged["reference_id"] = hashlib.sha256(
            canonical_json(damaged).encode("utf-8")
        ).hexdigest()
        reference_path.write_text(json.dumps(damaged), encoding="utf-8")
        with pytest.raises(MonitoringError):
            _load_reference(reference_path)


def test_numeric_drift_uses_inclusive_upper_edges_and_explicit_missing_buckets(tmp_path):
    reference_path = tmp_path / "drift-reference.json"
    _write_reference(reference_path)
    reference, _ = _load_reference(reference_path)
    assert reference is not None
    spec = reference["features"]["OverallQual"]
    first_edge, final_edge = spec["bins"]["edges"][0], spec["bins"]["edges"][-1]
    minimum = spec["minimum"]

    result = _feature_histogram(
        [
            {"OverallQual": None},
            {"OverallQual": minimum - 1},
            {"OverallQual": first_edge},
            {"OverallQual": final_edge},
            {"OverallQual": final_edge + 1},
        ],
        "OverallQual",
        spec,
    )

    assert result["missing_count"] == 1
    assert result["underflow_count"] == 1
    assert result["overflow_count"] == 1
    assert result["nonmissing_count"] == 4
