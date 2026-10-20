from __future__ import annotations

import json
import uuid

import pandas as pd
import pytest

from backend import event_store
from model_training.data_partitions import DEFAULT_DATASET, write_manifest
from model_training.simulate_truth import TruthImportConflict, import_truth


def _seed_request(db_path, *, row, source_id, dataset_hash, run_id, request_id=None, changed=False):
    split = __import__("model_training.data_partitions", fromlist=["load_manifest"]).load_manifest(
        db_path.parent / "split_manifest.json"
    )
    feature_names = split["source"]["feature_names"]
    numeric_features = split["source"]["numeric_features"]
    features = {name: None if pd.isna(row[name]) else row[name].item() if hasattr(row[name], "item") else row[name] for name in feature_names}
    if changed:
        features["GrLivArea"] = float(features["GrLivArea"] or 0) + 1
    canonical = event_store.canonical_record(features, feature_names=feature_names, numeric_features=numeric_features)
    fingerprint = event_store.feature_fingerprint(features, feature_names=feature_names, numeric_features=numeric_features)
    request_id = request_id or str(uuid.uuid4())
    connection = event_store.open_database(db_path)
    connection.execute(
        "INSERT INTO requests (request_id, received_at, completed_at, method, path, status_code, lifecycle, "
        "handler_duration_ms, body_text, body_truncated, parse_state, features_json, feature_fingerprint, "
        "prediction, model_version, cache_outcome, dataset_sha256, source_id, run_id, terminal_reason, logging_failed) "
        "VALUES (?, ?, ?, 'POST', '/predict', 200, 'completed', 1.0, '{}', 0, 'valid', ?, ?, 150000.0, 'bootstrap', 'miss', ?, ?, ?, NULL, 0)",
        (request_id, 100.0, 101.0, event_store.canonical_json(canonical), fingerprint, dataset_hash, source_id, run_id),
    )
    connection.commit()
    connection.close()
    return request_id


def test_delayed_truth_import_is_provenance_checked_and_idempotent(tmp_path):
    split_path = tmp_path / "split_manifest.json"
    split = write_manifest(split_path)
    event_store.initialize_database(tmp_path / "events.sqlite3")
    frame = pd.read_csv(DEFAULT_DATASET).set_index("Id", drop=False)
    source_id = split["partitions"]["adaptation_ids"][0]
    run_id = str(uuid.uuid4())
    request_id = _seed_request(
        tmp_path / "events.sqlite3",
        row=frame.loc[source_id],
        source_id=source_id,
        dataset_hash=split["source"]["sha256"],
        run_id=run_id,
    )
    private_path = tmp_path / "private-truth.json"
    private_path.write_text(json.dumps({
        "format": "house-prices-delayed-truth-v1",
        "run_id": run_id,
        "dataset_sha256": split["source"]["sha256"],
        "delay_seconds": 50.0,
        "labels": [{
            "source_id": source_id,
            "sale_price": float(frame.loc[source_id, "SalePrice"]),
            "request_ids": [request_id],
            "available_at": 152.0,
        }],
    }), encoding="utf-8")

    pending = import_truth(private_path, db_path=tmp_path / "events.sqlite3", split_manifest_path=split_path, now=151.0)
    imported = import_truth(private_path, db_path=tmp_path / "events.sqlite3", split_manifest_path=split_path, now=153.0)
    again = import_truth(private_path, db_path=tmp_path / "events.sqlite3", split_manifest_path=split_path, now=154.0)
    assert pending["not_yet_available"] == 1
    assert imported["imported"] == 1
    assert again["already_imported"] == 1
    connection = event_store.open_database(tmp_path / "events.sqlite3")
    try:
        assert connection.execute("SELECT COUNT(*) FROM labels").fetchone()[0] == 1
    finally:
        connection.close()
    connection = event_store.open_database(tmp_path / "events.sqlite3")
    connection.execute("UPDATE labels SET sale_price = sale_price + 1 WHERE source_id = ?", (source_id,))
    connection.commit()
    connection.close()
    with pytest.raises(TruthImportConflict, match="Conflicting previously imported label"):
        import_truth(private_path, db_path=tmp_path / "events.sqlite3", split_manifest_path=split_path, now=155.0)


def test_delayed_truth_refuses_changed_features_and_non_adaptation_ids(tmp_path):
    split_path = tmp_path / "split_manifest.json"
    split = write_manifest(split_path)
    db_path = tmp_path / "events.sqlite3"
    event_store.initialize_database(db_path)
    frame = pd.read_csv(DEFAULT_DATASET).set_index("Id", drop=False)
    source_id = split["partitions"]["adaptation_ids"][0]
    run_id = str(uuid.uuid4())
    changed_id = _seed_request(db_path, row=frame.loc[source_id], source_id=source_id, dataset_hash=split["source"]["sha256"], run_id=run_id, changed=True)
    path = tmp_path / "truth.json"
    path.write_text(json.dumps({
        "format": "house-prices-delayed-truth-v1",
        "run_id": run_id,
        "dataset_sha256": split["source"]["sha256"],
        "delay_seconds": 0.0,
        "labels": [{"source_id": source_id, "sale_price": float(frame.loc[source_id, "SalePrice"]), "request_ids": [changed_id], "available_at": 1.0}],
    }), encoding="utf-8")
    result = import_truth(path, db_path=db_path, split_manifest_path=split_path, now=2.0)
    assert result["ineligible"] == 1
    connection = event_store.open_database(db_path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM labels").fetchone()[0] == 0
    finally:
        connection.close()

    fit_id = split["partitions"]["original_fit_ids"][0]
    path.write_text(json.dumps({
        "format": "house-prices-delayed-truth-v1",
        "run_id": run_id,
        "dataset_sha256": split["source"]["sha256"],
        "delay_seconds": 0.0,
        "labels": [{"source_id": fit_id, "sale_price": float(frame.loc[fit_id, "SalePrice"]), "request_ids": [], "available_at": 1.0}],
    }), encoding="utf-8")
    with pytest.raises(TruthImportConflict, match="adaptation partition"):
        import_truth(path, db_path=db_path, split_manifest_path=split_path, now=2.0)


def test_truth_import_rejects_supplied_dataset_that_disagrees_with_manifest(tmp_path):
    split_path = tmp_path / "split_manifest.json"
    write_manifest(split_path)
    changed_dataset = tmp_path / "changed-train.csv"
    frame = pd.read_csv(DEFAULT_DATASET)
    frame.loc[0, "SalePrice"] = float(frame.loc[0, "SalePrice"]) + 1
    frame.to_csv(changed_dataset, index=False)
    private_path = tmp_path / "truth.json"
    private_path.write_text(json.dumps({
        "format": "house-prices-delayed-truth-v1",
        "run_id": "workload-run",
        "dataset_sha256": "not-the-original-hash",
        "delay_seconds": 0,
        "labels": [],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 does not match"):
        import_truth(
            private_path,
            db_path=tmp_path / "events.sqlite3",
            dataset_path=changed_dataset,
            split_manifest_path=split_path,
            now=10,
        )


def test_truth_delay_requires_completed_at_plus_configured_delay(tmp_path):
    split_path = tmp_path / "split_manifest.json"
    split = write_manifest(split_path)
    db_path = tmp_path / "events.sqlite3"
    event_store.initialize_database(db_path)
    frame = pd.read_csv(DEFAULT_DATASET).set_index("Id", drop=False)
    source_id = split["partitions"]["adaptation_ids"][0]
    run_id = str(uuid.uuid4())
    request_id = _seed_request(
        db_path,
        row=frame.loc[source_id],
        source_id=source_id,
        dataset_hash=split["source"]["sha256"],
        run_id=run_id,
    )
    private_path = tmp_path / "truth.json"
    private_path.write_text(json.dumps({
        "format": "house-prices-delayed-truth-v1",
        "run_id": run_id,
        "dataset_sha256": split["source"]["sha256"],
        "delay_seconds": 50,
        "labels": [{
            "source_id": source_id,
            "sale_price": float(frame.loc[source_id, "SalePrice"]),
            "request_ids": [request_id],
            "available_at": 150.99,
        }],
    }), encoding="utf-8")
    result = import_truth(private_path, db_path=db_path, split_manifest_path=split_path, now=151)
    assert result["ineligible"] == 1
    assert result["imported"] == 0
    connection = event_store.open_database(db_path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM labels").fetchone()[0] == 0
    finally:
        connection.close()
