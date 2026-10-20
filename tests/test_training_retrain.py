from __future__ import annotations

import json
import threading
import time
import urllib.error

import pandas as pd
import pytest

from backend import event_store
from backend.artifacts import schema_sha256 as backend_schema_sha256
from model_training import retrain
from model_training.bootstrap_registry import bootstrap
from model_training.data_partitions import load_manifest, write_manifest
from model_training.retrain import TrainerLeaseLost, _eligible_labels, run_once


def _seed_adaptation_labels(db_path, split_path, source_ids, *, completed_at=100.0, available_at=101.0):
    split = load_manifest(split_path)
    frame = pd.read_csv("dataset/train.csv").set_index("Id", drop=False)
    feature_names = split["source"]["feature_names"]
    numeric_features = split["source"]["numeric_features"]
    event_store.initialize_database(db_path)
    run_id = "workload-fixture"
    connection = event_store.open_database(db_path)
    try:
        for source_id in source_ids:
            row = frame.loc[source_id]
            raw = {
                name: None if pd.isna(row[name]) else row[name].item() if hasattr(row[name], "item") else row[name]
                for name in feature_names
            }
            canonical = event_store.canonical_record(raw, feature_names=feature_names, numeric_features=numeric_features)
            fingerprint = event_store.feature_fingerprint(raw, feature_names=feature_names, numeric_features=numeric_features)
            request_id = f"request-{source_id}"
            connection.execute(
                "INSERT INTO requests (request_id, received_at, completed_at, method, path, status_code, lifecycle, "
                "handler_duration_ms, body_text, body_truncated, parse_state, features_json, feature_fingerprint, "
                "prediction, model_version, cache_outcome, dataset_sha256, source_id, run_id, terminal_reason, logging_failed) "
                "VALUES (?, ?, ?, 'POST', '/predict', 200, 'completed', 1.0, '{}', 0, 'valid', ?, ?, 150000.0, 'bootstrap-fixture', 'miss', ?, ?, ?, NULL, 0)",
                (request_id, completed_at - 1, completed_at, event_store.canonical_json(canonical), fingerprint,
                 split["source"]["sha256"], source_id, run_id),
            )
            connection.execute(
                "INSERT INTO labels (observation_id, request_id, dataset_sha256, source_id, features_json, feature_fingerprint, sale_price, available_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (f"{split['source']['sha256']}:{source_id}", request_id, split["source"]["sha256"], source_id,
                 event_store.canonical_json(canonical), fingerprint, float(row["SalePrice"]), available_at),
            )
        connection.commit()
    finally:
        connection.close()


def test_retrainer_records_one_controlled_insufficient_label_skip(tmp_path):
    split_path = tmp_path / "split_manifest.json"
    reference_path = tmp_path / "drift_reference.json"
    registry_path = tmp_path / "registry"
    state_path = tmp_path / "state"
    version, _, bootstrap_manifest = bootstrap(
        registry_path, split_manifest_path=split_path, drift_reference_path=reference_path
    )
    assert version.startswith("bootstrap-")
    assert bootstrap_manifest["schema_sha256"] == backend_schema_sha256("models/feature_schema.json")
    db_path = tmp_path / "events.sqlite3"

    first = run_once(
        db_path=db_path,
        registry_path=registry_path,
        state_path=state_path,
        split_manifest_path=split_path,
        now=1_800_000_000.0,
    )
    second = run_once(
        db_path=db_path,
        registry_path=registry_path,
        state_path=state_path,
        split_manifest_path=split_path,
        now=1_800_000_001.0,
    )
    assert first["status"] == "skipped"
    assert first["reason"] == "insufficient_new_labels"
    assert second["status"] == "skipped"
    assert second["already_recorded"] is True
    connection = event_store.open_database(db_path)
    try:
        rows = connection.execute("SELECT status, details_json FROM runs").fetchall()
        assert len(rows) == 1
        assert rows[0]["status"] == "skipped"
        assert json.loads(rows[0]["details_json"])["new_ids"] == []
    finally:
        connection.close()


def test_retrainer_honors_live_singleton_lease(tmp_path):
    split_path = tmp_path / "split_manifest.json"
    registry_path = tmp_path / "registry"
    bootstrap(registry_path, split_manifest_path=split_path, drift_reference_path=tmp_path / "drift.json")
    db_path = tmp_path / "events.sqlite3"
    event_store.initialize_database(db_path)
    connection = event_store.open_database(db_path)
    connection.execute(
        "INSERT INTO runs (run_id, parent_version, snapshot_hash, status, started_at, completed_at, metrics_json, details_json) "
        "VALUES ('active-run', 'bootstrap-placeholder', NULL, 'running', ?, NULL, NULL, '{}')",
        (1_800_000_000.0,),
    )
    connection.execute(
        "INSERT INTO trainer_leases (lease_name, owner_id, acquired_at, expires_at, snapshot_hash, details_json) "
        "VALUES ('retrain', 'active-run', ?, ?, NULL, '{}')",
        (1_800_000_000.0, 1_800_000_100.0),
    )
    connection.commit()
    connection.close()
    result = run_once(
        db_path=db_path,
        registry_path=registry_path,
        state_path=tmp_path / "state",
        split_manifest_path=split_path,
        now=1_800_000_001.0,
        lease_seconds=100.0,
    )
    assert result == {"status": "skipped", "reason": "trainer_lease_held", "run_id": "active-run"}


def test_activation_client_requires_acknowledged_commit_and_queries_lost_ack(monkeypatch):
    class Response:
        def __init__(self, value):
            self.value = json.dumps(value).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            return self.value

    activation_id = "activation-fixture"
    active_version = "model-fixture"
    acknowledgement = {
        "activation_id": activation_id,
        "status": "committed",
        "active_version": active_version,
    }
    requests = []

    def committed_post(request, timeout):
        requests.append(request)
        assert timeout == 15.0
        assert request.get_header("Authorization") == "Bearer private-token"
        return Response(acknowledgement)

    monkeypatch.setattr(retrain.urllib.request, "urlopen", committed_post)
    result = retrain._request_activation(
        base_url="http://api",
        token="private-token",
        version=active_version,
        expected_parent="bootstrap-parent",
        activation_id=activation_id,
    )
    assert result == acknowledgement
    assert requests[0].get_method() == "POST"
    assert json.loads(requests[0].data) == {
        "candidate_version": active_version,
        "expected_parent": "bootstrap-parent",
        "activation_id": activation_id,
    }

    calls = []

    def lost_post_then_status(request, timeout):
        calls.append(request.get_method())
        if request.get_method() == "POST":
            raise urllib.error.URLError("ack lost")
        return Response(acknowledgement)

    monkeypatch.setattr(retrain.urllib.request, "urlopen", lost_post_then_status)
    status = retrain._request_activation(
        base_url="http://api",
        token="private-token",
        version=active_version,
        expected_parent="bootstrap-parent",
        activation_id=activation_id,
    )
    assert status == acknowledgement
    assert calls == ["POST", "GET"]


def test_activation_client_sends_trainer_run_fence(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            return json.dumps({"activation_id": "activation-1", "status": "committed", "active_version": "model-1"}).encode()

    requests = []
    monkeypatch.setattr(retrain.urllib.request, "urlopen", lambda request, timeout: (requests.append(request) or Response()))
    retrain._request_activation(
        base_url="http://api",
        token="private-token",
        version="model-1",
        expected_parent="bootstrap-1",
        activation_id="activation-1",
        trainer_run_id="trainer-run-1",
    )
    assert json.loads(requests[0].data)["trainer_run_id"] == "trainer-run-1"


def test_lease_heartbeat_renews_only_while_owned(tmp_path):
    db_path = tmp_path / "events.sqlite3"
    event_store.initialize_database(db_path)
    now = time.time()
    connection = event_store.open_database(db_path)
    connection.execute(
        "INSERT INTO runs (run_id, parent_version, snapshot_hash, status, started_at, completed_at, metrics_json, details_json) "
        "VALUES ('lease-run', 'parent', 'snapshot', 'running', ?, NULL, NULL, '{}')",
        (now,),
    )
    connection.execute(
        "INSERT INTO trainer_leases (lease_name, owner_id, acquired_at, expires_at, snapshot_hash, details_json) "
        "VALUES ('retrain', 'lease-run', ?, ?, 'snapshot', '{}')",
        (now, now + 0.3),
    )
    connection.commit()
    connection.close()
    lease = retrain.TrainerLease(db_path, "lease-run", 0.3, now)
    lease.start()
    time.sleep(0.15)
    try:
        lease.assert_owned()
        connection = event_store.open_database(db_path)
        try:
            expiry = connection.execute("SELECT expires_at FROM trainer_leases WHERE lease_name='retrain'").fetchone()[0]
        finally:
            connection.close()
        assert expiry > now + 0.3
    finally:
        lease.stop()


def test_lease_heartbeat_never_renews_an_expired_owner(tmp_path):
    db_path = tmp_path / "events.sqlite3"
    event_store.initialize_database(db_path)
    now = time.time()
    connection = event_store.open_database(db_path)
    connection.execute(
        "INSERT INTO runs (run_id, parent_version, snapshot_hash, status, started_at, completed_at, metrics_json, details_json) "
        "VALUES ('expired-run', 'parent', 'snapshot', 'running', ?, NULL, NULL, '{}')",
        (now - 2,),
    )
    connection.execute(
        "INSERT INTO trainer_leases (lease_name, owner_id, acquired_at, expires_at, snapshot_hash, details_json) "
        "VALUES ('retrain', 'expired-run', ?, ?, 'snapshot', '{}')",
        (now - 2, now - 1),
    )
    connection.commit()
    connection.close()
    lease = retrain.TrainerLease(db_path, "expired-run", 0.06, now)
    lease.start()
    try:
        time.sleep(0.04)
        assert lease.lost
        connection = event_store.open_database(db_path)
        try:
            expiry = connection.execute("SELECT expires_at FROM trainer_leases WHERE lease_name='retrain'").fetchone()[0]
        finally:
            connection.close()
        assert expiry == pytest.approx(now - 1)
    finally:
        lease.stop()


def test_expiration_joins_inflight_heartbeat_after_sqlite_lock_spans_lease_expiry(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "events.sqlite3"
    event_store.initialize_database(db_path)
    now = time.time()
    initial_expiry = now + 0.08
    connection = event_store.open_database(db_path)
    connection.execute(
        "INSERT INTO runs (run_id, parent_version, snapshot_hash, status, started_at, completed_at, metrics_json, details_json) "
        "VALUES ('race-run', 'parent', 'snapshot', 'running', ?, NULL, NULL, '{}')",
        (now,),
    )
    connection.execute(
        "INSERT INTO trainer_leases (lease_name, owner_id, acquired_at, expires_at, snapshot_hash, details_json) "
        "VALUES ('retrain', 'race-run', ?, ?, 'snapshot', '{}')",
        (now, initial_expiry),
    )
    connection.commit()
    connection.close()

    heartbeat_opened = threading.Event()
    original_open = event_store.open_database

    def track_heartbeat_connection(path):
        if threading.current_thread().name == "trainer-lease-race-run":
            heartbeat_opened.set()
        return original_open(path)

    monkeypatch.setattr(retrain.event_store, "open_database", track_heartbeat_connection)
    blocker = original_open(db_path)
    blocker.execute("BEGIN IMMEDIATE")
    blocker_active = True
    lease = retrain.TrainerLease(db_path, "race-run", 0.06, now)
    lease.start()
    expire_done = threading.Event()

    def expire_owner():
        try:
            lease.expire_owned_lease()
        finally:
            expire_done.set()

    expirer = threading.Thread(target=expire_owner)
    try:
        assert heartbeat_opened.wait(timeout=2.0)
        time.sleep(0.03)  # heartbeat is now waiting on the held BEGIN IMMEDIATE
        expirer.start()
        time.sleep(0.08)  # the lease expires while heartbeat/expiration wait
        blocker.commit()
        blocker.close()
        blocker_active = False
        assert expire_done.wait(timeout=3.0)
        assert lease.lost
        assert not lease._thread.is_alive()
        connection = original_open(db_path)
        try:
            expiry = connection.execute(
                "SELECT expires_at FROM trainer_leases WHERE lease_name='retrain'"
            ).fetchone()[0]
        finally:
            connection.close()
        assert expiry == pytest.approx(initial_expiry)
    finally:
        if blocker_active:
            blocker.rollback()
            blocker.close()
        lease.stop()
        if expirer.ident is not None:
            expirer.join(timeout=3.0)


def test_cutoff_rejects_post_cutoff_completion_and_normalizes_missing_values(tmp_path):
    split_path = tmp_path / "split_manifest.json"
    split = write_manifest(split_path)
    db_path = tmp_path / "events.sqlite3"
    adaptation_ids = split["partitions"]["adaptation_ids"][:2]
    _seed_adaptation_labels(db_path, split_path, adaptation_ids)
    source = pd.read_csv("dataset/train.csv").set_index("Id", drop=False)
    connection = event_store.open_database(db_path)
    try:
        labels = _eligible_labels(
            connection,
            cutoff=102.0,
            dataset_sha256=split["source"]["sha256"],
            source_frame=source,
            split=split,
        )
        assert len(labels) == 2
        assert labels[0]["features_json"] == event_store.canonical_json(
            event_store.canonical_record(
                {name: None if pd.isna(source.loc[893, name]) else source.loc[893, name] for name in split["source"]["feature_names"]},
                feature_names=split["source"]["feature_names"],
                numeric_features=split["source"]["numeric_features"],
            )
        )
        connection.execute("UPDATE requests SET completed_at=200 WHERE source_id=?", (adaptation_ids[1],))
        connection.commit()
        with pytest.raises(retrain.TrainingDataError, match="successful persisted prediction"):
            _eligible_labels(
                connection,
                cutoff=150.0,
                dataset_sha256=split["source"]["sha256"],
                source_frame=source,
                split=split,
            )
    finally:
        connection.close()


def test_retry_backoff_cannot_be_bypassed_by_later_labels(tmp_path):
    split_path = tmp_path / "split_manifest.json"
    split = write_manifest(split_path)
    registry_path = tmp_path / "registry"
    parent, _, _ = bootstrap(
        registry_path, split_manifest_path=split_path, drift_reference_path=tmp_path / "drift.json"
    )
    db_path = tmp_path / "events.sqlite3"
    ids = split["partitions"]["adaptation_ids"][:32]
    _seed_adaptation_labels(db_path, split_path, ids)
    now = 1_800_000_000.0
    connection = event_store.open_database(db_path)
    connection.execute(
        "INSERT INTO runs (run_id, parent_version, snapshot_hash, status, started_at, completed_at, metrics_json, details_json) "
        "VALUES ('failed-snapshot', ?, 'old-snapshot', 'failed', ?, ?, NULL, ?)",
        (parent, now - 1, now - 1, event_store.canonical_json({"identity_hash": "older-label-set", "cutoff": now - 50, "retry_after": now + 60, "retry_number": 1})),
    )
    connection.commit()
    connection.close()
    result = run_once(
        db_path=db_path,
        registry_path=registry_path,
        state_path=tmp_path / "state",
        split_manifest_path=split_path,
        now=now,
    )
    assert result["status"] == "skipped"
    assert result["reason"] == "retry_backoff"
    assert result["run_id"] == "failed-snapshot"


def test_reconciliation_queries_activation_status_outside_database_write_lock(tmp_path, monkeypatch):
    db_path = tmp_path / "events.sqlite3"
    state_path = tmp_path / "state"
    state_path.mkdir()
    (state_path / "current.json").write_text(
        json.dumps({"version": "parent", "activation_id": None, "previous_version": None})
    )
    event_store.initialize_database(db_path)
    connection = event_store.open_database(db_path)
    run_details = event_store.canonical_json(
        {
            "activation_id": "activation-1",
            "candidate_version": "model-1",
            "snapshot_hash": "snapshot",
        }
    )
    connection.execute(
        "INSERT INTO runs (run_id, parent_version, snapshot_hash, status, started_at, completed_at, metrics_json, details_json) "
        "VALUES ('stale-run', 'parent', 'snapshot', 'running', 1, NULL, NULL, ?)",
        (run_details,),
    )
    connection.execute(
        "INSERT INTO activations "
        "(activation_id, expected_parent, candidate_version, status, created_at, details_json) "
        "VALUES ('activation-1', 'parent', 'model-1', 'intent', 1, ?)",
        (event_store.canonical_json({"rollback": False, "trainer_run_id": "stale-run"}),),
    )
    connection.execute(
        "INSERT INTO trainer_leases (lease_name, owner_id, acquired_at, expires_at, snapshot_hash, details_json) "
        "VALUES ('retrain', 'stale-run', 1, 2, 'snapshot', ?) ",
        (run_details,),
    )
    connection.commit()
    connection.close()

    def status_with_write(_base_url, _token, _activation_id):
        separate = event_store.open_database(db_path)
        try:
            separate.execute("BEGIN IMMEDIATE")
            separate.execute("UPDATE runs SET details_json=details_json WHERE run_id='stale-run'")
            separate.commit()
        finally:
            separate.close()
        return None

    monkeypatch.setattr(retrain, "_activation_status", status_with_write)
    pending = retrain._reconcile_stale_runs(
        db_path,
        now=3.0,
        lease_seconds=10.0,
        api_base_url="http://api",
        admin_token="private-token",
        state_path=state_path,
    )
    assert pending == [{"run_id": "stale-run", "activation_id": "activation-1"}]
    connection = event_store.open_database(db_path)
    try:
        assert connection.execute("SELECT status FROM runs WHERE run_id='stale-run'").fetchone()[0] == "running"
        assert connection.execute("SELECT COUNT(*) FROM trainer_leases").fetchone()[0] == 1
    finally:
        connection.close()

    result = run_once(
        db_path=db_path,
        registry_path=tmp_path / "registry-not-read",
        state_path=state_path,
        now=3.1,
        lease_seconds=10.0,
        api_base_url="http://api",
        admin_token="private-token",
    )
    assert result["status"] == "skipped"
    assert result["reason"] == "activation_reconciliation_pending"
    assert result["pending_run_ids"] == ["stale-run"]
    connection = event_store.open_database(db_path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM trainer_leases").fetchone()[0] == 1
    finally:
        connection.close()


def test_stale_activation_survives_status_outage_and_recovers_shared_commit(tmp_path, monkeypatch):
    db_path = tmp_path / "events.sqlite3"
    state_path = tmp_path / "state"
    state_path.mkdir()
    pointer_path = state_path / "current.json"
    pointer_path.write_text(
        json.dumps({"version": "parent", "activation_id": None, "previous_version": None})
    )
    event_store.initialize_database(db_path)
    run_details = event_store.canonical_json(
        {
            "activation_id": "activation-recover",
            "candidate_version": "model-recover",
            "snapshot_hash": "snapshot-recover",
            "identity_hash": "identity-recover",
            "cutoff": 10.0,
            "eligible_ids": [11, 12],
        }
    )
    connection = event_store.open_database(db_path)
    connection.execute(
        "INSERT INTO runs (run_id, parent_version, snapshot_hash, status, started_at, completed_at, metrics_json, details_json) "
        "VALUES ('recover-run', 'parent', 'snapshot-recover', 'running', 1, NULL, NULL, ?)",
        (run_details,),
    )
    connection.execute(
        "INSERT INTO activations "
        "(activation_id, expected_parent, candidate_version, status, created_at, details_json) "
        "VALUES ('activation-recover', 'parent', 'model-recover', 'intent', 1, ?)",
        (event_store.canonical_json({"rollback": False, "trainer_run_id": "recover-run"}),),
    )
    connection.execute(
        "INSERT INTO trainer_leases (lease_name, owner_id, acquired_at, expires_at, snapshot_hash, details_json) "
        "VALUES ('retrain', 'recover-run', 1, 2, 'snapshot-recover', ?)",
        (run_details,),
    )
    connection.commit()
    connection.close()

    status_calls = []

    def unavailable_then_committed(_base_url, _token, activation_id, timeout=10.0):
        status_calls.append((activation_id, timeout))
        if len(status_calls) < 3:
            return None
        return {
            "activation_id": activation_id,
            "candidate_version": "model-recover",
            "status": "committed",
            "active_version": "a-later-model",
        }

    monkeypatch.setattr(retrain, "_activation_status", unavailable_then_committed)
    for attempt, current_time in enumerate((3.0, 4.0), start=1):
        pending = retrain._reconcile_stale_runs(
            db_path,
            now=current_time,
            lease_seconds=10.0,
            api_base_url="http://api",
            admin_token="private-token",
            state_path=state_path,
        )
        assert pending == [{"run_id": "recover-run", "activation_id": "activation-recover"}]
        connection = event_store.open_database(db_path)
        try:
            assert connection.execute(
                "SELECT status FROM runs WHERE run_id='recover-run'"
            ).fetchone()[0] == "running"
            assert connection.execute(
                "SELECT COUNT(*) FROM trainer_leases WHERE owner_id='recover-run'"
            ).fetchone()[0] == 1
        finally:
            connection.close()

    # A matching shared audit row is the durable authority even if the current
    # active pointer has since advanced beyond this historical candidate.
    connection = event_store.open_database(db_path)
    connection.execute(
        "UPDATE activations SET status='committed', completed_at=4, details_json=? "
        "WHERE activation_id='activation-recover'",
        (event_store.canonical_json({
            "rollback": False,
            "trainer_run_id": "recover-run",
            "active_version": "model-recover",
        }),),
    )
    pointer_path.write_text(
        json.dumps({"version": "a-later-model", "activation_id": "later-id", "previous_version": "model-recover"})
    )
    connection.commit()
    connection.close()
    pending = retrain._reconcile_stale_runs(
        db_path,
        now=5.0,
        lease_seconds=10.0,
        api_base_url="http://api",
        admin_token="private-token",
        state_path=state_path,
    )
    assert pending == []
    assert status_calls == [("activation-recover", 10.0)] * 3
    connection = event_store.open_database(db_path)
    try:
        run = connection.execute(
            "SELECT status, details_json FROM runs WHERE run_id='recover-run'"
        ).fetchone()
        assert run["status"] == "promoted"
        details = json.loads(run["details_json"])
        assert details["candidate_version"] == "model-recover"
        assert details["activation_id"] == "activation-recover"
        assert details["snapshot_hash"] == "snapshot-recover"
        assert details["eligible_ids"] == [11, 12]
        assert connection.execute(
            "SELECT COUNT(*) FROM trainer_leases WHERE owner_id='recover-run'"
        ).fetchone()[0] == 0
    finally:
        connection.close()


def test_expired_run_before_activation_request_is_terminalized_without_pointer(tmp_path):
    db_path = tmp_path / "events.sqlite3"
    state_path = tmp_path / "missing-state"
    event_store.initialize_database(db_path)
    details = event_store.canonical_json(
        {"snapshot_hash": "snapshot-before-activation", "label_count": 32}
    )
    connection = event_store.open_database(db_path)
    connection.execute(
        "INSERT INTO runs (run_id, parent_version, snapshot_hash, status, started_at, completed_at, metrics_json, details_json) "
        "VALUES ('pre-activation-run', 'parent', 'snapshot-before-activation', 'running', 1, NULL, NULL, ?)",
        (details,),
    )
    connection.execute(
        "INSERT INTO trainer_leases (lease_name, owner_id, acquired_at, expires_at, snapshot_hash, details_json) "
        "VALUES ('retrain', 'pre-activation-run', 1, 2, 'snapshot-before-activation', ?)",
        (details,),
    )
    connection.commit()
    connection.close()

    pending = retrain._reconcile_stale_runs(
        db_path,
        now=3.0,
        lease_seconds=10.0,
        api_base_url="http://api",
        admin_token=None,
        state_path=state_path,
    )
    assert pending == []
    connection = event_store.open_database(db_path)
    try:
        run = connection.execute(
            "SELECT status, details_json FROM runs WHERE run_id='pre-activation-run'"
        ).fetchone()
        assert run["status"] == "failed"
        assert json.loads(run["details_json"])["snapshot_hash"] == "snapshot-before-activation"
        assert connection.execute(
            "SELECT COUNT(*) FROM trainer_leases WHERE owner_id='pre-activation-run'"
        ).fetchone()[0] == 0
    finally:
        connection.close()


def test_missing_activation_intent_requires_pointer_still_at_parent(tmp_path, monkeypatch):
    db_path = tmp_path / "events.sqlite3"
    state_path = tmp_path / "state"
    state_path.mkdir()
    (state_path / "current.json").write_text(
        json.dumps({"version": "parent", "activation_id": None, "previous_version": None})
    )
    event_store.initialize_database(db_path)
    details = event_store.canonical_json(
        {
            "activation_id": "never-recorded-activation",
            "candidate_version": "candidate-never-recorded",
            "snapshot_hash": "snapshot-no-intent",
        }
    )
    connection = event_store.open_database(db_path)
    connection.execute(
        "INSERT INTO runs (run_id, parent_version, snapshot_hash, status, started_at, completed_at, metrics_json, details_json) "
        "VALUES ('no-intent-run', 'parent', 'snapshot-no-intent', 'running', 1, NULL, NULL, ?)",
        (details,),
    )
    connection.execute(
        "INSERT INTO trainer_leases (lease_name, owner_id, acquired_at, expires_at, snapshot_hash, details_json) "
        "VALUES ('retrain', 'no-intent-run', 1, 2, 'snapshot-no-intent', ?)",
        (details,),
    )
    connection.commit()
    connection.close()
    monkeypatch.setattr(retrain, "_activation_status", lambda *_args, **_kwargs: None)

    pending = retrain._reconcile_stale_runs(
        db_path,
        now=3.0,
        lease_seconds=10.0,
        api_base_url="http://api",
        admin_token="private-token",
        state_path=state_path,
    )
    assert pending == []
    connection = event_store.open_database(db_path)
    try:
        run = connection.execute(
            "SELECT status, details_json FROM runs WHERE run_id='no-intent-run'"
        ).fetchone()
        assert run["status"] == "failed"
        assert json.loads(run["details_json"])["activation_id"] == "never-recorded-activation"
        assert connection.execute(
            "SELECT COUNT(*) FROM trainer_leases WHERE owner_id='no-intent-run'"
        ).fetchone()[0] == 0
    finally:
        connection.close()


def test_real_training_with_missing_categorical_values_cannot_activate_after_lease_loss(tmp_path, monkeypatch):
    split_path = tmp_path / "split_manifest.json"
    split = write_manifest(split_path)
    registry_path = tmp_path / "registry"
    bootstrap(registry_path, split_manifest_path=split_path, drift_reference_path=tmp_path / "drift.json")
    db_path = tmp_path / "events.sqlite3"
    _seed_adaptation_labels(db_path, split_path, split["partitions"]["adaptation_ids"][:32])

    calls = {"assertions": 0, "activations": 0}
    original_assert = retrain.TrainerLease.assert_owned

    def lose_before_activation(lease):
        calls["assertions"] += 1
        if calls["assertions"] == 3:
            connection = event_store.open_database(db_path)
            try:
                connection.execute("DELETE FROM trainer_leases WHERE lease_name='retrain' AND owner_id=?", (lease.run_id,))
                connection.commit()
            finally:
                connection.close()
            lease._lost.set()
            raise TrainerLeaseLost("simulated takeover before activation")
        original_assert(lease)

    def activation_must_not_run(**_kwargs):
        calls["activations"] += 1
        raise AssertionError("activation request must be fenced after ownership loss")

    monkeypatch.setattr(retrain.TrainerLease, "assert_owned", lose_before_activation)
    monkeypatch.setattr(retrain, "_request_activation", activation_must_not_run)
    result = run_once(
        db_path=db_path,
        registry_path=registry_path,
        state_path=tmp_path / "state",
        split_manifest_path=split_path,
        admin_token="private-token",
        now=1_800_000_000.0,
    )
    assert calls["assertions"] == 3
    assert calls["activations"] == 0
    assert result["reason"] == "trainer_lease_lost"
    connection = event_store.open_database(db_path)
    try:
        run = connection.execute("SELECT status, details_json FROM runs WHERE run_id=?", (result["run_id"],)).fetchone()
        details = json.loads(run["details_json"])
        assert run["status"] == "running"
        assert details["label_count"] == 32
        assert details["candidate_version"]
        assert connection.execute("SELECT COUNT(*) FROM labels").fetchone()[0] == 32
    finally:
        connection.close()
    candidate_manifest = json.loads((registry_path / details["candidate_version"] / "manifest.json").read_text())
    assert candidate_manifest["training"]["run_id"] == result["run_id"]
    assert candidate_manifest["training"]["snapshot_hash"] == details["snapshot_hash"]
