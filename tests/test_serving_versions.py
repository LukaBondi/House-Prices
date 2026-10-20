"""Activation, rollback, request-version capture, and cache isolation tests."""

import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

from fastapi.testclient import TestClient

import backend.app as api
from backend.artifacts import (
    atomic_write_pointer,
    load_bootstrap_snapshot,
    preprocessing_sha256,
    read_archive_metadata,
    schema_sha256,
    sha256_file,
)
from backend.event_store import initialize_database, open_database

ROOT = Path(__file__).resolve().parents[1]
FEATURES = {"OverallQual": 7, "GrLivArea": 1710, "Neighborhood": "CollgCr"}


def _configure_paths(monkeypatch, state):
    monkeypatch.setenv("DATA_DIR", str(state))
    monkeypatch.setenv("DB_PATH", str(state / "requests.sqlite3"))
    monkeypatch.setenv("MODEL_REGISTRY_PATH", str(state / "artifacts"))
    monkeypatch.setenv("MODEL_STATE_PATH", str(state / "state"))
    monkeypatch.setenv("DRIFT_REFERENCE_PATH", str(state / "drift-reference.json"))
    monkeypatch.setenv("ADMIN_TOKEN", "activation-test-secret")


def _write_candidate(registry, version, parent_version, *, training=None):
    bootstrap_archive = ROOT / "models" / "house_prices_native.zip"
    schema_path = ROOT / "models" / "feature_schema.json"
    destination = registry / version
    destination.mkdir(parents=True)
    archive = destination / "model.zip"
    shutil.copyfile(bootstrap_archive, archive)
    metadata = read_archive_metadata(archive)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    manifest = {
        "format": "house-prices-registry-v1",
        "version": version,
        "artifact_sha256": sha256_file(archive),
        "preprocessing_sha256": preprocessing_sha256(metadata),
        "schema_sha256": schema_sha256(schema),
        "parent_version": parent_version,
        "gate": {
            "policy": {"minimum_rmsle_gain": 0.01},
            "results": {"accepted": True},
            "accepted": True,
        },
        "training": training or {"parameters": {}, "snapshot": "candidate-test"},
    }
    (destination / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_activation_cas_idempotency_rollback_and_version_qualified_cache(tmp_path, monkeypatch):
    state = tmp_path / "state"
    _configure_paths(monkeypatch, state)
    registry = state / "artifacts"
    registry.mkdir(parents=True)
    bootstrap = load_bootstrap_snapshot(
        ROOT / "models" / "house_prices_native.zip",
        ROOT / "models" / "feature_schema.json",
    )
    _write_candidate(registry, "candidate-one", bootstrap.version)
    _write_candidate(registry, "candidate-two", bootstrap.version)
    token = {"Authorization": "Bearer activation-test-secret"}

    with TestClient(api.app) as client:
        baseline_version = client.get("/health").json()["model_version"]
        assert baseline_version == bootstrap.version
        unauthenticated = client.post(
            "/admin/activate",
            json={
                "candidate_version": "candidate-one",
                "expected_parent": baseline_version,
                "activation_id": "unauthorized-activation",
            },
        )
        assert unauthenticated.status_code == 401
        baseline_first = client.post("/predict", json={"features": FEATURES})
        baseline_hit = client.post("/predict", json={"features": FEATURES})
        assert baseline_first.status_code == baseline_hit.status_code == 200

        activated = client.post(
            "/admin/activate",
            headers=token,
            json={
                "candidate_version": "candidate-one",
                "expected_parent": baseline_version,
                "activation_id": "activation-one",
            },
        )
        assert activated.status_code == 200, activated.text
        assert activated.json()["status"] == "committed"
        assert activated.json()["active_version"] == "candidate-one"

        candidate_first = client.post("/predict", json={"features": FEATURES})
        candidate_hit = client.post("/predict", json={"features": FEATURES})
        assert candidate_first.json()["model_version"] == "candidate-one"
        assert candidate_first.json()["sale_price"] == baseline_first.json()["sale_price"]
        assert candidate_hit.json()["request_id"] == candidate_hit.headers["x-request-id"]

        repeated = client.post(
            "/admin/activate",
            headers=token,
            json={
                "candidate_version": "candidate-one",
                "expected_parent": baseline_version,
                "activation_id": "activation-one",
            },
        )
        assert repeated.status_code == 200
        assert repeated.json()["idempotent"] is True

        reused_id = client.post(
            "/admin/activate",
            headers=token,
            json={
                "candidate_version": "candidate-two",
                "expected_parent": baseline_version,
                "activation_id": "activation-one",
            },
        )
        assert reused_id.status_code == 409

        stale_parent = client.post(
            "/admin/activate",
            headers=token,
            json={
                "candidate_version": "candidate-two",
                "expected_parent": baseline_version,
                "activation_id": "activation-stale",
            },
        )
        assert stale_parent.status_code == 409

        status = client.get("/admin/activation/activation-one", headers=token)
        assert status.status_code == 200
        assert status.json()["status"] == "committed"

        rolled_back = client.post(
            "/admin/rollback",
            headers=token,
            json={
                "target_version": baseline_version,
                "expected_parent": "candidate-one",
                "activation_id": "rollback-one",
            },
        )
        assert rolled_back.status_code == 200, rolled_back.text
        assert rolled_back.json()["active_version"] == baseline_version
        rollback_replay = client.post(
            "/admin/rollback",
            headers=token,
            json={
                "target_version": baseline_version,
                "expected_parent": "candidate-one",
                "activation_id": "rollback-one",
            },
        )
        assert rollback_replay.status_code == 200
        assert rollback_replay.json()["idempotent"] is True
        rollback_as_promotion = client.post(
            "/admin/activate",
            headers=token,
            json={
                "candidate_version": baseline_version,
                "expected_parent": "candidate-one",
                "activation_id": "rollback-one",
            },
        )
        assert rollback_as_promotion.status_code == 409
        baseline_again = client.post("/predict", json={"features": FEATURES})
        assert baseline_again.json()["model_version"] == baseline_version

    connection = open_database(state / "requests.sqlite3")
    try:
        prediction_rows = connection.execute(
            "SELECT model_version, cache_outcome FROM requests WHERE path='/predict' "
            "ORDER BY rowid"
        ).fetchall()
        assert [row["model_version"] for row in prediction_rows] == [
            baseline_version,
            baseline_version,
            "candidate-one",
            "candidate-one",
            baseline_version,
        ]
        assert [row["cache_outcome"] for row in prediction_rows] == [
            "miss",
            "hit",
            "miss",
            "hit",
            "hit",
        ]
        activation_rows = connection.execute(
            "SELECT activation_id, status FROM activations ORDER BY created_at, rowid"
        ).fetchall()
        assert {row["activation_id"]: row["status"] for row in activation_rows} == {
            "activation-one": "committed",
            "activation-stale": "rejected",
            "rollback-one": "committed",
        }
        admin_rows = connection.execute(
            "SELECT body_text FROM requests WHERE path LIKE '/admin/%'"
        ).fetchall()
        assert all(row["body_text"] is None for row in admin_rows)
    finally:
        connection.close()


def test_startup_reconciles_pointer_commit_and_prepublication_intent(tmp_path, monkeypatch):
    state = tmp_path / "state"
    _configure_paths(monkeypatch, state)
    registry = state / "artifacts"
    registry.mkdir(parents=True)
    bootstrap = load_bootstrap_snapshot(
        ROOT / "models" / "house_prices_native.zip",
        ROOT / "models" / "feature_schema.json",
    )
    _write_candidate(registry, "candidate-recovery", bootstrap.version)
    database = state / "requests.sqlite3"
    initialize_database(database)
    connection = open_database(database)
    now = time.time()
    try:
        connection.executemany(
            "INSERT INTO activations "
            "(activation_id, expected_parent, candidate_version, status, created_at, details_json) "
            "VALUES (?, ?, ?, 'intent', ?, ?)",
            [
                (
                    "crash-after-pointer",
                    bootstrap.version,
                    "candidate-recovery",
                    now,
                    json.dumps(
                        {
                            "rollback": False,
                            "trainer_run_id": "recovered-run",
                            "snapshot_hash": "c" * 64,
                        }
                    ),
                ),
                (
                    "crash-before-pointer",
                    bootstrap.version,
                    "candidate-recovery",
                    now + 0.001,
                    "{}",
                ),
            ],
        )
        connection.commit()
    finally:
        connection.close()
    atomic_write_pointer(
        state / "state" / "current.json",
        {
            "version": "candidate-recovery",
            "activation_id": "crash-after-pointer",
            "previous_version": bootstrap.version,
        },
    )

    with TestClient(api.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_version"] == "candidate-recovery"
        headers = {"Authorization": "Bearer activation-test-secret"}
        committed = client.get(
            "/admin/activation/crash-after-pointer", headers=headers
        )
        failed = client.get("/admin/activation/crash-before-pointer", headers=headers)
        assert committed.json()["status"] == "committed"
        assert committed.json()["details"]["trainer_run_id"] == "recovered-run"
        assert failed.json()["status"] == "failed"

    connection = open_database(database)
    try:
        statuses = {
            row["activation_id"]: row["status"]
            for row in connection.execute("SELECT activation_id, status FROM activations")
        }
        assert statuses == {
            "crash-after-pointer": "committed",
            "crash-before-pointer": "failed",
        }
    finally:
        connection.close()


def test_trainer_activation_is_fenced_by_run_identity_and_live_retrain_lease(
    tmp_path, monkeypatch
):
    state = tmp_path / "state"
    _configure_paths(monkeypatch, state)
    registry = state / "artifacts"
    registry.mkdir(parents=True)
    bootstrap = load_bootstrap_snapshot(
        ROOT / "models" / "house_prices_native.zip",
        ROOT / "models" / "feature_schema.json",
    )
    snapshot_hash = "b" * 64
    _write_candidate(
        registry,
        "candidate-trained",
        bootstrap.version,
        training={"run_id": "run-live", "snapshot_hash": snapshot_hash},
    )
    _write_candidate(
        registry,
        "candidate-expired",
        "candidate-trained",
        training={"run_id": "run-expired", "snapshot_hash": snapshot_hash},
    )
    token = {"Authorization": "Bearer activation-test-secret"}

    with TestClient(api.app) as client:
        assert client.get("/health").json()["model_version"] == bootstrap.version
        database = api._runtime_paths()["database"]
        connection = open_database(database)
        now = time.time()
        try:
            connection.executemany(
                "INSERT INTO runs (run_id, parent_version, snapshot_hash, status, started_at, details_json) "
                "VALUES (?, ?, ?, 'running', ?, ?)",
                [
                    (
                        "run-live",
                        bootstrap.version,
                        snapshot_hash,
                        now,
                        json.dumps(
                            {
                                "candidate_version": "candidate-trained",
                                "activation_id": "activation-trained",
                            }
                        ),
                    ),
                    (
                        "run-expired",
                        "candidate-trained",
                        snapshot_hash,
                        now,
                        json.dumps(
                            {
                                "candidate_version": "candidate-expired",
                                "activation_id": "activation-expired",
                            }
                        ),
                    ),
                ],
            )
            connection.executemany(
                "INSERT INTO trainer_leases "
                "(lease_name, owner_id, acquired_at, expires_at, snapshot_hash, details_json) "
                "VALUES ('retrain', ?, ?, ?, ?, '{}')",
                [("run-live", now, now + 30.0, snapshot_hash)],
            )
            connection.commit()
        finally:
            connection.close()

        live = client.post(
            "/admin/activate",
            headers=token,
            json={
                "candidate_version": "candidate-trained",
                "expected_parent": bootstrap.version,
                "activation_id": "activation-trained",
                "trainer_run_id": "run-live",
            },
        )
        assert live.status_code == 200, live.text
        assert live.json()["active_version"] == "candidate-trained"

        # A committed retry remains idempotent after lease expiry, while a changed
        # operation identity is rejected before an idempotent success is returned.
        replay = client.post(
            "/admin/activate",
            headers=token,
            json={
                "candidate_version": "candidate-trained",
                "expected_parent": bootstrap.version,
                "activation_id": "activation-trained",
                "trainer_run_id": "run-live",
            },
        )
        assert replay.status_code == 200
        assert replay.json()["idempotent"] is True
        changed_owner = client.post(
            "/admin/activate",
            headers=token,
            json={
                "candidate_version": "candidate-trained",
                "expected_parent": bootstrap.version,
                "activation_id": "activation-trained",
                "trainer_run_id": "other-run",
            },
        )
        assert changed_owner.status_code == 409

        connection = open_database(database)
        try:
            connection.execute(
                "UPDATE trainer_leases SET owner_id=?, acquired_at=?, expires_at=?, snapshot_hash=? "
                "WHERE lease_name='retrain'",
                ("run-expired", now, now + 60.0, snapshot_hash),
            )
            connection.commit()
        finally:
            connection.close()

        original_write_pointer = api.atomic_write_pointer

        def expire_lease_before_replace(path, pointer, *, before_replace=None):
            real_clock = api.time.time
            api.time.time = lambda: real_clock() + 120.0
            try:
                return original_write_pointer(
                    path, pointer, before_replace=before_replace
                )
            finally:
                api.time.time = real_clock

        monkeypatch.setattr(api, "atomic_write_pointer", expire_lease_before_replace)
        expired = client.post(
            "/admin/activate",
            headers=token,
            json={
                "candidate_version": "candidate-expired",
                "expected_parent": "candidate-trained",
                "activation_id": "activation-expired",
                "trainer_run_id": "run-expired",
            },
        )
        assert expired.status_code == 409, expired.text
        assert client.get("/health").json()["model_version"] == "candidate-trained"
        monkeypatch.setattr(api, "atomic_write_pointer", original_write_pointer)

    connection = open_database(api._runtime_paths()["database"])
    try:
        rows = {
            row["activation_id"]: (row["status"], json.loads(row["details_json"] or "{}"))
            for row in connection.execute(
                "SELECT activation_id, status, details_json FROM activations"
            )
        }
        assert rows["activation-trained"][0] == "committed"
        assert rows["activation-trained"][1]["trainer_run_id"] == "run-live"
        assert rows["activation-trained"][1]["lease_name"] == "retrain"
        assert rows["activation-expired"][0] == "rejected"
    finally:
        connection.close()


def test_manifest_parent_must_match_activation_compare_and_swap(tmp_path, monkeypatch):
    state = tmp_path / "state"
    _configure_paths(monkeypatch, state)
    registry = state / "artifacts"
    registry.mkdir(parents=True)
    bootstrap = load_bootstrap_snapshot(
        ROOT / "models" / "house_prices_native.zip",
        ROOT / "models" / "feature_schema.json",
    )
    _write_candidate(registry, "candidate-wrong-parent", "unserved-parent")

    with TestClient(api.app) as client:
        response = client.post(
            "/admin/activate",
            headers={"Authorization": "Bearer activation-test-secret"},
            json={
                "candidate_version": "candidate-wrong-parent",
                "expected_parent": bootstrap.version,
                "activation_id": "wrong-manifest-parent",
            },
        )
        assert response.status_code == 409
        assert client.get("/health").json()["model_version"] == bootstrap.version


def test_pointer_failure_after_replace_reconciles_memory_audit_and_restart(tmp_path, monkeypatch):
    state = tmp_path / "state"
    _configure_paths(monkeypatch, state)
    registry = state / "artifacts"
    registry.mkdir(parents=True)
    bootstrap = load_bootstrap_snapshot(
        ROOT / "models" / "house_prices_native.zip",
        ROOT / "models" / "feature_schema.json",
    )
    _write_candidate(registry, "candidate-after-replace", bootstrap.version)
    pointer_path = state / "state" / "current.json"
    original_write = api.atomic_write_pointer

    with TestClient(api.app) as client:
        assert pointer_path.is_file(), "startup must publish the initial baseline pointer"

        def replace_then_fail(path, pointer):
            original_write(path, pointer)
            raise OSError("injected directory fsync failure after replace")

        monkeypatch.setattr(api, "atomic_write_pointer", replace_then_fail)
        response = client.post(
            "/admin/activate",
            headers={"Authorization": "Bearer activation-test-secret"},
            json={
                "candidate_version": "candidate-after-replace",
                "expected_parent": bootstrap.version,
                "activation_id": "activation-after-replace",
            },
        )
        assert response.status_code == 503
        assert api.capture_serving_snapshot().version == "candidate-after-replace"
        assert json.loads(pointer_path.read_text(encoding="utf-8"))["version"] == (
            "candidate-after-replace"
        )
        status = client.get(
            "/admin/activation/activation-after-replace",
            headers={"Authorization": "Bearer activation-test-secret"},
        )
        assert status.status_code == 200
        assert status.json()["status"] == "committed"

    monkeypatch.setattr(api, "atomic_write_pointer", original_write)
    with TestClient(api.app) as restarted:
        health = restarted.get("/health")
        assert health.status_code == 200
        assert health.json()["model_version"] == "candidate-after-replace"


def test_inflight_prediction_keeps_its_captured_model_version(tmp_path, monkeypatch):
    state = tmp_path / "state"
    _configure_paths(monkeypatch, state)
    registry = state / "artifacts"
    registry.mkdir(parents=True)
    bootstrap = load_bootstrap_snapshot(
        ROOT / "models" / "house_prices_native.zip",
        ROOT / "models" / "feature_schema.json",
    )
    _write_candidate(registry, "candidate-during-flight", bootstrap.version)
    entered = Event()
    release = Event()

    with TestClient(api.app) as client:
        baseline_version = client.get("/health").json()["model_version"]
        original_predict = api.model.predict_records

        def blocked_predict(records):
            entered.set()
            if not release.wait(timeout=5):
                raise TimeoutError("test inference was not released")
            return original_predict(records)

        api.model.predict_records = blocked_predict
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                coalesced_before = api.metrics["in_flight_coalesced"]
                pending = executor.submit(
                    client.post, "/predict", json={"features": FEATURES}
                )
                assert entered.wait(timeout=3)
                duplicate = executor.submit(
                    client.post, "/predict", json={"features": FEATURES}
                )
                deadline = time.time() + 3
                while (
                    api.metrics["in_flight_coalesced"] == coalesced_before
                    and time.time() < deadline
                ):
                    time.sleep(0.005)
                assert api.metrics["in_flight_coalesced"] > coalesced_before
                activation = client.post(
                    "/admin/activate",
                    headers={"Authorization": "Bearer activation-test-secret"},
                    json={
                        "candidate_version": "candidate-during-flight",
                        "expected_parent": baseline_version,
                        "activation_id": "activation-during-flight",
                    },
                )
                assert activation.status_code == 200, activation.text
                release.set()
                old_response = pending.result(timeout=5)
                duplicate_response = duplicate.result(timeout=5)
        finally:
            release.set()

        assert old_response.status_code == 200
        assert old_response.json()["model_version"] == baseline_version
        assert duplicate_response.status_code == 200
        assert duplicate_response.json()["model_version"] == baseline_version
        assert duplicate_response.json()["request_id"] != old_response.json()["request_id"]
        next_response = client.post("/predict", json={"features": FEATURES})
        assert next_response.status_code == 200
        assert next_response.json()["model_version"] == "candidate-during-flight"
        candidate_hit = client.post("/predict", json={"features": FEATURES})
        assert candidate_hit.status_code == 200
        assert candidate_hit.json()["model_version"] == "candidate-during-flight"

    connection = open_database(state / "requests.sqlite3")
    try:
        rows = connection.execute(
            "SELECT model_version, cache_outcome FROM requests WHERE path='/predict' "
            "ORDER BY rowid"
        ).fetchall()
        assert [row["model_version"] for row in rows] == [
            baseline_version,
            baseline_version,
            "candidate-during-flight",
            "candidate-during-flight",
        ]
        assert [row["cache_outcome"] for row in rows] == [
            "miss",
            "coalesced",
            "miss",
            "hit",
        ]
    finally:
        connection.close()
