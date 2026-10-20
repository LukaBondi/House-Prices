"""Run a scheduled, provenance-checked challenger training and activation cycle."""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from backend import event_store
from backend.artifacts import ArtifactError, read_pointer
from backend.native_model import NativeModel
from model_training.candidate_export import (
    build_candidate_archive,
    evaluate_gate,
    verify_candidate_parity,
)
from model_training.data_partitions import (
    DEFAULT_DATASET,
    DEFAULT_MANIFEST,
    FROZEN_BASELINE_SHA256,
    IDENTITY,
    TARGET,
    load_manifest,
    sha256_file,
)
from model_training.registry import (
    archive_metadata,
    canonical_json,
    current_version,
    preprocessing_sha256,
    schema_sha256,
    sha256_bytes,
    write_immutable_version,
)
from model_training.registry import load_manifest as load_registry_manifest
from model_training.training_config import (
    BASELINE_ARCHIVE,
    DB_PATH,
    MODEL_REGISTRY_PATH,
    MODEL_STATE_PATH,
    SCHEMA_PATH,
)

MIN_NEW_LABELS = 32
BOOST_ROUNDS = 25
LEARNING_RATE = 0.01
MIN_RELATIVE_RMSLE_GAIN = 0.01
TREE_GROWTH_CAP = int(os.getenv("TRAINER_TREE_GROWTH_CAP", "600"))
LEASE_SECONDS = float(os.getenv("TRAINER_LEASE_SECONDS", "1800"))
RETRY_BASE_SECONDS = float(os.getenv("TRAINER_RETRY_BASE_SECONDS", "30"))
RETRY_MAX_SECONDS = float(os.getenv("TRAINER_RETRY_MAX_SECONDS", "3600"))


class TrainingDataError(ValueError):
    """An imported label or champion is incompatible with the frozen contract."""


class TrainerLeaseLost(RuntimeError):
    """The run no longer owns a live singleton trainer lease."""


class ActivationUncertain(RuntimeError):
    """Activation may have committed, but the trainer has no durable acknowledgement."""


class TrainerLease:
    """Heartbeat and verify a live lease using short, independent DB connections."""

    def __init__(self, db_path: str | Path, run_id: str, lease_seconds: float, acquired_at: float) -> None:
        if not math.isfinite(lease_seconds) or lease_seconds <= 0:
            raise ValueError("lease_seconds must be finite and positive")
        self.db_path = db_path
        self.run_id = run_id
        self.lease_seconds = lease_seconds
        self._clock_started = time.monotonic()
        self._clock_base = acquired_at
        self._stopping = threading.Event()
        self._lost = threading.Event()
        self._thread = threading.Thread(target=self._heartbeat, name=f"trainer-lease-{run_id[:8]}", daemon=True)

    def now(self) -> float:
        return self._clock_base + (time.monotonic() - self._clock_started)

    def start(self) -> None:
        self._thread.start()

    @property
    def lost(self) -> bool:
        return self._lost.is_set()

    def _heartbeat(self) -> None:
        interval = max(0.01, self.lease_seconds / 3.0)
        while not self._stopping.wait(interval):
            connection = None
            try:
                connection = event_store.open_database(self.db_path)
                connection.execute("BEGIN IMMEDIATE")
                # SQLite may have made us wait for another writer. Sample only
                # after acquiring the write fence so an expired lease cannot be
                # renewed with a stale pre-lock timestamp.
                now = self.now()
                cursor = connection.execute(
                    "UPDATE trainer_leases SET expires_at=? "
                    "WHERE lease_name='retrain' AND owner_id=? AND expires_at>?",
                    (now + self.lease_seconds, self.run_id, now),
                )
                connection.commit()
                if cursor.rowcount != 1:
                    self._lost.set()
                    return
            except Exception:
                if connection is not None:
                    try:
                        connection.rollback()
                    except Exception:
                        pass
                self._lost.set()
                return
            finally:
                if connection is not None:
                    connection.close()

    def assert_owned(self) -> None:
        if self._lost.is_set():
            raise TrainerLeaseLost(f"Trainer lease for run {self.run_id} was lost")
        connection = event_store.open_database(self.db_path)
        try:
            row = connection.execute(
                "SELECT 1 FROM trainer_leases WHERE lease_name='retrain' AND owner_id=? AND expires_at>?",
                (self.run_id, self.now()),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            self._lost.set()
            raise TrainerLeaseLost(f"Trainer lease for run {self.run_id} is no longer live")

    def expire_owned_lease(self) -> None:
        """Stop future API commits and let the next scheduler tick reconcile the activation ID."""
        # Join before touching the row: an in-flight heartbeat must not renew
        # the owner after this method intentionally expires it.
        self.stop()
        connection = event_store.open_database(self.db_path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            now = self.now()
            connection.execute(
                "UPDATE trainer_leases SET expires_at=? "
                "WHERE lease_name='retrain' AND owner_id=? AND expires_at>?",
                (now, self.run_id, now),
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def stop(self) -> None:
        self._stopping.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join()


def _read_active_version(registry_path: str | Path, state_path: str | Path, frozen_archive: str | Path) -> str:
    pointer = Path(state_path) / "current.json"
    if pointer.exists():
        return current_version(registry_path, state_path)
    bootstrap = f"bootstrap-{sha256_file(frozen_archive)[:12]}"
    load_registry_manifest(registry_path, bootstrap)
    return bootstrap


def _read_original_source(dataset_path: str | Path, split_manifest_path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    split = load_manifest(split_manifest_path)
    if sha256_file(dataset_path) != split["source"]["sha256"]:
        raise TrainingDataError("Training CSV hash no longer matches split manifest")
    frame = pd.read_csv(dataset_path)
    if frame.shape[0] != 1460 or sha256_file(Path(__file__).resolve().parents[1] / "models" / "baseline_xgb_pipeline.pkl") != FROZEN_BASELINE_SHA256:
        raise TrainingDataError("Frozen original training source changed")
    return frame.set_index(IDENTITY, drop=False), split


def _eligible_labels(
    connection: Any,
    *,
    cutoff: float,
    dataset_sha256: str,
    source_frame: pd.DataFrame,
    split: dict[str, Any],
) -> list[dict[str, Any]]:
    feature_names = split["source"]["feature_names"]
    numeric_features = split["source"]["numeric_features"]
    adaptation_ids = set(split["partitions"]["adaptation_ids"])
    rows = connection.execute(
        "SELECT l.observation_id, l.request_id, l.dataset_sha256, l.source_id, l.features_json, "
        "l.feature_fingerprint, l.sale_price, l.available_at, r.status_code, r.lifecycle, "
        "r.method, r.path, r.completed_at AS request_completed_at, r.logging_failed, r.dataset_sha256 AS request_dataset_sha256, r.source_id AS request_source_id, "
        "r.features_json AS request_features_json, r.feature_fingerprint AS request_feature_fingerprint "
        "FROM labels AS l JOIN requests AS r ON r.request_id = l.request_id "
        "WHERE l.available_at <= ? AND l.dataset_sha256 = ? "
        "ORDER BY l.source_id, l.available_at, l.request_id",
        (cutoff, dataset_sha256),
    ).fetchall()
    labels: list[dict[str, Any]] = []
    observed_ids: set[int] = set()
    for row in rows:
        source_id = int(row["source_id"])
        if source_id not in adaptation_ids:
            raise TrainingDataError(f"Label Source Id {source_id} is outside the adaptation partition")
        if source_id in observed_ids:
            raise TrainingDataError(f"Duplicate labeled observation for Source Id {source_id}")
        observed_ids.add(source_id)
        try:
            available_at = float(row["available_at"])
            completed_at = float(row["request_completed_at"])
        except (TypeError, ValueError) as exc:
            raise TrainingDataError(f"Label Source Id {source_id} has invalid timestamps") from exc
        if (
            row["method"] != "POST"
            or row["path"] != "/predict"
            or row["request_dataset_sha256"] != dataset_sha256
            or row["request_source_id"] != source_id
            or row["status_code"] != 200
            or row["lifecycle"] != "completed"
            or row["logging_failed"]
            or row["request_completed_at"] is None
            or not math.isfinite(available_at)
            or not math.isfinite(completed_at)
            or completed_at > cutoff
            or available_at > cutoff
            or available_at < completed_at
            or not row["request_features_json"]
        ):
            raise TrainingDataError(f"Label Source Id {source_id} has no successful persisted prediction")
        raw = _format_records(source_frame.loc[[source_id]], feature_names)[0]
        expected = event_store.canonical_record(raw, feature_names=feature_names, numeric_features=numeric_features)
        fingerprint = event_store.feature_fingerprint(raw, feature_names=feature_names, numeric_features=numeric_features)
        try:
            label_record = event_store.canonical_record(
                json.loads(row["features_json"]), feature_names=feature_names, numeric_features=numeric_features
            )
            request_record = event_store.canonical_record(
                json.loads(row["request_features_json"]), feature_names=feature_names, numeric_features=numeric_features
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise TrainingDataError(f"Invalid stored features for Source Id {source_id}") from exc
        if (
            label_record != expected
            or request_record != expected
            or row["feature_fingerprint"] != fingerprint
            or row["request_feature_fingerprint"] != fingerprint
        ):
            raise TrainingDataError(f"Stored features changed for Source Id {source_id}")
        sale_price = float(row["sale_price"])
        source_target = float(source_frame.loc[source_id, TARGET])
        if not math.isfinite(sale_price) or sale_price < 0 or sale_price != source_target:
            raise TrainingDataError(f"Label does not match original SalePrice for Source Id {source_id}")
        labels.append({
            "observation_id": str(row["observation_id"]),
            "request_id": str(row["request_id"]),
            "dataset_sha256": dataset_sha256,
            "source_id": source_id,
            "features_json": event_store.canonical_json(expected),
            "feature_fingerprint": fingerprint,
            "sale_price": sale_price,
            "available_at": float(row["available_at"]),
        })
    return labels


def _snapshot_hash(cutoff: float, labels: list[dict[str, Any]]) -> tuple[str, str]:
    identity = [
        {
            "dataset_sha256": row["dataset_sha256"],
            "source_id": row["source_id"],
            "feature_fingerprint": row["feature_fingerprint"],
            "sale_price": row["sale_price"],
            "available_at": row["available_at"],
        }
        for row in labels
    ]
    identity_hash = sha256_bytes(canonical_json(identity))
    snapshot_hash = sha256_bytes(canonical_json({"cutoff": cutoff, "identity_hash": identity_hash}))
    return snapshot_hash, identity_hash


def _run_rows(connection: Any) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(
        "SELECT run_id, parent_version, snapshot_hash, status, started_at, completed_at, metrics_json, details_json "
        "FROM runs ORDER BY started_at DESC"
    ).fetchall()]


def _row_details(row: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(row.get("details_json") or "{}")
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


def _save_run(
    db_path: str | Path,
    run_id: str,
    *,
    status: str,
    completed_at: float,
    metrics: dict[str, Any] | None,
    details: dict[str, Any],
    lease: TrainerLease,
) -> None:
    connection = event_store.open_database(db_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        now = lease.now()
        cursor = connection.execute(
            "UPDATE runs SET status=?, completed_at=?, metrics_json=?, details_json=? "
            "WHERE run_id=? AND status='running' AND EXISTS ("
            "SELECT 1 FROM trainer_leases WHERE lease_name='retrain' AND owner_id=? AND expires_at>?)",
            (
                status,
                completed_at,
                event_store.canonical_json(metrics) if metrics is not None else None,
                event_store.canonical_json(details),
                run_id,
                run_id,
                now,
            ),
        )
        if cursor.rowcount != 1:
            connection.rollback()
            raise TrainerLeaseLost(f"Run {run_id} cannot be finalized without its live trainer lease")
        deleted = connection.execute(
            "DELETE FROM trainer_leases WHERE lease_name='retrain' AND owner_id=? AND expires_at>?",
            (run_id, now),
        )
        if deleted.rowcount != 1:
            connection.rollback()
            raise TrainerLeaseLost(f"Run {run_id} lost its lease while finalizing")
        connection.commit()
        lease.stop()
    finally:
        connection.close()


def _store_running_details(
    db_path: str | Path,
    run_id: str,
    details: dict[str, Any],
    lease: TrainerLease,
    metrics: dict[str, Any] | None = None,
) -> None:
    connection = event_store.open_database(db_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        now = lease.now()
        updated_run = connection.execute(
            "UPDATE runs SET details_json=?, metrics_json=? WHERE run_id=? AND status='running' "
            "AND EXISTS (SELECT 1 FROM trainer_leases WHERE lease_name='retrain' AND owner_id=? AND expires_at>?)",
            (event_store.canonical_json(details), event_store.canonical_json(metrics) if metrics is not None else None, run_id, run_id, now),
        )
        updated_lease = connection.execute(
            "UPDATE trainer_leases SET snapshot_hash=?, details_json=? "
            "WHERE lease_name='retrain' AND owner_id=? AND expires_at>?",
            (details.get("snapshot_hash"), event_store.canonical_json(details), run_id, now),
        )
        if updated_run.rowcount != 1 or updated_lease.rowcount != 1:
            connection.rollback()
            raise TrainerLeaseLost(f"Run {run_id} no longer owns a live trainer lease")
        connection.commit()
    finally:
        connection.close()


def _load_baseline_pipeline(path: str | Path | None = None) -> Any:
    baseline = Path(path) if path is not None else Path(__file__).resolve().parents[1] / "models" / "baseline_xgb_pipeline.pkl"
    if sha256_file(baseline) != FROZEN_BASELINE_SHA256:
        raise TrainingDataError("Frozen baseline pipeline checksum changed")
    pipeline = joblib.load(baseline)
    return pipeline


def _native_edge_records(schema_path: str | Path) -> list[dict[str, Any]]:
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8-sig"))
    feature_names = schema["features"]
    sample = dict(schema.get("sample", {}))
    sample.update({"OverallQual": 0, "GrLivArea": None, "Neighborhood": "not-a-known-neighborhood"})
    all_null = {name: None for name in feature_names}
    partial = {"OverallQual": None, "GrLivArea": 0, "Neighborhood": "not-a-known-neighborhood"}
    return [sample, all_null, partial, dict(reversed(list(sample.items())))]


def _training_parameters(original_booster: xgb.Booster) -> dict[str, Any]:
    config = json.loads(original_booster.save_config())
    learner = config["learner"]
    gradient = learner["gradient_booster"]
    tree = gradient.get("tree_train_param", {})
    model = learner["learner_model_param"]
    base_raw = model["base_score"]
    try:
        parsed = json.loads(base_raw) if isinstance(base_raw, str) else base_raw
        base_score = float(parsed[0] if isinstance(parsed, list) else parsed)
    except (ValueError, TypeError, IndexError):
        base_score = float(str(base_raw).strip("[]"))
    objective = learner["objective"]["name"]
    return {
        "objective": objective,
        "base_score": base_score,
        "seed": 42,
        "nthread": 1,
        "max_depth": int(tree.get("max_depth", 4)),
        "subsample": float(tree.get("subsample", 0.8)),
        "colsample_bytree": float(tree.get("colsample_bytree", 0.8)),
        "min_child_weight": float(tree.get("min_child_weight", 1.0)),
        "lambda": float(tree.get("lambda", tree.get("reg_lambda", 1.0))),
        "alpha": float(tree.get("alpha", tree.get("reg_alpha", 0.0))),
        "gamma": float(tree.get("gamma", 0.0)),
        "learning_rate": LEARNING_RATE,
        "num_boost_round": BOOST_ROUNDS,
        "tree_method": "hist",
    }


def _activation_status(base_url: str, token: str, activation_id: str, timeout: float = 10.0) -> dict[str, Any] | None:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/admin/activation/{activation_id}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read(65536).decode("utf-8"))
        return value if isinstance(value, dict) else None
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {"activation_id": activation_id, "status": "not_found"}
        return None
    except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError):
        return None


def _is_activation_committed(value: dict[str, Any] | None, activation_id: str, version: str) -> bool:
    if not value or value.get("activation_id") != activation_id or value.get("status") != "committed":
        return False
    candidate = value.get("candidate_version")
    if candidate is not None:
        return candidate == version
    details = value.get("details")
    if isinstance(details, dict) and details.get("active_version") is not None:
        return details.get("active_version") == version
    # The direct POST acknowledgement predates candidate_version in its reply.
    return value.get("active_version") == version


def _request_activation(
    *,
    base_url: str,
    token: str,
    version: str,
    expected_parent: str,
    activation_id: str,
    trainer_run_id: str | None = None,
    timeout: float = 15.0,
) -> dict[str, Any]:
    payload_value = {
        "candidate_version": version,
        "expected_parent": expected_parent,
        "activation_id": activation_id,
    }
    if trainer_run_id is not None:
        payload_value["trainer_run_id"] = trainer_run_id
    payload = json.dumps(
        payload_value,
        separators=(",", ":"),
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/admin/activate",
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read(65536).decode("utf-8"))
        if _is_activation_committed(result, activation_id, version):
            return result
    except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
        result = None
    status = _activation_status(base_url, token, activation_id)
    if _is_activation_committed(status, activation_id, version):
        return status or {}
    raise ActivationUncertain("Activation acknowledgement is unavailable; the activation ID will be reconciled")


def _reconcile_stale_runs(
    db_path: str | Path,
    *,
    now: float,
    lease_seconds: float,
    api_base_url: str,
    admin_token: str | None,
    state_path: str | Path = MODEL_STATE_PATH,
) -> list[dict[str, str | None]]:
    """Reconcile expired owners without discarding uncertain activation lineage.

    Status requests happen before the write transaction. The shared activation row
    and pointer are then re-read under ``BEGIN IMMEDIATE`` after rechecking the
    expired owner, so a trainer/API activation still holding the DB fence must
    finish before this reconciliation can make a terminal decision.
    """
    connection = event_store.open_database(db_path)
    try:
        expired_rows = [
            dict(row)
            for row in connection.execute(
                "SELECT l.lease_name, l.owner_id, l.expires_at, l.details_json AS lease_details, "
                "r.run_id, r.parent_version, r.started_at, r.status, r.details_json AS run_details "
                "FROM trainer_leases AS l LEFT JOIN runs AS r ON r.run_id=l.owner_id "
                "WHERE l.lease_name='retrain' AND l.expires_at<=?",
                (now,),
            ).fetchall()
        ]
    finally:
        connection.close()

    pending: dict[str, dict[str, str | None]] = {}

    def details_for(row: dict[str, Any]) -> dict[str, Any]:
        for key in ("run_details", "lease_details", "details_json"):
            try:
                value = json.loads(row.get(key) or "{}")
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(value, dict):
                return value
        return {}

    def note_pending(run_id: str, activation_id: str | None) -> None:
        pending[run_id] = {"run_id": run_id, "activation_id": activation_id}

    def reconcile_row(row: dict[str, Any], *, requires_expired_lease: bool) -> None:
        run_id = row.get("owner_id") or row.get("run_id")
        if not isinstance(run_id, str):
            return
        initial_details = details_for(row)
        activation_id = initial_details.get("activation_id")
        candidate = initial_details.get("candidate_version")

        # The GET can repair a pointer-matching activation intent on the backend.
        # It is deliberately outside SQLite's write transaction.
        if admin_token and isinstance(activation_id, str) and activation_id:
            _activation_status(api_base_url, admin_token, activation_id)

        connection = event_store.open_database(db_path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            current_run = connection.execute(
                "SELECT run_id, parent_version, started_at, status, details_json "
                "FROM runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
            if current_run is None or current_run["status"] != "running":
                if requires_expired_lease:
                    connection.execute(
                        "DELETE FROM trainer_leases WHERE lease_name='retrain' "
                        "AND owner_id=? AND expires_at<=?",
                        (run_id, now),
                    )
                connection.commit()
                return

            if current_run["parent_version"] != row.get("parent_version") or (
                current_run["details_json"] != row.get("run_details", row.get("details_json"))
            ):
                # Do not apply an HTTP verdict to a changed run/activation identity.
                connection.commit()
                note_pending(run_id, activation_id if isinstance(activation_id, str) else None)
                return

            if requires_expired_lease:
                lease = connection.execute(
                    "SELECT owner_id FROM trainer_leases WHERE lease_name='retrain' "
                    "AND owner_id=? AND expires_at<=?",
                    (run_id, now),
                ).fetchone()
                if lease is None:
                    connection.commit()
                    return
            else:
                current_lease = connection.execute(
                    "SELECT owner_id FROM trainer_leases WHERE lease_name='retrain' "
                    "AND owner_id=?",
                    (run_id,),
                ).fetchone()
                if current_lease is not None or float(current_run["started_at"]) > now - lease_seconds:
                    connection.commit()
                    return

            details = details_for(
                {"run_details": current_run["details_json"], "lease_details": row.get("lease_details")}
            )
            activation_id = details.get("activation_id")
            candidate = details.get("candidate_version")
            parent_version = current_run["parent_version"]
            try:
                pointer = read_pointer(Path(state_path) / "current.json")
            except (ArtifactError, OSError):
                pointer = None

            outcome: str | None = None
            activation = None
            if isinstance(activation_id, str) and activation_id:
                activation = connection.execute(
                    "SELECT expected_parent, candidate_version, status, details_json "
                    "FROM activations WHERE activation_id=?",
                    (activation_id,),
                ).fetchone()

            if activation is not None:
                try:
                    activation_details = json.loads(activation["details_json"] or "{}")
                except (json.JSONDecodeError, TypeError):
                    activation_details = {}
                linked = (
                    isinstance(activation_details, dict)
                    and activation_details.get("rollback") is False
                    and activation_details.get("trainer_run_id") == run_id
                    and activation["expected_parent"] == parent_version
                    and activation["candidate_version"] == candidate
                )
                if linked and activation["status"] == "committed":
                    outcome = "promoted"
                elif linked and activation["status"] in {"rejected", "failed"}:
                    outcome = "failed"
                elif (
                    linked
                    and activation["status"] == "intent"
                    and pointer is not None
                    and pointer.get("activation_id") == activation_id
                    and pointer.get("version") == candidate
                    and pointer.get("previous_version") == parent_version
                ):
                    # Pointer replacement is durable even if the backend could not
                    # finish the activation audit update before it stopped.
                    outcome = "promoted"
            elif not (isinstance(activation_id, str) and activation_id):
                # _store_running_details persists activation_id before the HTTP
                # request; its absence proves this run never requested activation.
                outcome = "failed"
            elif (
                parent_version is not None
                and pointer is not None
                and pointer.get("version") == parent_version
                and pointer.get("activation_id") != activation_id
            ):
                # Shared DB has no activation intent and the durable serving
                # pointer is still the run's parent after the owner fence.
                outcome = "failed"

            if outcome is None:
                connection.commit()
                note_pending(run_id, activation_id if isinstance(activation_id, str) else None)
                return

            if outcome == "promoted":
                details["reconciled_after_restart"] = True
                details["activation_acknowledgement"] = {
                    "activation_id": activation_id,
                    "candidate_version": candidate,
                    "status": "committed",
                }
            else:
                failures = int(details.get("retry_number", 0))
                reason = (
                    "activation_rejected_or_failed"
                    if activation is not None
                    else "activation_not_present_after_lease_expiry"
                )
                details.update(
                    {
                        "failure": reason,
                        "retry_number": failures + 1,
                        "retry_after": now
                        + min(RETRY_MAX_SECONDS, RETRY_BASE_SECONDS * (2**failures)),
                    }
                )
            connection.execute(
                "UPDATE runs SET status=?, completed_at=?, details_json=? "
                "WHERE run_id=? AND status='running'",
                (outcome, now, event_store.canonical_json(details), run_id),
            )
            if requires_expired_lease:
                connection.execute(
                    "DELETE FROM trainer_leases WHERE lease_name='retrain' "
                    "AND owner_id=? AND expires_at<=?",
                    (run_id, now),
                )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    for row in expired_rows:
        reconcile_row(row, requires_expired_lease=True)

    connection = event_store.open_database(db_path)
    try:
        stale_rows = [
            dict(row)
            for row in connection.execute(
                "SELECT run_id, parent_version, started_at, status, details_json "
                "FROM runs WHERE status='running' AND started_at<=? "
                "AND run_id NOT IN (SELECT owner_id FROM trainer_leases WHERE lease_name='retrain')",
                (now - lease_seconds,),
            ).fetchall()
        ]
    finally:
        connection.close()
    for row in stale_rows:
        reconcile_row(row, requires_expired_lease=False)
    return list(pending.values())


def _record_skip(
    connection: Any,
    *,
    parent_version: str,
    now: float,
    reason: str,
    details: dict[str, Any],
) -> dict[str, Any]:
    identity_hash = details.get("identity_hash")
    existing = _run_rows(connection)
    for row in existing:
        if row["parent_version"] == parent_version and row["status"] == "skipped" and _row_details(row).get("identity_hash") == identity_hash:
            connection.commit()
            return {"status": "skipped", "reason": reason, "already_recorded": True, "run_id": row["run_id"]}
    run_id = str(uuid.uuid4())
    details = {**details, "reason": reason}
    connection.execute(
        "INSERT INTO runs (run_id, parent_version, snapshot_hash, status, started_at, completed_at, metrics_json, details_json) "
        "VALUES (?, ?, ?, 'skipped', ?, ?, NULL, ?)",
        (run_id, parent_version, details.get("snapshot_hash"), now, now, event_store.canonical_json(details)),
    )
    connection.commit()
    return {"status": "skipped", "reason": reason, "already_recorded": False, "run_id": run_id}


def _pending_failed_retry(rows: list[dict[str, Any]], parent_version: str) -> dict[str, Any] | None:
    latest_by_identity: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("parent_version") != parent_version:
            continue
        identity_hash = _row_details(row).get("identity_hash")
        if isinstance(identity_hash, str) and identity_hash not in latest_by_identity:
            latest_by_identity[identity_hash] = row
    pending = [row for row in latest_by_identity.values() if row.get("status") == "failed"]
    if not pending:
        return None
    return max(pending, key=lambda row: float(row.get("started_at") or 0.0))


def _format_records(frame: pd.DataFrame, feature_names: list[str]) -> list[dict[str, Any]]:
    rows = []
    for _, row in frame.loc[:, feature_names].iterrows():
        rows.append({name: None if pd.isna(row[name]) else row[name].item() if hasattr(row[name], "item") else row[name] for name in feature_names})
    return rows


def run_once(
    *,
    db_path: str | Path = DB_PATH,
    registry_path: str | Path = MODEL_REGISTRY_PATH,
    state_path: str | Path = MODEL_STATE_PATH,
    dataset_path: str | Path = DEFAULT_DATASET,
    split_manifest_path: str | Path = DEFAULT_MANIFEST,
    frozen_archive: str | Path = BASELINE_ARCHIVE,
    schema_path: str | Path = SCHEMA_PATH,
    api_base_url: str | None = None,
    admin_token: str | None = None,
    now: float | None = None,
    minimum_new_labels: int = MIN_NEW_LABELS,
    lease_seconds: float = LEASE_SECONDS,
    tree_growth_cap: int = TREE_GROWTH_CAP,
) -> dict[str, Any]:
    now = time.time() if now is None else float(now)
    if not math.isfinite(now):
        raise ValueError("now must be finite")
    if minimum_new_labels < 1:
        raise ValueError("minimum_new_labels must be positive")
    if not math.isfinite(lease_seconds) or lease_seconds <= 0:
        raise ValueError("lease_seconds must be finite and positive")
    if tree_growth_cap < BOOST_ROUNDS:
        raise ValueError("tree_growth_cap must allow at least one continuation")
    api_base_url = api_base_url or os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
    admin_token = admin_token if admin_token is not None else os.getenv("ADMIN_TOKEN")
    event_store.initialize_database(db_path)
    pending_reconciliations = _reconcile_stale_runs(
        db_path,
        now=now,
        lease_seconds=lease_seconds,
        api_base_url=api_base_url,
        admin_token=admin_token,
        state_path=state_path,
    )
    if pending_reconciliations:
        return {
            "status": "skipped",
            "reason": "activation_reconciliation_pending",
            "run_id": pending_reconciliations[0]["run_id"],
            "activation_id": pending_reconciliations[0]["activation_id"],
            "pending_run_ids": [item["run_id"] for item in pending_reconciliations],
        }
    source_frame, split = _read_original_source(dataset_path, split_manifest_path)
    parent_version = _read_active_version(registry_path, state_path, frozen_archive)
    parent_dir = Path(registry_path) / parent_version
    parent_manifest = load_registry_manifest(registry_path, parent_version)
    schema_hash = schema_sha256(schema_path)
    if parent_manifest.get("schema_sha256") != schema_hash:
        raise TrainingDataError("Active champion schema does not match the original feature schema")
    connection = event_store.open_database(db_path)
    run_id = str(uuid.uuid4())
    acquired = False
    lease: TrainerLease | None = None
    details: dict[str, Any] = {}
    labels: list[dict[str, Any]] = []
    cutoff = now
    identity_hash = ""
    snapshot_hash = ""
    try:
        connection.execute("BEGIN IMMEDIATE")
        running = connection.execute(
            "SELECT owner_id, expires_at FROM trainer_leases WHERE lease_name='retrain'"
        ).fetchone()
        if running is not None and float(running["expires_at"]) > now:
            connection.commit()
            return {"status": "skipped", "reason": "trainer_lease_held", "run_id": running["owner_id"]}
        parent_training = parent_manifest.get("training", {})
        parent_ids = {int(value) for value in parent_training.get("eligible_ids", [])}
        historical = _run_rows(connection)
        failed = _pending_failed_retry(historical, parent_version)
        failed_details: dict[str, Any] = {}
        if failed is not None:
            failed_details = _row_details(failed)
            try:
                retry_after = float(failed_details["retry_after"])
                cutoff = float(failed_details["cutoff"])
            except (KeyError, TypeError, ValueError) as exc:
                raise TrainingDataError("Failed snapshot is missing its frozen cutoff/backoff") from exc
            if not math.isfinite(retry_after) or not math.isfinite(cutoff):
                raise TrainingDataError("Failed snapshot has invalid cutoff/backoff timestamps")
            if now < retry_after:
                connection.commit()
                return {"status": "skipped", "reason": "retry_backoff", "retry_after": retry_after, "run_id": failed["run_id"]}
        labels = _eligible_labels(
            connection,
            cutoff=cutoff,
            dataset_sha256=split["source"]["sha256"],
            source_frame=source_frame,
            split=split,
        )
        snapshot_hash, identity_hash = _snapshot_hash(cutoff, labels)
        if failed is not None and (
            identity_hash != failed_details.get("identity_hash")
            or snapshot_hash != failed.get("snapshot_hash")
        ):
            raise TrainingDataError("Frozen failed snapshot could not be reconstructed exactly")
        eligible_ids = sorted(int(row["source_id"]) for row in labels)
        new_ids = sorted(set(eligible_ids) - parent_ids)
        details = {
            "cutoff": cutoff,
            "snapshot_hash": snapshot_hash,
            "identity_hash": identity_hash,
            "eligible_ids": eligible_ids,
            "new_ids": new_ids,
            "label_count": len(labels),
            "minimum_new_labels": minimum_new_labels,
        }
        if failed is not None:
            details.update({
                "retry_of": failed["run_id"],
                "retry_number": int(failed_details.get("retry_number", 0)),
            })
        latest_rejected = next((row for row in historical if row["parent_version"] == parent_version and row["status"] == "rejected"), None)
        if latest_rejected:
            rejected_ids = {int(value) for value in _row_details(latest_rejected).get("eligible_ids", [])}
            cooldown_new = sorted(set(eligible_ids) - rejected_ids)
            details["labels_since_rejection"] = cooldown_new
            if identity_hash == _row_details(latest_rejected).get("identity_hash"):
                connection.commit()
                return {"status": "skipped", "reason": "rejected_snapshot_already_attempted", "run_id": latest_rejected["run_id"]}
            if len(cooldown_new) < minimum_new_labels:
                result = _record_skip(connection, parent_version=parent_version, now=now, reason="rejection_cooldown", details=details)
                return result
        if len(new_ids) < minimum_new_labels:
            result = _record_skip(connection, parent_version=parent_version, now=now, reason="insufficient_new_labels", details=details)
            return result
        already_decided = next((row for row in historical if row["parent_version"] == parent_version and row["status"] in {"rejected", "promoted"} and _row_details(row).get("identity_hash") == identity_hash), None)
        if already_decided:
            connection.commit()
            return {"status": "skipped", "reason": "snapshot_already_attempted", "run_id": already_decided["run_id"]}
        connection.execute(
            "INSERT INTO runs (run_id, parent_version, snapshot_hash, status, started_at, completed_at, metrics_json, details_json) "
            "VALUES (?, ?, ?, 'running', ?, NULL, NULL, ?)",
            (run_id, parent_version, snapshot_hash, now, event_store.canonical_json(details)),
        )
        connection.execute(
            "INSERT INTO trainer_leases (lease_name, owner_id, acquired_at, expires_at, snapshot_hash, details_json) "
            "VALUES ('retrain', ?, ?, ?, ?, ?) "
            "ON CONFLICT(lease_name) DO UPDATE SET owner_id=excluded.owner_id, acquired_at=excluded.acquired_at, "
            "expires_at=excluded.expires_at, snapshot_hash=excluded.snapshot_hash, details_json=excluded.details_json "
            "WHERE trainer_leases.expires_at<=excluded.acquired_at",
            (run_id, now, now + lease_seconds, snapshot_hash, event_store.canonical_json(details)),
        )
        lease_row = connection.execute(
            "SELECT owner_id FROM trainer_leases WHERE lease_name='retrain'"
        ).fetchone()
        if lease_row is None or lease_row["owner_id"] != run_id:
            connection.rollback()
            return {"status": "skipped", "reason": "trainer_lease_held"}
        connection.commit()
        acquired = True
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()

    if not acquired:
        return {"status": "skipped", "reason": "no_snapshot"}
    lease = TrainerLease(db_path, run_id, lease_seconds, now)
    lease.start()
    try:
        champion_path = parent_dir / "model.zip"
        champion = NativeModel(champion_path)
        original_pipeline = _load_baseline_pipeline()
        preprocessor = original_pipeline.named_steps["preprocessor"]
        original_booster = original_pipeline.named_steps["regressor"].get_booster()
        expected_preprocessing = parent_manifest["preprocessing_sha256"]
        champion_metadata, _ = archive_metadata(champion_path.read_bytes())
        if preprocessing_sha256(champion_metadata) != expected_preprocessing:
            raise TrainingDataError("Current champion preprocessing fingerprint is invalid")
        frozen_metadata, _ = archive_metadata(Path(frozen_archive).read_bytes())
        if preprocessing_sha256(frozen_metadata) != expected_preprocessing:
            raise TrainingDataError("Champion preprocessing differs from the frozen baseline coordinates")
        original_params = _training_parameters(original_booster)
        parent_rounds = len(champion.booster.get_dump())
        if parent_rounds + BOOST_ROUNDS > tree_growth_cap:
            details.update({"reason": "tree_growth_cap", "parent_tree_count": parent_rounds, "tree_growth_cap": tree_growth_cap})
            _save_run(db_path, run_id, status="skipped", completed_at=lease.now(), metrics=None, details=details, lease=lease)
            return {"status": "skipped", "reason": "tree_growth_cap", "run_id": run_id}

        feature_names = split["source"]["feature_names"]
        fit_ids = split["partitions"]["original_fit_ids"]
        fit_rows = source_frame.loc[fit_ids]
        adaptation_rows = source_frame.loc[eligible_ids]
        fit_X = fit_rows.loc[:, feature_names]
        fit_y = fit_rows[TARGET].to_numpy(dtype=np.float64)
        adaptation_X = adaptation_rows.loc[:, feature_names]
        adaptation_y = np.asarray([next(row["sale_price"] for row in labels if row["source_id"] == source_id) for source_id in eligible_ids], dtype=np.float64)
        training_X = pd.concat([fit_X.reset_index(drop=True), adaptation_X.reset_index(drop=True)], ignore_index=True)
        training_y = np.concatenate([fit_y, adaptation_y])
        training_matrix = preprocessor.transform(training_X)
        dtrain = xgb.DMatrix(training_matrix, label=training_y, nthread=1)
        booster_copy = xgb.Booster()
        booster_copy.load_model(bytearray(champion.booster.save_raw(raw_format="ubj")))
        params = {
            "objective": original_params["objective"],
            "base_score": original_params["base_score"],
            "seed": original_params["seed"],
            "nthread": original_params["nthread"],
            "max_depth": original_params["max_depth"],
            "subsample": original_params["subsample"],
            "colsample_bytree": original_params["colsample_bytree"],
            "min_child_weight": original_params["min_child_weight"],
            "lambda": original_params["lambda"],
            "alpha": original_params["alpha"],
            "gamma": original_params["gamma"],
            "eta": LEARNING_RATE,
            "tree_method": original_params["tree_method"],
            "eval_metric": "rmse",
        }
        challenger_booster = xgb.train(params, dtrain, num_boost_round=BOOST_ROUNDS, xgb_model=booster_copy, verbose_eval=False)
        if len(challenger_booster.get_dump()) != parent_rounds + BOOST_ROUNDS:
            raise RuntimeError("Continuation did not add the declared number of trees")
        if len(challenger_booster.get_dump()) > tree_growth_cap:
            raise RuntimeError("Continuation exceeded cumulative tree growth cap")
        lease.assert_owned()

        gate_ids = split["partitions"]["promotion_gate_ids"]
        gate_rows = source_frame.loc[gate_ids]
        gate_X = gate_rows.loc[:, feature_names]
        gate_matrix = preprocessor.transform(gate_X)
        dgate = xgb.DMatrix(gate_matrix, nthread=1)
        y_gate = gate_rows[TARGET].to_numpy(dtype=np.float64)
        champion_predictions = np.asarray(champion.booster.predict(dgate), dtype=np.float64)
        challenger_predictions = np.asarray(challenger_booster.predict(dgate), dtype=np.float64)
        gate = evaluate_gate(y_gate, champion_predictions, challenger_predictions, minimum_relative_rmsle_gain=MIN_RELATIVE_RMSLE_GAIN)
        gate["evaluation_rows"] = gate_ids
        candidate_gate_valid = bool(np.isfinite(challenger_predictions).all() and np.all(challenger_predictions >= 0))
        gate["paired_errors"] = [
            {
                "source_id": int(source_id),
                "actual": float(actual),
                "champion_prediction": float(champion_prediction),
                "challenger_prediction": float(challenger_prediction),
                "champion_absolute_error": float(abs(actual - champion_prediction)),
                "challenger_absolute_error": float(abs(actual - challenger_prediction)),
                "champion_squared_log_error": float((np.log1p(max(champion_prediction, 0.0)) - np.log1p(actual)) ** 2),
                "challenger_squared_log_error": float((np.log1p(max(challenger_prediction, 0.0)) - np.log1p(actual)) ** 2),
            }
            for source_id, actual, champion_prediction, challenger_prediction in zip(gate_ids, y_gate, champion_predictions, challenger_predictions)
        ] if candidate_gate_valid else []
        if not candidate_gate_valid:
            gate["accepted"] = False
            gate["rejection_reason"] = "candidate_gate_predictions_not_finite_and_nonnegative"
            details.update({"gate": gate, "parent_tree_count": parent_rounds, "candidate_tree_count": len(challenger_booster.get_dump()), "parameters": params})
            _save_run(db_path, run_id, status="rejected", completed_at=lease.now(), metrics=gate, details=details, lease=lease)
            return {"status": "rejected", "run_id": run_id, "gate": gate}
        try:
            archive = build_candidate_archive(champion_path, challenger_booster)
            edge_records = _native_edge_records(schema_path)
            candidate_records = _format_records(gate_X, feature_names) + edge_records
            edge_matrix = preprocessor.transform(pd.DataFrame([
                {name: np.nan if record.get(name) is None or record.get(name) == "" else record.get(name) for name in feature_names}
                for record in edge_records
            ], columns=feature_names))
            edge_predictions = np.asarray(challenger_booster.predict(xgb.DMatrix(edge_matrix, nthread=1)), dtype=np.float64)
            if not np.isfinite(edge_predictions).all() or np.any(edge_predictions < 0):
                raise ValueError("candidate edge predictions are not finite and nonnegative")
            parity = verify_candidate_parity(archive, challenger_booster, preprocessor, candidate_records)
            candidate_metadata, _ = archive_metadata(archive)
            if preprocessing_sha256(candidate_metadata) != expected_preprocessing:
                raise TrainingDataError("candidate preprocessing fingerprint differs from champion")
        except Exception as exc:
            gate["accepted"] = False
            gate["rejection_reason"] = f"candidate_export_or_serving_check_failed: {type(exc).__name__}: {exc}"
            details.update({"gate": gate, "parent_tree_count": parent_rounds, "candidate_tree_count": len(challenger_booster.get_dump()), "parameters": params})
            _save_run(db_path, run_id, status="rejected", completed_at=lease.now(), metrics=gate, details=details, lease=lease)
            return {"status": "rejected", "run_id": run_id, "gate": gate}
        gate["serving_parity"] = parity
        gate["accepted"] = bool(gate["accepted"] and np.isfinite(challenger_predictions).all() and np.all(challenger_predictions >= 0))
        details.update({"gate": gate, "parent_tree_count": parent_rounds, "candidate_tree_count": len(challenger_booster.get_dump()), "parameters": params, "serving_parity": parity})
        if not gate["accepted"]:
            _save_run(db_path, run_id, status="rejected", completed_at=lease.now(), metrics=gate, details=details, lease=lease)
            return {"status": "rejected", "run_id": run_id, "gate": gate}

        training = {
            "parameters": {**params, "num_boost_round": BOOST_ROUNDS, "tree_growth_cap": tree_growth_cap},
            "run_id": run_id,
            "cutoff": cutoff,
            "snapshot_hash": snapshot_hash,
            "eligible_ids": eligible_ids,
            "new_ids": new_ids,
            "environment": {"python": __import__("sys").version.split()[0], "xgboost": xgb.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        }
        gate_manifest = {
            "policy": gate["policy"],
            "results": {key: value for key, value in gate.items() if key not in {"policy", "accepted", "paired_errors", "evaluation_rows"}},
            "accepted": True,
        }
        lease.assert_owned()
        version, candidate_dir, candidate_manifest = write_immutable_version(
            registry_path,
            archive,
            parent_version=parent_version,
            gate=gate_manifest,
            training=training,
            schema_sha256=schema_hash,
        )
        if candidate_manifest["preprocessing_sha256"] != expected_preprocessing:
            raise RuntimeError("Registry writer changed the prevalidated candidate preprocessing fingerprint")
        activation_id = str(uuid.uuid4())
        details.update({"candidate_version": version, "candidate_directory": str(candidate_dir), "activation_id": activation_id})
        _store_running_details(db_path, run_id, details, lease=lease, metrics=gate)
        if not admin_token:
            raise RuntimeError("ADMIN_TOKEN is required to activate an accepted candidate")
        lease.assert_owned()
        activation = _request_activation(
            base_url=api_base_url,
            token=admin_token,
            version=version,
            expected_parent=parent_version,
            activation_id=activation_id,
            trainer_run_id=run_id,
        )
        details["activation_acknowledgement"] = activation
        _save_run(db_path, run_id, status="promoted", completed_at=lease.now(), metrics=gate, details=details, lease=lease)
        return {"status": "promoted", "run_id": run_id, "version": version, "activation_id": activation_id, "gate": gate}
    except BaseException as exc:
        if isinstance(exc, ActivationUncertain):
            lease.expire_owned_lease()
            return {
                "status": "pending_reconciliation",
                "run_id": run_id,
                "activation_id": details.get("activation_id"),
            }
        retry_number = int(details.get("retry_number", 0)) + 1
        retry_after = time.time() + min(RETRY_MAX_SECONDS, RETRY_BASE_SECONDS * (2 ** (retry_number - 1)))
        details.update({
            "failure": f"{type(exc).__name__}: {exc}",
            "retry_number": retry_number,
            "retry_after": retry_after,
            "cutoff": cutoff,
            "snapshot_hash": snapshot_hash,
            "identity_hash": identity_hash,
        })
        try:
            _save_run(db_path, run_id, status="failed", completed_at=lease.now(), metrics=details.get("gate"), details=details, lease=lease)
        except TrainerLeaseLost:
            return {"status": "failed", "reason": "trainer_lease_lost", "run_id": run_id}
        return {"status": "failed", "run_id": run_id, "retry_after": retry_after, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        lease.stop()


def run_interval(interval_seconds: float, **kwargs: Any) -> None:
    if not math.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError("Interval must be finite and positive")
    while True:
        try:
            result = run_once(**kwargs)
        except Exception as exc:
            result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(result, allow_nan=False, default=str), flush=True)
        time.sleep(interval_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--interval", type=float, metavar="SECONDS")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--registry", type=Path, default=MODEL_REGISTRY_PATH)
    parser.add_argument("--state", type=Path, default=MODEL_STATE_PATH)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reference", type=Path, default=BASELINE_ARCHIVE)
    parser.add_argument("--schema", type=Path, default=SCHEMA_PATH)
    parser.add_argument("--api-base-url", default=None)
    parser.add_argument("--minimum-new-labels", type=int, default=MIN_NEW_LABELS)
    parser.add_argument("--lease-seconds", type=float, default=LEASE_SECONDS)
    parser.add_argument("--tree-growth-cap", type=int, default=TREE_GROWTH_CAP)
    args = parser.parse_args()
    kwargs = {
        "db_path": args.db,
        "registry_path": args.registry,
        "state_path": args.state,
        "dataset_path": args.dataset,
        "split_manifest_path": args.split_manifest,
        "frozen_archive": args.reference,
        "schema_path": args.schema,
        "api_base_url": args.api_base_url,
        "minimum_new_labels": args.minimum_new_labels,
        "lease_seconds": args.lease_seconds,
        "tree_growth_cap": args.tree_growth_cap,
    }
    if args.once:
        print(json.dumps(run_once(**kwargs), indent=2, allow_nan=False, default=str))
    else:
        run_interval(args.interval, **kwargs)


if __name__ == "__main__":
    main()
