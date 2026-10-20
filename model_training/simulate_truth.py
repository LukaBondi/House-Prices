"""Import delayed Ames ground truth only after persisted provenance verification."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from backend import event_store
from model_training.data_partitions import (
    DEFAULT_DATASET,
    DEFAULT_MANIFEST,
    IDENTITY,
    TARGET,
    load_manifest,
)
from model_training.training_config import DB_PATH


class TruthImportConflict(ValueError):
    """A label manifest conflicts with source data or a previously imported label."""


def _source_records(dataset_path: str | Path, manifest_path: str | Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    import pandas as pd

    manifest = load_manifest(manifest_path, dataset_path=dataset_path)
    frame = pd.read_csv(dataset_path)
    source_records: dict[int, dict[str, Any]] = {}
    feature_names = manifest["source"]["feature_names"]
    for _, row in frame.iterrows():
        source_id = int(row[IDENTITY])
        source_records[source_id] = {
            "features": {
                name: (None if pd.isna(row[name]) else row[name].item() if hasattr(row[name], "item") else row[name])
                for name in feature_names
            },
            "sale_price": float(row[TARGET]),
        }
    return source_records, manifest


def _verified_request(
    connection: Any,
    request_ids: list[str],
    *,
    run_id: str,
    dataset_sha256: str,
    source_id: int,
    expected_record: dict[str, Any],
    expected_fingerprint: str,
    feature_names: list[str],
    numeric_features: list[str],
    latest_completed_at: float,
) -> Any | None:
    successful = []
    for request_id in request_ids:
        row = connection.execute("SELECT * FROM requests WHERE request_id = ?", (request_id,)).fetchone()
        if row is None:
            continue
        try:
            completed_at = float(row["completed_at"])
        except (TypeError, ValueError):
            continue
        if (
            row["method"] != "POST"
            or row["path"] != "/predict"
            or row["status_code"] != 200
            or row["lifecycle"] != "completed"
            or row["logging_failed"]
            or row["dataset_sha256"] != dataset_sha256
            or row["source_id"] != source_id
            or row["run_id"] != run_id
            or row["completed_at"] is None
            or not math.isfinite(completed_at)
            or not math.isfinite(latest_completed_at)
            or completed_at > latest_completed_at
            or not row["features_json"]
        ):
            continue
        try:
            persisted = json.loads(row["features_json"])
            canonical = event_store.canonical_record(
                persisted,
                feature_names=feature_names,
                numeric_features=numeric_features,
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if canonical != expected_record:
            continue
        fingerprint = event_store.feature_fingerprint(
            persisted,
            feature_names=feature_names,
            numeric_features=numeric_features,
        )
        if fingerprint != expected_fingerprint or row["feature_fingerprint"] != expected_fingerprint:
            continue
        successful.append(row)
    if not successful:
        return None
    return min(successful, key=lambda row: (float(row["completed_at"]), row["request_id"]))


def import_truth(
    manifest_path: str | Path,
    *,
    db_path: str | Path = DB_PATH,
    dataset_path: str | Path = DEFAULT_DATASET,
    split_manifest_path: str | Path = DEFAULT_MANIFEST,
    now: float | None = None,
) -> dict[str, Any]:
    """Insert elapsed labels after exact source, partition and HTTP checks."""
    now = time.time() if now is None else float(now)
    if not math.isfinite(now):
        raise ValueError("now must be finite")
    source_records, split = _source_records(dataset_path, split_manifest_path)
    private = json.loads(Path(manifest_path).read_text(encoding="utf-8-sig"))
    if private.get("format") != "house-prices-delayed-truth-v1":
        raise ValueError("Unsupported delayed truth manifest format")
    run_id = private.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("Delayed truth manifest requires run_id")
    dataset_sha256 = private.get("dataset_sha256")
    if dataset_sha256 != split["source"]["sha256"]:
        raise TruthImportConflict("Private labels refer to a different source dataset hash")
    feature_names = split["source"]["feature_names"]
    numeric_features = split["source"]["numeric_features"]
    try:
        delay_seconds = float(private.get("delay_seconds"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Delayed truth manifest requires a numeric delay_seconds") from exc
    if not math.isfinite(delay_seconds) or delay_seconds < 0:
        raise ValueError("delay_seconds must be finite and nonnegative")
    adaptation_ids = set(split["partitions"]["adaptation_ids"])
    delayed = imported = idempotent = ineligible = 0
    skipped: list[dict[str, Any]] = []
    event_store.initialize_database(db_path)
    connection = event_store.open_database(db_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        for item in private.get("labels", []):
            source_id = item.get("source_id")
            try:
                source_id = int(source_id)
            except (TypeError, ValueError):
                raise TruthImportConflict(f"Invalid source_id: {source_id!r}")
            if source_id not in adaptation_ids:
                raise TruthImportConflict(f"Source Id {source_id} is not in the adaptation partition")
            source = source_records.get(source_id)
            if source is None:
                raise TruthImportConflict(f"Source Id {source_id} does not exist in original train.csv")
            try:
                sale_price = float(item["sale_price"])
                available_at = float(item["available_at"])
            except (KeyError, TypeError, ValueError):
                ineligible += 1
                skipped.append({"source_id": source_id, "reason": "label has no successful request or availability time"})
                continue
            if not math.isfinite(sale_price) or sale_price < 0 or sale_price != source["sale_price"]:
                raise TruthImportConflict(f"Label for Source Id {source_id} does not match original SalePrice")
            if not math.isfinite(available_at):
                ineligible += 1
                skipped.append({"source_id": source_id, "reason": "label availability time must be finite"})
                continue
            if available_at > now:
                delayed += 1
                continue
            earliest_completion = available_at - delay_seconds
            if not math.isfinite(earliest_completion):
                ineligible += 1
                skipped.append({"source_id": source_id, "reason": "label delay cutoff must be finite"})
                continue
            request_ids = item.get("request_ids")
            if not isinstance(request_ids, list) or not request_ids or any(not isinstance(value, str) for value in request_ids):
                ineligible += 1
                skipped.append({"source_id": source_id, "reason": "no mapped server request UUID"})
                continue
            expected_record = event_store.canonical_record(
                source["features"],
                feature_names=feature_names,
                numeric_features=numeric_features,
            )
            fingerprint = event_store.feature_fingerprint(
                source["features"],
                feature_names=feature_names,
                numeric_features=numeric_features,
            )
            request = _verified_request(
                connection,
                request_ids,
                run_id=run_id,
                dataset_sha256=dataset_sha256,
                source_id=source_id,
                expected_record=expected_record,
                expected_fingerprint=fingerprint,
                feature_names=feature_names,
                numeric_features=numeric_features,
                latest_completed_at=earliest_completion,
            )
            if request is None:
                ineligible += 1
                skipped.append({"source_id": source_id, "reason": "no persisted successful unchanged request with matching provenance"})
                continue
            observation_id = f"{dataset_sha256}:{source_id}"
            existing = connection.execute(
                "SELECT * FROM labels WHERE dataset_sha256 = ? AND source_id = ?",
                (dataset_sha256, source_id),
            ).fetchone()
            if existing is not None:
                if (
                    float(existing["sale_price"]) != sale_price
                    or existing["features_json"] != event_store.canonical_json(expected_record)
                    or existing["feature_fingerprint"] != fingerprint
                ):
                    raise TruthImportConflict(f"Conflicting previously imported label for Source Id {source_id}")
                idempotent += 1
                continue
            connection.execute(
                "INSERT INTO labels (observation_id, request_id, dataset_sha256, source_id, features_json, feature_fingerprint, sale_price, available_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    observation_id,
                    request["request_id"],
                    dataset_sha256,
                    source_id,
                    event_store.canonical_json(expected_record),
                    fingerprint,
                    sale_price,
                    available_at,
                ),
            )
            imported += 1
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()
    return {
        "run_id": run_id,
        "dataset_sha256": dataset_sha256,
        "imported": imported,
        "already_imported": idempotent,
        "not_yet_available": delayed,
        "ineligible": ineligible,
        "skipped": skipped,
        "checked_at": now,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, required=True, help="Private delayed-truth manifest from generate_workload")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--now", type=float, default=None, help="Controlled UNIX timestamp for tests/reproducibility")
    args = parser.parse_args()
    print(json.dumps(import_truth(args.run_manifest, db_path=args.db, dataset_path=args.dataset, split_manifest_path=args.split_manifest, now=args.now), indent=2))


if __name__ == "__main__":
    main()
