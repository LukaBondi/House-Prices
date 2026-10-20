"""Consistent SQLite monitoring summaries and fixed-reference drift metrics."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.distance import jensenshannon

from backend.event_store import canonical_json, open_database

ALLOWED_WINDOWS_MINUTES = (15, 60, 1440)
MINIMUM_DRIFT_SAMPLES = 100
DEFAULT_DRIFT_THRESHOLD = 0.10
FROZEN_SOURCE_DATASET_SHA256 = "1e18addf81e5e4d347cc17ee6075bbe4a42b7fa26b9e5b063e8f692a5f929d41"
FROZEN_SPLIT_MANIFEST_SHA256 = "ee20e8982d27f4953035f627e3a8edb75078586cda919b742443a804e9c78b64"
FROZEN_FIT_IDS_SHA256 = "29fd7b75ce44cdfbf6375347b498426520c8b89511e72d8af5b8188773ac6be3"
FROZEN_FIT_ROW_COUNT = 1168
FROZEN_FEATURE_COUNT = 79


class MonitoringError(RuntimeError):
    """Monitoring data could not be read or validated."""


def _normalise_counts(values: Sequence[float], expected: int, label: str) -> np.ndarray:
    try:
        counts = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MonitoringError(f"invalid {label} reference counts") from exc
    if counts.ndim != 1 or len(counts) != expected:
        raise MonitoringError(f"{label} reference count vector has the wrong length")
    if np.any(~np.isfinite(counts)) or np.any(counts < 0) or counts.sum() <= 0:
        raise MonitoringError(f"{label} reference counts must be finite and nonzero")
    return counts / counts.sum()


def _numeric_reference(spec: Mapping[str, Any]) -> tuple[float | None, list[float], np.ndarray]:
    bins_value = spec.get("bins")
    bins = bins_value if isinstance(bins_value, Mapping) else {}
    edges_value = bins.get("edges", bins.get("upper_bounds", spec.get("edges", [])))
    minimum = bins.get("minimum", spec.get("minimum"))
    if edges_value is None:
        edges_value = []
    if not isinstance(edges_value, Sequence) or isinstance(edges_value, (str, bytes)):
        raise MonitoringError("numeric drift edges must be a list")
    try:
        edges = [float(value) for value in edges_value]
        minimum_value = float(minimum) if minimum is not None else None
    except (TypeError, ValueError, OverflowError) as exc:
        raise MonitoringError("numeric drift edges must be numeric") from exc
    if any(not math.isfinite(value) for value in edges) or (
        minimum_value is not None and not math.isfinite(minimum_value)
    ):
        raise MonitoringError("numeric drift edges must be finite")
    if edges != sorted(set(edges)):
        raise MonitoringError("numeric drift edges must be sorted and deduplicated")
    counts = bins.get(
        "counts",
        bins.get("probabilities", spec.get("counts", spec.get("probabilities"))),
    )
    if counts is None:
        raise MonitoringError("numeric drift reference is missing fixed bucket counts")
    expected = len(edges) + 3
    reference = _normalise_counts(counts, expected, "numeric")
    return minimum_value, edges, reference


def _categorical_reference(
    spec: Mapping[str, Any],
) -> tuple[list[str], np.ndarray]:
    categories = spec.get("categories", [])
    if not isinstance(categories, Sequence) or isinstance(categories, (str, bytes)):
        raise MonitoringError("categorical drift categories must be a list")
    vocabulary = [str(value) for value in categories]
    if len(vocabulary) != len(set(vocabulary)):
        raise MonitoringError("categorical drift vocabulary must be unique")
    counts = spec.get("counts", spec.get("probabilities"))
    if counts is None:
        raise MonitoringError("categorical drift reference is missing fixed bucket counts")
    reference = _normalise_counts(counts, len(vocabulary) + 2, "categorical")
    return vocabulary, reference


def _feature_histogram(
    records: Sequence[Mapping[str, Any]],
    feature: str,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    kind = str(spec.get("kind", "")).lower()
    missing = 0
    unknown = 0
    underflow = 0
    overflow = 0
    nonmissing = 0

    if kind in {"numeric", "number", "continuous"}:
        minimum, edges, reference = _numeric_reference(spec)
        counts = np.zeros(len(edges) + 3, dtype=np.float64)
        for record in records:
            value = record.get(feature)
            if value is None or value == "":
                counts[0] += 1
                missing += 1
                continue
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError):
                counts[0] += 1
                missing += 1
                continue
            if not math.isfinite(number):
                counts[0] += 1
                missing += 1
                continue
            nonmissing += 1
            if minimum is not None and number < minimum:
                counts[1] += 1
                underflow += 1
            else:
                index = int(np.searchsorted(edges, number, side="left"))
                if index >= len(edges):
                    counts[-1] += 1
                    overflow += 1
                else:
                    counts[2 + index] += 1
        if counts.sum() == 0:
            return {
                "distance": None,
                "observed_count": 0,
                "nonmissing_count": 0,
                "missing_count": 0,
                "missing_rate": None,
                "unknown_count": 0,
                "underflow_count": 0,
                "overflow_count": 0,
            }
        observed = counts / counts.sum()
        distance = float(jensenshannon(reference, observed, base=2.0))
        return {
            "distance": distance,
            "observed_count": int(counts.sum()),
            "nonmissing_count": nonmissing,
            "missing_count": missing,
            "missing_rate": missing / int(counts.sum()),
            "unknown_count": 0,
            "underflow_count": underflow,
            "overflow_count": overflow,
        }

    if kind in {"categorical", "category", "categorical_feature"}:
        vocabulary, reference = _categorical_reference(spec)
        category_index = {value: index for index, value in enumerate(vocabulary)}
        counts = np.zeros(len(vocabulary) + 2, dtype=np.float64)
        for record in records:
            value = record.get(feature)
            if value is None or value == "":
                counts[0] += 1
                missing += 1
                continue
            nonmissing += 1
            index = category_index.get(str(value))
            if index is None:
                counts[-1] += 1
                unknown += 1
            else:
                counts[index + 1] += 1
        if counts.sum() == 0:
            return {
                "distance": None,
                "observed_count": 0,
                "nonmissing_count": 0,
                "missing_count": 0,
                "missing_rate": None,
                "unknown_count": 0,
                "unknown_rate": None,
                "underflow_count": 0,
                "overflow_count": 0,
            }
        observed = counts / counts.sum()
        distance = float(jensenshannon(reference, observed, base=2.0))
        return {
            "distance": distance,
            "observed_count": int(counts.sum()),
            "nonmissing_count": nonmissing,
            "missing_count": missing,
            "missing_rate": missing / int(counts.sum()),
            "unknown_count": unknown,
            "unknown_rate": unknown / nonmissing if nonmissing else None,
            "underflow_count": 0,
            "overflow_count": 0,
        }
    raise MonitoringError(f"unsupported drift feature kind for {feature!r}")


def _load_reference(path: str | Path) -> tuple[dict[str, Any] | None, str | None]:
    reference_path = Path(path)
    if not reference_path.is_file():
        return None, None
    try:
        value = json.loads(reference_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MonitoringError(f"could not read drift reference: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("features"), dict):
        raise MonitoringError("drift reference must contain a feature map")
    if value.get("format") != "house-prices-drift-v1":
        raise MonitoringError("unsupported drift reference format")
    source_hash = value.get("source_dataset_sha256")
    dataset_hash = value.get("dataset_sha256")
    if (
        source_hash != FROZEN_SOURCE_DATASET_SHA256
        or dataset_hash != FROZEN_SOURCE_DATASET_SHA256
    ):
        raise MonitoringError("drift reference does not match the frozen source dataset")
    if value.get("source_split_manifest_sha256") != FROZEN_SPLIT_MANIFEST_SHA256:
        raise MonitoringError("drift reference does not match the frozen split manifest")
    if value.get("target_excluded") != "SalePrice" or value.get("identity_excluded") != "Id":
        raise MonitoringError("drift reference must exclude target and row identity")
    fit_ids = value.get("fit_ids")
    if (
        not isinstance(fit_ids, list)
        or len(fit_ids) != FROZEN_FIT_ROW_COUNT
        or any(not isinstance(row_id, int) or isinstance(row_id, bool) for row_id in fit_ids)
        or len(set(fit_ids)) != FROZEN_FIT_ROW_COUNT
        or any(row_id < 1 or row_id > 1460 for row_id in fit_ids)
    ):
        raise MonitoringError("drift reference must identify the 1,168 original fit rows")
    fit_ids_hash = hashlib.sha256(canonical_json(fit_ids).encode("utf-8")).hexdigest()
    if fit_ids_hash != FROZEN_FIT_IDS_SHA256 or value.get("fit_ids_sha256") != fit_ids_hash:
        raise MonitoringError("drift reference fit ID checksum does not match")
    if value.get("distance") != "jensen-shannon-distance" or value.get("log_base") != 2:
        raise MonitoringError("drift reference must declare base-2 Jensen-Shannon distance")
    features = value["features"]
    if (
        len(features) != FROZEN_FEATURE_COUNT
        or "Id" in features
        or "SalePrice" in features
        or any(not isinstance(name, str) or not name for name in features)
    ):
        raise MonitoringError("drift reference feature set does not match the frozen schema")
    for name, spec in features.items():
        if not isinstance(spec, Mapping):
            raise MonitoringError(f"drift reference feature {name!r} must be an object")
        kind = spec.get("kind")
        if kind == "numeric":
            _validate_numeric_spec(name, spec)
        elif kind == "categorical":
            _validate_categorical_spec(name, spec)
        else:
            raise MonitoringError(f"drift reference feature {name!r} has an unsupported kind")
    reference_id = value.get("reference_id")
    if not isinstance(reference_id, str) or len(reference_id) != 64:
        raise MonitoringError("drift reference identity is missing or invalid")
    identity = {key: item for key, item in value.items() if key != "reference_id"}
    expected_id = hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()
    if reference_id != expected_id:
        raise MonitoringError("drift reference identity checksum does not match its contents")
    return value, reference_id


def _count_vector(
    values: Any,
    expected_length: int,
    *,
    label: str,
) -> list[int]:
    if (
        not isinstance(values, list)
        or len(values) != expected_length
        or any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in values)
        or sum(values) != FROZEN_FIT_ROW_COUNT
    ):
        raise MonitoringError(
            f"{label} reference counts must be nonnegative integer source observations"
        )
    return values


def _probability_vector(
    values: Any,
    counts: list[int],
    *,
    label: str,
) -> None:
    if not isinstance(values, list) or len(values) != len(counts):
        raise MonitoringError(f"{label} reference probabilities have the wrong length")
    try:
        probabilities = [float(item) for item in values]
    except (TypeError, ValueError, OverflowError) as exc:
        raise MonitoringError(f"{label} reference probabilities must be numeric") from exc
    if any(not math.isfinite(item) or item < 0 for item in probabilities):
        raise MonitoringError(f"{label} reference probabilities must be finite and nonnegative")
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise MonitoringError(f"{label} reference probabilities must sum to one")
    if any(
        not math.isclose(actual, count / FROZEN_FIT_ROW_COUNT, rel_tol=0.0, abs_tol=1e-12)
        for actual, count in zip(probabilities, counts)
    ):
        raise MonitoringError(f"{label} reference probabilities do not match the source counts")


def _validate_common_counts(
    spec: Mapping[str, Any], counts: list[int], *, label: str
) -> None:
    if spec.get("sample_count") != FROZEN_FIT_ROW_COUNT:
        raise MonitoringError(f"{label} sample count does not match the original fit rows")
    if spec.get("nonmissing_count") != FROZEN_FIT_ROW_COUNT - counts[0]:
        raise MonitoringError(f"{label} nonmissing count does not match its missing bucket")
    _count_vector(spec.get("reference_counts"), len(counts), label=label)
    _count_vector(spec.get("counts"), len(counts), label=label)
    if spec["reference_counts"] != counts or spec["counts"] != counts:
        raise MonitoringError(f"{label} reference count aliases disagree")
    _probability_vector(spec.get("reference_probabilities"), counts, label=label)
    _probability_vector(spec.get("probabilities"), counts, label=label)
    if spec["reference_probabilities"] != spec["probabilities"]:
        raise MonitoringError(f"{label} reference probability aliases disagree")


def _validate_numeric_spec(name: str, spec: Mapping[str, Any]) -> None:
    bins = spec.get("bins")
    if not isinstance(bins, Mapping):
        raise MonitoringError(f"numeric drift feature {name!r} is missing its bins object")
    try:
        minimum = float(spec["minimum"])
        maximum = float(spec["maximum"])
        bins_minimum = float(bins["minimum"])
        edges = [float(item) for item in bins["edges"]]
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise MonitoringError(f"numeric drift feature {name!r} has invalid boundaries") from exc
    if (
        not math.isfinite(minimum)
        or not math.isfinite(maximum)
        or not math.isfinite(bins_minimum)
        or minimum != bins_minimum
        or minimum > maximum
        or not edges
        or any(not math.isfinite(edge) for edge in edges)
        or edges != sorted(set(edges))
        or edges[0] < minimum
        or edges[-1] != maximum
        or spec.get("bin_edges") != bins["edges"]
    ):
        raise MonitoringError(f"numeric drift feature {name!r} has inconsistent boundaries")
    expected_order = [
        "missing",
        "underflow",
        *[f"bin_{index}" for index in range(len(edges))],
        "overflow",
    ]
    if (
        spec.get("bucket_order") != expected_order
        or spec.get("missing_bucket") != "missing"
        or spec.get("underflow_bucket") != "underflow"
        or spec.get("overflow_bucket") != "overflow"
    ):
        raise MonitoringError(f"numeric drift feature {name!r} has an invalid bucket layout")
    counts = _count_vector(
        bins.get("counts"), len(edges) + 3, label=f"numeric feature {name!r}"
    )
    _validate_common_counts(spec, counts, label=f"numeric feature {name!r}")
    _count_vector(bins.get("reference_counts", counts), len(counts), label=f"numeric feature {name!r}")
    if bins.get("reference_counts", counts) != counts:
        raise MonitoringError(f"numeric drift feature {name!r} bin counts disagree")
    _probability_vector(bins.get("probabilities"), counts, label=f"numeric feature {name!r}")
    _probability_vector(
        bins.get("reference_probabilities", bins.get("probabilities")),
        counts,
        label=f"numeric feature {name!r}",
    )


def _validate_categorical_spec(name: str, spec: Mapping[str, Any]) -> None:
    categories = spec.get("categories")
    if (
        not isinstance(categories, list)
        or any(not isinstance(value, str) for value in categories)
        or categories != sorted(set(categories))
    ):
        raise MonitoringError(f"categorical drift feature {name!r} has an invalid vocabulary")
    expected_order = [
        "missing",
        *[f"category_{index}" for index in range(len(categories))],
        "unknown",
    ]
    if (
        spec.get("bucket_order") != expected_order
        or spec.get("missing_bucket") != "missing"
        or spec.get("unknown_bucket") != "unknown"
    ):
        raise MonitoringError(f"categorical drift feature {name!r} has an invalid bucket layout")
    counts = _count_vector(
        spec.get("counts"), len(categories) + 2, label=f"categorical feature {name!r}"
    )
    _validate_common_counts(spec, counts, label=f"categorical feature {name!r}")


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def build_monitoring_summary(
    database_path: str | Path,
    drift_reference_path: str | Path,
    *,
    window_minutes: int = 15,
    current_model_version: str | None = None,
    logging_status: Mapping[str, Any] | None = None,
    as_of: float | None = None,
    drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
) -> dict[str, Any]:
    if window_minutes not in ALLOWED_WINDOWS_MINUTES:
        raise ValueError(f"window_minutes must be one of {ALLOWED_WINDOWS_MINUTES}")
    now = time.time() if as_of is None else float(as_of)
    start = now - window_minutes * 60
    connection = open_database(database_path)
    try:
        connection.execute("BEGIN")
        request_rows = connection.execute(
            "SELECT request_id, received_at, completed_at, status_code, lifecycle, "
            "handler_duration_ms, feature_fingerprint, features_json, cache_outcome, path, "
            "model_version FROM requests WHERE path='/predict' AND method='POST' "
            "AND received_at>=? AND received_at<=?",
            (start, now),
        ).fetchall()
        label_rows = connection.execute(
            "SELECT observation_id FROM labels WHERE available_at>=? AND available_at<=?",
            (start, now),
        ).fetchall()
        last_run = connection.execute(
            "SELECT run_id, parent_version, status, started_at, completed_at, metrics_json, details_json "
            "FROM runs ORDER BY COALESCE(completed_at, started_at) DESC LIMIT 1"
        ).fetchone()
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()

    total = len(request_rows)
    completed = [
        row
        for row in request_rows
        if row["lifecycle"] in {"completed", "logging_failed"}
        and row["status_code"] is not None
    ]
    pending = sum(row["lifecycle"] == "pending" for row in request_rows)
    aborted = sum(row["lifecycle"] == "aborted" for row in request_rows)
    logging_failed = sum(row["lifecycle"] == "logging_failed" for row in request_rows)
    errors_4xx = sum(
        row["status_code"] is not None and 400 <= row["status_code"] < 500
        for row in completed
    )
    errors_5xx = sum(
        row["status_code"] is not None and 500 <= row["status_code"] < 600
        for row in completed
    )
    durations = [
        float(row["handler_duration_ms"])
        for row in completed
        if row["handler_duration_ms"] is not None
        and math.isfinite(float(row["handler_duration_ms"]))
    ]
    fingerprints = {
        row["feature_fingerprint"]
        for row in request_rows
        if row["feature_fingerprint"] is not None
    }
    cache_hits = sum(row["cache_outcome"] == "hit" for row in request_rows)
    cache_misses = sum(row["cache_outcome"] == "miss" for row in request_rows)
    coalesced = sum(row["cache_outcome"] == "coalesced" for row in request_rows)
    validated_records: list[Mapping[str, Any]] = []
    for row in request_rows:
        if row["features_json"] is None or row["feature_fingerprint"] is None:
            continue
        try:
            record = json.loads(row["features_json"])
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(record, dict):
            validated_records.append(record)

    reference, reference_id = _load_reference(drift_reference_path)
    if reference is None:
        drift = {
            "status": "unavailable",
            "reference_id": None,
            "sample_count": len(validated_records),
            "distinct_inputs": len(fingerprints),
            "minimum_samples": MINIMUM_DRIFT_SAMPLES,
            "threshold": drift_threshold,
            "features": {},
        }
    else:
        reference_features = reference["features"]
        feature_results: dict[str, Any] = {}
        for feature, spec in reference_features.items():
            if not isinstance(spec, Mapping):
                continue
            feature_results[str(feature)] = _feature_histogram(
                validated_records,
                str(feature),
                spec,
            )
        distances = [
            values["distance"]
            for values in feature_results.values()
            if values["distance"] is not None
        ]
        if len(validated_records) < MINIMUM_DRIFT_SAMPLES:
            drift_status = "insufficient_data"
        elif not distances:
            drift_status = "unavailable"
        elif any(distance >= drift_threshold for distance in distances):
            drift_status = "advisory"
        else:
            drift_status = "ok"
        drift = {
            "status": drift_status,
            "reference_id": reference_id,
            "dataset_sha256": reference.get("dataset_sha256"),
            "sample_count": len(validated_records),
            "distinct_inputs": len(fingerprints),
            "minimum_samples": MINIMUM_DRIFT_SAMPLES,
            "threshold": drift_threshold,
            "features": feature_results,
        }

    last_run_value = None
    if last_run is not None:
        def decode_json(value: Any) -> Any:
            if value is None:
                return None
            try:
                return json.loads(value)
            except (TypeError, json.JSONDecodeError):
                return None

        last_run_value = {
            "run_id": last_run["run_id"],
            "parent_version": last_run["parent_version"],
            "status": last_run["status"],
            "started_at": last_run["started_at"],
            "completed_at": last_run["completed_at"],
            "metrics": decode_json(last_run["metrics_json"]),
            "details": decode_json(last_run["details_json"]),
        }
    logging = dict(logging_status or {"enabled": True, "healthy": True, "failures": 0})
    return {
        "as_of": now,
        "window": {"minutes": window_minutes, "start": start, "end": now},
        "requests": {
            "total": total,
            "completed": len(completed),
            "pending": pending,
            "aborted": aborted,
            "logging_failed": logging_failed,
            "rate_per_minute": total / window_minutes if total else None,
            "distinct_inputs": len(fingerprints),
        },
        "latency": {
            "handler_p50_ms": _percentile(durations, 50),
            "handler_p95_ms": _percentile(durations, 95),
        },
        "errors": {
            "4xx": {
                "count": errors_4xx,
                "rate": errors_4xx / len(completed) if completed else None,
            },
            "5xx": {
                "count": errors_5xx,
                "rate": errors_5xx / len(completed) if completed else None,
            },
        },
        "cache": {"hits": cache_hits, "misses": cache_misses, "coalesced": coalesced},
        "logging": logging,
        "model": {"version": current_model_version},
        "labels": {
            "count": len(label_rows),
            "distinct_observations": len({row["observation_id"] for row in label_rows}),
        },
        "retraining": {"last_run": last_run_value},
        "drift": drift,
    }
