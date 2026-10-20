"""Package and parity-check a continued booster without changing frozen export."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from backend.native_model import NativeModel
from model_training.registry import archive_metadata


def build_candidate_archive(parent_archive: str | Path, booster: xgb.Booster) -> bytes:
    parent_bytes = Path(parent_archive).read_bytes()
    _, metadata_bytes = archive_metadata(parent_bytes)
    booster_bytes = bytes(booster.save_raw(raw_format="ubj"))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, content in (("metadata.json", metadata_bytes), ("booster.ubj", booster_bytes)):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info._compresslevel = 9
            archive.writestr(info, content)
    return buffer.getvalue()


def _frame(records: list[dict[str, Any]], feature_names: list[str]) -> pd.DataFrame:
    rows = []
    for record in records:
        row = {}
        for name in feature_names:
            value = record.get(name)
            row[name] = np.nan if value is None or value == "" else value
        rows.append(row)
    return pd.DataFrame(rows, columns=feature_names)


def verify_candidate_parity(
    archive_bytes: bytes,
    booster: xgb.Booster,
    preprocessor: Any,
    records: list[dict[str, Any]],
    *,
    tolerance: float = 1e-6,
) -> dict[str, float | int]:
    """Compare the in-memory challenger with the candidate native serving path."""
    if not records:
        raise ValueError("Candidate parity requires at least one record")
    feature_names = list(preprocessor.feature_names_in_)
    matrix = preprocessor.transform(_frame(records, feature_names))
    expected = np.asarray(booster.predict(xgb.DMatrix(matrix)), dtype=np.float64)
    with __import__("tempfile").NamedTemporaryFile(suffix=".zip", delete=False) as stream:
        path = Path(stream.name)
        stream.write(archive_bytes)
    try:
        actual = NativeModel(path).predict_records(records)
    finally:
        path.unlink(missing_ok=True)
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("Candidate parity produced nonfinite predictions")
    difference = np.abs(expected - actual)
    max_absolute = float(difference.max(initial=0.0))
    if not np.allclose(expected, actual, rtol=tolerance, atol=1e-4):
        raise AssertionError(f"Native candidate parity failed; max absolute difference={max_absolute}")
    if [round(float(value), 2) for value in expected] != [round(float(value), 2) for value in actual]:
        raise AssertionError("Native candidate does not match trainer to rounded cents")
    return {"rows": len(records), "max_absolute_difference": max_absolute}


def gate_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if y_true.ndim != 1 or prediction.shape != y_true.shape or len(y_true) == 0:
        raise ValueError("Gate actuals and predictions must be matching nonempty vectors")
    if not np.isfinite(y_true).all() or not np.isfinite(prediction).all() or np.any(y_true < 0) or np.any(prediction < 0):
        raise ValueError("Gate actuals and predictions must be finite and nonnegative")
    residual = prediction - y_true
    return {
        "n": int(len(y_true)),
        "rmsle": float(np.sqrt(np.mean((np.log1p(prediction) - np.log1p(y_true)) ** 2))),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
    }


def evaluate_gate(
    y_true: np.ndarray,
    champion_prediction: np.ndarray,
    challenger_prediction: np.ndarray,
    *,
    minimum_relative_rmsle_gain: float = 0.01,
) -> dict[str, Any]:
    champion = gate_metrics(y_true, champion_prediction)
    try:
        challenger = gate_metrics(y_true, challenger_prediction)
    except ValueError as exc:
        return {
            "policy": {
                "minimum_relative_rmsle_gain": minimum_relative_rmsle_gain,
                "challenger_rmse_must_not_exceed_champion": True,
                "predictions_finite_nonnegative": True,
            },
            "champion": champion,
            "challenger": None,
            "relative_rmsle_gain": None,
            "accepted": False,
            "rejection_reason": f"invalid_challenger_predictions: {exc}",
        }
    gain = (champion["rmsle"] - challenger["rmsle"]) / champion["rmsle"] if champion["rmsle"] > 0 else 0.0
    accepted = bool(
        gain >= minimum_relative_rmsle_gain
        and challenger["rmse"] <= champion["rmse"]
    )
    return {
        "policy": {
            "minimum_relative_rmsle_gain": minimum_relative_rmsle_gain,
            "challenger_rmse_must_not_exceed_champion": True,
            "predictions_finite_nonnegative": True,
        },
        "champion": champion,
        "challenger": challenger,
        "relative_rmsle_gain": float(gain),
        "accepted": accepted,
    }
