"""Export and parity-check the frozen Phase 2 model as a compact native archive."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from backend.native_model import NativeModel

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "models" / "baseline_xgb_pipeline.pkl"
OUTPUT = ROOT / "models" / "house_prices_native.zip"
SCHEMA_PATH = ROOT / "models" / "feature_schema.json"
EXPECTED_SOURCE_SHA256 = "0eed717dd49aa5b5e307577d14b8e11bca813c131f35dc39fb9e41a7c9c26d42"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def metadata_from_pipeline(pipeline: Any) -> dict[str, Any]:
    preprocessor = pipeline.named_steps["preprocessor"]
    numeric_imputer = preprocessor.named_transformers_["num"]
    categorical_pipeline = preprocessor.named_transformers_["cat"]
    categorical_imputer = categorical_pipeline.named_steps["imputer"]
    encoder = categorical_pipeline.named_steps["onehot"]
    numeric_features = list(preprocessor.transformers_[0][2])
    categorical_features = list(preprocessor.transformers_[1][2])
    categories = [
        [json_value(value) for value in values]
        for values in encoder.categories_
    ]
    category_sizes = [len(values) for values in categories]
    offsets = np.cumsum([0, *category_sizes[:-1]]).astype(int).tolist()
    return {
        "format": "house-prices-xgboost-native-v1",
        "source_model": SOURCE.name,
        "source_sha256": sha256(SOURCE),
        "feature_schema_sha256": sha256(SCHEMA_PATH),
        "feature_names": list(pipeline.feature_names_in_),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "numeric_medians": [float(value) for value in numeric_imputer.statistics_],
        "categorical_modes": [
            json_value(value) for value in categorical_imputer.statistics_
        ],
        "categories": categories,
        "categorical_offsets": offsets,
        "n_output_features": int(pipeline.named_steps["regressor"].n_features_in_),
        "booster_features": int(pipeline.named_steps["regressor"].get_booster().num_features()),
        "sparse_output": bool(preprocessor.sparse_output_),
    }


def export() -> None:
    if sha256(SOURCE) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("The frozen baseline model hash changed; refusing export")
    pipeline = joblib.load(SOURCE)
    metadata = metadata_from_pipeline(pipeline)
    if not metadata["sparse_output"]:
        raise RuntimeError("Expected the baseline preprocessing output to be sparse")
    metadata["xgboost_version"] = __import__("xgboost").__version__
    booster_bytes = bytes(pipeline.named_steps["regressor"].get_booster().save_raw(raw_format="ubj"))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, content in (
            ("metadata.json", json.dumps(metadata, separators=(",", ":"))),
            ("booster.ubj", booster_bytes),
        ):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 26, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info._compresslevel = 9
            archive.writestr(info, content)

    native = NativeModel(OUTPUT)
    datasets = [ROOT / "dataset" / name for name in ("train.csv", "test.csv")]
    worst_abs = 0.0
    for path in datasets:
        frame = pd.read_csv(path)
        records = frame.loc[:, metadata["feature_names"]].to_dict(orient="records")
        expected = np.asarray(pipeline.predict(frame.loc[:, metadata["feature_names"]]))
        actual = native.predict_records(records)
        max_abs = float(np.max(np.abs(expected - actual)))
        worst_abs = max(worst_abs, max_abs)
        if not np.allclose(expected, actual, rtol=1e-6, atol=1e-4):
            raise AssertionError(f"Parity failed for {path.name}; max abs error={max_abs}")
        if [round(float(value), 2) for value in expected] != [round(float(value), 2) for value in actual]:
            raise AssertionError(f"Rounded-cent parity failed for {path.name}")

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    sample = dict(schema.get("sample", {}))
    sample.update({"OverallQual": 0, "GrLivArea": None, "Neighborhood": "not-a-known-neighborhood"})
    edge = [sample, {key: None for key in metadata["feature_names"]}, dict(reversed(list(sample.items())))]
    edge_frame = pd.DataFrame(edge, columns=metadata["feature_names"])
    edge_frame = edge_frame.replace({None: np.nan, "": np.nan})
    expected_edge = np.asarray(pipeline.predict(edge_frame))
    actual_edge = native.predict_records(edge)
    if not np.allclose(expected_edge, actual_edge, rtol=1e-6, atol=1e-4):
        raise AssertionError("Parity failed for partial, unknown, null, zero, or reordered rows")
    if [round(float(value), 2) for value in expected_edge] != [round(float(value), 2) for value in actual_edge]:
        raise AssertionError("Rounded-cent parity failed for edge cases")
    print(json.dumps({
        "artifact": str(OUTPUT),
        "artifact_bytes": OUTPUT.stat().st_size,
        "source_sha256": metadata["source_sha256"],
        "train_rows": 1460,
        "test_rows": 1459,
        "edge_rows": len(edge),
        "max_absolute_error": worst_abs,
    }, indent=2))


if __name__ == "__main__":
    export()
