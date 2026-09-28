"""Small XGBoost runtime for the frozen Phase 2 baseline."""

from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse
import xgboost as xgb


class NativeModel:
    """Apply the fitted baseline preprocessing metadata and predict from CSR."""

    def __init__(self, path: str | Path, *, nthread: int = 1) -> None:
        self.path = Path(path)
        with zipfile.ZipFile(self.path) as archive:
            self.metadata = json.loads(archive.read("metadata.json"))
            self.booster = xgb.Booster()
            self.booster.load_model(bytearray(archive.read("booster.ubj")))
        self.booster.set_param({"nthread": nthread})
        self.feature_names = self.metadata["feature_names"]
        self.numeric_features = self.metadata["numeric_features"]
        self.categorical_features = self.metadata["categorical_features"]
        self.medians = np.asarray(self.metadata["numeric_medians"], dtype=np.float64)
        self.modes = self.metadata["categorical_modes"]
        self.category_maps = [
            {self._category_key(value): index for index, value in enumerate(categories)}
            for categories in self.metadata["categories"]
        ]
        self.categorical_offsets = [int(value) for value in self.metadata["categorical_offsets"]]
        self.n_features = int(self.metadata["n_output_features"])

    @staticmethod
    def _category_key(value: Any) -> str:
        # The fitted sklearn OneHotEncoder sees the same string categories.
        return str(value)

    def build_csr(self, records: list[dict[str, Any]]) -> scipy.sparse.csr_matrix:
        cols: list[int] = []
        data: list[float] = []
        indptr = [0]
        numeric_count = len(self.numeric_features)
        for record in records:
            for column_index, feature in enumerate(self.numeric_features):
                value = record.get(feature)
                if value is None or value == "":
                    numeric_value = self.medians[column_index]
                else:
                    try:
                        numeric_value = float(value)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"{feature} must be numeric") from exc
                    if math.isnan(numeric_value):
                        numeric_value = self.medians[column_index]
                if not math.isfinite(numeric_value):
                    raise ValueError(f"{feature} must be finite")
                # ColumnTransformer's dense numeric block is converted to CSR;
                # explicit zeros are eliminated by sklearn's sparse path.
                if numeric_value != 0.0:
                    cols.append(column_index)
                    data.append(float(numeric_value))

            for column_index, feature in enumerate(self.categorical_features):
                value = record.get(feature)
                if value is None or value == "" or (
                    isinstance(value, (float, np.floating)) and math.isnan(value)
                ):
                    value = self.modes[column_index]
                elif not isinstance(value, str):
                    raise ValueError(f"{feature} must be a string")
                category_index = self.category_maps[column_index].get(
                    self._category_key(value)
                )
                # OneHotEncoder(handle_unknown="ignore") emits an all-zero block.
                if category_index is not None:
                    cols.append(
                        numeric_count
                        + self.categorical_offsets[column_index]
                        + category_index
                    )
                    data.append(1.0)
            indptr.append(len(data))

        return scipy.sparse.csr_matrix(
            (
                np.asarray(data, dtype=np.float32),
                np.asarray(cols, dtype=np.int32),
                np.asarray(indptr, dtype=np.int32),
            ),
            shape=(len(records), self.n_features),
            dtype=np.float32,
        )

    def validate_record(self, record: dict[str, Any]) -> None:
        """Reject malformed known values before they enter the batch queue."""
        for feature in self.numeric_features:
            value = record.get(feature)
            if value is None or value == "":
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{feature} must be numeric") from exc
            if not math.isfinite(numeric_value):
                raise ValueError(f"{feature} must be finite")
        for feature in self.categorical_features:
            value = record.get(feature)
            if value is not None and value != "" and not isinstance(value, str):
                raise ValueError(f"{feature} must be a string")

    def predict_records(self, records: list[dict[str, Any]]) -> np.ndarray:
        if not records:
            return np.empty(0, dtype=np.float32)
        matrix = self.build_csr(records)
        return np.asarray(self.booster.inplace_predict(matrix), dtype=np.float64)
