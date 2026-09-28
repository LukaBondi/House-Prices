"""Verify stdlib benchmark CSV parsing matches the dataset's pandas semantics."""

import math

import pandas as pd

from model_training.benchmark_models import _read_dataset


def test_benchmark_records_match_pandas_csv_values_and_missing_cells():
    columns, records, targets = _read_dataset()
    frame = pd.read_csv("dataset/train.csv")
    feature_frame = frame.drop(columns=["Id", "SalePrice"])

    assert columns == feature_frame.columns.tolist()
    assert len(records) == len(feature_frame) == len(targets)

    missing_count = 0
    for row_index, record in enumerate(records):
        for column in columns:
            expected = feature_frame.iloc[row_index][column]
            actual = record[column]
            if pd.isna(expected):
                missing_count += 1
                assert actual is None, (row_index, column, actual)
            elif isinstance(expected, str):
                assert actual == expected, (row_index, column, actual, expected)
            else:
                assert math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=0), (
                    row_index,
                    column,
                    actual,
                    expected,
                )

    assert missing_count > 0
