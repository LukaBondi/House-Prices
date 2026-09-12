"""Unit tests for backend helpers."""

import backend.app as api


def test_build_feature_frame_fills_missing_with_nan():
    original = list(api.feature_schema.get("features", []))
    api.feature_schema["features"] = ["OverallQual", "GrLivArea", "Neighborhood"]
    try:
        frame = api.build_feature_frame({"OverallQual": 7, "Neighborhood": "CollgCr"})
        assert list(frame.columns) == ["OverallQual", "GrLivArea", "Neighborhood"]
        assert frame.loc[0, "OverallQual"] == 7
        assert frame.loc[0, "Neighborhood"] == "CollgCr"
        assert frame.loc[0, "GrLivArea"] != frame.loc[0, "GrLivArea"]  # NaN check
    finally:
        api.feature_schema["features"] = original


def test_build_feature_frame_treats_empty_as_nan():
    original = list(api.feature_schema.get("features", []))
    api.feature_schema["features"] = ["LotArea"]
    try:
        frame = api.build_feature_frame({"LotArea": ""})
        assert frame.loc[0, "LotArea"] != frame.loc[0, "LotArea"]
    finally:
        api.feature_schema["features"] = original
