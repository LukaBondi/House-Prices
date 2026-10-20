"""Integration tests for the prediction API and UI static assets."""

import builtins
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app import app

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / "frontend"


@pytest.fixture(autouse=True)
def isolate_api_state(tmp_path, monkeypatch):
    data_dir = tmp_path / "api-state"
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("DB_PATH", str(data_dir / "requests.sqlite3"))
    monkeypatch.setenv("MODEL_REGISTRY_PATH", str(data_dir / "artifacts"))
    monkeypatch.setenv("MODEL_STATE_PATH", str(data_dir / "state"))
    monkeypatch.setenv("DRIFT_REFERENCE_PATH", str(data_dir / "drift-reference.json"))
    monkeypatch.setenv("ADMIN_TOKEN", "test-admin-token")


def test_predict_returns_200_on_valid_input():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "features": {
                    "OverallQual": 7,
                    "GrLivArea": 1710,
                    "YearBuilt": 2003,
                    "TotalBsmtSF": 856,
                    "GarageCars": 2,
                    "Neighborhood": "CollgCr",
                    "MSZoning": "RL",
                    "KitchenQual": "Gd",
                    "ExterQual": "Gd",
                }
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert "sale_price" in body
        assert isinstance(body["sale_price"], (int, float))
        assert body["sale_price"] > 0


def test_invalid_numeric_feature_is_rejected_before_prediction_queue():
    with TestClient(app) as client:
        before = client.get("/metrics").json()["batch_requests"]
        response = client.post("/predict", json={"features": {"LotArea": "not numeric"}})
        after = client.get("/metrics").json()["batch_requests"]
        assert response.status_code == 400
        assert "LotArea must be numeric" in response.json()["detail"]
        assert after == before


def test_unexpected_inference_failure_is_recorded_as_operational_500(monkeypatch):
    import backend.app as api
    from backend.event_store import open_database

    async def fail_inference(*args, **kwargs):
        raise RuntimeError("native runtime unavailable")

    monkeypatch.setattr(api, "_run_records", fail_inference)
    with TestClient(app) as client:
        response = client.post("/predict", json={"features": {"OverallQual": 7}})
        assert response.status_code == 500
        request_id = response.headers.get("x-request-id")
        assert request_id
    connection = open_database(api._runtime_paths()["database"])
    try:
        row = connection.execute(
            "SELECT status_code, lifecycle FROM requests WHERE request_id=?", (request_id,)
        ).fetchone()
        assert tuple(row) == (500, "completed")
    finally:
        connection.close()


def test_full_app_startup_and_prediction_do_not_import_training_stack(monkeypatch):
    forbidden = {"pandas", "sklearn", "joblib", "model_training"}
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in forbidden:
            raise AssertionError(f"runtime attempted to import {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"features": {"OverallQual": 7, "Neighborhood": "CollgCr"}},
        )
        assert response.status_code == 200
        assert response.json()["sale_price"] > 0


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["model_loaded"] is True


def test_ui_index_contains_predict_controls():
    html = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    assert 'id="predict-form"' in html
    assert 'id="predict-btn"' in html
    assert "SalePrice" in html or "sale price" in html.lower()


def test_ui_js_calls_predict_endpoint():
    js = (FRONTEND_DIR / "app.js").read_text(encoding="utf-8")
    assert "/predict" in js
    assert "fetch" in js


def test_concurrent_predictions_are_batched_and_shutdown_drains_work(monkeypatch):
    from fastapi.testclient import TestClient

    import backend.app as api

    monkeypatch.setattr(api, "DYNAMIC_BATCHING_ENABLED", True)

    with TestClient(app) as client:
        payloads = [
            {"features": {"OverallQual": 4 + index % 5, "GrLivArea": 1300 + index * 11}}
            for index in range(8)
        ]
        with ThreadPoolExecutor(max_workers=8) as executor:
            responses = list(
                executor.map(lambda item: client.post("/predict", json=item), payloads)
            )
        assert all(response.status_code == 200 for response in responses)
        assert all(response.json()["sale_price"] > 0 for response in responses)
        stats = client.get("/metrics").json()
        assert stats["batch_requests"] >= len(payloads)
        assert stats["batch_count"] > 0
        assert stats["queue_overloads"] == 0
    import backend.app as api

    assert api.batch_task is None
    assert api.inference_queue is None
    assert not api.in_flight
