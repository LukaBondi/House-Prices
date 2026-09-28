"""Integration tests for the prediction API and UI static assets."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app import app

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / "frontend"


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
