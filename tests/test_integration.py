"""Integration tests for the prediction API and UI static assets."""

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
