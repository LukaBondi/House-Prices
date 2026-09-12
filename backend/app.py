from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("house-prices-api")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "baseline_xgb_pipeline.pkl"
DEFAULT_SCHEMA_PATH = ROOT / "models" / "feature_schema.json"

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
SCHEMA_PATH = Path(os.getenv("SCHEMA_PATH", str(DEFAULT_SCHEMA_PATH)))

app = FastAPI(
    title="House Prices API",
    description="Predict SalePrice from Ames housing features.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
feature_schema: dict[str, Any] = {}


class PredictRequest(BaseModel):
    features: dict[str, Any] = Field(
        ...,
        description="House feature map. Partial inputs are allowed; missing values are imputed.",
    )


class PredictResponse(BaseModel):
    sale_price: float
    currency: str = "USD"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


def load_artifacts() -> None:
    global model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    feature_schema.clear()
    if SCHEMA_PATH.exists():
        feature_schema.update(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))
    else:
        feature_schema["features"] = []
    logger.info("Loaded model from %s", MODEL_PATH)


def build_feature_frame(payload: dict[str, Any]) -> pd.DataFrame:
    feature_names = feature_schema.get("features") or list(payload.keys())
    row: dict[str, Any] = {}
    for name in feature_names:
        value = payload.get(name, np.nan)
        if value is None or value == "":
            value = np.nan
        row[name] = value
    return pd.DataFrame([row], columns=feature_names)


@app.on_event("startup")
def startup() -> None:
    load_artifacts()


@app.get("/")
def root() -> RedirectResponse:
    """Browsers hitting the API root get the interactive docs instead of a bare 404."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=model is not None)


@app.get("/schema")
def schema() -> dict[str, Any]:
    if not feature_schema:
        raise HTTPException(status_code=503, detail="Feature schema not loaded")
    return feature_schema


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not request.features:
        raise HTTPException(
            status_code=422,
            detail="No features provided. Send a non-empty 'features' object.",
        )

    try:
        frame = build_feature_frame(request.features)
        prediction = float(model.predict(frame)[0])
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    logger.info("Prediction=%s for %d input keys", round(prediction, 2), len(request.features))
    return PredictResponse(sale_price=round(prediction, 2))
