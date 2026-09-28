from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from backend.cache import TTLCache
from backend.native_model import NativeModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("house-prices-api")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "house_prices_native.zip"
DEFAULT_SCHEMA_PATH = ROOT / "models" / "feature_schema.json"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
SCHEMA_PATH = Path(os.getenv("SCHEMA_PATH", str(DEFAULT_SCHEMA_PATH)))
CACHE_ENABLED = os.getenv("PREDICTION_CACHE_ENABLED", "1").lower() not in {"0", "false", "no"}
CACHE_MAX_ENTRIES = int(os.getenv("PREDICTION_CACHE_MAX_ENTRIES", "1024"))
CACHE_TTL_SECONDS = float(os.getenv("PREDICTION_CACHE_TTL_SECONDS", "300"))
DYNAMIC_BATCHING_ENABLED = os.getenv("DYNAMIC_BATCHING_ENABLED", "0").lower() not in {
    "0", "false", "no"
}
BATCH_MAX_SIZE = max(1, int(os.getenv("BATCH_MAX_SIZE", "16")))
BATCH_MAX_DELAY_SECONDS = max(0.0, float(os.getenv("BATCH_MAX_DELAY_MS", "0")) / 1000)
BATCH_QUEUE_MAX_SIZE = max(1, int(os.getenv("BATCH_QUEUE_MAX_SIZE", "256")))
INFERENCE_WORKERS = max(1, int(os.getenv("INFERENCE_WORKERS", "1")))

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

model: Any = None
feature_schema: dict[str, Any] = {}
prediction_cache: TTLCache[float] = TTLCache(
    max_entries=CACHE_MAX_ENTRIES,
    ttl_seconds=CACHE_TTL_SECONDS,
)
inference_queue: asyncio.Queue["QueuedPrediction"] | None = None
batch_task: asyncio.Task[None] | None = None
batch_tasks: list[asyncio.Task[None]] = []
direct_tasks: set[asyncio.Task[None]] = set()
inference_executor: ThreadPoolExecutor | None = None
in_flight: dict[str, asyncio.Future[float]] = {}
metrics: dict[str, int] = {
    "batch_requests": 0,
    "batch_count": 0,
    "batch_rows": 0,
    "batch_max_size": 0,
    "queue_overloads": 0,
    "in_flight_coalesced": 0,
}


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


@dataclass
class QueuedPrediction:
    key: str
    record: dict[str, Any]
    result: asyncio.Future[float]


class PicklePipelineAdapter:
    """Reference path for model-level comparisons; imports training stack lazily."""

    def __init__(self, path: Path) -> None:
        import joblib

        self.pipeline = joblib.load(path)
        self.feature_names = list(self.pipeline.feature_names_in_)
        preprocessor = self.pipeline.named_steps["preprocessor"]
        self.numeric_features = list(preprocessor.transformers_[0][2])
        self.categorical_features = list(preprocessor.transformers_[1][2])

    def validate_record(self, record: dict[str, Any]) -> None:
        for feature in self.numeric_features:
            value = record.get(feature)
            if value is None or value == "":
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{feature} must be numeric") from exc
            if not np.isfinite(numeric_value):
                raise ValueError(f"{feature} must be finite")
        for feature in self.categorical_features:
            value = record.get(feature)
            if value is not None and value != "" and not isinstance(value, str):
                raise ValueError(f"{feature} must be a string")

    def _frame(self, records: list[dict[str, Any]]):
        import pandas as pd

        rows = []
        for payload in records:
            row = {}
            for name in self.feature_names:
                value = payload.get(name, np.nan)
                row[name] = np.nan if value is None or value == "" else value
            rows.append(row)
        return pd.DataFrame(rows, columns=self.feature_names)

    def predict_records(self, records: list[dict[str, Any]]) -> np.ndarray:
        return np.asarray(self.pipeline.predict(self._frame(records)), dtype=np.float64)


def load_artifacts() -> None:
    global model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    if MODEL_PATH.suffix.lower() == ".zip":
        model = NativeModel(MODEL_PATH, nthread=1)
    else:
        model = PicklePipelineAdapter(MODEL_PATH)
    prediction_cache.clear()
    in_flight.clear()
    feature_schema.clear()
    if SCHEMA_PATH.exists():
        feature_schema.update(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))
    else:
        feature_schema["features"] = list(getattr(model, "feature_names", []))
    logger.info("Loaded model from %s", MODEL_PATH)


def build_feature_frame(payload: dict[str, Any]):
    """Backward-compatible helper for callers that need the DataFrame adapter."""
    import pandas as pd

    feature_names = feature_schema.get("features") or list(payload.keys())
    row: dict[str, Any] = {}
    for name in feature_names:
        value = payload.get(name, np.nan)
        if value is None or value == "":
            value = np.nan
        row[name] = value
    return pd.DataFrame([row], columns=feature_names)


def make_prediction_cache_key(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def canonicalize_record(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep only modeled fields and normalize values that predict identically."""
    names = getattr(model, "feature_names", None) or feature_schema.get("features", [])
    numeric = set(getattr(model, "numeric_features", []))
    record: dict[str, Any] = {}
    for name in names:
        value = payload.get(name)
        if value is None or value == "":
            record[name] = None
        elif name in numeric:
            record[name] = float(value)
        else:
            record[name] = value
    return record


def _resolve_failure(future: asyncio.Future[float], error: Exception) -> None:
    if not future.done():
        future.set_exception(error)


async def _run_records(records: list[dict[str, Any]], *, use_batch_executor: bool) -> np.ndarray:
    loop = asyncio.get_running_loop()
    if use_batch_executor and inference_executor is not None:
        return await loop.run_in_executor(inference_executor, model.predict_records, records)
    return await asyncio.to_thread(model.predict_records, records)


def _settle_item(item: QueuedPrediction, value: float) -> None:
    if CACHE_ENABLED:
        prediction_cache.set(item.key, value)
    if not item.result.done():
        item.result.set_result(value)
    in_flight.pop(item.key, None)


async def _finish_batch(batch: list[QueuedPrediction]) -> None:
    active = [item for item in batch if not item.result.done()]
    if not active:
        return
    try:
        predictions = await _run_records([item.record for item in active], use_batch_executor=True)
        if len(predictions) != len(active):
            raise RuntimeError("Model returned an unexpected prediction count")
        for item, value in zip(active, predictions):
            _settle_item(item, float(value))
    except Exception:
        # Isolate an unexpected row/model failure so valid requests in the batch still succeed.
        for item in active:
            try:
                result = await _run_records([item.record], use_batch_executor=True)
                _settle_item(item, float(result[0]))
            except Exception as exc:  # noqa: BLE001
                if not item.result.done():
                    item.result.set_exception(exc)
                in_flight.pop(item.key, None)


async def _batch_dispatcher() -> None:
    assert inference_queue is not None
    while True:
        first = await inference_queue.get()
        if first is None:
            inference_queue.task_done()
            return
        batch = [first]
        while len(batch) < BATCH_MAX_SIZE:
            try:
                batch.append(inference_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        deadline = asyncio.get_running_loop().time() + BATCH_MAX_DELAY_SECONDS
        while len(batch) < BATCH_MAX_SIZE:
            timeout = deadline - asyncio.get_running_loop().time()
            if timeout <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(inference_queue.get(), timeout))
            except TimeoutError:
                break
        metrics["batch_requests"] += len(batch)
        metrics["batch_count"] += 1
        metrics["batch_rows"] += len(batch)
        metrics["batch_max_size"] = max(metrics["batch_max_size"], len(batch))
        try:
            await _finish_batch(batch)
        finally:
            for _ in batch:
                inference_queue.task_done()


@app.on_event("startup")
async def startup() -> None:
    global inference_queue, batch_task, batch_tasks, direct_tasks, inference_executor
    load_artifacts()
    direct_tasks.clear()
    for key in metrics:
        metrics[key] = 0
    if DYNAMIC_BATCHING_ENABLED:
        inference_queue = asyncio.Queue(maxsize=BATCH_QUEUE_MAX_SIZE)
        inference_executor = ThreadPoolExecutor(
            max_workers=INFERENCE_WORKERS,
            thread_name_prefix="house-price-inference",
        )
        batch_tasks = [
            asyncio.create_task(_batch_dispatcher()) for _ in range(INFERENCE_WORKERS)
        ]
        batch_task = batch_tasks[0]


@app.on_event("shutdown")
async def shutdown() -> None:
    global batch_task, batch_tasks, direct_tasks, inference_queue, inference_executor
    if inference_queue is not None and batch_task is not None and not batch_task.done():
        # Let work already accepted finish before stopping the dispatcher.
        await inference_queue.join()
        for _ in batch_tasks:
            await inference_queue.put(None)
        await asyncio.gather(*batch_tasks)
    else:
        for task in batch_tasks:
            if not task.done():
                task.cancel()
        if batch_tasks:
            await asyncio.gather(*batch_tasks, return_exceptions=True)
    if inference_queue is not None:
        while not inference_queue.empty():
            item = inference_queue.get_nowait()
            if item is not None:
                _resolve_failure(item.result, RuntimeError("Server is shutting down"))
                in_flight.pop(item.key, None)
            inference_queue.task_done()
    if direct_tasks:
        await asyncio.gather(*direct_tasks, return_exceptions=True)
        direct_tasks.clear()
    for future in in_flight.values():
        _resolve_failure(future, RuntimeError("Server is shutting down"))
    in_flight.clear()
    if inference_executor is not None:
        await asyncio.to_thread(inference_executor.shutdown, wait=True, cancel_futures=False)
    batch_task = None
    batch_tasks = []
    inference_queue = None
    inference_executor = None


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=model is not None)


@app.get("/schema")
def schema() -> dict[str, Any]:
    if not feature_schema:
        raise HTTPException(status_code=503, detail="Feature schema not loaded")
    return feature_schema


async def _predict_value(payload: dict[str, Any]) -> float:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not payload:
        raise HTTPException(status_code=422, detail="No features provided. Send a non-empty 'features' object.")
    try:
        model.validate_record(payload)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    record = canonicalize_record(payload)
    cache_key = make_prediction_cache_key(record)
    if CACHE_ENABLED:
        cached = prediction_cache.get(cache_key)
        if cached is not None:
            return cached
        existing = in_flight.get(cache_key)
        if existing is not None:
            metrics["in_flight_coalesced"] += 1
            return await asyncio.shield(existing)

    loop = asyncio.get_running_loop()
    result: asyncio.Future[float] = loop.create_future()
    result.add_done_callback(
        lambda future: future.exception() if not future.cancelled() else None
    )
    if CACHE_ENABLED:
        in_flight[cache_key] = result

    if DYNAMIC_BATCHING_ENABLED:
        assert inference_queue is not None
        try:
            inference_queue.put_nowait(QueuedPrediction(cache_key, record, result))
        except asyncio.QueueFull as exc:
            metrics["queue_overloads"] += 1
            in_flight.pop(cache_key, None)
            raise HTTPException(status_code=503, detail="Prediction queue is full") from exc
        return await asyncio.shield(result)

    async def predict_and_settle() -> None:
        try:
            predictions = await _run_records([record], use_batch_executor=False)
            _settle_item(QueuedPrediction(cache_key, record, result), float(predictions[0]))
        except Exception as exc:  # noqa: BLE001
            _resolve_failure(result, exc)
            in_flight.pop(cache_key, None)

    task = asyncio.create_task(predict_and_settle())
    direct_tasks.add(task)
    task.add_done_callback(direct_tasks.discard)
    try:
        return await asyncio.shield(result)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    try:
        value = await _predict_value(request.features)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc
    return PredictResponse(sale_price=round(value, 2))


@app.get("/metrics")
def get_metrics() -> dict[str, Any]:
    stats = prediction_cache.stats()
    return {
        "model_path": str(MODEL_PATH),
        "cache_enabled": CACHE_ENABLED,
        "cache_hits": stats.hits,
        "cache_misses": stats.misses,
        "cache_size": stats.size,
        "dynamic_batching_enabled": DYNAMIC_BATCHING_ENABLED,
        "batch_requests": metrics["batch_requests"],
        "batch_count": metrics["batch_count"],
        "batch_rows": metrics["batch_rows"],
        "batch_max_size": metrics["batch_max_size"],
        "queue_size": inference_queue.qsize() if inference_queue is not None else 0,
        "queue_overloads": metrics["queue_overloads"],
        "in_flight_coalesced": metrics["in_flight_coalesced"],
    }
