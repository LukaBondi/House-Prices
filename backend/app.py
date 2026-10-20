from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import sqlite3
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from backend.artifacts import (
    ArtifactError,
    ServingSnapshot,
    atomic_write_pointer,
    load_bootstrap_snapshot,
    load_version_snapshot,
    make_snapshot,
    read_pointer,
    safe_version,
    sha256_file,
)
from backend.cache import TTLCache
from backend.event_store import (
    StoreUnavailable,
    canonical_json,
    feature_fingerprint,
    initialize_database,
    open_database,
)
from backend.monitoring import ALLOWED_WINDOWS_MINUTES, MonitoringError, build_monitoring_summary
from backend.request_logging import (
    RequestEventLogger,
    RequestLoggingMiddleware,
    current_request_context,
)

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
REQUEST_LOGGING_ENABLED = os.getenv("REQUEST_LOGGING_ENABLED", "1").lower() not in {
    "0", "false", "no"
}

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
    expose_headers=["X-Request-ID"],
)
app.add_middleware(RequestLoggingMiddleware, app_state=app.state)

model: Any = None
feature_schema: dict[str, Any] = {}
serving_snapshot: ServingSnapshot | None = None
active_pointer: dict[str, Any] | None = None
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
activation_mutex = asyncio.Lock()
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
    observation: dict[str, Any] | None = None
    run_id: str | None = Field(default=None, max_length=128)


class PredictResponse(BaseModel):
    sale_price: float
    currency: str = "USD"
    request_id: str
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
    ready: bool = False
    logging_enabled: bool = True
    logging_healthy: bool = False


@dataclass
class QueuedPrediction:
    key: str
    record: dict[str, Any]
    result: asyncio.Future[float]
    snapshot: ServingSnapshot | None = None


class ActivationRequest(BaseModel):
    candidate_version: str = Field(min_length=1, max_length=64)
    expected_parent: str = Field(min_length=1, max_length=64)
    activation_id: str = Field(min_length=1, max_length=128)
    trainer_run_id: str | None = Field(default=None, max_length=128)


class RollbackRequest(BaseModel):
    target_version: str = Field(min_length=1, max_length=64)
    expected_parent: str = Field(min_length=1, max_length=64)
    activation_id: str = Field(min_length=1, max_length=128)


class ActivationRejected(RuntimeError):
    """A compare-and-swap or trainer lease fence rejected publication."""


class ActivationPublishedUncertain(RuntimeError):
    """Pointer and memory switched, but the activation DB commit was uncertain."""


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


def _default_data_dir() -> Path:
    explicit = os.getenv("DATA_DIR")
    if explicit:
        return Path(explicit).expanduser()
    local_base = os.getenv("LOCALAPPDATA") or os.getenv("XDG_STATE_HOME") or tempfile.gettempdir()
    return Path(local_base) / "sys-304" / "house-prices"


def _runtime_paths() -> dict[str, Path]:
    data_dir = _default_data_dir()
    return {
        "data_dir": data_dir,
        "database": Path(os.getenv("DB_PATH", str(data_dir / "requests.sqlite3"))).expanduser(),
        "registry": Path(os.getenv("MODEL_REGISTRY_PATH", str(data_dir / "artifacts"))).expanduser(),
        "state": Path(os.getenv("MODEL_STATE_PATH", str(data_dir / "state"))).expanduser(),
        "drift_reference": Path(
            os.getenv("DRIFT_REFERENCE_PATH", str(ROOT / "models" / "drift_reference.json"))
        ).expanduser(),
    }


def _publish_snapshot(snapshot: ServingSnapshot) -> None:
    """Swap the request default and legacy compatibility references without an await."""
    global model, serving_snapshot
    model = snapshot.model
    feature_schema.clear()
    feature_schema.update(snapshot.schema)
    serving_snapshot = snapshot


def capture_serving_snapshot() -> ServingSnapshot:
    """Capture the current generation once, with a legacy test/helper fallback."""
    current = serving_snapshot
    if current is not None and model is current.model:
        return current
    if model is None:
        raise RuntimeError("Model not loaded")
    schema = dict(feature_schema)
    legacy_id = hashlib.sha256(
        f"{type(model).__name__}:{id(model)}".encode("utf-8")
    ).hexdigest()[:12]
    digest = hashlib.sha256(f"legacy:{legacy_id}".encode("utf-8")).hexdigest()
    return make_snapshot(
        model,
        schema,
        version=f"legacy-{legacy_id}",
        artifact_hash=digest,
        preprocessor_hash=digest,
    )


def load_artifacts() -> None:
    """Load the frozen default model for legacy callers and process startup."""
    global active_pointer
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    schema = (
        json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        if SCHEMA_PATH.exists()
        else {"features": []}
    )
    if MODEL_PATH.suffix.lower() == ".zip":
        snapshot = load_bootstrap_snapshot(MODEL_PATH, SCHEMA_PATH)
    else:
        legacy_model = PicklePipelineAdapter(MODEL_PATH)
        digest = sha256_file(MODEL_PATH)
        snapshot = make_snapshot(
            legacy_model,
            schema,
            version=f"legacy-{digest[:16]}",
            artifact_hash=digest,
            preprocessor_hash=digest,
        )
    _publish_snapshot(snapshot)
    active_pointer = None
    prediction_cache.clear()
    in_flight.clear()
    logger.info("Loaded model version %s from %s", snapshot.version, MODEL_PATH)


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


def make_prediction_cache_key(payload: dict[str, Any], version: str | None = None) -> str:
    value = {"version": version, "features": payload} if version is not None else payload
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def canonicalize_record(
    payload: dict[str, Any], snapshot: ServingSnapshot | None = None
) -> dict[str, Any]:
    """Keep only modeled fields and normalize values that predict identically."""
    serving = snapshot or capture_serving_snapshot()
    from backend.event_store import canonical_record

    return canonical_record(
        payload,
        feature_names=serving.feature_names,
        numeric_features=serving.numeric_features,
    )


def _resolve_failure(future: asyncio.Future[float], error: Exception) -> None:
    if not future.done():
        future.set_exception(error)


async def _run_records(
    records: list[dict[str, Any]],
    *,
    use_batch_executor: bool,
    snapshot: ServingSnapshot | None = None,
) -> np.ndarray:
    loop = asyncio.get_running_loop()
    target_model = snapshot.model if snapshot is not None else model
    if target_model is None:
        raise RuntimeError("Model not loaded")
    if use_batch_executor and inference_executor is not None:
        return await loop.run_in_executor(
            inference_executor, target_model.predict_records, records
        )
    return await asyncio.to_thread(target_model.predict_records, records)


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
    generations: dict[tuple[str, int], list[QueuedPrediction]] = {}
    for item in active:
        generation = (
            item.snapshot.version,
            id(item.snapshot.model),
        ) if item.snapshot is not None else ("legacy", id(model))
        generations.setdefault(generation, []).append(item)

    for generation_items in generations.values():
        snapshot = generation_items[0].snapshot
        try:
            predictions = await _run_records(
                [item.record for item in generation_items],
                use_batch_executor=True,
                snapshot=snapshot,
            )
            if len(predictions) != len(generation_items):
                raise RuntimeError("Model returned an unexpected prediction count")
            for item, value in zip(generation_items, predictions):
                _settle_item(item, float(value))
        except Exception:
            # Isolate an unexpected row/model failure while retaining each snapshot.
            for item in generation_items:
                try:
                    result = await _run_records(
                        [item.record],
                        use_batch_executor=True,
                        snapshot=item.snapshot,
                    )
                    _settle_item(item, float(result[0]))
                except Exception as exc:  # noqa: BLE001
                    if not item.result.done():
                        item.result.set_exception(exc)
                    in_flight.pop(item.key, None)


def _item_generation(item: QueuedPrediction) -> tuple[str, int]:
    if item.snapshot is not None:
        return (item.snapshot.version, id(item.snapshot.model))
    return ("legacy", id(model))


async def _batch_dispatcher() -> None:
    assert inference_queue is not None
    deferred: list[QueuedPrediction | None] = []
    while True:
        first = deferred.pop(0) if deferred else await inference_queue.get()
        if first is None:
            inference_queue.task_done()
            return
        batch = [first]
        generation = _item_generation(first)

        def collect_available() -> bool:
            try:
                candidate = inference_queue.get_nowait()
            except asyncio.QueueEmpty:
                return False
            if candidate is None:
                deferred.append(None)
                return False
            if _item_generation(candidate) == generation:
                batch.append(candidate)
            else:
                deferred.append(candidate)
            return True

        while len(batch) < BATCH_MAX_SIZE and collect_available():
            pass
        deadline = asyncio.get_running_loop().time() + BATCH_MAX_DELAY_SECONDS
        while len(batch) < BATCH_MAX_SIZE:
            timeout = deadline - asyncio.get_running_loop().time()
            if timeout <= 0:
                break
            try:
                candidate = await asyncio.wait_for(inference_queue.get(), timeout)
            except TimeoutError:
                break
            if candidate is None:
                deferred.append(None)
                break
            if _item_generation(candidate) == generation:
                batch.append(candidate)
            else:
                deferred.append(candidate)
        metrics["batch_requests"] += len(batch)
        metrics["batch_count"] += 1
        metrics["batch_rows"] += len(batch)
        metrics["batch_max_size"] = max(metrics["batch_max_size"], len(batch))
        try:
            await _finish_batch(batch)
        finally:
            for _ in batch:
                inference_queue.task_done()


def _reconcile_activation_intents(database_path: Path, pointer: dict[str, Any] | None) -> None:
    connection = open_database(database_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        intents = connection.execute(
            "SELECT activation_id, candidate_version, details_json "
            "FROM activations WHERE status='intent'"
        ).fetchall()
        now = time.time()
        pointer_activation = pointer.get("activation_id") if pointer else None
        pointer_version = pointer.get("version") if pointer else None
        for row in intents:
            committed = (
                row["activation_id"] == pointer_activation
                and row["candidate_version"] == pointer_version
            )
            status = "committed" if committed else "failed"
            try:
                details = json.loads(row["details_json"] or "{}")
            except json.JSONDecodeError:
                details = {}
            if not isinstance(details, dict):
                details = {}
            details.update(
                {
                    "reconciled_at": now,
                    "reason": "pointer_matches_intent"
                    if committed
                    else "pointer_does_not_match_intent",
                    "active_version": pointer_version,
                }
            )
            connection.execute(
                "UPDATE activations SET status=?, completed_at=?, details_json=? "
                "WHERE activation_id=? AND status='intent'",
                (status, now, canonical_json(details), row["activation_id"]),
            )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


@app.on_event("startup")
async def startup() -> None:
    global inference_queue, batch_task, batch_tasks, direct_tasks, inference_executor
    global active_pointer
    paths = _runtime_paths()
    event_logger = RequestEventLogger(
        paths["database"],
        enabled=REQUEST_LOGGING_ENABLED,
    )
    app.state.runtime_paths = paths
    app.state.request_event_logger = event_logger
    try:
        if REQUEST_LOGGING_ENABLED:
            await event_logger.start()
        else:
            await asyncio.to_thread(initialize_database, paths["database"])
        pointer_path = paths["state"] / "current.json"
        pointer = await asyncio.to_thread(read_pointer, pointer_path)
        if pointer is None:
            if MODEL_PATH.suffix.lower() == ".zip":
                snapshot = await asyncio.to_thread(
                    load_bootstrap_snapshot, MODEL_PATH, SCHEMA_PATH
                )
            else:
                await asyncio.to_thread(load_artifacts)
                snapshot = serving_snapshot
                if snapshot is None:
                    raise ArtifactError("could not load configured model")
            pointer = {
                "version": snapshot.version,
                "activation_id": None,
                "previous_version": None,
            }
            try:
                atomic_write_pointer(pointer_path, pointer)
                app.state.pointer_durability_error = None
            except OSError as exc:
                actual = None
                try:
                    actual = read_pointer(pointer_path)
                except ArtifactError:
                    pass
                if actual != pointer:
                    raise
                app.state.pointer_durability_error = str(exc)[:300]
        else:
            snapshot = await asyncio.to_thread(
                load_version_snapshot,
                pointer["version"],
                registry_path=paths["registry"],
                bootstrap_path=MODEL_PATH,
                schema_path=SCHEMA_PATH,
                require_accepted=True,
            )
            app.state.pointer_durability_error = None
        _publish_snapshot(snapshot)
        prediction_cache.clear()
        in_flight.clear()
        active_pointer = pointer
        app.state.serving_pointer_path = pointer_path
        await asyncio.to_thread(_reconcile_activation_intents, paths["database"], pointer)
    except BaseException:
        await event_logger.stop()
        app.state.request_event_logger = None
        raise

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
    event_logger: RequestEventLogger | None = getattr(
        app.state, "request_event_logger", None
    )
    if event_logger is not None:
        await event_logger.stop()
        app.state.request_event_logger = None
    batch_task = None
    batch_tasks = []
    inference_queue = None
    inference_executor = None


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
def health(response: Response) -> HealthResponse:
    snapshot = serving_snapshot
    event_logger: RequestEventLogger | None = getattr(
        app.state, "request_event_logger", None
    )
    logging_enabled = event_logger.enabled if event_logger is not None else REQUEST_LOGGING_ENABLED
    logging_healthy = event_logger.healthy if event_logger is not None else False
    pointer_healthy = getattr(app.state, "pointer_durability_error", None) is None
    ready = snapshot is not None and (logging_healthy or not logging_enabled) and pointer_healthy
    response.status_code = 200 if ready else 503
    return HealthResponse(
        status="ok" if ready else "degraded",
        model_loaded=snapshot is not None,
        model_version=snapshot.version if snapshot is not None else None,
        ready=ready,
        logging_enabled=logging_enabled,
        logging_healthy=logging_healthy,
    )


@app.get("/schema")
def schema() -> dict[str, Any]:
    snapshot = serving_snapshot
    if snapshot is None or not snapshot.feature_names:
        raise HTTPException(status_code=503, detail="Feature schema not loaded")
    return snapshot.schema


async def _predict_value(
    payload: dict[str, Any], snapshot: ServingSnapshot | None = None
) -> float:
    snapshot = snapshot or capture_serving_snapshot()
    context = current_request_context.get()
    if context is not None:
        context.model_version = snapshot.version
    target_model = snapshot.model
    if target_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not payload:
        raise HTTPException(status_code=422, detail="No features provided. Send a non-empty 'features' object.")
    try:
        target_model.validate_record(payload)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    try:
        record = canonicalize_record(payload, snapshot)
        fingerprint = feature_fingerprint(
            record,
            feature_names=snapshot.feature_names,
            numeric_features=snapshot.numeric_features,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc
    if context is not None:
        context.features = record
        context.feature_fingerprint = fingerprint
    cache_key = make_prediction_cache_key(record, snapshot.version)
    if CACHE_ENABLED:
        cached = prediction_cache.get(cache_key)
        if cached is not None:
            if context is not None:
                context.cache_outcome = "hit"
                context.prediction = float(cached)
            return cached
        existing = in_flight.get(cache_key)
        if existing is not None:
            metrics["in_flight_coalesced"] += 1
            if context is not None:
                context.cache_outcome = "coalesced"
            value = await asyncio.shield(existing)
            if context is not None:
                context.prediction = float(value)
            return value
        if context is not None:
            context.cache_outcome = "miss"
    elif context is not None:
        context.cache_outcome = "disabled"

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
            inference_queue.put_nowait(
                QueuedPrediction(cache_key, record, result, snapshot)
            )
        except asyncio.QueueFull as exc:
            metrics["queue_overloads"] += 1
            in_flight.pop(cache_key, None)
            raise HTTPException(status_code=503, detail="Prediction queue is full") from exc
        value = await asyncio.shield(result)
        if context is not None:
            context.prediction = float(value)
        return value

    async def predict_and_settle() -> None:
        try:
            predictions = await _run_records(
                [record], use_batch_executor=False, snapshot=snapshot
            )
            _settle_item(
                QueuedPrediction(cache_key, record, result, snapshot),
                float(predictions[0]),
            )
        except Exception as exc:  # noqa: BLE001
            _resolve_failure(result, exc)
            in_flight.pop(cache_key, None)

    task = asyncio.create_task(predict_and_settle())
    direct_tasks.add(task)
    task.add_done_callback(direct_tasks.discard)
    try:
        value = await asyncio.shield(result)
        if context is not None:
            context.prediction = float(value)
        return value
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction inference failed")
        raise HTTPException(status_code=500, detail="Prediction inference failed") from exc


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    try:
        snapshot = capture_serving_snapshot()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="Model not loaded") from exc
    context = current_request_context.get()
    if context is not None:
        context.model_version = snapshot.version
        context.run_id = request.run_id
        if isinstance(request.observation, dict):
            dataset_hash = request.observation.get("dataset_sha256")
            source_id = request.observation.get("source_id")
            if isinstance(dataset_hash, str) and len(dataset_hash) == 64:
                try:
                    int(dataset_hash, 16)
                except ValueError:
                    pass
                else:
                    context.dataset_sha256 = dataset_hash.lower()
                    if isinstance(source_id, int) and not isinstance(source_id, bool):
                        context.source_id = source_id
    try:
        value = await _predict_value(request.features, snapshot)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
    return PredictResponse(
        sale_price=round(value, 2),
        request_id=context.request_id if context is not None else "",
        model_version=snapshot.version,
    )


def _database_execute_sync(
    database_path: Path, sql: str, parameters: tuple[Any, ...]
) -> int:
    connection = open_database(database_path)
    try:
        cursor = connection.execute(sql, parameters)
        connection.commit()
        return cursor.rowcount
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


async def _database_execute(sql: str, parameters: tuple[Any, ...]) -> int:
    paths = getattr(app.state, "runtime_paths", _runtime_paths())
    event_logger: RequestEventLogger | None = getattr(
        app.state, "request_event_logger", None
    )
    if event_logger is not None and event_logger.enabled:
        started = time.perf_counter()
        try:
            changed = await event_logger.writer.execute(sql, parameters)
        except BaseException as exc:
            if not isinstance(exc, asyncio.CancelledError):
                event_logger._failure(exc)
            raise
        event_logger._success(
            phase="activation",
            duration_ms=(time.perf_counter() - started) * 1000,
        )
        return changed
    return await asyncio.to_thread(_database_execute_sync, paths["database"], sql, parameters)


def _query_activation(database_path: Path, activation_id: str) -> dict[str, Any] | None:
    connection = open_database(database_path)
    try:
        row = connection.execute(
            "SELECT activation_id, expected_parent, candidate_version, status, created_at, "
            "completed_at, details_json FROM activations WHERE activation_id=?",
            (activation_id,),
        ).fetchone()
        return dict(row) if row is not None else None
    finally:
        connection.close()


def _admin_authorize(authorization: str | None) -> None:
    configured = os.getenv("ADMIN_TOKEN", "")
    if not configured:
        raise HTTPException(status_code=503, detail="Admin activation is not configured")
    if not authorization:
        raise HTTPException(status_code=401, detail="Bearer token required")
    scheme, separator, supplied = authorization.partition(" ")
    if not separator or scheme.lower() != "bearer" or not hmac.compare_digest(
        supplied.encode("utf-8"), configured.encode("utf-8")
    ):
        raise HTTPException(status_code=403, detail="Invalid bearer token")


def _activation_response(
    activation_id: str,
    status: str,
    active_version: str | None,
    *,
    idempotent: bool = False,
) -> dict[str, Any]:
    return {
        "activation_id": activation_id,
        "status": status,
        "active_version": active_version,
        "idempotent": idempotent,
    }


async def _set_activation_terminal(
    activation_id: str,
    status: str,
    details: dict[str, Any],
) -> None:
    paths = getattr(app.state, "runtime_paths", _runtime_paths())
    started = time.perf_counter()
    changed = await asyncio.to_thread(
        _set_activation_terminal_sync,
        paths["database"],
        activation_id,
        status,
        details,
    )
    event_logger: RequestEventLogger | None = getattr(
        app.state, "request_event_logger", None
    )
    if event_logger is not None and event_logger.enabled:
        event_logger._success(
            phase="activation", duration_ms=(time.perf_counter() - started) * 1000
        )
    if changed != 1:
        raise StoreUnavailable("activation intent could not be finalized")


def _set_activation_terminal_sync(
    database_path: Path,
    activation_id: str,
    status: str,
    updates: dict[str, Any],
) -> int:
    connection = open_database(database_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT status, details_json FROM activations WHERE activation_id=?",
            (activation_id,),
        ).fetchone()
        if row is None or row["status"] != "intent":
            connection.rollback()
            return 0
        try:
            details = json.loads(row["details_json"] or "{}")
        except json.JSONDecodeError:
            details = {}
        if not isinstance(details, dict):
            details = {}
        details.update(updates)
        cursor = connection.execute(
            "UPDATE activations SET status=?, completed_at=?, details_json=? "
            "WHERE activation_id=? AND status='intent'",
            (status, time.time(), canonical_json(details), activation_id),
        )
        connection.commit()
        return cursor.rowcount
    except sqlite3.Error as exc:
        connection.rollback()
        raise StoreUnavailable(f"activation store write failed: {exc}") from exc
    finally:
        connection.close()


async def _activation_rejected(
    activation_id: str,
    expected_parent: str,
    candidate_version: str,
    reason: str,
    *,
    rollback: bool = False,
    trainer_run_id: str | None = None,
) -> None:
    now = time.time()
    await _database_execute(
        "INSERT OR IGNORE INTO activations "
        "(activation_id, expected_parent, candidate_version, status, created_at, "
        "completed_at, details_json) VALUES (?, ?, ?, 'rejected', ?, ?, ?)",
        (
            activation_id,
            expected_parent,
            candidate_version,
            now,
            now,
            canonical_json(
                {
                    "reason": reason,
                    "rollback": rollback,
                    "trainer_run_id": trainer_run_id,
                }
            ),
        ),
    )


def _commit_fenced_activation(
    *,
    database_path: Path,
    pointer_path: Path,
    snapshot: ServingSnapshot,
    expected_parent: str,
    activation_id: str,
    trainer_run_id: str,
    rollback: bool,
) -> str | None:
    """Publish a trainer candidate while its live DB lease cannot be replaced."""
    global active_pointer
    current = serving_snapshot
    if current is None:
        raise ActivationRejected("serving model is not loaded")
    pointer = {
        "version": snapshot.version,
        "activation_id": activation_id,
        "previous_version": current.version,
    }
    connection = open_database(database_path)
    pointer_before: dict[str, Any] | None = None
    try:
        connection.execute("BEGIN IMMEDIATE")
        activation = connection.execute(
            "SELECT expected_parent, candidate_version, status, details_json FROM activations "
            "WHERE activation_id=?",
            (activation_id,),
        ).fetchone()
        if (
            activation is None
            or activation["status"] != "intent"
            or activation["expected_parent"] != expected_parent
            or activation["candidate_version"] != snapshot.version
        ):
            raise ActivationRejected("activation intent is missing or changed")
        try:
            activation_details = json.loads(activation["details_json"] or "{}")
        except json.JSONDecodeError as exc:
            raise ActivationRejected("activation intent details are invalid") from exc
        if not isinstance(activation_details, dict) or (
            activation_details.get("rollback") is not rollback
            or activation_details.get("trainer_run_id") != trainer_run_id
        ):
            raise ActivationRejected("activation intent owner or operation identity changed")
        manifest = snapshot.manifest or {}
        if manifest.get("parent_version") != expected_parent:
            raise ActivationRejected("candidate manifest parent differs from expected parent")
        if current.version != expected_parent:
            raise ActivationRejected("active version changed before fenced publication")
        try:
            pointer_now = read_pointer(pointer_path)
        except ArtifactError as exc:
            app.state.pointer_durability_error = "serving pointer became invalid during activation"
            raise ActivationRejected("serving pointer became invalid") from exc
        if pointer_now is None or pointer_now.get("version") != expected_parent:
            app.state.pointer_durability_error = "serving pointer differs from in-memory version"
            raise ActivationRejected("serving pointer does not match the expected parent")
        pointer_before = pointer_now

        run = connection.execute(
            "SELECT run_id, parent_version, snapshot_hash, status, details_json "
            "FROM runs WHERE run_id=?",
            (trainer_run_id,),
        ).fetchone()
        lease = connection.execute(
            "SELECT lease_name, owner_id, expires_at, snapshot_hash FROM trainer_leases "
            "WHERE owner_id=? AND lease_name='retrain'",
            (trainer_run_id,),
        ).fetchone()
        now = time.time()
        training = (snapshot.manifest or {}).get("training", {})
        if (
            run is None
            or run["status"] != "running"
            or run["parent_version"] != expected_parent
            or not run["snapshot_hash"]
            or lease is None
            or float(lease["expires_at"]) <= now
            or lease["snapshot_hash"] != run["snapshot_hash"]
            or training.get("run_id") != trainer_run_id
            or training.get("snapshot_hash") != run["snapshot_hash"]
        ):
            raise ActivationRejected("trainer run has no matching live lease and snapshot")
        try:
            run_details = json.loads(run["details_json"] or "{}")
        except json.JSONDecodeError as exc:
            raise ActivationRejected("trainer run details are invalid") from exc
        if (
            run_details.get("candidate_version") != snapshot.version
            or run_details.get("activation_id") != activation_id
        ):
            raise ActivationRejected("trainer run is not assigned this candidate activation")

        pointer_sync_error: str | None = None
        def require_live_lease_at_publication() -> None:
            if float(lease["expires_at"]) <= time.time():
                raise ActivationRejected("trainer lease expired before pointer publication")

        try:
            atomic_write_pointer(
                pointer_path,
                pointer,
                before_replace=require_live_lease_at_publication,
            )
        except ActivationRejected:
            raise
        except Exception as exc:
            actual = None
            try:
                actual = read_pointer(pointer_path)
            except ArtifactError:
                pass
            if actual != pointer:
                raise StoreUnavailable(f"could not publish serving pointer: {exc}") from exc
            pointer_sync_error = str(exc)[:300]

        # No await between pointer publication and memory switch while the run lease is fenced.
        _publish_snapshot(snapshot)
        active_pointer = pointer
        app.state.pointer_durability_error = pointer_sync_error
        details = {
            **activation_details,
            "active_version": snapshot.version,
            "previous_version": current.version,
            "rollback": rollback,
            "trainer_run_id": trainer_run_id,
            "lease_name": lease["lease_name"],
            "lease_expires_at": lease["expires_at"],
            "pointer_durability_error": pointer_sync_error,
        }
        changed = connection.execute(
            "UPDATE activations SET status='committed', completed_at=?, details_json=? "
            "WHERE activation_id=? AND status='intent'",
            (time.time(), canonical_json(details), activation_id),
        ).rowcount
        if changed != 1:
            raise RuntimeError("activation intent was not updated")
        connection.commit()
        return pointer_sync_error
    except BaseException as exc:
        connection.rollback()
        try:
            actual = read_pointer(pointer_path)
        except ArtifactError:
            actual = None
        if actual == pointer:
            # A failed DB commit after replace is recoverable from the durable pointer.
            _publish_snapshot(snapshot)
            active_pointer = pointer
            app.state.pointer_durability_error = (
                f"activation audit commit is uncertain: {str(exc)[:200]}"
            )
            raise ActivationPublishedUncertain(str(exc)) from exc
        if pointer_before is not None and actual != pointer_before:
            app.state.pointer_durability_error = (
                f"serving pointer publication state is uncertain: {str(exc)[:200]}"
            )
        raise
    finally:
        connection.close()


async def _activate_version(
    *,
    candidate_version: str,
    expected_parent: str,
    activation_id: str,
    rollback: bool,
    trainer_run_id: str | None = None,
) -> dict[str, Any]:
    global active_pointer
    try:
        candidate_version = safe_version(candidate_version)
        expected_parent = safe_version(expected_parent)
    except ArtifactError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    paths = app.state.runtime_paths

    prior = await asyncio.to_thread(
        _query_activation, paths["database"], activation_id
    )
    if prior is not None:
        try:
            prior_details = json.loads(prior["details_json"] or "{}")
        except json.JSONDecodeError:
            prior_details = {}
        if not isinstance(prior_details, dict):
            prior_details = {}
        if (
            prior["expected_parent"] != expected_parent
            or prior["candidate_version"] != candidate_version
            or prior_details.get("rollback") is not rollback
            or prior_details.get("trainer_run_id") != trainer_run_id
        ):
            raise HTTPException(
                status_code=409,
                detail="activation_id was already used for a different operation",
            )
        if prior["status"] == "committed":
            return _activation_response(
                activation_id,
                "committed",
                prior_details.get("active_version", candidate_version),
                idempotent=True,
            )
        if prior["status"] in {"rejected", "failed"}:
            raise HTTPException(
                status_code=409,
                detail=f"activation is already {prior['status']}",
            )

    try:
        staged = await asyncio.to_thread(
            load_version_snapshot,
            candidate_version,
            registry_path=paths["registry"],
            bootstrap_path=MODEL_PATH,
            schema_path=SCHEMA_PATH,
            require_accepted=not rollback,
        )
        if not rollback and staged.manifest is None:
            raise ArtifactError("normal activation requires an accepted registry candidate")
        manifest_training = (staged.manifest or {}).get("training", {})
        manifest_run_id = (
            manifest_training.get("run_id")
            if isinstance(manifest_training, dict)
            else None
        )
        if not rollback and manifest_run_id is not None:
            if trainer_run_id is None or trainer_run_id != manifest_run_id:
                raise HTTPException(
                    status_code=409,
                    detail="trainer_run_id must match the candidate manifest",
                )
        elif not rollback and trainer_run_id is not None:
            raise HTTPException(
                status_code=409,
                detail="candidate manifest is not authored by the supplied trainer run",
            )
        if (
            not rollback
            and staged.manifest is not None
            and staged.manifest.get("parent_version") != expected_parent
        ):
            raise HTTPException(
                status_code=409,
                detail="candidate parent_version does not match expected_parent",
            )
    except ArtifactError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Could not stage model version %s", candidate_version)
        raise HTTPException(status_code=422, detail="Candidate failed staged validation") from exc

    async with activation_mutex:
        existing = await asyncio.to_thread(
            _query_activation, paths["database"], activation_id
        )
        existing_details: dict[str, Any] = {}
        if existing is not None:
            if (
                existing["expected_parent"] != expected_parent
                or existing["candidate_version"] != candidate_version
            ):
                raise HTTPException(
                    status_code=409,
                    detail="activation_id was already used with different values",
                )
            if existing["details_json"]:
                try:
                    existing_details = json.loads(existing["details_json"])
                except json.JSONDecodeError:
                    existing_details = {}
            if not isinstance(existing_details, dict):
                existing_details = {}
            if (
                existing_details.get("rollback") is not rollback
                or existing_details.get("trainer_run_id") != trainer_run_id
            ):
                raise HTTPException(
                    status_code=409,
                    detail="activation_id was already used for a different operation",
                )
            if existing["status"] == "committed":
                return _activation_response(
                    activation_id,
                    "committed",
                    existing_details.get("active_version", candidate_version),
                    idempotent=True,
                )
            if existing["status"] in {"rejected", "failed"}:
                raise HTTPException(
                    status_code=409,
                    detail=f"activation is already {existing['status']}",
                )
            if existing["status"] == "intent":
                pointer = await asyncio.to_thread(
                    read_pointer, app.state.serving_pointer_path
                )
                if (
                    pointer is not None
                    and pointer.get("activation_id") == activation_id
                    and pointer.get("version") == candidate_version
                    and pointer.get("previous_version") == expected_parent
                ):
                    await _set_activation_terminal(
                        activation_id,
                        "committed",
                        {
                            "active_version": candidate_version,
                            "previous_version": pointer.get("previous_version"),
                            "reconciled": True,
                            "rollback": rollback,
                            "trainer_run_id": trainer_run_id,
                        },
                    )
                    if serving_snapshot is None or serving_snapshot.version != candidate_version:
                        _publish_snapshot(staged)
                    active_pointer = pointer
                    return _activation_response(
                        activation_id, "committed", candidate_version, idempotent=True
                    )

        current = capture_serving_snapshot()
        if rollback:
            pointer_now = active_pointer or {}
            bootstrap = await asyncio.to_thread(
                load_bootstrap_snapshot, MODEL_PATH, SCHEMA_PATH
            )
            allowed_targets = {
                value
                for value in (pointer_now.get("previous_version"), bootstrap.version)
                if value is not None
            }
            if candidate_version not in allowed_targets:
                if existing is None:
                    await _activation_rejected(
                        activation_id,
                        expected_parent,
                        candidate_version,
                        "rollback target is not the previous or bootstrap version",
                        rollback=True,
                    )
                raise HTTPException(
                    status_code=409,
                    detail="rollback target must be the previous or bootstrap version",
                )

        if current.version != expected_parent:
            if existing is None:
                await _activation_rejected(
                    activation_id,
                    expected_parent,
                    candidate_version,
                    "expected parent is stale",
                    rollback=rollback,
                    trainer_run_id=trainer_run_id,
                )
            else:
                await _set_activation_terminal(
                    activation_id,
                    "failed",
                    {"reason": "expected parent is stale", "active_version": current.version},
                )
            raise HTTPException(status_code=409, detail="expected_parent does not match active version")

        if (
            staged.preprocessing_sha256 != current.preprocessing_sha256
            or staged.schema_sha256 != current.schema_sha256
            or staged.feature_names != current.feature_names
        ):
            if existing is None:
                await _activation_rejected(
                    activation_id,
                    expected_parent,
                    candidate_version,
                    "candidate schema or preprocessing differs from active model",
                    rollback=rollback,
                    trainer_run_id=trainer_run_id,
                )
            else:
                await _set_activation_terminal(
                    activation_id,
                    "failed",
                    {"reason": "schema or preprocessing mismatch"},
                )
            raise HTTPException(
                status_code=409,
                detail="candidate schema or preprocessing differs from active model",
            )

        if existing is None:
            manifest = staged.manifest or {}
            intent_details = {
                "rollback": rollback,
                "trainer_run_id": trainer_run_id,
                "artifact_sha256": staged.artifact_sha256,
                "preprocessing_sha256": staged.preprocessing_sha256,
                "schema_sha256": staged.schema_sha256,
                "gate": manifest.get("gate"),
            }
            await _database_execute(
                "INSERT INTO activations "
                "(activation_id, expected_parent, candidate_version, status, created_at, details_json) "
                "VALUES (?, ?, ?, 'intent', ?, ?)",
                (
                    activation_id,
                    expected_parent,
                    candidate_version,
                    time.time(),
                    canonical_json(intent_details),
                ),
            )

        if trainer_run_id is not None:
            try:
                pointer_sync_error = _commit_fenced_activation(
                    database_path=paths["database"],
                    pointer_path=app.state.serving_pointer_path,
                    snapshot=staged,
                    expected_parent=expected_parent,
                    activation_id=activation_id,
                    trainer_run_id=trainer_run_id,
                    rollback=rollback,
                )
            except ActivationRejected as exc:
                await _set_activation_terminal(
                    activation_id,
                    "rejected",
                    {"reason": str(exc), "active_version": capture_serving_snapshot().version},
                )
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except ActivationPublishedUncertain as exc:
                raise HTTPException(
                    status_code=503,
                    detail="activation was published; query its activation_id for status",
                ) from exc
            except StoreUnavailable as exc:
                try:
                    await _set_activation_terminal(
                        activation_id,
                        "failed",
                        {"reason": str(exc), "active_version": capture_serving_snapshot().version},
                    )
                except StoreUnavailable:
                    pass
                raise HTTPException(status_code=503, detail="activation store is unavailable") from exc
            except Exception as exc:  # noqa: BLE001
                logger.exception("Fenced activation %s failed", activation_id)
                try:
                    await _set_activation_terminal(
                        activation_id,
                        "failed",
                        {"reason": str(exc)[:200]},
                    )
                except StoreUnavailable:
                    pass
                raise HTTPException(status_code=503, detail="could not publish serving pointer") from exc
            if pointer_sync_error is not None:
                raise HTTPException(
                    status_code=503,
                    detail="serving pointer published; directory durability is uncertain; query activation status",
                )
            return _activation_response(activation_id, "committed", candidate_version)

        pointer = {
            "version": candidate_version,
            "activation_id": activation_id,
            "previous_version": current.version,
        }
        try:
            pointer_before = read_pointer(app.state.serving_pointer_path)
        except ArtifactError as exc:
            app.state.pointer_durability_error = "serving pointer became invalid before activation"
            raise HTTPException(status_code=503, detail="serving pointer is invalid") from exc
        if pointer_before is None or pointer_before.get("version") != current.version:
            app.state.pointer_durability_error = "serving pointer differs from active model before activation"
            raise HTTPException(status_code=409, detail="serving pointer does not match active model")
        pointer_sync_error: str | None = None
        try:
            atomic_write_pointer(app.state.serving_pointer_path, pointer)
        except Exception as exc:  # noqa: BLE001
            try:
                actual_pointer = read_pointer(app.state.serving_pointer_path)
            except ArtifactError:
                actual_pointer = None
            if actual_pointer != pointer:
                if actual_pointer != pointer_before:
                    app.state.pointer_durability_error = (
                        f"serving pointer publication state is uncertain: {str(exc)[:200]}"
                    )
                try:
                    await _set_activation_terminal(
                        activation_id,
                        "failed",
                        {"reason": "pointer publication failed", "error": str(exc)[:200]},
                    )
                except Exception:
                    logger.exception("Could not record failed pointer publication")
                raise HTTPException(status_code=503, detail="could not publish serving pointer") from exc
            pointer_sync_error = str(exc)[:300]

        # No await between publishing the durable pointer and switching the live snapshot.
        _publish_snapshot(staged)
        active_pointer = pointer
        app.state.pointer_durability_error = pointer_sync_error
        details = {
            "rollback": rollback,
            "trainer_run_id": trainer_run_id,
            "active_version": candidate_version,
            "previous_version": current.version,
            "pointer_durability_error": pointer_sync_error,
        }
        try:
            await _set_activation_terminal(activation_id, "committed", details)
        except StoreUnavailable as exc:
            logger.error("Activation %s published but its audit acknowledgement failed", activation_id)
            raise HTTPException(
                status_code=503,
                detail="activation published; query its activation_id for status",
            ) from exc
        if pointer_sync_error is not None:
            raise HTTPException(
                status_code=503,
                detail="serving pointer published; directory durability is uncertain; query activation status",
            )
        return _activation_response(activation_id, "committed", candidate_version)


@app.post("/admin/activate")
async def activate(
    request: ActivationRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _admin_authorize(authorization)
    try:
        return await _activate_version(
            candidate_version=request.candidate_version,
            expected_parent=request.expected_parent,
            activation_id=request.activation_id,
            rollback=False,
            trainer_run_id=request.trainer_run_id,
        )
    except StoreUnavailable as exc:
        raise HTTPException(status_code=503, detail="activation store is unavailable") from exc


@app.post("/admin/rollback")
async def rollback(
    request: RollbackRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _admin_authorize(authorization)
    try:
        return await _activate_version(
            candidate_version=request.target_version,
            expected_parent=request.expected_parent,
            activation_id=request.activation_id,
            rollback=True,
        )
    except StoreUnavailable as exc:
        raise HTTPException(status_code=503, detail="activation store is unavailable") from exc


@app.get("/admin/activation/{activation_id}")
async def activation_status(
    activation_id: str,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _admin_authorize(authorization)
    if len(activation_id) > 128:
        raise HTTPException(status_code=422, detail="activation_id is too long")
    paths = app.state.runtime_paths
    row = await asyncio.to_thread(_query_activation, paths["database"], activation_id)
    if row is None:
        raise HTTPException(status_code=404, detail="activation_id not found")
    pointer = await asyncio.to_thread(read_pointer, app.state.serving_pointer_path)
    status = row["status"]
    details: dict[str, Any] = {}
    if row["details_json"]:
        try:
            details = json.loads(row["details_json"])
        except json.JSONDecodeError:
            details = {}
    if (
        status == "intent"
        and pointer is not None
        and pointer.get("activation_id") == activation_id
        and pointer.get("version") == row["candidate_version"]
    ):
        details = {
            **details,
            "active_version": pointer["version"],
            "previous_version": pointer.get("previous_version"),
            "reconciled": True,
        }
        try:
            await _set_activation_terminal(activation_id, "committed", details)
            status = "committed"
        except StoreUnavailable:
            status = "committed"
    return {
        "activation_id": activation_id,
        "status": status,
        "expected_parent": row["expected_parent"],
        "candidate_version": row["candidate_version"],
        "active_version": pointer.get("version") if pointer else (serving_snapshot.version if serving_snapshot else None),
        "details": details,
    }


@app.get("/monitoring/summary")
async def monitoring_summary(
    window_minutes: int = Query(default=15),
) -> dict[str, Any]:
    if window_minutes not in ALLOWED_WINDOWS_MINUTES:
        raise HTTPException(
            status_code=422,
            detail=f"window_minutes must be one of {ALLOWED_WINDOWS_MINUTES}",
        )
    paths = getattr(app.state, "runtime_paths", _runtime_paths())
    event_logger: RequestEventLogger | None = getattr(
        app.state, "request_event_logger", None
    )
    logging_status = (
        event_logger.snapshot()
        if event_logger is not None
        else {"enabled": REQUEST_LOGGING_ENABLED, "healthy": False, "failures": 0}
    )
    snapshot = serving_snapshot
    try:
        return await asyncio.to_thread(
            build_monitoring_summary,
            paths["database"],
            paths["drift_reference"],
            window_minutes=window_minutes,
            current_model_version=snapshot.version if snapshot is not None else None,
            logging_status=logging_status,
        )
    except (MonitoringError, OSError) as exc:
        raise HTTPException(status_code=503, detail="monitoring summary unavailable") from exc


@app.get("/metrics")
def get_metrics() -> dict[str, Any]:
    stats = prediction_cache.stats()
    event_logger: RequestEventLogger | None = getattr(
        app.state, "request_event_logger", None
    )
    return {
        "model_path": str(MODEL_PATH),
        "model_version": serving_snapshot.version if serving_snapshot is not None else None,
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
        "request_logging": event_logger.snapshot()
        if event_logger is not None
        else {"enabled": REQUEST_LOGGING_ENABLED, "healthy": False, "failures": 0},
    }
