"""Unit tests for backend helpers."""

import asyncio
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import backend.app as api
from backend.cache import TTLCache


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


def test_ttl_cache_is_bounded_and_tracks_hits_and_misses():
    cache = TTLCache[int](max_entries=1, ttl_seconds=60)
    cache.set("first", 1)
    assert cache.get("first") == 1
    cache.set("second", 2)
    assert cache.get("first") is None
    assert cache.get("second") == 2
    stats = cache.stats()
    assert stats.hits == 2
    assert stats.misses == 1
    assert stats.size == 1


def test_ttl_cache_expires_and_clear_resets_counters(monkeypatch):
    now = 100.0
    monkeypatch.setattr("backend.cache.time.monotonic", lambda: now)
    cache = TTLCache[int](max_entries=3, ttl_seconds=5)
    cache.set("one", 1)
    assert cache.get("one") == 1
    now += 5
    assert cache.get("one") is None
    cache.clear()
    assert cache.stats().hits == 0
    assert cache.stats().misses == 0
    assert cache.stats().size == 0


def test_batched_fallback_keeps_successes_mapped_to_their_requests(monkeypatch):
    class FailingBatchModel:
        def predict_records(self, records):
            if len(records) > 1:
                raise ValueError("batch contained a bad row")
            if records[0]["value"] == "bad":
                raise ValueError("bad row")
            return [float(records[0]["value"])]

    async def run():
        old_model = api.model
        old_cache_enabled = api.CACHE_ENABLED
        api.model = FailingBatchModel()
        api.CACHE_ENABLED = False
        loop = asyncio.get_running_loop()
        batch = [
            api.QueuedPrediction(str(value), {"value": value}, loop.create_future())
            for value in (3, "bad", 9)
        ]
        try:
            await api._finish_batch(batch)
            assert batch[0].result.result() == 3
            try:
                batch[1].result.result()
            except ValueError as exc:
                assert str(exc) == "bad row"
            else:
                raise AssertionError("failed row should receive its own inference error")
            assert batch[2].result.result() == 9
        finally:
            api.model = old_model
            api.CACHE_ENABLED = old_cache_enabled

    asyncio.run(run())


def test_full_prediction_queue_returns_503_and_keeps_accepted_request(monkeypatch):
    class StubModel:
        feature_names = ["value"]
        numeric_features = ["value"]

        def validate_record(self, record):
            return None

    async def run():
        old_model = api.model
        old_queue = api.inference_queue
        old_cache_enabled = api.CACHE_ENABLED
        old_enabled = api.DYNAMIC_BATCHING_ENABLED
        api.model = StubModel()
        api.inference_queue = asyncio.Queue(maxsize=1)
        api.CACHE_ENABLED = True
        api.DYNAMIC_BATCHING_ENABLED = True
        api.prediction_cache.clear()
        api.in_flight.clear()
        try:
            accepted = asyncio.create_task(api._predict_value({"value": 1}))
            await asyncio.sleep(0)
            try:
                await api._predict_value({"value": 2})
            except api.HTTPException as exc:
                assert exc.status_code == 503
                assert exc.detail == "Prediction queue is full"
            else:
                raise AssertionError("full prediction queue should reject new work")
            queued = api.inference_queue.get_nowait()
            api._settle_item(queued, 123.0)
            api.inference_queue.task_done()
            assert await accepted == 123.0
        finally:
            api.in_flight.clear()
            api.inference_queue = old_queue
            api.model = old_model
            api.CACHE_ENABLED = old_cache_enabled
            api.DYNAMIC_BATCHING_ENABLED = old_enabled

    asyncio.run(run())


def test_shutdown_drains_an_accepted_prediction():
    class StubModel:
        feature_names = ["value"]
        numeric_features = ["value"]

        def validate_record(self, record):
            return None

        def predict_records(self, records):
            return [float(record["value"]) * 2 for record in records]

    async def run():
        old_model = api.model
        old_queue = api.inference_queue
        old_task = api.batch_task
        old_tasks = api.batch_tasks
        old_executor = api.inference_executor
        old_cache_enabled = api.CACHE_ENABLED
        old_batch_enabled = api.DYNAMIC_BATCHING_ENABLED
        api.model = StubModel()
        api.CACHE_ENABLED = False
        api.DYNAMIC_BATCHING_ENABLED = True
        api.inference_queue = asyncio.Queue(maxsize=4)
        api.inference_executor = ThreadPoolExecutor(max_workers=1)
        api.batch_tasks = [asyncio.create_task(api._batch_dispatcher())]
        api.batch_task = api.batch_tasks[0]
        api.in_flight.clear()
        request = asyncio.create_task(api._predict_value({"value": 6}))
        await asyncio.sleep(0)
        try:
            await api.shutdown()
            assert await request == 12
            assert api.batch_task is None
            assert api.inference_queue is None
            assert not api.in_flight
        finally:
            api.model = old_model
            api.inference_queue = old_queue
            api.batch_task = old_task
            api.batch_tasks = old_tasks
            api.inference_executor = old_executor
            api.CACHE_ENABLED = old_cache_enabled
            api.DYNAMIC_BATCHING_ENABLED = old_batch_enabled

    asyncio.run(run())


def test_native_runtime_imports_and_predicts_without_training_dependencies():
    root = Path(__file__).resolve().parents[1]
    script = r'''
import importlib.abc
import sys
from pathlib import Path

blocked = {"pandas", "sklearn", "joblib", "model_training"}
class BlockTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in blocked:
            raise ModuleNotFoundError(fullname)
        return None
sys.meta_path.insert(0, BlockTrainingImports())
sys.path.insert(0, str(Path.cwd()))
from backend.native_model import NativeModel
model = NativeModel(Path("models/house_prices_native.zip"))
prediction = model.predict_records([{"OverallQual": 7, "GrLivArea": 1710, "Neighborhood": "CollgCr"}])
assert prediction.shape == (1,) and prediction[0] > 0
assert not (blocked & set(sys.modules))
print(float(prediction[0]))
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
