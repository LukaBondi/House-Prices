"""HTTP-level audit, backpressure, and interrupted-request tests."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend import request_logging
from backend.app import app
from backend.event_store import StoreUnavailable, open_database


@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    state = tmp_path / "service-state"
    monkeypatch.setenv("DATA_DIR", str(state))
    monkeypatch.setenv("DB_PATH", str(state / "requests.sqlite3"))
    monkeypatch.setenv("MODEL_REGISTRY_PATH", str(state / "artifacts"))
    monkeypatch.setenv("MODEL_STATE_PATH", str(state / "state"))
    monkeypatch.setenv("DRIFT_REFERENCE_PATH", str(state / "drift-reference.json"))
    monkeypatch.setenv("ADMIN_TOKEN", "test-admin-token")
    return state


def test_every_route_is_audited_and_prediction_bodies_are_bounded(isolated_state):
    with TestClient(app) as client:
        first = client.post(
            "/predict",
            json={"features": {"OverallQual": 7, "Neighborhood": "CollgCr"}},
        )
        second = client.post(
            "/predict",
            json={"features": {"OverallQual": 7.0, "Neighborhood": "CollgCr"}},
        )
        malformed = client.post(
            "/predict",
            content=b'{"features":',
            headers={"content-type": "application/json"},
        )
        oversized = client.post(
            "/predict",
            content=b"x" * (64 * 1024 + 5),
            headers={"content-type": "application/json"},
        )
        missing = client.get("/not-a-route")
        preflight = client.options(
            "/predict",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        summary = client.get("/monitoring/summary")

        assert first.status_code == second.status_code == 200
        assert first.headers["x-request-id"] == first.json()["request_id"]
        assert second.headers["x-request-id"] == second.json()["request_id"]
        assert first.json()["model_version"] == second.json()["model_version"]
        assert malformed.status_code == 422
        assert oversized.status_code == 413
        assert missing.status_code == 404
        assert preflight.status_code == 200
        assert summary.status_code == 200
        assert summary.json()["requests"]["total"] == 4
        assert all("x-request-id" in response.headers for response in (malformed, oversized, missing, preflight))

    connection = open_database(isolated_state / "requests.sqlite3")
    try:
        rows = connection.execute(
            "SELECT path, method, status_code, lifecycle, body_text, body_truncated, "
            "parse_state, features_json, feature_fingerprint, prediction, model_version, "
            "cache_outcome, request_id FROM requests ORDER BY rowid"
        ).fetchall()
        assert len(rows) == 7
        prediction_rows = [
            row for row in rows if row["path"] == "/predict" and row["method"] == "POST"
        ]
        assert [row["status_code"] for row in prediction_rows] == [200, 200, 422, 413]
        assert [row["cache_outcome"] for row in prediction_rows[:2]] == ["miss", "hit"]
        assert all(row["lifecycle"] == "completed" for row in rows)
        assert prediction_rows[0]["features_json"] is not None
        assert prediction_rows[0]["feature_fingerprint"]
        assert prediction_rows[0]["prediction"] > 0
        assert prediction_rows[0]["model_version"]
        assert prediction_rows[2]["parse_state"] == "invalid_json"
        assert prediction_rows[3]["parse_state"] == "oversized"
        assert prediction_rows[3]["body_truncated"] == 1
        assert len(prediction_rows[3]["body_text"].encode("utf-8")) <= 64 * 1024
        assert rows[-3]["status_code"] == 404
        assert rows[-2]["method"] == "OPTIONS"
        assert rows[-1]["path"] == "/monitoring/summary"
    finally:
        connection.close()


def test_store_admission_failure_returns_bounded_503(isolated_state, monkeypatch):
    async def unavailable(self, context):
        raise StoreUnavailable("simulated full writer")

    monkeypatch.setattr(request_logging.RequestEventLogger, "admit", unavailable)
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 503
    assert response.headers.get("x-request-id")
    assert response.json()["detail"] == "request logging unavailable"


def test_completion_write_failure_replaces_success_with_503(isolated_state, monkeypatch):
    async def unavailable(self, context, status_code):
        raise StoreUnavailable("simulated final commit failure")

    monkeypatch.setattr(request_logging.RequestEventLogger, "complete", unavailable)
    with TestClient(app) as client:
        response = client.post("/predict", json={"features": {"OverallQual": 7}})
        assert response.status_code == 503
        assert response.json()["detail"] == "request logging unavailable"
        request_id = response.json()["request_id"]

    connection = open_database(isolated_state / "requests.sqlite3")
    try:
        row = connection.execute(
            "SELECT status_code, lifecycle, logging_failed FROM requests WHERE request_id=?",
            (request_id,),
        ).fetchone()
        assert row is not None
        assert tuple(row) == (503, "logging_failed", 1)
    finally:
        connection.close()


def test_cancelled_body_read_is_persisted_as_aborted(tmp_path):
    async def cancelled_receive():
        raise asyncio.CancelledError

    async def never_reached(scope, receive, send):
        raise AssertionError("request body cancellation should happen before routing")

    async def run():
        database = tmp_path / "cancel.sqlite3"
        event_logger = request_logging.RequestEventLogger(database)
        await event_logger.start()
        middleware = request_logging.RequestLoggingMiddleware(
            never_reached,
            app_state=SimpleNamespace(request_event_logger=event_logger),
        )
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/predict",
            "headers": [],
            "query_string": b"",
            "http_version": "1.1",
            "scheme": "http",
            "server": ("test", 80),
            "client": ("test", 1234),
        }

        async def send(message):
            return None

        with pytest.raises(asyncio.CancelledError):
            await middleware(scope, cancelled_receive, send)
        await event_logger.stop()
        connection = open_database(database)
        try:
            row = connection.execute(
                "SELECT lifecycle, terminal_reason FROM requests"
            ).fetchone()
            assert tuple(row) == ("aborted", "request_cancelled")
        finally:
            connection.close()

    asyncio.run(run())


def test_unhandled_application_failure_gets_audited_500_with_request_id(tmp_path):
    async def failing_app(scope, receive, send):
        raise RuntimeError("simulated handler failure")

    async def unused_receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def run():
        database = tmp_path / "server-error.sqlite3"
        event_logger = request_logging.RequestEventLogger(database)
        await event_logger.start()
        middleware = request_logging.RequestLoggingMiddleware(
            failing_app,
            app_state=SimpleNamespace(request_event_logger=event_logger),
        )
        messages = []
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/broken",
            "headers": [],
            "query_string": b"",
            "http_version": "1.1",
            "scheme": "http",
            "server": ("test", 80),
            "client": ("test", 1234),
        }

        async def send(message):
            messages.append(message)

        await middleware(scope, unused_receive, send)
        await event_logger.stop()
        start = next(message for message in messages if message["type"] == "http.response.start")
        body = next(message for message in messages if message["type"] == "http.response.body")
        headers = dict(start["headers"])
        request_id = headers[b"x-request-id"].decode()
        assert start["status"] == 500
        assert request_id.encode() in body["body"]
        connection = open_database(database)
        try:
            row = connection.execute(
                "SELECT request_id, status_code, lifecycle, terminal_reason FROM requests"
            ).fetchone()
            assert tuple(row) == (
                request_id,
                500,
                "completed",
                "unhandled_application_exception",
            )
        finally:
            connection.close()

    asyncio.run(run())


def test_final_ack_timeout_sends_only_503_and_late_200_is_corrected_with_full_queue(tmp_path):
    async def run():
        database = tmp_path / "timeout-response.sqlite3"
        event_logger = request_logging.RequestEventLogger(
            database, queue_size=1, ack_timeout=0.05
        )
        await event_logger.start()
        connection = open_database(database)
        middleware_messages = []

        async def receive():
            return {
                "type": "http.request",
                "body": b'{"features":{"OverallQual":7}}',
                "more_body": False,
            }

        async def successful_app(scope, downstream_receive, send):
            connection.execute("BEGIN IMMEDIATE")

            async def occupy_the_full_queue():
                await asyncio.sleep(0.01)
                return await event_logger.writer.execute(
                    "UPDATE runs SET status=status WHERE run_id=?",
                    ("missing",),
                    timeout=1.0,
                )

            filler = asyncio.create_task(occupy_the_full_queue())
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": b'{"ok":true}'})
            connection.rollback()
            await filler

        async def send(message):
            middleware_messages.append(message)

        middleware = request_logging.RequestLoggingMiddleware(
            successful_app,
            app_state=SimpleNamespace(request_event_logger=event_logger),
        )
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/predict",
            "headers": [],
            "query_string": b"",
            "http_version": "1.1",
            "scheme": "http",
            "server": ("test", 80),
            "client": ("test", 1234),
        }
        try:
            await middleware(scope, receive, send)
        finally:
            connection.close()
            await event_logger.stop()

        starts = [
            message
            for message in middleware_messages
            if message["type"] == "http.response.start"
        ]
        bodies = [
            message
            for message in middleware_messages
            if message["type"] == "http.response.body"
        ]
        assert len(starts) == 1
        assert starts[0]["status"] == 503
        assert len(bodies) == 1
        request_id = json.loads(bodies[0]["body"])["request_id"]
        connection = open_database(database)
        try:
            row = connection.execute(
                "SELECT status_code, lifecycle, logging_failed FROM requests WHERE request_id=?",
                (request_id,),
            ).fetchone()
            assert tuple(row) == (503, "logging_failed", 1)
        finally:
            connection.close()

    asyncio.run(run())
