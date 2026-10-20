"""Bounded ASGI request capture and durable lifecycle auditing."""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

from backend.event_store import SQLiteEventWriter, StoreUnavailable, canonical_json

logger = logging.getLogger("house-prices-api.audit")
MAX_PREDICTION_BODY_BYTES = 64 * 1024


@dataclass
class RequestContext:
    request_id: str
    received_at: float
    method: str
    path: str
    body_text: str | None = None
    body_truncated: bool = False
    parse_state: str = "not_captured"
    features: dict[str, Any] | None = None
    feature_fingerprint: str | None = None
    prediction: float | None = None
    model_version: str | None = None
    cache_outcome: str | None = None
    dataset_sha256: str | None = None
    source_id: int | None = None
    run_id: str | None = None
    terminal_reason: str | None = None
    status_code: int | None = None
    handler_duration_ms: float | None = None
    _raw_body: bytes = field(default=b"", repr=False)
    _started: float = field(default_factory=time.perf_counter, repr=False)
    _finalized: bool = field(default=False, repr=False)


current_request_context: contextvars.ContextVar[RequestContext | None] = (
    contextvars.ContextVar("house_prices_request_context", default=None)
)


class RequestEventLogger:
    """Own the bounded writer and process-local health diagnostics."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        enabled: bool = True,
        queue_size: int = 256,
        ack_timeout: float = 0.25,
    ) -> None:
        self.database_path = Path(database_path)
        self.enabled = enabled
        self.writer = SQLiteEventWriter(
            self.database_path,
            queue_size=queue_size,
            ack_timeout=ack_timeout,
        )
        self._lock = Lock()
        self.failures = 0
        self.last_failure: str | None = None
        self.last_failure_at: float | None = None
        self.last_success_at: float | None = None
        self.admission_wait_ms_total = 0.0
        self.completion_wait_ms_total = 0.0
        self.admission_writes = 0
        self.completion_writes = 0

    async def start(self) -> None:
        if self.enabled:
            await self.writer.start()

    async def stop(self) -> None:
        if self.enabled:
            await self.writer.stop()

    @property
    def healthy(self) -> bool:
        if not self.enabled:
            return False
        return self.writer.running and self.last_failure is None

    @property
    def queue_depth(self) -> int:
        return self.writer.queue_depth if self.enabled else 0

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "healthy": self.healthy,
                "failures": self.failures,
                "last_failure": self.last_failure,
                "last_failure_at": self.last_failure_at,
                "queue_depth": self.queue_depth,
                "admission_wait_ms_total": round(self.admission_wait_ms_total, 3),
                "completion_wait_ms_total": round(self.completion_wait_ms_total, 3),
                "admission_writes": self.admission_writes,
                "completion_writes": self.completion_writes,
            }

    def _success(self, *, phase: str, duration_ms: float) -> None:
        with self._lock:
            self.last_success_at = time.time()
            self.last_failure = None
            self.last_failure_at = None
            if phase == "admission":
                self.admission_writes += 1
                self.admission_wait_ms_total += duration_ms
            elif phase == "completion":
                self.completion_writes += 1
                self.completion_wait_ms_total += duration_ms

    def _failure(self, error: BaseException) -> None:
        with self._lock:
            self.failures += 1
            self.last_failure = str(error)[:300]
            self.last_failure_at = time.time()

    async def admit(self, context: RequestContext) -> None:
        if not self.enabled:
            return
        started = time.perf_counter()
        try:
            await self.writer.execute(
                "INSERT INTO requests "
                "(request_id, received_at, method, path, lifecycle, parse_state) "
                "VALUES (?, ?, ?, ?, 'pending', 'awaiting_body')",
                (
                    context.request_id,
                    context.received_at,
                    context.method,
                    context.path,
                ),
                correct_request_id=context.request_id,
            )
        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError):
                raise
            self._failure(exc)
            raise StoreUnavailable(str(exc)) from exc
        self._success(
            phase="admission",
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    async def complete(self, context: RequestContext, status_code: int) -> None:
        if not self.enabled:
            return
        context.status_code = status_code
        context.handler_duration_ms = max(0.0, (time.perf_counter() - context._started) * 1000)
        features_json = (
            canonical_json(context.features) if context.features is not None else None
        )
        started = time.perf_counter()
        sql = (
            "UPDATE requests SET completed_at=?, status_code=?, lifecycle='completed', "
            "handler_duration_ms=?, body_text=?, body_truncated=?, parse_state=?, "
            "features_json=?, feature_fingerprint=?, prediction=?, model_version=?, "
            "cache_outcome=?, dataset_sha256=?, source_id=?, run_id=?, terminal_reason=?, "
            "logging_failed=0 WHERE request_id=? AND lifecycle='pending'"
        )
        parameters = (
            time.time(),
            status_code,
            context.handler_duration_ms,
            context.body_text,
            int(context.body_truncated),
            context.parse_state,
            features_json,
            context.feature_fingerprint,
            context.prediction,
            context.model_version,
            context.cache_outcome,
            context.dataset_sha256,
            context.source_id,
            context.run_id,
            context.terminal_reason,
            context.request_id,
        )
        try:
            changed = await self.writer.execute(
                sql,
                parameters,
                correct_request_id=context.request_id,
            )
            if changed != 1:
                raise StoreUnavailable("request admission row was not pending")
        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError):
                raise
            self._failure(exc)
            raise StoreUnavailable(str(exc)) from exc
        self._success(
            phase="completion",
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    async def abort(self, context: RequestContext, reason: str) -> None:
        context.terminal_reason = reason[:120]
        if not self.enabled:
            return
        sql = (
            "UPDATE requests SET completed_at=?, lifecycle='aborted', body_text=?, "
            "body_truncated=?, parse_state=?, terminal_reason=? "
            "WHERE request_id=? AND lifecycle='pending'"
        )
        try:
            await self.writer.execute(
                sql,
                (
                    time.time(),
                    context.body_text,
                    int(context.body_truncated),
                    context.parse_state,
                    context.terminal_reason,
                    context.request_id,
                ),
            )
        except BaseException as exc:
            if not isinstance(exc, asyncio.CancelledError):
                self._failure(exc)

    async def mark_logging_failed(self, request_id: str) -> None:
        if not self.enabled:
            return
        try:
            await self.writer.execute(
                "UPDATE requests SET status_code=503, lifecycle='logging_failed', "
                "logging_failed=1, terminal_reason='completion_ack_failed', completed_at=? "
                "WHERE request_id=?",
                (time.time(), request_id),
                timeout=self.writer.ack_timeout,
                correct_request_id=request_id,
            )
        except BaseException as exc:
            if not isinstance(exc, asyncio.CancelledError):
                self._failure(exc)


class RequestLoggingMiddleware:
    """Audit all HTTP requests, capturing only bounded prediction request bodies."""

    def __init__(self, app: Any, *, app_state: Any) -> None:
        self.app = app
        self.app_state = app_state

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        request_id = str(uuid.uuid4())
        context = RequestContext(
            request_id=request_id,
            received_at=time.time(),
            method=str(scope.get("method", "GET")),
            path=str(scope.get("path", "/")),
        )
        token = current_request_context.set(context)
        event_logger: RequestEventLogger | None = getattr(
            self.app_state, "request_event_logger", None
        )

        if event_logger is None:
            await self._send_json(
                send,
                503,
                {"detail": "request logger is not ready", "request_id": request_id},
                request_id,
            )
            current_request_context.reset(token)
            return

        try:
            try:
                await event_logger.admit(context)
            except StoreUnavailable:
                await self._send_json(
                    send,
                    503,
                    {"detail": "request logging unavailable", "request_id": request_id},
                    request_id,
                )
                return

            downstream_receive = receive
            if context.path == "/predict":
                captured = await self._capture_prediction_body(context, receive)
                if captured == "disconnected":
                    await self._bounded_abort(event_logger, context, "client_disconnected")
                    return
                if captured == "oversized":
                    try:
                        await event_logger.complete(context, 413)
                        context._finalized = True
                    except StoreUnavailable:
                        context._finalized = True
                        asyncio.create_task(
                            event_logger.mark_logging_failed(context.request_id)
                        )
                        await self._send_json(
                            send,
                            503,
                            {
                                "detail": "request logging unavailable",
                                "request_id": request_id,
                            },
                            request_id,
                        )
                        return
                    await self._send_json(
                        send,
                        413,
                        {"detail": "prediction request body exceeds 65536 bytes"},
                        request_id,
                    )
                    return
                downstream_receive = self._replay_body(receive, context._raw_body)

            response_overridden = False
            response_started = False

            async def send_with_audit(message: dict[str, Any]) -> None:
                nonlocal response_overridden, response_started
                if response_overridden:
                    return
                if message.get("type") == "http.response.start":
                    status_code = int(message.get("status", 500))
                    context.status_code = status_code
                    if status_code >= 500 and context.terminal_reason is None:
                        context.terminal_reason = "http_5xx"
                    elif status_code >= 400 and context.terminal_reason is None:
                        context.terminal_reason = "http_4xx"
                    if not context._finalized:
                        try:
                            await event_logger.complete(context, status_code)
                            context._finalized = True
                        except StoreUnavailable:
                            context._finalized = True
                            response_overridden = True
                            asyncio.create_task(
                                event_logger.mark_logging_failed(context.request_id)
                            )
                            await self._send_json(
                                send,
                                503,
                                {
                                    "detail": "request logging unavailable",
                                    "request_id": request_id,
                                },
                                request_id,
                            )
                            return
                    headers = [
                        (name, value)
                        for name, value in message.get("headers", [])
                        if name.lower() != b"x-request-id"
                    ]
                    headers.append((b"x-request-id", request_id.encode("ascii")))
                    message = {**message, "headers": headers}
                    response_started = True
                await send(message)

            try:
                await self.app(scope, downstream_receive, send_with_audit)
            except Exception:
                logger.exception("Unhandled ASGI request failure for %s", request_id)
                if not context._finalized:
                    try:
                        context.terminal_reason = "unhandled_application_exception"
                        await event_logger.complete(context, 500)
                        context._finalized = True
                    except StoreUnavailable:
                        context._finalized = True
                        await self._send_json(
                            send,
                            503,
                            {
                                "detail": "request logging unavailable",
                                "request_id": request_id,
                            },
                            request_id,
                        )
                        return
                if response_started:
                    raise
                await self._send_json(
                    send,
                    500,
                    {"detail": "Internal Server Error", "request_id": request_id},
                    request_id,
                )
        except asyncio.CancelledError:
            if not context._finalized:
                await self._bounded_abort(event_logger, context, "request_cancelled")
            raise
        except Exception:
            if not context._finalized:
                await self._bounded_abort(event_logger, context, "middleware_or_transport_error")
            raise
        finally:
            current_request_context.reset(token)

    async def _capture_prediction_body(
        self, context: RequestContext, receive: Any
    ) -> str | None:
        body = bytearray()
        while True:
            message = await receive()
            message_type = message.get("type")
            if message_type == "http.disconnect":
                context.parse_state = "client_disconnected"
                context.body_text, text_truncated = self._bounded_body_text(bytes(body))
                context.body_truncated = context.body_truncated or text_truncated
                context._raw_body = bytes(body)
                return "disconnected"
            if message_type != "http.request":
                continue
            part = message.get("body", b"")
            remaining = MAX_PREDICTION_BODY_BYTES - len(body)
            if len(part) > remaining:
                if remaining > 0:
                    body.extend(part[:remaining])
                context.body_truncated = True
                context.parse_state = "oversized"
                context.terminal_reason = "body_too_large"
                context.body_text, _ = self._bounded_body_text(bytes(body))
                context._raw_body = bytes(body)
                return "oversized"
            body.extend(part)
            if len(body) > MAX_PREDICTION_BODY_BYTES:
                context.body_truncated = True
                context.parse_state = "oversized"
                context.terminal_reason = "body_too_large"
                context.body_text, _ = self._bounded_body_text(
                    bytes(body[:MAX_PREDICTION_BODY_BYTES])
                )
                context._raw_body = bytes(body[:MAX_PREDICTION_BODY_BYTES])
                return "oversized"
            if not message.get("more_body", False):
                break

        context._raw_body = bytes(body)
        context.body_text, text_truncated = self._bounded_body_text(bytes(body))
        context.body_truncated = context.body_truncated or text_truncated
        if not body:
            context.parse_state = "empty"
        else:
            try:
                json.loads(body.decode("utf-8"))
                context.parse_state = "valid_json"
            except UnicodeDecodeError:
                context.parse_state = "invalid_utf8"
            except json.JSONDecodeError:
                context.parse_state = "invalid_json"
        return None

    @staticmethod
    def _bounded_body_text(body: bytes) -> tuple[str | None, bool]:
        text = body.decode("utf-8", errors="replace")
        encoded = text.encode("utf-8")
        if len(encoded) <= MAX_PREDICTION_BODY_BYTES:
            return text or None, False
        bounded = encoded[:MAX_PREDICTION_BODY_BYTES].decode("utf-8", errors="ignore")
        return bounded or None, True

    @staticmethod
    def _replay_body(receive: Any, body: bytes) -> Any:
        replayed = False

        async def replay() -> dict[str, Any]:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        return replay

    @staticmethod
    async def _bounded_abort(
        event_logger: RequestEventLogger, context: RequestContext, reason: str
    ) -> None:
        try:
            task = asyncio.create_task(event_logger.abort(context, reason))
            await asyncio.wait_for(asyncio.shield(task), timeout=0.3)
        except BaseException:
            logger.warning("Could not acknowledge aborted request %s", context.request_id)
        finally:
            context._finalized = True

    @staticmethod
    async def _send_json(send: Any, status: int, data: dict[str, Any], request_id: str) -> None:
        body = json.dumps(data, separators=(",", ":")).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                    (b"x-request-id", request_id.encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
