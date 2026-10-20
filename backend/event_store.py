"""SQLite event storage shared by the API and the offline trainer."""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import math
import queue
import sqlite3
import threading
import time
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
BUSY_TIMEOUT_SECONDS = 0.1
DEFAULT_ACK_TIMEOUT_SECONDS = 0.25
DEFAULT_QUEUE_SIZE = 256

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY,
    received_at REAL NOT NULL,
    completed_at REAL,
    method TEXT NOT NULL,
    path TEXT NOT NULL,
    status_code INTEGER,
    lifecycle TEXT NOT NULL,
    handler_duration_ms REAL,
    body_text TEXT,
    body_truncated INTEGER NOT NULL DEFAULT 0,
    parse_state TEXT,
    features_json TEXT,
    feature_fingerprint TEXT,
    prediction REAL,
    model_version TEXT,
    cache_outcome TEXT,
    dataset_sha256 TEXT,
    source_id INTEGER,
    run_id TEXT,
    terminal_reason TEXT,
    logging_failed INTEGER NOT NULL DEFAULT 0,
    admission_write_ms REAL,
    completion_write_ms REAL
);
CREATE INDEX IF NOT EXISTS idx_requests_time_path ON requests(received_at, path);
CREATE INDEX IF NOT EXISTS idx_requests_version ON requests(model_version, received_at);
CREATE INDEX IF NOT EXISTS idx_requests_observation ON requests(dataset_sha256, source_id);
CREATE INDEX IF NOT EXISTS idx_requests_fingerprint ON requests(feature_fingerprint, received_at);

CREATE TABLE IF NOT EXISTS labels (
    observation_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL REFERENCES requests(request_id),
    dataset_sha256 TEXT NOT NULL,
    source_id INTEGER NOT NULL,
    features_json TEXT NOT NULL,
    feature_fingerprint TEXT NOT NULL,
    sale_price REAL NOT NULL CHECK(sale_price >= 0),
    available_at REAL NOT NULL,
    UNIQUE(dataset_sha256, source_id)
);
CREATE INDEX IF NOT EXISTS idx_labels_available ON labels(available_at);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    parent_version TEXT,
    snapshot_hash TEXT,
    status TEXT NOT NULL,
    started_at REAL NOT NULL,
    completed_at REAL,
    metrics_json TEXT,
    details_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at DESC);

CREATE TABLE IF NOT EXISTS activations (
    activation_id TEXT PRIMARY KEY,
    expected_parent TEXT,
    candidate_version TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    completed_at REAL,
    details_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_activations_created ON activations(created_at DESC);

CREATE TABLE IF NOT EXISTS trainer_leases (
    lease_name TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    acquired_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    snapshot_hash TEXT,
    details_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_trainer_leases_expires ON trainer_leases(expires_at);
"""


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically and reject NaN/Infinity."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_record(
    features: Mapping[str, Any],
    *,
    feature_names: Sequence[str],
    numeric_features: Collection[str],
) -> dict[str, Any]:
    """Project and normalize one raw input row before fitted imputation."""
    if not isinstance(features, Mapping):
        raise TypeError("features must be a mapping")

    def is_missing(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return value == ""
        try:
            return math.isnan(float(value))
        except (TypeError, ValueError, OverflowError):
            return False

    numeric = set(numeric_features)
    result: dict[str, Any] = {}
    for name in feature_names:
        value = features.get(name)
        if is_missing(value):
            result[name] = None
        elif name in numeric:
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} must be numeric") from exc
            if not math.isfinite(number):
                raise ValueError(f"{name} must be finite")
            result[name] = 0.0 if number == 0 else number
        elif isinstance(value, str):
            result[name] = value
        else:
            raise ValueError(f"{name} must be a string")
    return result


def feature_fingerprint(
    features: Mapping[str, Any],
    *,
    feature_names: Sequence[str],
    numeric_features: Collection[str],
) -> str:
    """Return the stable SHA-256 fingerprint used by requests and labels."""
    record = canonical_record(
        features,
        feature_names=feature_names,
        numeric_features=numeric_features,
    )
    return hashlib.sha256(canonical_json(record).encode("utf-8")).hexdigest()


def open_database(path: str | Path) -> sqlite3.Connection:
    """Open one configured SQLite connection; callers own and close it."""
    db_path = Path(path)
    if str(db_path) != ":memory:":
        db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        str(db_path),
        timeout=BUSY_TIMEOUT_SECONDS,
        isolation_level="DEFERRED",
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(f"PRAGMA busy_timeout = {int(BUSY_TIMEOUT_SECONDS * 1000)}")
    connection.execute("PRAGMA synchronous = FULL")
    return connection


def initialize_database(path: str | Path) -> None:
    """Create or upgrade the shared schema before API/trainer use."""
    db_path = Path(path)
    if str(db_path) != ":memory:":
        db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        str(db_path),
        timeout=BUSY_TIMEOUT_SECONDS,
        isolation_level="DEFERRED",
    )
    try:
        connection.execute(f"PRAGMA busy_timeout = {int(BUSY_TIMEOUT_SECONDS * 1000)}")
        if str(db_path) != ":memory:":
            connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(_SCHEMA)
        # Additive migration path for databases created by earlier development builds.
        request_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(requests)")
        }
        for name, declaration in (
            ("logging_failed", "INTEGER NOT NULL DEFAULT 0"),
            ("admission_write_ms", "REAL"),
            ("completion_write_ms", "REAL"),
        ):
            if name not in request_columns:
                connection.execute(f"ALTER TABLE requests ADD COLUMN {name} {declaration}")
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        connection.commit()
    finally:
        connection.close()


def archive_old_requests(
    source_path: str | Path,
    archive_path: str | Path,
    *,
    before: float,
    limit: int = 1000,
) -> int:
    """Archive one bounded batch of old unlabelled request rows, then prune them.

    Copy commits before deletion, so a crash can leave a duplicate archive row but
    cannot lose the only copy. Requests with adaptation provenance, labels, or an
    incomplete lifecycle are retained in the live store.
    """
    if limit < 1 or limit > 10_000:
        raise ValueError("archive limit must be between 1 and 10000")
    source = Path(source_path).resolve()
    archive = Path(archive_path).resolve()
    if source == archive:
        raise ValueError("source and archive databases must be different files")
    initialize_database(archive)
    source_connection = open_database(source)
    archive_connection = open_database(archive)
    try:
        rows = source_connection.execute(
            "SELECT request.* FROM requests AS request "
            "WHERE request.completed_at IS NOT NULL AND request.completed_at < ? "
            "AND request.lifecycle IN ('completed', 'aborted', 'logging_failed') "
            "AND request.dataset_sha256 IS NULL AND request.source_id IS NULL "
            "AND NOT EXISTS (SELECT 1 FROM labels WHERE labels.request_id=request.request_id) "
            "ORDER BY request.completed_at, request.request_id LIMIT ?",
            (float(before), limit),
        ).fetchall()
        if not rows:
            return 0
        columns = [column[1] for column in source_connection.execute("PRAGMA table_info(requests)")]
        placeholders = ",".join("?" for _ in columns)
        quoted_columns = ",".join(f'"{name}"' for name in columns)
        insert_sql = (
            f"INSERT OR IGNORE INTO requests ({quoted_columns}) VALUES ({placeholders})"
        )
        archive_connection.executemany(
            insert_sql,
            [tuple(row[name] for name in columns) for row in rows],
        )
        archive_connection.commit()

        request_ids = [row["request_id"] for row in rows]
        request_placeholders = ",".join("?" for _ in request_ids)
        source_connection.execute("BEGIN IMMEDIATE")
        cursor = source_connection.execute(
            "DELETE FROM requests WHERE request_id IN ("
            + request_placeholders
            + ") AND dataset_sha256 IS NULL AND source_id IS NULL "
            "AND lifecycle IN ('completed', 'aborted', 'logging_failed') "
            "AND NOT EXISTS (SELECT 1 FROM labels WHERE labels.request_id=requests.request_id)",
            tuple(request_ids),
        )
        source_connection.commit()
        return max(0, cursor.rowcount)
    except BaseException:
        source_connection.rollback()
        archive_connection.rollback()
        raise
    finally:
        source_connection.close()
        archive_connection.close()


def checkpoint_database(path: str | Path, *, truncate: bool = False) -> tuple[int, int, int]:
    """Run an explicit passive WAL checkpoint without deleting event history."""
    connection = open_database(path)
    try:
        mode = "TRUNCATE" if truncate else "PASSIVE"
        row = connection.execute(f"PRAGMA wal_checkpoint({mode})").fetchone()
        return tuple(int(value) for value in row)
    finally:
        connection.close()


class StoreUnavailable(RuntimeError):
    """A bounded store operation could not be acknowledged."""


@dataclass
class _Command:
    sql: str
    parameters: tuple[Any, ...]
    future: concurrent.futures.Future[int]
    deadline: float
    cancelled: threading.Event = field(default_factory=threading.Event)
    correct_request_id: str | None = None
    finished_at: float | None = None


class SQLiteEventWriter:
    """One bounded queue and dedicated SQLite thread for acknowledged writes."""

    def __init__(
        self,
        path: str | Path,
        *,
        queue_size: int = DEFAULT_QUEUE_SIZE,
        ack_timeout: float = DEFAULT_ACK_TIMEOUT_SECONDS,
    ) -> None:
        self.path = Path(path)
        self.ack_timeout = float(ack_timeout)
        if queue_size < 1:
            raise ValueError("writer queue_size must be positive")
        self._queue: queue.Queue[_Command | None] = queue.Queue(
            maxsize=min(int(queue_size), DEFAULT_QUEUE_SIZE)
        )
        self._thread: threading.Thread | None = None
        self._started = False
        self._closing = False
        self._active_calls = 0
        self._pending_corrections = 0
        self._state_lock = threading.Lock()
        # Corrections are safety-critical terminal updates. Keep them outside the
        # bounded admission queue so backpressure cannot discard a 503 correction.
        self._corrections: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._lifecycle_lock = asyncio.Lock()

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def running(self) -> bool:
        with self._state_lock:
            return self._is_running_locked()

    def _is_running_locked(self) -> bool:
        return (
            self._thread is not None
            and self._thread.is_alive()
            and self._started
            and not self._closing
        )

    async def start(self) -> None:
        async with self._lifecycle_lock:
            with self._state_lock:
                thread = self._thread
                closing = self._closing
            if thread is not None and thread.is_alive():
                if closing:
                    raise StoreUnavailable("event writer is stopping")
                return
            if thread is not None:
                await asyncio.to_thread(thread.join)
            self._discard_stale_commands()
            await asyncio.to_thread(initialize_database, self.path)
            thread = threading.Thread(
                target=self._run,
                name="house-prices-sqlite-writer",
                daemon=True,
            )
            with self._state_lock:
                self._thread = thread
                self._started = True
                self._closing = False
            thread.start()

    async def stop(self) -> None:
        async with self._lifecycle_lock:
            with self._state_lock:
                thread = self._thread
            if thread is None:
                return
            with self._state_lock:
                self._closing = True
            # The worker polls the queue while closing and exits only after all
            # accepted callers and their corrections have settled. No sentinel can
            # be stranded behind a full queue or consumed by a later worker.
            await asyncio.to_thread(thread.join)
            with self._state_lock:
                self._started = False
                if not thread.is_alive():
                    self._thread = None
                    self._closing = False

    async def execute(
        self,
        sql: str,
        parameters: Sequence[Any] = (),
        *,
        timeout: float | None = None,
        correct_request_id: str | None = None,
    ) -> int:
        budget = self.ack_timeout if timeout is None else max(0.001, timeout)
        command = _Command(
            sql=sql,
            parameters=tuple(parameters),
            future=concurrent.futures.Future(),
            deadline=time.monotonic() + budget,
            correct_request_id=correct_request_id,
        )
        with self._state_lock:
            if not self._is_running_locked():
                raise StoreUnavailable("event writer is not running")
            self._active_calls += 1
            try:
                self._queue.put_nowait(command)
            except queue.Full as exc:
                if correct_request_id is not None:
                    self._corrections.put(correct_request_id)
                    self._pending_corrections += 1
                self._active_calls -= 1
                raise StoreUnavailable("event writer queue is full") from exc
        wrapped = asyncio.wrap_future(command.future)
        wrapped.add_done_callback(
            lambda future: future.exception() if not future.cancelled() else None
        )
        try:
            return await asyncio.wait_for(asyncio.shield(wrapped), timeout=budget)
        except TimeoutError as exc:
            if command.future.done() and command.finished_at is not None:
                if command.finished_at <= command.deadline:
                    try:
                        return command.future.result()
                    except sqlite3.Error as store_exc:
                        raise StoreUnavailable(f"event write failed: {store_exc}") from store_exc
                try:
                    command.future.result()
                except BaseException:
                    pass
            command.cancelled.set()
            if correct_request_id is not None:
                self._queue_correction(correct_request_id)
            raise StoreUnavailable("event write acknowledgement timed out") from exc
        except asyncio.CancelledError:
            command.cancelled.set()
            if correct_request_id is not None:
                self._queue_correction(correct_request_id)
            raise
        except sqlite3.Error as exc:
            command.cancelled.set()
            if correct_request_id is not None:
                self._queue_correction(correct_request_id)
            raise StoreUnavailable(f"event write failed: {exc}") from exc
        finally:
            with self._state_lock:
                self._active_calls -= 1

    def _queue_correction(self, request_id: str) -> None:
        with self._state_lock:
            self._corrections.put(request_id)
            self._pending_corrections += 1

    @staticmethod
    def _apply_correction(connection: sqlite3.Connection, request_id: str) -> None:
        connection.execute(
            "UPDATE requests SET status_code=503, lifecycle='logging_failed', "
            "logging_failed=1, terminal_reason='completion_ack_timeout', "
            "completed_at=COALESCE(completed_at, ?) WHERE request_id=? "
            "AND lifecycle IN ('pending', 'completed', 'logging_failed')",
            (time.time(), request_id),
        )
        connection.commit()

    def _apply_pending_corrections(self, connection: sqlite3.Connection) -> None:
        while True:
            try:
                request_id = self._corrections.get_nowait()
            except queue.Empty:
                return
            try:
                self._apply_correction(connection, request_id)
            except sqlite3.Error:
                try:
                    connection.rollback()
                except sqlite3.Error:
                    pass
                # Retain the safety update for the next bounded busy-timeout
                # window instead of expiring or dropping it under SQLite pressure.
                self._corrections.put(request_id)
                return
            with self._state_lock:
                self._pending_corrections -= 1

    def _discard_stale_commands(self) -> None:
        """Fail leftovers only after the previous consumer has fully exited."""
        while True:
            try:
                command = self._queue.get_nowait()
            except queue.Empty:
                return
            if command is not None and not command.future.done():
                command.future.set_exception(StoreUnavailable("event writer restarted"))
            self._queue.task_done()

    def _run(self) -> None:
        connection: sqlite3.Connection | None = None
        try:
            connection = open_database(self.path)
            while True:
                self._apply_pending_corrections(connection)
                try:
                    command = self._queue.get(timeout=0.02)
                except queue.Empty:
                    with self._state_lock:
                        should_stop = (
                            self._closing
                            and self._active_calls == 0
                            and self._pending_corrections == 0
                        )
                    if should_stop and self._queue.empty() and self._corrections.empty():
                        with self._state_lock:
                            if (
                                self._closing
                                and self._active_calls == 0
                                and self._pending_corrections == 0
                                and self._queue.empty()
                                and self._corrections.empty()
                            ):
                                break
                    continue
                try:
                    if command.cancelled.is_set() or time.monotonic() > command.deadline:
                        if command.correct_request_id:
                            self._apply_correction(connection, command.correct_request_id)
                        raise StoreUnavailable("event write expired before execution")
                    cursor = connection.execute(command.sql, command.parameters)
                    connection.commit()
                    rowcount = cursor.rowcount
                    late = command.cancelled.is_set() or time.monotonic() > command.deadline
                    if late and command.correct_request_id:
                        self._apply_correction(connection, command.correct_request_id)
                    if late:
                        command.finished_at = time.monotonic()
                        raise StoreUnavailable("event write completed after its acknowledgement deadline")
                    command.finished_at = time.monotonic()
                    command.future.set_result(rowcount)
                except BaseException as exc:  # keep the writer alive after a bad command
                    try:
                        connection.rollback()
                    except sqlite3.Error:
                        pass
                    if command.correct_request_id is not None:
                        self._queue_correction(command.correct_request_id)
                    command.finished_at = time.monotonic()
                    if not command.future.done():
                        command.future.set_exception(exc)
                finally:
                    self._queue.task_done()
        except BaseException as exc:  # surfaced on the next bounded write
            while True:
                try:
                    command = self._queue.get_nowait()
                except queue.Empty:
                    break
                if command is not None and not command.future.done():
                    command.future.set_exception(StoreUnavailable(str(exc)))
                self._queue.task_done()
        finally:
            if connection is not None:
                connection.close()
