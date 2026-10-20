"""Focused tests for durable request and shared SQLite contracts."""

import asyncio

import pytest

import backend.event_store as event_store
from backend.event_store import (
    SQLiteEventWriter,
    StoreUnavailable,
    archive_old_requests,
    canonical_json,
    canonical_record,
    checkpoint_database,
    feature_fingerprint,
    initialize_database,
    open_database,
)


def test_canonical_record_normalizes_numbers_and_drops_unmodeled_fields():
    names = ["OverallQual", "GrLivArea", "Neighborhood"]
    numeric = {"OverallQual", "GrLivArea"}
    left = canonical_record(
        {"OverallQual": 7, "GrLivArea": "1700", "Neighborhood": "CollgCr", "extra": 99},
        feature_names=names,
        numeric_features=numeric,
    )
    right = canonical_record(
        {"OverallQual": 7.0, "GrLivArea": 1700.0, "Neighborhood": "CollgCr"},
        feature_names=names,
        numeric_features=numeric,
    )

    assert left == right
    assert left["OverallQual"] == 7.0
    assert "extra" not in left
    assert feature_fingerprint(
        left, feature_names=names, numeric_features=numeric
    ) == feature_fingerprint(right, feature_names=names, numeric_features=numeric)


def test_canonical_record_normalizes_missing_and_rejects_nonfinite_values():
    names = ["LotArea", "Neighborhood"]
    missing = canonical_record(
        {"LotArea": "", "Neighborhood": None},
        feature_names=names,
        numeric_features={"LotArea"},
    )
    assert missing == {"LotArea": None, "Neighborhood": None}
    dataframe_missing = canonical_record(
        {"LotArea": float("nan"), "Neighborhood": float("nan")},
        feature_names=names,
        numeric_features={"LotArea"},
    )
    assert dataframe_missing == missing
    with pytest.raises(ValueError, match="must be finite"):
        canonical_record(
            {"LotArea": float("inf")},
            feature_names=names,
            numeric_features={"LotArea"},
        )
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_database_schema_and_acknowledged_writer_persist_across_restart(tmp_path):
    database = tmp_path / "events.sqlite3"
    initialize_database(database)
    connection = open_database(database)
    try:
        names = {
            table[0]
            for table in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert {"requests", "labels", "runs", "activations", "trainer_leases"} <= names
        request_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(requests)")
        }
        assert {
            "request_id",
            "received_at",
            "status_code",
            "lifecycle",
            "body_text",
            "features_json",
            "model_version",
            "cache_outcome",
            "logging_failed",
        } <= request_columns
    finally:
        connection.close()

    async def first_writer():
        writer = SQLiteEventWriter(database)
        await writer.start()
        changed = await writer.execute(
            "INSERT INTO requests "
            "(request_id, received_at, method, path, lifecycle, parse_state) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("request-1", 100.0, "POST", "/predict", "pending", "awaiting_body"),
        )
        assert changed == 1
        await writer.stop()

    asyncio.run(first_writer())

    async def second_writer():
        writer = SQLiteEventWriter(database)
        await writer.start()
        changed = await writer.execute(
            "UPDATE requests SET status_code=?, lifecycle=?, completed_at=? "
            "WHERE request_id=?",
            (200, "completed", 101.0, "request-1"),
        )
        assert changed == 1
        await writer.stop()

    asyncio.run(second_writer())
    connection = open_database(database)
    try:
        row = connection.execute(
            "SELECT status_code, lifecycle, completed_at FROM requests WHERE request_id=?",
            ("request-1",),
        ).fetchone()
        assert tuple(row) == (200, "completed", 101.0)
    finally:
        connection.close()


def test_explicit_archive_is_bounded_and_preserves_pending_or_observation_rows(tmp_path):
    source = tmp_path / "source.sqlite3"
    archive = tmp_path / "archive.sqlite3"
    initialize_database(source)
    connection = open_database(source)
    try:
        connection.executemany(
            "INSERT INTO requests "
            "(request_id, received_at, completed_at, method, path, status_code, lifecycle, "
            "dataset_sha256, source_id) VALUES (?, ?, ?, 'POST', '/predict', 200, ?, ?, ?)",
            [
                ("old-unlabelled", 1.0, 10.0, "completed", None, None),
                ("old-observation", 1.0, 10.0, "completed", "a" * 64, 42),
                ("pending", 1.0, None, "pending", None, None),
            ],
        )
        connection.execute(
            "INSERT INTO labels "
            "(observation_id, request_id, dataset_sha256, source_id, features_json, "
            "feature_fingerprint, sale_price, available_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "observation-42",
                "old-observation",
                "a" * 64,
                42,
                "{}",
                "fingerprint",
                100000.0,
                11.0,
            ),
        )
        connection.commit()
    finally:
        connection.close()

    assert archive_old_requests(source, archive, before=100.0, limit=1) == 1
    assert archive_old_requests(source, archive, before=100.0, limit=1) == 0
    assert len(checkpoint_database(source)) == 3

    live = open_database(source)
    archived = open_database(archive)
    try:
        live_ids = {
            row[0] for row in live.execute("SELECT request_id FROM requests")
        }
        archive_ids = {
            row[0] for row in archived.execute("SELECT request_id FROM requests")
        }
        assert live_ids == {"old-observation", "pending"}
        assert archive_ids == {"old-unlabelled"}
    finally:
        live.close()
        archived.close()


def test_writer_queue_cannot_be_unbounded_and_late_completion_is_marked_503(tmp_path):
    database = tmp_path / "late.sqlite3"
    initialize_database(database)
    connection = open_database(database)
    connection.execute(
        "INSERT INTO requests (request_id, received_at, method, path, lifecycle) "
        "VALUES (?, ?, ?, ?, ?)",
        ("late-request", 1.0, "GET", "/health", "pending"),
    )
    connection.commit()

    async def run_writer():
        writer = SQLiteEventWriter(database, queue_size=1, ack_timeout=0.05)
        assert writer._queue.maxsize == 1
        assert SQLiteEventWriter(database, queue_size=10_000)._queue.maxsize == 256
        await writer.start()
        connection.execute("BEGIN IMMEDIATE")
        completion = asyncio.create_task(
            writer.execute(
                "UPDATE requests SET status_code=200, lifecycle='completed' "
                "WHERE request_id=?",
                ("late-request",),
                correct_request_id="late-request",
            )
        )
        for _ in range(20):
            await asyncio.sleep(0.001)
            if writer._queue.empty():
                break
        assert writer._queue.empty(), "completion should be executing under the held DB lock"
        filler = asyncio.create_task(
            writer.execute("UPDATE runs SET status=status WHERE run_id=?", ("missing",), timeout=1)
        )
        await asyncio.sleep(0.06)
        try:
            with pytest.raises(StoreUnavailable, match="timed out"):
                await completion
            assert writer._queue.full(), "late correction should face a full bounded queue"
        finally:
            connection.rollback()
        assert await filler == 0
        await writer.stop()

    try:
        asyncio.run(run_writer())
    finally:
        connection.close()
    connection = open_database(database)
    try:
        row = connection.execute(
            "SELECT status_code, lifecycle, logging_failed FROM requests WHERE request_id=?",
            ("late-request",),
        ).fetchone()
        assert tuple(row) == (503, "logging_failed", 1)
    finally:
        connection.close()


def test_concurrent_writer_stop_drains_before_restart_without_blocking_loop(tmp_path):
    database = tmp_path / "writer-stop.sqlite3"
    initialize_database(database)
    connection = open_database(database)

    async def run_writer():
        writer = SQLiteEventWriter(database, queue_size=1, ack_timeout=2.0)
        await writer.start()
        connection.execute("BEGIN IMMEDIATE")
        command = asyncio.create_task(
            writer.execute(
                "UPDATE runs SET status=status WHERE run_id=?",
                ("missing",),
                timeout=2.0,
            )
        )
        for _ in range(100):
            await asyncio.sleep(0.001)
            if writer._queue.empty():
                break
        first_stop = asyncio.create_task(writer.stop())
        second_stop = asyncio.create_task(writer.stop())
        heartbeat = asyncio.create_task(asyncio.sleep(0.02))
        await heartbeat
        assert not first_stop.done()
        assert not second_stop.done()
        assert not command.done()
        connection.rollback()
        assert await command == 0
        await asyncio.gather(first_stop, second_stop)
        assert not writer.running

        await writer.start()
        assert await writer.execute(
            "UPDATE runs SET status=status WHERE run_id=?", ("still-missing",)
        ) == 0
        await writer.stop()

    try:
        asyncio.run(run_writer())
    finally:
        connection.close()


def test_cancellation_after_completion_commit_queues_nonexpiring_correction(
    tmp_path, monkeypatch
):
    database = tmp_path / "cancel-after-commit.sqlite3"
    initialize_database(database)
    connection = open_database(database)
    connection.execute(
        "INSERT INTO requests (request_id, received_at, method, path, lifecycle) "
        "VALUES ('cancelled-completion', 1.0, 'POST', '/predict', 'pending')"
    )
    connection.commit()
    connection.close()

    async def run_writer():
        writer = SQLiteEventWriter(database, ack_timeout=1.0)
        await writer.start()
        original_wait_for = asyncio.wait_for

        async def cancel_after_ack(awaitable, timeout):
            await original_wait_for(awaitable, timeout)
            raise asyncio.CancelledError

        monkeypatch.setattr(event_store.asyncio, "wait_for", cancel_after_ack)
        try:
            with pytest.raises(asyncio.CancelledError):
                await writer.execute(
                    "UPDATE requests SET completed_at=?, status_code=200, lifecycle='completed' "
                    "WHERE request_id=?",
                    (2.0, "cancelled-completion"),
                    correct_request_id="cancelled-completion",
                )
        finally:
            monkeypatch.setattr(event_store.asyncio, "wait_for", original_wait_for)
        await writer.stop()

    asyncio.run(run_writer())
    connection = open_database(database)
    try:
        row = connection.execute(
            "SELECT status_code, lifecycle, logging_failed FROM requests "
            "WHERE request_id='cancelled-completion'"
        ).fetchone()
        assert tuple(row) == (503, "logging_failed", 1)
    finally:
        connection.close()
