"""SQLite schema for rebuildable operational-v2 projections."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from memo.errors import OperationalError, OperationalErrorCode

OPERATIONAL_VIEW_SCHEMA_VERSION = 2

_DDL = """
CREATE TABLE IF NOT EXISTS view_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS applied_events (
    event_id TEXT PRIMARY KEY,
    origin_device TEXT NOT NULL,
    origin_sequence INTEGER NOT NULL,
    event_hash TEXT NOT NULL,
    event_json TEXT NOT NULL,
    applied_at TEXT NOT NULL,
    UNIQUE(origin_device, origin_sequence)
) STRICT;

CREATE TABLE IF NOT EXISTS origin_cursors (
    origin_device TEXT PRIMARY KEY,
    origin_sequence INTEGER NOT NULL,
    event_hash TEXT NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS idempotency (
    scope TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    event_id TEXT NOT NULL,
    result_json TEXT NOT NULL,
    PRIMARY KEY(scope, idempotency_key),
    FOREIGN KEY(event_id) REFERENCES applied_events(event_id)
) STRICT;

CREATE TABLE IF NOT EXISTS focus (
    project TEXT PRIMARY KEY,
    row_json TEXT NOT NULL,
    updated_event_id TEXT NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS handoffs (
    id TEXT PRIMARY KEY,
    project TEXT NOT NULL,
    row_json TEXT NOT NULL,
    updated_event_id TEXT NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS attention (
    id TEXT PRIMARY KEY,
    project TEXT NOT NULL,
    row_json TEXT NOT NULL,
    updated_event_id TEXT NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS conflicts (
    id TEXT PRIMARY KEY,
    row_json TEXT NOT NULL,
    updated_event_id TEXT NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS outcomes (
    task_id TEXT PRIMARY KEY,
    row_json TEXT NOT NULL,
    updated_event_id TEXT NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    project TEXT NOT NULL,
    workspace TEXT NOT NULL,
    status TEXT NOT NULL,
    row_json TEXT NOT NULL,
    updated_event_id TEXT NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS session_local_artifacts (
    session_id TEXT NOT NULL,
    artifact_uri TEXT NOT NULL,
    row_json TEXT NOT NULL,
    PRIMARY KEY(session_id, artifact_uri)
) STRICT;

CREATE TABLE IF NOT EXISTS durable_outbox (
    promotion_id TEXT PRIMARY KEY,
    operation_key TEXT NOT NULL,
    status TEXT NOT NULL,
    attempt_count INTEGER NOT NULL,
    retry_at TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    row_json TEXT NOT NULL,
    updated_event_id TEXT NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS durable_outbox_ready
ON durable_outbox(status, retry_at, promotion_id);

CREATE TABLE IF NOT EXISTS quarantined_events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    reason TEXT NOT NULL,
    event_json TEXT NOT NULL,
    quarantined_at TEXT NOT NULL
) STRICT;
"""


def _storage_failure(message: str) -> OperationalError:
    return OperationalError(
        OperationalErrorCode.STORAGE_UNAVAILABLE,
        message,
        retryable=False,
    )


def connect_operational_db(path: Path) -> sqlite3.Connection:
    """Open one derived operational database with deterministic settings."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        destination,
        timeout=30.0,
        isolation_level=None,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = FULL")
    return connection


def ensure_operational_schema(connection: sqlite3.Connection) -> None:
    """Create the current schema or make the rebuildable v1 history upgradeable."""
    connection.execute(
        "CREATE TABLE IF NOT EXISTS view_meta "
        "(key TEXT PRIMARY KEY, value TEXT NOT NULL) STRICT"
    )
    row = connection.execute(
        "SELECT value FROM view_meta WHERE key = 'schema_version'"
    ).fetchone()
    if row is not None and row["value"] == "1":
        applied_table = connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table' AND name = 'applied_events'
            """
        ).fetchone()
        if applied_table is not None:
            columns = {
                str(column["name"])
                for column in connection.execute("PRAGMA table_info(applied_events)")
            }
            if "event_json" not in columns:
                connection.execute(
                    "ALTER TABLE applied_events "
                    "ADD COLUMN event_json TEXT NOT NULL DEFAULT ''"
                )
        connection.execute(
            """
            INSERT INTO view_meta(key, value) VALUES('rebuild_required', '1')
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """
        )
        connection.execute(
            "UPDATE view_meta SET value = ? WHERE key = 'schema_version'",
            (str(OPERATIONAL_VIEW_SCHEMA_VERSION),),
        )
        row = connection.execute(
            "SELECT value FROM view_meta WHERE key = 'schema_version'"
        ).fetchone()
    if row is not None and row["value"] != str(OPERATIONAL_VIEW_SCHEMA_VERSION):
        raise _storage_failure(
            f"unsupported operational view schema: {row['value']}"
        )
    connection.executescript(_DDL)
    connection.execute(
        "INSERT OR IGNORE INTO view_meta(key, value) VALUES('schema_version', ?)",
        (str(OPERATIONAL_VIEW_SCHEMA_VERSION),),
    )
    history_incomplete = (
        connection.execute(
            "SELECT 1 FROM applied_events WHERE event_json = '' LIMIT 1"
        ).fetchone()
        is not None
    )
    if history_incomplete:
        connection.execute(
            """
            INSERT INTO view_meta(key, value) VALUES('rebuild_required', '1')
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """
        )
    else:
        connection.execute(
            """
            INSERT OR IGNORE INTO view_meta(key, value)
            VALUES('rebuild_required', '0')
            """
        )


__all__ = [
    "OPERATIONAL_VIEW_SCHEMA_VERSION",
    "connect_operational_db",
    "ensure_operational_schema",
]
