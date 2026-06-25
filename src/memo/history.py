"""History/audit log — separate sqlite DB recording every save/update/delete.

Why a separate DB (`history.db`) instead of a table inside `memvec.db`:

- The vec store is hot-path read-heavy under search; the history is
  append-only writes. Splitting the WAL avoids history writes blocking
  vec reads under contention.
- A history corruption never threatens the index (which is the
  authoritative retrieval surface). Reset is safe: `rm history.db`.

## Schema

```
CREATE TABLE events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,        -- ISO8601 with offset
    op          TEXT NOT NULL,        -- save / update / delete
    record_id   TEXT NOT NULL,        -- references meta.id
    title       TEXT,                 -- snapshot of title at time of op
    type        TEXT,                 -- snapshot of type
    delta_json  TEXT,                 -- JSON-encoded {field: [old, new]} for updates
    device_id   TEXT                  -- ID of the device that created the event
);
```

`delta_json` is `null` for `save` (initial creation — no prior state)
and `delete` (the "after" state is gone). For `update` it carries only
the fields that changed, with `[old, new]` pairs.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    op          TEXT NOT NULL,
    record_id   TEXT NOT NULL,
    title       TEXT,
    type        TEXT,
    delta_json  TEXT,
    device_id   TEXT
);
CREATE TABLE IF NOT EXISTS sync_state (
    device_id   TEXT PRIMARY KEY,
    last_lsn    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_record ON events(record_id);
CREATE INDEX IF NOT EXISTS idx_events_ts     ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_op     ON events(op);
CREATE INDEX IF NOT EXISTS idx_events_device ON events(device_id);
"""


class HistoryStore:
    """Append-only audit log. Never raises on log failure — losing an
    audit row is preferable to crashing a save/update/delete operation.
    """

    def __init__(self, db_path: Path, device_id: str | None = None) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Unique ID for this device to distinguish its events during sync.
        self.device_id = device_id or "unknown"

        self._conn = sqlite3.connect(str(db_path), timeout=10.0, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # WAL = concurrent readers + a writer; one shared connection across the
        # FastMCP threadpool, so serialise `_tx()` — two threads issuing
        # BEGIN IMMEDIATE on the same connection raise "transaction within a
        # transaction" and silently drop the audit row.
        with suppress(sqlite3.Error):
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._tx_lock = threading.Lock()
        with self._conn:
            # Additive migration: an `events` table created before `device_id`
            # existed must gain the column before _SCHEMA_DDL's
            # `idx_events_device` index references events(device_id) — otherwise
            # executescript raises "no such column: device_id" and Memory()
            # construction crashes. A fresh DB has no events table yet, so the
            # ALTER is skipped and executescript creates it with the column.
            existing = self._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='events'"
            ).fetchone()
            if existing is not None:
                cols = {row[1] for row in self._conn.execute("PRAGMA table_info(events)")}
                if "device_id" not in cols:
                    self._conn.execute("ALTER TABLE events ADD COLUMN device_id TEXT")
            self._conn.executescript(_SCHEMA_DDL)
        # Count of swallowed log failures. Surfaced via `error_count` and
        # `Memory.stats()` so silent audit drops don't disappear from sight.
        self._error_count = 0

    @property
    def error_count(self) -> int:
        """Total log_* failures swallowed since construction."""
        return self._error_count

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._tx_lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def log_save(
        self,
        *,
        ts: str,
        record_id: str,
        title: str,
        type_: str,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        # `provenance` is an optional bag of synapse_* / agent_* keys
        # (trace_id, route_reason, write_policy_schema, agent_id,
        # agent_signature, write_target). Stored as a {"_provenance": {...}}
        # envelope inside `delta_json` so the events schema stays unchanged
        # and `list_recent` exposes it transparently.
        delta_json = None
        if provenance:
            try:
                delta_json = json.dumps(
                    {"_provenance": provenance},
                    default=str,
                    ensure_ascii=False,
                )
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                _log.warning(
                    "history log_save provenance encode failed (id=%s): %s",
                    record_id[:8],
                    exc,
                )
        try:
            with self._tx() as cx:
                cx.execute(
                    "INSERT INTO events (ts, op, record_id, title, type, delta_json, device_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (ts, "save", record_id, title, type_, delta_json, self.device_id),
                )
        except sqlite3.Error as exc:
            self._error_count += 1
            _log.warning("history log_save failed (id=%s): %s", record_id[:8], exc)

    def log_update(
        self,
        *,
        ts: str,
        record_id: str,
        title: str,
        type_: str,
        delta: dict[str, tuple[Any, Any]],
    ) -> None:
        try:
            # Normalise delta to JSON-friendly: tuples → lists; coerce
            # tags lists already JSON-friendly. Skip empty diff.
            if not delta:
                return
            payload = {k: [v[0], v[1]] for k, v in delta.items()}
            with self._tx() as cx:
                cx.execute(
                    "INSERT INTO events (ts, op, record_id, title, type, delta_json, device_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        ts,
                        "update",
                        record_id,
                        title,
                        type_,
                        json.dumps(payload, default=str, ensure_ascii=False),
                        self.device_id,
                    ),
                )
        except (sqlite3.Error, TypeError, ValueError) as exc:
            self._error_count += 1
            _log.warning("history log_update failed (id=%s): %s", record_id[:8], exc)

    def log_delete(self, *, ts: str, record_id: str, title: str, type_: str) -> None:
        try:
            with self._tx() as cx:
                cx.execute(
                    "INSERT INTO events (ts, op, record_id, title, type, device_id) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (ts, "delete", record_id, title, type_, self.device_id),
                )
        except sqlite3.Error as exc:
            self._error_count += 1
            _log.warning("history log_delete failed (id=%s): %s", record_id[:8], exc)

    def list_recent(
        self,
        *,
        limit: int = 50,
        op: str | None = None,
        record_id: str | None = None,
        after_lsn: int | None = None,
        device_id: str | None = None,
    ) -> list[dict[str, Any]]:
        sql = "SELECT id, ts, op, record_id, title, type, delta_json, device_id FROM events"
        clauses: list[str] = []
        params: list[Any] = []
        if op:
            clauses.append("op = ?")
            params.append(op)
        if record_id:
            clauses.append("record_id = ?")
            params.append(record_id)
        if after_lsn:
            clauses.append("id > ?")
            params.append(after_lsn)
        if device_id:
            clauses.append("device_id = ?")
            params.append(device_id)

        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            if d.get("delta_json"):
                try:
                    d["delta"] = json.loads(d["delta_json"])
                except (ValueError, TypeError):
                    d["delta"] = None
            else:
                d["delta"] = None
            d.pop("delta_json", None)
            out.append(d)
        return out

    def get_sync_state(self, device_id: str) -> int:
        """Get the last processed LSN for a remote device."""
        row = self._conn.execute(
            "SELECT last_lsn FROM sync_state WHERE device_id = ?", (device_id,)
        ).fetchone()
        return row[0] if row else 0

    def update_sync_state(self, device_id: str, lsn: int) -> None:
        """Update the last processed LSN for a remote device."""
        try:
            with self._tx() as cx:
                cx.execute(
                    "INSERT INTO sync_state (device_id, last_lsn) VALUES (?, ?) "
                    "ON CONFLICT(device_id) DO UPDATE SET last_lsn = excluded.last_lsn",
                    (device_id, lsn),
                )
        except sqlite3.Error as exc:
            _log.warning("history update_sync_state failed (device=%s): %s", device_id, exc)

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]

    def close(self) -> None:
        with suppress(Exception):
            self._conn.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        with suppress(Exception):
            self.close()


__all__ = ["HistoryStore"]
