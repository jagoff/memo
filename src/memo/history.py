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
    delta_json  TEXT                  -- JSON-encoded {field: [old, new]} for updates
);
```

`delta_json` is `null` for `save` (initial creation — no prior state)
and `delete` (the "after" state is gone). For `update` it carries only
the fields that changed, with `[old, new]` pairs.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    op          TEXT NOT NULL,
    record_id   TEXT NOT NULL,
    title       TEXT,
    type        TEXT,
    delta_json  TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_record ON events(record_id);
CREATE INDEX IF NOT EXISTS idx_events_ts     ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_op     ON events(op);
"""


class HistoryStore:
    """Append-only audit log. Never raises on log failure — losing an
    audit row is preferable to crashing a save/update/delete operation.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), timeout=10.0, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.executescript(_SCHEMA_DDL)

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def log_save(self, *, ts: str, record_id: str, title: str, type_: str) -> None:
        try:
            with self._tx() as cx:
                cx.execute(
                    "INSERT INTO events (ts, op, record_id, title, type) VALUES (?, ?, ?, ?, ?)",
                    (ts, "save", record_id, title, type_),
                )
        except Exception:
            pass

    def log_update(
        self, *, ts: str, record_id: str, title: str, type_: str,
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
                    "INSERT INTO events (ts, op, record_id, title, type, delta_json) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (ts, "update", record_id, title, type_,
                     json.dumps(payload, default=str, ensure_ascii=False)),
                )
        except Exception:
            pass

    def log_delete(self, *, ts: str, record_id: str, title: str, type_: str) -> None:
        try:
            with self._tx() as cx:
                cx.execute(
                    "INSERT INTO events (ts, op, record_id, title, type) VALUES (?, ?, ?, ?, ?)",
                    (ts, "delete", record_id, title, type_),
                )
        except Exception:
            pass

    def list_recent(
        self, *, limit: int = 50, op: str | None = None, record_id: str | None = None,
    ) -> list[dict[str, Any]]:
        sql = "SELECT id, ts, op, record_id, title, type, delta_json FROM events"
        clauses: list[str] = []
        params: list[Any] = []
        if op:
            clauses.append("op = ?")
            params.append(op)
        if record_id:
            clauses.append("record_id = ?")
            params.append(record_id)
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
                except Exception:
                    d["delta"] = None
            else:
                d["delta"] = None
            d.pop("delta_json", None)
            out.append(d)
        return out

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


__all__ = ["HistoryStore"]
