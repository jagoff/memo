"""Temporal fact-edge sidecar store.

This is memo's first explicit substrate for facts that can change over time.
Markdown memories remain the source of truth; this store is a queryable,
rebuildable sidecar that records extracted or manually asserted fact edges with
validity windows.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import weakref
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .connection import _ConnectionMixin


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _normalize_ts(value: str | datetime | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat()


def _fact_id(
    *,
    subject: str,
    predicate: str,
    object_: str,
    source_record_id: str | None,
    valid_at: str,
) -> str:
    payload = json.dumps(
        {
            "subject": subject,
            "predicate": predicate,
            "object": object_,
            "source_record_id": source_record_id or "",
            "valid_at": valid_at,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _row_to_fact(row: Any) -> dict[str, Any]:
    data = dict(row)
    for key in ("provenance_json", "metadata_json"):
        out_key = key.removesuffix("_json")
        raw = data.pop(key, None)
        if not raw:
            data[out_key] = {}
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {}
        data[out_key] = parsed if isinstance(parsed, dict) else {}
    return data


class FactEdgeStore(_ConnectionMixin):
    """sqlite-backed temporal fact edge store.

    Edges are live at ``as_of`` when ``valid_at <= as_of`` and neither
    ``invalid_at`` nor ``expired_at`` has passed.
    """

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self._local = threading.local()
        self._conn_holders: weakref.WeakSet[object] = weakref.WeakSet()
        self._conn_holders_lock = threading.Lock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _load_vec0(self, conn: sqlite3.Connection) -> None:
        # Fact edges are scalar sqlite rows; no vector extension is needed.
        return None

    def _ensure_schema(self) -> None:
        conn = self._conn
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS fact_edges (
                id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                source_record_id TEXT,
                valid_at TEXT NOT NULL,
                invalid_at TEXT,
                expired_at TEXT,
                confidence REAL NOT NULL DEFAULT 1.0,
                provenance_json TEXT,
                metadata_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_fact_edges_subject
                ON fact_edges(subject, valid_at);
            CREATE INDEX IF NOT EXISTS idx_fact_edges_predicate
                ON fact_edges(predicate, valid_at);
            CREATE INDEX IF NOT EXISTS idx_fact_edges_object
                ON fact_edges(object, valid_at);
            CREATE INDEX IF NOT EXISTS idx_fact_edges_source
                ON fact_edges(source_record_id);
            """
        )
        conn.commit()

    def upsert_fact(
        self,
        *,
        subject: str,
        predicate: str,
        object: str,
        source_record_id: str | None = None,
        valid_at: str | datetime | None = None,
        invalid_at: str | datetime | None = None,
        expired_at: str | datetime | None = None,
        confidence: float = 1.0,
        provenance: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        supersedes: list[str] | None = None,
    ) -> str:
        subject = subject.strip()
        predicate = predicate.strip()
        object_ = object.strip()
        if not subject:
            raise ValueError("fact subject cannot be empty")
        if not predicate:
            raise ValueError("fact predicate cannot be empty")
        if not object_:
            raise ValueError("fact object cannot be empty")
        valid = _normalize_ts(valid_at) or _now_iso()
        invalid = _normalize_ts(invalid_at)
        expired = _normalize_ts(expired_at)
        now = _now_iso()
        id_ = _fact_id(
            subject=subject,
            predicate=predicate,
            object_=object_,
            source_record_id=source_record_id,
            valid_at=valid,
        )
        with self._tx() as cx:
            cx.execute(
                "INSERT INTO fact_edges (id, subject, predicate, object, source_record_id, "
                "valid_at, invalid_at, expired_at, confidence, provenance_json, metadata_json, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET subject=excluded.subject, "
                "predicate=excluded.predicate, object=excluded.object, "
                "source_record_id=excluded.source_record_id, valid_at=excluded.valid_at, "
                "invalid_at=excluded.invalid_at, expired_at=excluded.expired_at, "
                "confidence=excluded.confidence, provenance_json=excluded.provenance_json, "
                "metadata_json=excluded.metadata_json, updated_at=excluded.updated_at",
                (
                    id_,
                    subject,
                    predicate,
                    object_,
                    source_record_id,
                    valid,
                    invalid,
                    expired,
                    float(confidence),
                    json.dumps(provenance or {}, ensure_ascii=False, sort_keys=True),
                    json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
                    now,
                    now,
                ),
            )
            for old_id in supersedes or []:
                cx.execute(
                    "UPDATE fact_edges SET invalid_at = COALESCE(invalid_at, ?), "
                    "updated_at = ? WHERE id = ?",
                    (valid, now, old_id),
                )
        return id_

    def get(self, id_: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM fact_edges WHERE id = ?", (id_,)).fetchone()
        return _row_to_fact(row) if row else None

    def query(
        self,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
        source_record_id: str | None = None,
        as_of: str | datetime | None = None,
        include_inactive: bool = False,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        target = _normalize_ts(as_of) or _now_iso()
        params: list[Any] = [
            subject,
            subject,
            predicate,
            predicate,
            object,
            object,
            source_record_id,
            source_record_id,
            1 if include_inactive else 0,
            target,
            target,
            target,
            max(1, int(limit)),
        ]
        rows = self._conn.execute(
            "SELECT * FROM fact_edges WHERE "
            "(? IS NULL OR subject = ?) AND "
            "(? IS NULL OR predicate = ?) AND "
            "(? IS NULL OR object = ?) AND "
            "(? IS NULL OR source_record_id = ?) AND "
            "(? = 1 OR (valid_at <= ? AND "
            "(invalid_at IS NULL OR invalid_at > ?) AND "
            "(expired_at IS NULL OR expired_at > ?))) "
            "ORDER BY valid_at DESC, confidence DESC, updated_at DESC LIMIT ?",
            params,
        ).fetchall()
        return [_row_to_fact(row) for row in rows]

    def invalidate(self, id_: str, *, invalid_at: str | datetime | None = None) -> bool:
        ts = _normalize_ts(invalid_at) or _now_iso()
        with self._tx() as cx:
            cur = cx.execute(
                "UPDATE fact_edges SET invalid_at = ?, updated_at = ? WHERE id = ?",
                (ts, _now_iso(), id_),
            )
        return cur.rowcount > 0

    def delete_for_source(self, source_record_id: str) -> int:
        """Delete all fact edges derived from one source record."""
        with self._tx() as cx:
            cur = cx.execute(
                "DELETE FROM fact_edges WHERE source_record_id = ?",
                (source_record_id,),
            )
        return int(cur.rowcount)

    def clear(self) -> int:
        """Delete every fact edge. Used by markdown-derived rebuilds."""
        with self._tx() as cx:
            cur = cx.execute("DELETE FROM fact_edges")
        return int(cur.rowcount)


__all__ = ["FactEdgeStore"]
