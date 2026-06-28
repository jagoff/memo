"""EpisodeStore — the derived semantic index over work *sessions*.

Phase 1 of memo's episodic-memory layer (see
`docs/superpowers/specs/2026-06-27-semantic-resume-design.md`). One embedding
per `(agent, session_id)` of the session's prompt-arc, so `memo resume` can
search the FULL session history by meaning instead of recency/substring.

This is a **derived** index: the transcript is the source of truth, this store
is rebuildable (`memo episodes index --rebuild`). It is deliberately separate
from the `.md`-sourced memory store (`VecStore`) — episodes are not durable
facts, carry no `.md`, and never enter the recall hook. Folds into `memvec.db`
when `MEMO_SINGLE_DB` is on; otherwise lives in its own `episodes.db`.
"""

from __future__ import annotations

import json
import threading
import weakref
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..sqlite_compat import import_sqlite_vec
from .connection import _ConnectionMixin

serialize_float32 = import_sqlite_vec().serialize_float32


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class EpisodeStore(_ConnectionMixin):
    """sqlite-vec backed session index. One vector per `(agent, session_id)`.

    Reuses the trinity-proven connection plumbing (thread-local conns, WAL,
    vec0 load, `_tx`) from `_ConnectionMixin`, with its own two-table schema.
    """

    def __init__(self, db_path: Path | str, dims: int) -> None:
        self.db_path = Path(db_path)
        self.dims = dims
        self._local = threading.local()
        self._conn_holders: weakref.WeakSet[object] = weakref.WeakSet()
        self._conn_holders_lock = threading.Lock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        # vec0 CREATE is not transactional — run it outside `_tx` (matches the
        # main store's schema setup).
        conn = self._conn
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS episode_vec USING vec0("
            f"id TEXT PRIMARY KEY, embedding FLOAT[{self.dims}] distance_metric=cosine)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS episode_meta ("
            "id TEXT PRIMARY KEY, agent TEXT, session_id TEXT, content_hash TEXT, "
            "cwd TEXT, updated_at TEXT, summary TEXT, resume_command TEXT, "
            "turn_count INTEGER, indexed_at TEXT)"
        )
        conn.commit()

    @staticmethod
    def _id(agent: str, session_id: str) -> str:
        return f"{agent}:{session_id}"

    def content_hash_for(self, agent: str, session_id: str) -> str | None:
        """The indexed content_hash for a session, or None if not indexed.

        Lets the indexer skip the (expensive) re-embed when a session is
        unchanged since its last index.
        """
        row = self._conn.execute(
            "SELECT content_hash FROM episode_meta WHERE id = ?",
            (self._id(agent, session_id),),
        ).fetchone()
        return str(row["content_hash"]) if row else None

    def upsert(
        self,
        *,
        agent: str,
        session_id: str,
        content_hash: str,
        embedding: list[float],
        cwd: str,
        updated_at: str,
        summary: str,
        resume_command: list[str],
        turn_count: int,
    ) -> None:
        if len(embedding) != self.dims:
            raise ValueError(
                f"Episode embedding dim mismatch: got {len(embedding)}, store expects "
                f"{self.dims}. Usually a swapped model / MEMO_EMBEDDER_DIMS. Run "
                f"`memo episodes index --rebuild` after restoring the correct model."
            )
        norm = sum(x * x for x in embedding) ** 0.5
        if norm != norm or not (0.5 < norm < 1.5):  # NaN or non-unit
            raise ValueError(f"Episode embedding not L2-normalised (norm={norm}).")
        id_ = self._id(agent, session_id)
        with self._tx() as cx:
            # vec0 has no upsert — delete + insert, atomic within the tx.
            cx.execute("DELETE FROM episode_vec WHERE id = ?", (id_,))
            cx.execute(
                "INSERT INTO episode_vec (id, embedding) VALUES (?, ?)",
                (id_, serialize_float32(embedding)),
            )
            cx.execute(
                "INSERT INTO episode_meta (id, agent, session_id, content_hash, cwd, "
                "updated_at, summary, resume_command, turn_count, indexed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET content_hash=excluded.content_hash, "
                "cwd=excluded.cwd, updated_at=excluded.updated_at, summary=excluded.summary, "
                "resume_command=excluded.resume_command, turn_count=excluded.turn_count, "
                "indexed_at=excluded.indexed_at",
                (
                    id_,
                    agent,
                    session_id,
                    content_hash,
                    cwd,
                    updated_at,
                    summary,
                    json.dumps(resume_command),
                    turn_count,
                    _now_iso(),
                ),
            )

    def search(self, embedding: list[float], k: int = 50) -> list[dict[str, Any]]:
        """Top-k sessions by cosine. Each row carries a `score` (1 - distance)."""
        if len(embedding) != self.dims:
            raise ValueError(
                f"Episode query embedding dim mismatch: got {len(embedding)}, expected {self.dims}"
            )
        rows = self._conn.execute(
            "SELECT m.agent, m.session_id, m.cwd, m.updated_at, m.summary, "
            "m.resume_command, m.turn_count, v.distance AS distance "
            "FROM episode_vec v JOIN episode_meta m ON v.id = m.id "
            "WHERE v.embedding MATCH ? AND v.k = ? ORDER BY distance ASC LIMIT ?",
            (serialize_float32(embedding), max(1, int(k)), max(1, int(k))),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            d["resume_command"] = json.loads(d.get("resume_command") or "[]")
            d["score"] = max(0.0, 1.0 - float(r["distance"]))
            out.append(d)
        return out

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM episode_meta").fetchone()
        return int(row["n"]) if row else 0

    def clear(self) -> None:
        """Drop every episode (for `--rebuild`)."""
        with self._tx() as cx:
            cx.execute("DELETE FROM episode_vec")
            cx.execute("DELETE FROM episode_meta")
