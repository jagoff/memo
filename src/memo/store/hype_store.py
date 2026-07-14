"""HypeStore — the derived question-space index for HyPE (Hypothetical
Questions for Expansion).

Each durable memory gets a handful of LLM-generated hypothetical questions,
embedded and indexed here. The read-path fold (`MEMO_HYPE_ENABLED`) can then
match a live query against the QUESTION space instead of only the memory-body
space, closing the gap where a memory's wording never anticipates how it will
later be asked about.

This is a **derived** sidecar, always collocated with the main index at
`cfg.db_path` (memvec.db) so its kNN shares the same file as `vec` — never a
separate `.db`. It is fully rebuildable from the nightly HyPE pass (watermark
by `body_hash`) and prunable against the live memory id set.
"""

from __future__ import annotations

import hashlib
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


def _question_id(memory_id: str, text: str) -> str:
    return hashlib.sha256(f"{memory_id}:{text}".encode()).hexdigest()[:32]


class HypeStore(_ConnectionMixin):
    """sqlite-vec backed index of hypothetical questions per memory.

    Reuses the trinity-proven connection plumbing (thread-local conns, WAL,
    vec0 load, `_tx`) from `_ConnectionMixin`, with its own two-table schema:
    `hype_vec` (vector-only, keyed by `question_id`) and `hype_questions`
    (question text + provenance, keyed by `question_id`). The knn joins the
    two by `question_id` to recover `memory_id` — vec0 doesn't carry the
    mutable metadata column itself, to avoid duplicating it there.
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
            f"CREATE VIRTUAL TABLE IF NOT EXISTS hype_vec USING vec0("
            f"question_id TEXT PRIMARY KEY, "
            f"embedding FLOAT[{self.dims}] distance_metric=cosine)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS hype_questions ("
            "question_id TEXT PRIMARY KEY, memory_id TEXT NOT NULL, "
            "question TEXT NOT NULL, body_hash TEXT NOT NULL, model TEXT NOT NULL, "
            "created_at TEXT NOT NULL)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hype_mem ON hype_questions(memory_id)")
        conn.commit()

    def replace_for_memory(
        self,
        memory_id: str,
        body_hash: str,
        model: str,
        questions: list[tuple[str, list[float]]],
    ) -> int:
        """Delete old rows for `memory_id`, insert `(question_text, embedding)` rows.

        `question_id = sha256(f"{memory_id}:{text}").hexdigest()[:32]`.
        Returns the inserted count.
        """
        created_at = _now_iso()
        with self._tx() as cx:
            old_ids = [
                str(r["question_id"])
                for r in cx.execute(
                    "SELECT question_id FROM hype_questions WHERE memory_id = ?",
                    (memory_id,),
                ).fetchall()
            ]
            for old_id in old_ids:
                cx.execute("DELETE FROM hype_vec WHERE question_id = ?", (old_id,))
            cx.execute("DELETE FROM hype_questions WHERE memory_id = ?", (memory_id,))
            for text, embedding in questions:
                if len(embedding) != self.dims:
                    raise ValueError(
                        f"HyPE question embedding dim mismatch: got {len(embedding)}, "
                        f"store expects {self.dims}. Usually a swapped model / "
                        f"MEMO_EMBEDDER_DIMS."
                    )
                qid = _question_id(memory_id, text)
                cx.execute(
                    "INSERT INTO hype_vec (question_id, embedding) VALUES (?, ?)",
                    (qid, serialize_float32(embedding)),
                )
                cx.execute(
                    "INSERT INTO hype_questions "
                    "(question_id, memory_id, question, body_hash, model, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (qid, memory_id, text, body_hash, model, created_at),
                )
        return len(questions)

    def body_hash_for(self, memory_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT body_hash FROM hype_questions WHERE memory_id = ? LIMIT 1",
            (memory_id,),
        ).fetchone()
        return str(row["body_hash"]) if row else None

    def knn(self, embedding: list[float], k: int) -> list[dict[str, Any]]:
        """`[{memory_id, question, score}]`, best question per memory, sorted
        by score desc. `score = 1.0 - distance`."""
        if len(embedding) != self.dims:
            raise ValueError(
                f"HyPE query embedding dim mismatch: got {len(embedding)}, expected {self.dims}"
            )
        k = max(1, int(k))
        rows = self._conn.execute(
            "SELECT q.memory_id, q.question, v.distance AS distance "
            "FROM hype_vec v JOIN hype_questions q ON v.question_id = q.question_id "
            "WHERE v.embedding MATCH ? AND v.k = ? ORDER BY distance ASC LIMIT ?",
            (serialize_float32(embedding), k, k),
        ).fetchall()
        best_by_memory: dict[str, dict[str, Any]] = {}
        for r in rows:
            score = max(0.0, 1.0 - float(r["distance"]))
            memory_id = str(r["memory_id"])
            existing = best_by_memory.get(memory_id)
            if existing is None or score > existing["score"]:
                best_by_memory[memory_id] = {
                    "memory_id": memory_id,
                    "question": str(r["question"]),
                    "score": score,
                }
        return sorted(best_by_memory.values(), key=lambda d: d["score"], reverse=True)

    def prune_orphans(self, live_ids: set[str]) -> int:
        """Delete hype rows (vec + text) for memory ids not in `live_ids`.
        Returns the number of question rows removed."""
        with self._tx() as cx:
            rows = cx.execute("SELECT DISTINCT memory_id FROM hype_questions").fetchall()
            orphan_ids = [str(r["memory_id"]) for r in rows if str(r["memory_id"]) not in live_ids]
            removed = 0
            for memory_id in orphan_ids:
                qids = [
                    str(r["question_id"])
                    for r in cx.execute(
                        "SELECT question_id FROM hype_questions WHERE memory_id = ?",
                        (memory_id,),
                    ).fetchall()
                ]
                for qid in qids:
                    cx.execute("DELETE FROM hype_vec WHERE question_id = ?", (qid,))
                cur = cx.execute("DELETE FROM hype_questions WHERE memory_id = ?", (memory_id,))
                removed += cur.rowcount
        return removed

    def stats(self) -> dict[str, int]:
        memories = self._conn.execute(
            "SELECT COUNT(DISTINCT memory_id) AS n FROM hype_questions"
        ).fetchone()
        questions = self._conn.execute("SELECT COUNT(*) AS n FROM hype_questions").fetchone()
        return {
            "memories": int(memories["n"]) if memories else 0,
            "questions": int(questions["n"]) if questions else 0,
        }


__all__ = ["HypeStore"]
