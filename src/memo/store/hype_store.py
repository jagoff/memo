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
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..sqlite_compat import import_sqlite_vec
from .connection import _ConnectionHolder, _ConnectionMixin

serialize_float32 = import_sqlite_vec().serialize_float32


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _question_id(memory_id: str, text: str, view_kind: str = "hypothetical_question") -> str:
    return hashlib.sha256(f"{memory_id}:{view_kind}:{text}".encode()).hexdigest()[:32]


class HypeStore(_ConnectionMixin):
    """sqlite-vec backed index of hypothetical questions per memory.

    Reuses the trinity-proven connection plumbing (thread-local conns, WAL,
    vec0 load, `_tx`) from `_ConnectionMixin`, with its own two-table schema:
    `hype_vec` (vector-only, keyed by `question_id`) and `hype_questions`
    (question text + provenance, keyed by `question_id`). The knn joins the
    two by `question_id` to recover `memory_id` — vec0 doesn't carry the
    mutable metadata column itself, to avoid duplicating it there.
    """

    def __init__(
        self,
        db_path: Path | str,
        dims: int,
        embedder_model: str = "",
    ) -> None:
        self.db_path = Path(db_path)
        self.dims = dims
        self.embedder_model = embedder_model
        self._local = threading.local()
        # Strong references (matching VecStore): CPython does not guarantee the
        # order in which a terminating thread clears its ``threading.local``
        # values and finalizes sqlite connections, so ``close()`` (or the
        # dead-owner sweep in ``_connect``) is the single deterministic cleanup
        # boundary. A WeakSet would let a holder vanish before either fires.
        self._conn_holders: set[_ConnectionHolder] = set()
        self._conn_holders_lock = threading.Lock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        conn = self._conn
        conn.execute(
            "CREATE TABLE IF NOT EXISTS hype_questions ("
            "question_id TEXT PRIMARY KEY, memory_id TEXT NOT NULL, "
            "question TEXT NOT NULL, body_hash TEXT NOT NULL, model TEXT NOT NULL, "
            "created_at TEXT NOT NULL, variant TEXT NOT NULL DEFAULT 'query', "
            "view_kind TEXT NOT NULL DEFAULT 'hypothetical_question')"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hype_mem ON hype_questions(memory_id)")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS hype_attempts ("
            "memory_id TEXT PRIMARY KEY, body_hash TEXT NOT NULL, status TEXT NOT NULL, "
            "updated_at TEXT NOT NULL)"
        )
        # Inline ALTER-guard migration: a DB created before `variant` existed
        # needs the column backfilled — its rows were all embedded with the
        # query prefix (the only variant that ever existed then), so 'query'
        # is the honest default for pre-existing rows.
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(hype_questions)").fetchall()}
        if "variant" not in cols:
            conn.execute(
                "ALTER TABLE hype_questions ADD COLUMN variant TEXT NOT NULL DEFAULT 'query'"
            )
        if "view_kind" not in cols:
            conn.execute(
                "ALTER TABLE hype_questions ADD COLUMN view_kind TEXT NOT NULL "
                "DEFAULT 'hypothetical_question'"
            )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hype_kind ON hype_questions(view_kind)")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS hype_schema_meta ("
            "key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )

        vec_row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'hype_vec'"
        ).fetchone()
        actual_dims: int | None = None
        if vec_row is not None and vec_row["sql"]:
            match = re.search(r"FLOAT\[(\d+)\]", str(vec_row["sql"]), re.IGNORECASE)
            if match:
                actual_dims = int(match.group(1))
        stored = {
            str(row["key"]): str(row["value"])
            for row in conn.execute(
                "SELECT key, value FROM hype_schema_meta "
                "WHERE key IN ('embedder_model', 'embedder_dims')"
            ).fetchall()
        }
        current_model = self.embedder_model.strip()
        if not current_model:
            # HyPE is always collocated with VecStore.  Direct callers from
            # older integrations may omit the identity argument, but the main
            # DB is already self-describing; inherit its exact owner (including
            # an ST ``@revision`` suffix) instead of stamping an anonymous
            # sidecar that a later production open would have to invalidate.
            has_main_schema = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_meta'"
            ).fetchone()
            if has_main_schema is not None:
                main_model = conn.execute(
                    "SELECT value FROM schema_meta WHERE key = 'embedder_model'"
                ).fetchone()
                if main_model is not None and main_model["value"]:
                    current_model = str(main_model["value"])
        stamp_identity = bool(current_model and "stub" not in current_model.lower())
        identity_changed = stamp_identity and stored.get("embedder_model") != current_model
        dimensions_changed = actual_dims is not None and actual_dims != self.dims

        # HyPE is derived from canonical Markdown.  Unknown legacy identity is
        # treated as stale once a real identity is available: equal vector
        # width does not make embeddings from two model revisions compatible.
        if identity_changed or dimensions_changed:
            conn.execute("BEGIN IMMEDIATE")
            try:
                if vec_row is not None:
                    conn.execute("DROP TABLE hype_vec")
                conn.execute("DELETE FROM hype_questions")
                conn.execute(
                    f"CREATE VIRTUAL TABLE hype_vec USING vec0("
                    f"question_id TEXT PRIMARY KEY, "
                    f"embedding FLOAT[{self.dims}] distance_metric=cosine)"
                )
                if stamp_identity:
                    conn.execute(
                        "INSERT INTO hype_schema_meta (key, value) "
                        "VALUES ('embedder_model', ?) "
                        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                        (current_model,),
                    )
                    conn.execute(
                        "INSERT INTO hype_schema_meta (key, value) "
                        "VALUES ('embedder_dims', ?) "
                        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                        (str(self.dims),),
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        else:
            conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS hype_vec USING vec0("
                f"question_id TEXT PRIMARY KEY, "
                f"embedding FLOAT[{self.dims}] distance_metric=cosine)"
            )
            if stamp_identity:
                conn.execute(
                    "INSERT OR IGNORE INTO hype_schema_meta (key, value) "
                    "VALUES ('embedder_model', ?)",
                    (current_model,),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO hype_schema_meta (key, value) "
                    "VALUES ('embedder_dims', ?)",
                    (str(self.dims),),
                )
        conn.commit()

    def replace_for_memory(
        self,
        memory_id: str,
        body_hash: str,
        model: str,
        questions: list[tuple[str, list[float]]],
        variant: str = "query",
        view_kind: str = "hypothetical_question",
    ) -> int:
        """Delete old rows for `memory_id`, insert `(question_text, embedding)` rows.

        `question_id = sha256(f"{memory_id}:{text}").hexdigest()[:32]`.
        `variant` records which embedding scale the vectors were built in
        ("query" = `embed_query` prefix, "raw" = document-side, no prefix).
        Returns the inserted count.
        """
        created_at = _now_iso()
        with self._tx() as cx:
            old_ids = [
                str(r["question_id"])
                for r in cx.execute(
                    "SELECT question_id FROM hype_questions WHERE memory_id = ? AND view_kind = ?",
                    (memory_id, view_kind),
                ).fetchall()
            ]
            for old_id in old_ids:
                cx.execute("DELETE FROM hype_vec WHERE question_id = ?", (old_id,))
            cx.execute(
                "DELETE FROM hype_questions WHERE memory_id = ? AND view_kind = ?",
                (memory_id, view_kind),
            )
            for text, embedding in questions:
                if len(embedding) != self.dims:
                    raise ValueError(
                        f"HyPE question embedding dim mismatch: got {len(embedding)}, "
                        f"store expects {self.dims}. Usually a swapped model / "
                        f"MEMO_EMBEDDER_DIMS."
                    )
                qid = _question_id(memory_id, text, view_kind)
                cx.execute(
                    "INSERT INTO hype_vec (question_id, embedding) VALUES (?, ?)",
                    (qid, serialize_float32(embedding)),
                )
                cx.execute(
                    "INSERT INTO hype_questions "
                    "(question_id, memory_id, question, body_hash, model, created_at, variant, view_kind) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (qid, memory_id, text, body_hash, model, created_at, variant, view_kind),
                )
            if view_kind == "hypothetical_question":
                cx.execute(
                    "INSERT INTO hype_attempts (memory_id, body_hash, status, updated_at) "
                    "VALUES (?, ?, 'indexed', ?) "
                    "ON CONFLICT(memory_id) DO UPDATE SET body_hash=excluded.body_hash, "
                    "status=excluded.status, updated_at=excluded.updated_at",
                    (memory_id, body_hash, created_at),
                )
        return len(questions)

    def view_body_hash_for(self, memory_id: str, view_kind: str) -> str | None:
        """Return the watermark for one derived view kind."""
        row = self._conn.execute(
            "SELECT body_hash FROM hype_questions WHERE memory_id = ? AND view_kind = ? LIMIT 1",
            (memory_id, view_kind),
        ).fetchone()
        return str(row["body_hash"]) if row else None

    def mark_attempt(self, memory_id: str, body_hash: str, status: str) -> None:
        """Record a non-error derivation outcome, including a valid empty view."""
        with self._tx() as cx:
            cx.execute(
                "INSERT INTO hype_attempts (memory_id, body_hash, status, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(memory_id) DO UPDATE SET body_hash=excluded.body_hash, "
                "status=excluded.status, updated_at=excluded.updated_at",
                (memory_id, body_hash, status, _now_iso()),
            )

    def body_hash_for(self, memory_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT body_hash FROM hype_questions "
            "WHERE memory_id = ? AND view_kind = 'hypothetical_question' LIMIT 1",
            (memory_id,),
        ).fetchone()
        if row:
            return str(row["body_hash"])
        attempt = self._conn.execute(
            "SELECT body_hash, status FROM hype_attempts WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()
        if attempt and str(attempt["status"]) in {"indexed", "empty"}:
            return str(attempt["body_hash"])
        return None

    def knn(self, embedding: list[float], k: int) -> list[dict[str, Any]]:
        """`[{memory_id, question, score}]`, best question per memory, sorted
        by score desc. `score = 1.0 - distance`."""
        if len(embedding) != self.dims:
            raise ValueError(
                f"HyPE query embedding dim mismatch: got {len(embedding)}, expected {self.dims}"
            )
        k = max(1, int(k))
        rows = self._conn.execute(
            "SELECT q.memory_id, q.question, q.view_kind, v.distance AS distance "
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
                    "view_kind": str(r["view_kind"]),
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
                cx.execute("DELETE FROM hype_attempts WHERE memory_id = ?", (memory_id,))
                removed += cur.rowcount
        return removed

    def stats(self) -> dict[str, Any]:
        memories = self._conn.execute(
            "SELECT COUNT(DISTINCT memory_id) AS n FROM hype_questions"
        ).fetchone()
        questions = self._conn.execute("SELECT COUNT(*) AS n FROM hype_questions").fetchone()
        variant_rows = self._conn.execute(
            "SELECT variant, COUNT(*) AS n FROM hype_questions GROUP BY variant"
        ).fetchall()
        return {
            "memories": int(memories["n"]) if memories else 0,
            "questions": int(questions["n"]) if questions else 0,
            "by_variant": {str(r["variant"]): int(r["n"]) for r in variant_rows},
        }

    def memories_with_variant_other_than(self, variant: str) -> list[str]:
        """Distinct `memory_id`s that have at least one question row whose
        `variant` differs from `variant` — the reembed backlog."""
        rows = self._conn.execute(
            "SELECT DISTINCT memory_id FROM hype_questions WHERE variant != ?",
            (variant,),
        ).fetchall()
        return [str(r["memory_id"]) for r in rows]

    def questions_for_memory(self, memory_id: str) -> list[dict[str, Any]]:
        """All question rows for `memory_id` (text + body_hash + model),
        in insertion order — used by the reembed pass to recover the stored
        question text without needing the LLM again."""
        rows = self._conn.execute(
            "SELECT question, body_hash, model FROM hype_questions "
            "WHERE memory_id = ? ORDER BY rowid",
            (memory_id,),
        ).fetchall()
        return [
            {
                "question": str(r["question"]),
                "body_hash": str(r["body_hash"]),
                "model": str(r["model"]),
            }
            for r in rows
        ]


__all__ = ["HypeStore"]
