from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from ..sqlite_compat import import_sqlite_vec
from ._base import _StoreBase

serialize_float32 = import_sqlite_vec().serialize_float32


class _FeedbackMixin(_StoreBase):
    # -- source-level feedback (👍 / 👎) ------------------------------------

    def record_source_feedback(
        self,
        *,
        source_id: str,
        query_text: str,
        query_emb: list[float],
        rating: int,
        feedback_id: str | None = None,
        extra: dict[str, Any] | None = None,
        only_if_absent: bool = False,
    ) -> str:
        """Persist a 👍/👎 vote on `source_id` for `query_text`.

        Idempotent on `(source_id, query_text, rating)` — re-recording the
        same vote returns the existing feedback id. Changing rating for
        the same (source, query) replaces the old row (cancel-and-replace
        so the unique constraint stays clean).

        `only_if_absent=True` makes this a no-op when ANY row already exists
        for `(source_id, query_text)` — used by automated (outcome-loop)
        feedback so it never clobbers a manual 👍/👎 on the same pair.
        """
        if rating not in (-1, 1):
            raise ValueError(f"rating must be -1 or 1, got {rating!r}")
        if len(query_emb) != self.dims:
            raise ValueError(f"query_emb dim mismatch: got {len(query_emb)}, expected {self.dims}")
        now = datetime.now(UTC).isoformat()
        # Read + write inside one BEGIN IMMEDIATE so concurrent FastMCP threads
        # serialise — a plain SELECT then `with self._conn` (BEGIN DEFERRED) let
        # two threads both see existing=None and race the UNIQUE constraint.
        with self._tx() as cx:
            # Find any existing row for this (source, query) regardless of rating.
            existing = cx.execute(
                "SELECT id, rating FROM source_feedback WHERE source_id = ? AND query_text = ?",
                (source_id, query_text),
            ).fetchone()
            if existing and (only_if_absent or int(existing["rating"]) == rating):
                return str(existing["id"])
            if existing:
                # Replacing rating — drop old vec + row first.
                cx.execute(
                    "DELETE FROM source_feedback_vec WHERE feedback_id = ?",
                    (existing["id"],),
                )
                cx.execute(
                    "DELETE FROM source_feedback WHERE id = ?",
                    (existing["id"],),
                )
            fid = feedback_id or uuid.uuid4().hex
            cx.execute(
                "INSERT INTO source_feedback "
                "(id, source_id, query_text, rating, created_at, extra_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    fid,
                    source_id,
                    query_text,
                    rating,
                    now,
                    json.dumps(extra) if extra else None,
                ),
            )
            cx.execute(
                "INSERT INTO source_feedback_vec (feedback_id, source_id, query_emb) VALUES (?, ?, ?)",
                (fid, source_id, serialize_float32(query_emb)),
            )
        return fid

    def find_feedback_for_source(
        self,
        source_id: str,
        query_emb: list[float],
        *,
        threshold: float = 0.85,
        limit: int = 16,
    ) -> list[dict[str, Any]]:
        """Return prior feedback rows on `source_id` whose query embedding
        is cosine-similar to `query_emb` at >= `threshold`.

        Returns a list of dicts with keys: id, rating, query_text,
        similarity (float in [0, 1]), created_at. Empty list if none.
        """
        if len(query_emb) != self.dims:
            return []
        # `source_id` is a vec0 PARTITION KEY, so `fv.source_id = ?` pre-filters
        # the kNN to this source's rows BEFORE picking the top-k. The previous
        # shape (global kNN on `k`, then `fb.source_id = ?` in the join) could
        # silently drop a source's matches when they fell outside the global
        # top-k. The `meta`-style join still fetches the human-readable fields.
        rows = self._conn.execute(
            "SELECT fb.id, fb.rating, fb.query_text, fb.created_at, "
            "       fb.extra_json, fv.distance "
            "FROM source_feedback_vec fv "
            "JOIN source_feedback fb ON fb.id = fv.feedback_id "
            "WHERE fv.query_emb MATCH ? "
            "  AND k = ? "
            "  AND fv.source_id = ? "
            "ORDER BY fv.distance ASC",
            (serialize_float32(query_emb), int(limit), source_id),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            dist = float(r["distance"])
            sim = 1.0 - dist
            if sim < threshold:
                continue
            out.append(
                {
                    "id": r["id"],
                    "rating": int(r["rating"]),
                    "query_text": r["query_text"],
                    "created_at": r["created_at"],
                    "extra_json": r["extra_json"],
                    "similarity": sim,
                }
            )
        return out

    def sources_with_feedback(self, source_ids: list[str]) -> set[str]:
        """Return the subset of `source_ids` that have >=1 feedback row.

        Cheap existence check on `idx_source_feedback_source` so the kNN vec
        scan in `find_feedback_for_source` only runs for sources that actually
        have feedback. Most memories have none, so this collapses a per-hit
        N+1 of kNN queries into a single IN-list lookup.
        """
        ids = list(source_ids)
        if not ids:
            return set()
        placeholders = ",".join("?" * len(ids))
        rows = self._conn.execute(
            "SELECT DISTINCT source_id FROM source_feedback "  # noqa: S608
            f"WHERE source_id IN ({placeholders})",
            ids,
        ).fetchall()
        return {r["source_id"] for r in rows}

    def list_source_feedback(
        self,
        *,
        source_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if source_id:
            rows = self._conn.execute(
                "SELECT id, source_id, query_text, rating, created_at, extra_json "
                "FROM source_feedback WHERE source_id = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (source_id, int(limit)),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, source_id, query_text, rating, created_at, extra_json "
                "FROM source_feedback ORDER BY created_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [dict(r) for r in rows]

    def rebuild_feedback_vecs(self, embed_fn: object) -> int:
        """Re-embed all `source_feedback` rows into `source_feedback_vec`.

        Iterates feedback rows without a vector, embeds ``query_text`` via
        ``embed_fn(query_text: str) -> list[float]`` — pass the embedder's
        ``embed_query`` so the stored vector carries the asymmetric-retrieval
        query prefix, matching how feedback vecs are written on the normal
        path (rerank_ops) and how they're matched at search time. Idempotent —
        skips feedback ids that already have a vector row. Returns count of new
        vectors inserted.
        """
        rows = self._conn.execute(
            "SELECT sf.id, sf.source_id, sf.query_text "
            "FROM source_feedback sf "
            "LEFT JOIN source_feedback_vec fv ON fv.feedback_id = sf.id "
            "WHERE fv.feedback_id IS NULL"
        ).fetchall()
        if not rows:
            return 0
        vectors = [embed_fn(r["query_text"]) for r in rows]  # type: ignore[operator]
        count = 0
        with self._tx() as cx:
            for r, vec in zip(rows, vectors, strict=True):
                if len(vec) != self.dims:
                    continue
                cx.execute(
                    "INSERT OR IGNORE INTO source_feedback_vec "
                    "(feedback_id, source_id, query_emb) VALUES (?, ?, ?)",
                    (r["id"], r["source_id"], serialize_float32(vec)),
                )
                count += 1
        return count

    def clear_source_feedback(self, source_id: str) -> int:
        """Drop all feedback rows for a source. Returns count deleted."""
        with self._tx() as cx:
            ids = [
                r["id"]
                for r in cx.execute(
                    "SELECT id FROM source_feedback WHERE source_id = ?",
                    (source_id,),
                ).fetchall()
            ]
            if not ids:
                return 0
            placeholders = ",".join("?" * len(ids))
            cx.execute(
                f"DELETE FROM source_feedback_vec WHERE feedback_id IN ({placeholders})",  # noqa: S608
                ids,
            )
            cx.execute(
                f"DELETE FROM source_feedback WHERE id IN ({placeholders})",  # noqa: S608
                ids,
            )
        return len(ids)
