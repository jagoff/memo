from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlite_vec import serialize_float32

from ._base import _StoreBase


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
    ) -> str:
        """Persist a 👍/👎 vote on `source_id` for `query_text`.

        Idempotent on `(source_id, query_text, rating)` — re-recording the
        same vote returns the existing feedback id. Changing rating for
        the same (source, query) replaces the old row (cancel-and-replace
        so the unique constraint stays clean).
        """
        if rating not in (-1, 1):
            raise ValueError(f"rating must be -1 or 1, got {rating!r}")
        if len(query_emb) != self.dims:
            raise ValueError(
                f"query_emb dim mismatch: got {len(query_emb)}, expected {self.dims}"
            )
        now = datetime.now(UTC).isoformat()
        # Find any existing row for this (source, query) regardless of rating.
        existing = self._conn.execute(
            "SELECT id, rating FROM source_feedback "
            "WHERE source_id = ? AND query_text = ?",
            (source_id, query_text),
        ).fetchone()
        if existing and int(existing["rating"]) == rating:
            return str(existing["id"])
        with self._conn:
            if existing:
                # Replacing rating — drop old vec + row first.
                self._conn.execute(
                    "DELETE FROM source_feedback_vec WHERE feedback_id = ?",
                    (existing["id"],),
                )
                self._conn.execute(
                    "DELETE FROM source_feedback WHERE id = ?",
                    (existing["id"],),
                )
            fid = feedback_id or uuid.uuid4().hex
            self._conn.execute(
                "INSERT INTO source_feedback "
                "(id, source_id, query_text, rating, created_at, extra_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    fid, source_id, query_text, rating, now,
                    json.dumps(extra) if extra else None,
                ),
            )
            self._conn.execute(
                "INSERT INTO source_feedback_vec (feedback_id, query_emb) "
                "VALUES (?, ?)",
                (fid, serialize_float32(query_emb)),
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
        rows = self._conn.execute(
            "SELECT fb.id, fb.rating, fb.query_text, fb.created_at, "
            "       fb.extra_json, fv.distance "
            "FROM source_feedback fb "
            "JOIN source_feedback_vec fv ON fb.id = fv.feedback_id "
            "WHERE fb.source_id = ? "
            "  AND fv.query_emb MATCH ? "
            "  AND k = ? "
            "ORDER BY fv.distance ASC",
            (source_id, serialize_float32(query_emb), int(limit)),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            dist = float(r["distance"])
            sim = 1.0 - dist
            if sim < threshold:
                continue
            out.append({
                "id": r["id"],
                "rating": int(r["rating"]),
                "query_text": r["query_text"],
                "created_at": r["created_at"],
                "extra_json": r["extra_json"],
                "similarity": sim,
            })
        return out

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

    def clear_source_feedback(self, source_id: str) -> int:
        """Drop all feedback rows for a source. Returns count deleted."""
        ids = [
            r["id"] for r in self._conn.execute(
                "SELECT id FROM source_feedback WHERE source_id = ?",
                (source_id,),
            ).fetchall()
        ]
        if not ids:
            return 0
        with self._conn:
            placeholders = ",".join("?" * len(ids))
            self._conn.execute(
                f"DELETE FROM source_feedback_vec WHERE feedback_id IN ({placeholders})",
                ids,
            )
            self._conn.execute(
                f"DELETE FROM source_feedback WHERE id IN ({placeholders})",
                ids,
            )
        return len(ids)
