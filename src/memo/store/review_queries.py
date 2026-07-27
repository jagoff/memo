from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from memo.store._base import _StoreBase


class _ReviewQueriesMixin(_StoreBase):
    """Review scheduling/evidence signal queries preserved across rebuilds."""

    def update_review_state(
        self,
        *,
        id_: str,
        review_after: str | None,
        verification_state: str,
        verified_at: int | None,
        evidence: str | None = None,
        actor: str | None = None,
        reviewed_at: str | None = None,
    ) -> bool:
        with self._tx() as cx:
            cur = cx.execute(
                "UPDATE meta SET review_after=?, verification_state=?, verified_at=? WHERE id=?",
                (review_after, verification_state, verified_at, id_),
            )
            if cur.rowcount and reviewed_at is not None:
                cx.execute(
                    "INSERT INTO memory_reviews "
                    "(id, memory_id, reviewed_at, evidence, actor, next_review_after) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        uuid.uuid4().hex,
                        id_,
                        reviewed_at,
                        evidence,
                        actor,
                        review_after,
                    ),
                )
        return cur.rowcount > 0

    def list_due_reviews(
        self,
        *,
        now: str,
        namespace: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        params: list[Any] = [now]
        if namespace:
            params.append(namespace)
        params.append(max(1, min(int(limit), 1000)))
        query = (
            "SELECT m.id, m.path, m.title, m.type, m.tags, m.created, m.updated, "
            "m.review_after, m.verification_state, m.verified_at, m.valid_at, m.invalid_at, "
            "m.namespace, EXISTS(SELECT 1 FROM memory_relations r "
            "WHERE r.judgment_status='judged' AND r.relation='conflicts_with' "
            "AND (r.source_id=m.id OR r.target_id=m.id) "
            "AND EXISTS(SELECT 1 FROM meta peer "
            "WHERE peer.id=CASE WHEN r.source_id=m.id THEN r.target_id ELSE r.source_id END "
            "AND (peer.deleted_at IS NULL OR peer.deleted_at='') AND peer.invalid_at IS NULL) "
            "AND NOT EXISTS(SELECT 1 FROM memory_reviews reviewed "
            "WHERE reviewed.memory_id=m.id "
            "AND julianday(reviewed.reviewed_at) >= julianday(r.updated_at))) "
            "AS open_conflict FROM meta m WHERE "
            "(m.review_after IS NOT NULL AND m.review_after <= ? OR EXISTS("
            "SELECT 1 FROM memory_relations r2 "
            "WHERE r2.judgment_status='judged' AND r2.relation='conflicts_with' "
            "AND (r2.source_id=m.id OR r2.target_id=m.id) "
            "AND EXISTS(SELECT 1 FROM meta peer2 "
            "WHERE peer2.id=CASE "
            "WHEN r2.source_id=m.id THEN r2.target_id ELSE r2.source_id END "
            "AND (peer2.deleted_at IS NULL OR peer2.deleted_at='') "
            "AND peer2.invalid_at IS NULL) "
            "AND NOT EXISTS(SELECT 1 FROM memory_reviews reviewed2 "
            "WHERE reviewed2.memory_id=m.id "
            "AND julianday(reviewed2.reviewed_at) >= julianday(r2.updated_at)))) "
            "AND (m.deleted_at IS NULL OR m.deleted_at='') AND m.invalid_at IS NULL"
        )
        if namespace:
            query += " AND m.namespace = ?"
        query += " ORDER BY open_conflict DESC, m.review_after ASC LIMIT ?"
        rows = self._conn.execute(
            query,
            params,
        ).fetchall()
        return [dict(row) for row in rows]

    def review_evidence(self, memory_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, memory_id, reviewed_at, evidence, actor, next_review_after "
            "FROM memory_reviews WHERE memory_id=? ORDER BY reviewed_at DESC LIMIT ?",
            (memory_id, max(1, min(int(limit), 1000))),
        ).fetchall()
        return [dict(row) for row in rows]

    def review_diagnostics(self) -> dict[str, int]:
        now = datetime.now(UTC).isoformat()
        row = self._conn.execute(
            "SELECT COUNT(*) AS scheduled, "
            "SUM(CASE WHEN review_after <= ? THEN 1 ELSE 0 END) AS due "
            "FROM meta WHERE review_after IS NOT NULL AND invalid_at IS NULL "
            "AND (deleted_at IS NULL OR deleted_at='')",
            (now,),
        ).fetchone()
        return {
            "scheduled": int(row["scheduled"] or 0),
            "due": int(row["due"] or 0),
            "evidence": int(
                self._conn.execute("SELECT COUNT(*) FROM memory_reviews").fetchone()[0]
            ),
        }
