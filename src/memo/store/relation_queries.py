from __future__ import annotations

import hashlib
import json
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any

from memo.errors import NotFoundError, RelationConflictError, ValidationError

from ._base import _StoreBase

RELATION_VERBS = frozenset(
    {
        "supersedes",
        "conflicts_with",
        "compatible",
        "scoped",
        "related",
        "not_conflict",
    }
)
RELATION_STATES = frozenset({"pending", "judged", "orphaned"})


def relation_pair_key(source_id: str, target_id: str) -> str:
    """Stable unordered pair identity; direction remains on the stored row."""
    if not source_id or not target_id or source_id == target_id:
        raise ValidationError("a relation requires two distinct memory ids")
    a, b = sorted((source_id, target_id))
    return hashlib.sha256(f"{a}\0{b}".encode()).hexdigest()


def relation_id_for_pair(source_id: str, target_id: str) -> str:
    return f"rel-{relation_pair_key(source_id, target_id)[:24]}"


def _row_dict(row: Any) -> dict[str, Any]:
    out = dict(row)
    raw = out.pop("provenance_json", None)
    try:
        out["provenance"] = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        out["provenance"] = {}
    return out


class _RelationQueriesMixin(_StoreBase):
    """Signal-table ownership for pending and judged memory relations."""

    def create_relation_candidate(
        self,
        *,
        source_id: str,
        target_id: str,
        suggested_relation: str | None = None,
        reason: str | None = None,
        confidence: float | None = None,
        provenance: dict[str, Any] | None = None,
        migration_key: str | None = None,
        migrated_from: str | None = None,
    ) -> dict[str, Any]:
        if suggested_relation is not None and suggested_relation not in RELATION_VERBS:
            raise ValidationError(f"invalid relation verb: {suggested_relation!r}")
        pair_key = relation_pair_key(source_id, target_id)
        relation_id = relation_id_for_pair(source_id, target_id)
        now = datetime.now(UTC).isoformat()
        with self._tx() as cx:
            cx.execute(
                "INSERT OR IGNORE INTO memory_relations "
                "(id, pair_key, source_id, target_id, relation, judgment_status, "
                "reason, confidence, provenance_json, migration_key, migrated_from, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?)",
                (
                    relation_id,
                    pair_key,
                    source_id,
                    target_id,
                    suggested_relation,
                    reason,
                    confidence,
                    json.dumps(provenance or {}, default=str, ensure_ascii=False),
                    migration_key,
                    migrated_from,
                    now,
                    now,
                ),
            )
            row = cx.execute(
                "SELECT * FROM memory_relations WHERE pair_key = ? OR id = ? "
                "ORDER BY CASE WHEN pair_key = ? THEN 0 ELSE 1 END LIMIT 1",
                (pair_key, relation_id, pair_key),
            ).fetchone()
        if row is None:  # pragma: no cover - sqlite contract guard
            raise RuntimeError("relation insert committed without a readable row")
        return _row_dict(row)

    def get_relation(self, relation_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM memory_relations WHERE id = ? OR sync_id = ? LIMIT 1",
            (relation_id, relation_id),
        ).fetchone()
        return _row_dict(row) if row is not None else None

    def find_relation_pair(self, source_id: str, target_id: str) -> dict[str, Any] | None:
        pair_key = relation_pair_key(source_id, target_id)
        row = self._conn.execute(
            "SELECT * FROM memory_relations WHERE pair_key = ? LIMIT 1", (pair_key,)
        ).fetchone()
        return _row_dict(row) if row is not None else None

    def list_relations(
        self,
        *,
        status: str | None = None,
        memory_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if status is not None and status not in RELATION_STATES:
            raise ValidationError(f"invalid relation status: {status!r}")
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("judgment_status = ?")
            params.append(status)
        if memory_ids:
            placeholders = ",".join("?" for _ in memory_ids)
            clauses.append(f"(source_id IN ({placeholders}) OR target_id IN ({placeholders}))")
            params.extend(memory_ids)
            params.extend(memory_ids)
        sql = "SELECT * FROM memory_relations"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY COALESCE(updated_at, created_at) DESC LIMIT ?"
        params.append(max(1, min(int(limit), 1000)))
        return [_row_dict(row) for row in self._conn.execute(sql, params).fetchall()]

    def commit_relation_judgment(
        self,
        *,
        relation_id: str,
        relation: str,
        reason: str | None,
        confidence: float,
        actor: str | None = None,
        actor_kind: str | None = None,
        model: str | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if relation not in RELATION_VERBS:
            raise ValidationError(f"invalid relation verb: {relation!r}")
        confidence = max(0.0, min(1.0, float(confidence)))
        now = datetime.now(UTC).isoformat()
        with self._tx() as cx:
            row = cx.execute(
                "SELECT * FROM memory_relations WHERE id = ? OR sync_id = ? LIMIT 1",
                (relation_id, relation_id),
            ).fetchone()
            if row is None:
                raise NotFoundError(f"relation not found: {relation_id}")
            existing = str(row["relation"] or "")
            status = str(row["judgment_status"] or "pending")
            if status == "orphaned":
                raise ValidationError(f"relation {row['id']} has an orphaned endpoint")
            if status == "judged":
                if existing != relation:
                    raise RelationConflictError(str(row["id"]), existing, relation)
                return _row_dict(row)
            merged_provenance = {}
            with suppress(TypeError, ValueError):
                merged_provenance = json.loads(row["provenance_json"] or "{}")
            merged_provenance.update(provenance or {})
            cx.execute(
                "UPDATE memory_relations SET judgment_status='judged', relation=?, "
                "reason=?, confidence=?, actor=?, actor_kind=?, model=?, provenance_json=?, "
                "updated_at=? WHERE id=?",
                (
                    relation,
                    reason,
                    confidence,
                    actor,
                    actor_kind,
                    model,
                    json.dumps(merged_provenance, default=str, ensure_ascii=False),
                    now,
                    row["id"],
                ),
            )
            updated = cx.execute(
                "SELECT * FROM memory_relations WHERE id = ?", (row["id"],)
            ).fetchone()
        return _row_dict(updated)

    def reorient_pending_relation(
        self, relation_id: str, *, source_id: str, target_id: str
    ) -> dict[str, Any]:
        pair_key = relation_pair_key(source_id, target_id)
        with self._tx() as cx:
            row = cx.execute(
                "SELECT * FROM memory_relations WHERE id = ? LIMIT 1", (relation_id,)
            ).fetchone()
            if row is None:
                raise NotFoundError(f"relation not found: {relation_id}")
            if str(row["pair_key"] or "") != pair_key:
                raise ValidationError("relation endpoints do not match the canonical pair")
            if str(row["judgment_status"] or "pending") != "pending":
                raise ValidationError("only pending relations can be reoriented")
            cx.execute(
                "UPDATE memory_relations SET source_id=?, target_id=? WHERE id=?",
                (source_id, target_id, relation_id),
            )
            updated = cx.execute(
                "SELECT * FROM memory_relations WHERE id = ?", (relation_id,)
            ).fetchone()
        return _row_dict(updated)

    def reopen_relation(self, relation_id: str) -> bool:
        now = datetime.now(UTC).isoformat()
        with self._tx() as cx:
            cur = cx.execute(
                "UPDATE memory_relations SET judgment_status='pending', relation=NULL, "
                "actor=NULL, actor_kind=NULL, model=NULL, updated_at=? "
                "WHERE id=? AND judgment_status='judged'",
                (now, relation_id),
            )
        return bool(cur.rowcount)

    def orphan_relations_for(self, memory_id: str) -> int:
        now = datetime.now(UTC).isoformat()
        with self._tx() as cx:
            cur = cx.execute(
                "UPDATE memory_relations SET judgment_status='orphaned', updated_at=? "
                "WHERE (source_id=? OR target_id=?) AND judgment_status!='orphaned'",
                (now, memory_id, memory_id),
            )
        return int(cur.rowcount or 0)

    def relation_stats(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT judgment_status, COUNT(*) AS n FROM memory_relations GROUP BY judgment_status"
        ).fetchall()
        return {str(row["judgment_status"]): int(row["n"]) for row in rows}
