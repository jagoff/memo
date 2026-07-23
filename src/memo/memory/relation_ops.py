from __future__ import annotations

import contextlib
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

from memo.flags import flag_bool
from memo.identity import namespace_for_index
from memo.memory._base import _MemoryBase
from memo.memory.record import MemoryRecord
from memo.store.relation_queries import RELATION_VERBS
from memo.tiers import DURABLE_TYPES

_log = logging.getLogger(__name__)
_CANDIDATE_TAGS = frozenset({"architecture", "config", "policy"})


def _candidate_eligible(record: MemoryRecord) -> bool:
    tags = {str(tag).strip().lower() for tag in record.tags}
    return record.type in {"decision", "preference"} or (
        record.type in DURABLE_TYPES and bool(tags & _CANDIDATE_TAGS)
    )


def _allowed_candidate_namespaces(namespace: str) -> frozenset[str]:
    if namespace.startswith("project:"):
        return frozenset({namespace, "_global"})
    if namespace == "_global":
        return frozenset({"_global"})
    return frozenset({"_unscoped", "_global"})


class _RelationOpsMixin(_MemoryBase):
    """Candidate generation, auditable judgment, and relation annotations."""

    def detect_relation_candidates(
        self,
        record: MemoryRecord,
        *,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        if not _candidate_eligible(record):
            return []
        identity = self.store.get_identity_keys(record.id) or {}
        namespace = str(
            identity.get("namespace")
            or namespace_for_index(record.tags, path=record.path)
            or "_unscoped"
        )
        allowed = _allowed_candidate_namespaces(namespace)
        # Widen only the retrieval pool, never the persisted candidate cap.
        hits = self.search(
            f"{record.title}\n{record.body[:600]}",
            limit=max(12, limit * 4),
            disable_reranker=True,
            _track_usage=False,
        )
        candidates: list[dict[str, Any]] = []
        for hit in hits:
            if hit.id == record.id or hit.invalid_at is not None or not _candidate_eligible(hit):
                continue
            hit_identity = self.store.get_identity_keys(hit.id) or {}
            hit_namespace = str(
                hit_identity.get("namespace")
                or namespace_for_index(hit.tags, path=hit.path)
                or "_unscoped"
            )
            if hit_namespace not in allowed:
                continue
            row = self.store.create_relation_candidate(
                source_id=record.id,
                target_id=hit.id,
                reason="post-save semantic candidate",
                confidence=hit.score,
                provenance={
                    "generator": "memo.relation_ops",
                    "source_namespace": namespace,
                    "target_namespace": hit_namespace,
                },
            )
            candidates.append(self._compact_relation(row))
            if len(candidates) >= max(0, min(limit, 3)):
                break
        return candidates

    def attach_post_save_relations(self, record: MemoryRecord) -> MemoryRecord:
        if record.action not in {"created", "revised"}:
            return record
        if not flag_bool("MEMO_RELATION_CANDIDATES_ENABLED"):
            return replace(record, relation_candidates=[], relation_detection="disabled")
        try:
            candidates = self.detect_relation_candidates(record, limit=3)
            return replace(
                record,
                relation_candidates=candidates,
                relation_detection="ok",
            )
        except Exception as exc:
            _log.warning(
                "post-save relation detection unavailable for %s: %s",
                record.id[:8],
                exc,
            )
            return replace(record, relation_candidates=[], relation_detection="unavailable")

    def judge_relation(
        self,
        relation_id: str,
        relation: str,
        *,
        reason: str | None = None,
        confidence: float = 1.0,
        actor: str | None = None,
        actor_kind: str | None = None,
        model: str | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if relation not in RELATION_VERBS:
            from memo.errors import ValidationError

            raise ValidationError(f"invalid relation verb: {relation!r}")
        current = self.store.get_relation(relation_id)
        if current is None:
            from memo.errors import NotFoundError

            raise NotFoundError(f"relation not found: {relation_id}")
        if current.get("judgment_status") == "judged":
            # Store owns idempotency/conflict detection and returns the original
            # audit row without reapplying a temporal transition.
            return self.store.commit_relation_judgment(
                relation_id=relation_id,
                relation=relation,
                reason=reason,
                confidence=confidence,
                actor=actor,
                actor_kind=actor_kind,
                model=model,
                provenance=provenance,
            )

        validity_before: tuple[str | None, str | None] | None = None
        if relation == "supersedes":
            target = self.get(str(current["target_id"]))
            if target is None:
                self.store.orphan_relations_for(str(current["target_id"]))
                from memo.errors import NotFoundError

                raise NotFoundError(f"relation target not found: {current['target_id']}")
            validity_before = (target.valid_at, target.invalid_at)
            self.supersede(
                str(current["target_id"]),
                str(current["source_id"]),
                reason=reason or "relation judgment",
            )
        try:
            return self.store.commit_relation_judgment(
                relation_id=relation_id,
                relation=relation,
                reason=reason,
                confidence=confidence,
                actor=actor,
                actor_kind=actor_kind,
                model=model,
                provenance=provenance,
            )
        except Exception:
            if validity_before is not None:
                with contextlib.suppress(Exception):
                    self._set_validity_metadata(
                        str(current["target_id"]),
                        valid_at=validity_before[0],
                        invalid_at=validity_before[1],
                        reason="rollback failed relation judgment",
                    )
            raise

    def compare_memories(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        *,
        reason: str | None = None,
        confidence: float = 1.0,
        actor: str | None = None,
    ) -> dict[str, Any]:
        if self.get(source_id) is None:
            from memo.errors import NotFoundError

            raise NotFoundError(f"relation source not found: {source_id}")
        if self.get(target_id) is None:
            from memo.errors import NotFoundError

            raise NotFoundError(f"relation target not found: {target_id}")
        candidate = self.store.create_relation_candidate(
            source_id=source_id,
            target_id=target_id,
            reason=reason,
            confidence=confidence,
            provenance={"generator": "explicit_compare"},
        )
        return self.judge_relation(
            str(candidate["id"]),
            relation,
            reason=reason,
            confidence=confidence,
            actor=actor,
            actor_kind="agent" if actor else None,
        )

    def list_relation_reviews(self, *, limit: int = 50) -> list[dict[str, Any]]:
        return [
            self._compact_relation(row)
            for row in self.store.list_relations(status="pending", limit=limit)
        ]

    def annotate_relations(self, records: list[MemoryRecord]) -> list[MemoryRecord]:
        if not records or not flag_bool("MEMO_RELATION_ANNOTATIONS_ENABLED"):
            return records
        rows = self.store.list_relations(
            status="judged", memory_ids=[record.id for record in records], limit=200
        )
        by_id: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            if row.get("relation") == "not_conflict":
                continue
            source_id = str(row["source_id"])
            target_id = str(row["target_id"])
            by_id.setdefault(source_id, []).append(
                {**self._compact_relation(row), "direction": "out", "other_id": target_id}
            )
            by_id.setdefault(target_id, []).append(
                {**self._compact_relation(row), "direction": "in", "other_id": source_id}
            )
        out: list[MemoryRecord] = []
        for record in records:
            related = by_id.get(record.id)
            if not related:
                out.append(record)
                continue
            out.append(
                replace(
                    record,
                    extra={**(record.extra or {}), "memory_relations": related[:3]},
                )
            )
        return out

    def import_legacy_contradictions(self, *, limit: int = 10000) -> dict[str, int]:
        path = Path(self.cfg.contradictions_db)
        if not path.exists():
            return {"seen": 0, "imported": 0, "existing": 0}
        from memo.contradict import ContradictionStore

        legacy = ContradictionStore(path)
        seen = imported = existing = 0
        try:
            for pair in legacy.list_all(limit=limit):
                seen += 1
                before = self.store.find_relation_pair(pair.memory_id_a, pair.memory_id_b)
                suggested = (
                    "conflicts_with"
                    if pair.relationship == "contradiction"
                    else "related"
                )
                row = self.store.create_relation_candidate(
                    source_id=pair.memory_id_a,
                    target_id=pair.memory_id_b,
                    suggested_relation=suggested,
                    reason=pair.rationale,
                    confidence=pair.confidence,
                    provenance={"legacy_status": pair.status},
                    migration_key=f"legacy-contradiction:{pair.pair_id}",
                    migrated_from="contradictions.db",
                )
                if before is not None:
                    existing += 1
                    continue
                imported += 1
                resolved_relation = {
                    "dismissed": "not_conflict",
                    "competing": "conflicts_with",
                    "fused": "compatible",
                    "evolved": "related",
                    "kept_newer": "supersedes",
                    "kept_older": "supersedes",
                }.get(pair.status)
                if resolved_relation:
                    self.store.commit_relation_judgment(
                        relation_id=str(row["id"]),
                        relation=resolved_relation,
                        reason=pair.resolution_note or pair.rationale,
                        confidence=pair.confidence,
                        actor_kind="migration",
                        provenance={"legacy_status": pair.status},
                    )
        finally:
            legacy.close()
        return {"seen": seen, "imported": imported, "existing": existing}

    @staticmethod
    def _compact_relation(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": str(row.get("id") or ""),
            "source_id": str(row.get("source_id") or ""),
            "target_id": str(row.get("target_id") or ""),
            "relation": row.get("relation"),
            "status": row.get("judgment_status"),
            "reason": row.get("reason"),
            "confidence": row.get("confidence"),
        }
