"""Task-outcome feedback and deterministic procedural promotion."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from memo.atomic_io import authority_write_lock
from memo.contracts import TrustTier
from memo.durable_outbox import (
    DurableOutboxWorker,
    canonical_save_request_hash,
    promotion_operation_key,
)
from memo.errors import NotFoundError, ValidationError
from memo.memory._base import _MemoryBase
from memo.memory.record import MemoryRecord
from memo.operational_event_types import OUTCOME_RECORDED
from memo.util import utc_now_iso

_LEARNING_TYPES = {"procedure", "failure_pattern"}


def _stats(extra: dict[str, Any]) -> dict[str, Any]:
    raw = extra.get("outcome_stats")
    return dict(raw) if isinstance(raw, dict) else {}


def _normalized_memory_ids(memory_ids: list[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in memory_ids if value.strip()))


def _resolve_records(memory: Any, memory_ids: list[str]) -> tuple[list[str], list[Any]]:
    ids = _normalized_memory_ids(memory_ids)
    records: list[Any] = []
    for memory_id in ids:
        record = memory.get(memory_id)
        if record is None:
            raise NotFoundError(f"memory not found: {memory_id}")
        records.append(record)
    return ids, records


def _updated_outcome_extra(
    record: Any,
    *,
    outcomes: list[dict[str, Any]],
) -> dict[str, Any]:
    extra = dict(record.extra or {})
    relevant = [outcome for outcome in outcomes if record.id in (outcome.get("memory_ids") or ())]
    total = len(relevant)
    successes = sum(outcome.get("status") == "success" for outcome in relevant)
    failures = sum(outcome.get("status") == "failure" for outcome in relevant)
    partials = sum(outcome.get("status") == "partial" for outcome in relevant)
    utility = (successes + 0.5 * partials) / total
    latest = max(relevant, key=lambda row: str(row.get("recorded_at") or ""))
    extra["outcome_stats"] = {
        "total": total,
        "successes": successes,
        "failures": failures,
        "partials": partials,
        "utility": round(utility, 4),
        "last_status": latest["status"],
        "last_task_id": latest["task_id"],
        "last_recorded_at": latest["recorded_at"],
    }
    if total >= 2 and utility >= 0.8:
        extra["priority"] = "high"
    elif total >= 2 and failures / total > 0.5:
        extra["priority"] = "low"
    return extra


def _validated_task_id(task_id: str) -> str:
    normalized = task_id.strip()
    if not normalized:
        raise ValidationError("task_id cannot be empty")
    return normalized


def _canonical_timestamp(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError("source outcome has an invalid recorded_at timestamp") from exc
    if parsed.tzinfo is None:
        raise ValidationError("source outcome recorded_at timestamp has no timezone")
    return parsed.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _source_outcome_evidence(
    memory: Any,
    memory_ids: list[str],
) -> tuple[tuple[str, ...], str]:
    ledger = getattr(memory.operational, "ledger", None)
    validated_events = getattr(ledger, "validated_events", None)
    if not callable(validated_events):
        raise ValidationError("verified operational outcome events are unavailable")
    source_ids = set(memory_ids)
    covered_ids: set[str] = set()
    evidence: list[tuple[str, str]] = []
    for event in validated_events():
        if (
            getattr(event, "op", None) != "outcome.record"
            and getattr(event, "event_type", None) != OUTCOME_RECORDED
        ):
            continue
        payload = getattr(event, "payload", None)
        if not isinstance(payload, Mapping):
            continue
        raw_ids = payload.get("memory_ids")
        cited = {str(value) for value in raw_ids} if isinstance(raw_ids, (list, tuple)) else set()
        relevant_ids = cited.intersection(source_ids)
        if not relevant_ids:
            continue
        covered_ids.update(relevant_ids)
        recorded_at = str(
            payload.get("recorded_at")
            or getattr(event, "created_at", None)
            or getattr(event, "ts", "")
        )
        evidence.append((str(getattr(event, "event_id", "")), _canonical_timestamp(recorded_at)))
    evidence = sorted({item for item in evidence if item[0]})
    if not evidence or covered_ids != source_ids:
        raise ValidationError("source memories have no verified outcome event provenance")
    latest = max(timestamp for _event_id, timestamp in evidence)
    return tuple(event_id for event_id, _timestamp in evidence), latest


def _semantic_outcome_payload(
    *,
    task_id: str,
    status: str,
    memory_ids: list[str],
    actor_id: str,
    artifacts: list[str] | None,
    environment: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "task_id": _validated_task_id(task_id),
        "status": status,
        "memory_ids": sorted(_normalized_memory_ids(memory_ids)),
        "actor_id": actor_id,
        "artifacts": list(artifacts or ()),
        "environment": dict(environment or {}),
    }


def _stored_semantic_outcome_payload(outcome: dict[str, Any]) -> dict[str, Any]:
    raw_memory_ids = outcome.get("memory_ids")
    memory_ids = (
        [str(value) for value in raw_memory_ids] if isinstance(raw_memory_ids, list) else []
    )
    raw_artifacts = outcome.get("artifacts")
    artifacts = [str(value) for value in raw_artifacts] if isinstance(raw_artifacts, list) else []
    raw_environment = outcome.get("environment")
    environment = dict(raw_environment) if isinstance(raw_environment, dict) else {}
    return _semantic_outcome_payload(
        task_id=str(outcome.get("task_id") or ""),
        status=str(outcome.get("status") or ""),
        memory_ids=memory_ids,
        actor_id=str(outcome.get("actor_id") or "memo"),
        artifacts=artifacts,
        environment=environment,
    )


def _update_outcome_records(
    memory: Any,
    records: list[Any],
    *,
    outcomes: list[dict[str, Any]],
) -> list[str]:
    updated_ids: list[str] = []
    for record in records:
        current = memory.get(record.id)
        if current is None:
            raise NotFoundError(f"memory not found: {record.id}")
        extra = _updated_outcome_extra(current, outcomes=outcomes)
        if extra == dict(current.extra or {}):
            continue
        if memory.update(record.id, extra=extra) is not None:
            updated_ids.append(record.id)
    return updated_ids


class _OutcomeFeedbackOpsMixin(_MemoryBase):
    def record_task_outcome(
        self,
        *,
        task_id: str,
        status: str,
        memory_ids: list[str],
        actor_id: str = "memo",
        artifacts: list[str] | None = None,
        environment: dict[str, Any] | None = None,
        idempotency_key: str = "",
    ) -> dict[str, Any]:
        """Record a task result and feed it back into every cited memory.

        The operational journal is the event authority.  Per-memory aggregates
        are a rebuildable convenience projection stored in Markdown metadata.
        """
        task_id = _validated_task_id(task_id)
        with authority_write_lock(self.cfg.state_dir / "outcome-feedback"):
            ids, records = _resolve_records(self, memory_ids)
            outcomes_by_task = self.operational.state().get("outcomes", {})
            semantic_payload = _semantic_outcome_payload(
                task_id=task_id,
                status=status,
                memory_ids=ids,
                actor_id=actor_id,
                artifacts=artifacts,
                environment=environment,
            )
            matching_key = [
                outcome
                for outcome in outcomes_by_task.values()
                if isinstance(outcome, dict)
                and idempotency_key
                and outcome.get("idempotency_key") == idempotency_key
            ]
            replay = bool(matching_key)
            if replay and any(
                _stored_semantic_outcome_payload(outcome) != semantic_payload
                for outcome in matching_key
            ):
                raise ValueError("idempotency_key already exists with a different outcome payload")
            outcome = self.operational.record_outcome(
                task_id=task_id,
                status=status,
                memory_ids=ids,
                actor_id=actor_id,
                artifacts=artifacts,
                environment=environment,
                idempotency_key=idempotency_key,
            )
            outcomes = list(self.operational.state().get("outcomes", {}).values())
            updated_ids = _update_outcome_records(
                self,
                records,
                outcomes=outcomes,
            )
            return {
                **outcome,
                "updated_memory_ids": updated_ids,
                "idempotent_replay": replay,
            }

    def procedure_candidates(
        self,
        *,
        min_successes: int = 2,
        min_utility: float = 0.75,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return memories with enough observed utility to become procedures."""
        if min_successes < 1:
            raise ValueError("min_successes must be >= 1")
        if not 0.0 <= min_utility <= 1.0:
            raise ValueError("min_utility must be between 0 and 1")
        candidates: list[dict[str, Any]] = []
        total = max(0, int(self.store.count()))
        for record in self.list(limit=max(1, total + 1)):
            if record.type in _LEARNING_TYPES or record.type in {"secret", "reference"}:
                continue
            stats = _stats(dict(record.extra or {}))
            successes = int(stats.get("successes") or 0)
            utility = float(stats.get("utility") or 0.0)
            if successes < min_successes or utility < min_utility:
                continue
            candidates.append(
                {
                    "id": record.id,
                    "title": record.title,
                    "type": record.type,
                    "successes": successes,
                    "total": int(stats.get("total") or 0),
                    "utility": utility,
                }
            )
        candidates.sort(
            key=lambda row: (float(row["utility"]), int(row["successes"]), str(row["title"])),
            reverse=True,
        )
        return candidates[: max(0, limit)]

    def promote_learning(
        self,
        memory_ids: list[str],
        *,
        title: str,
        kind: str = "procedure",
        content: str | None = None,
        reason: str = "outcome-backed promotion",
        actor_id: str = "memo",
        idempotency_key: str,
    ) -> MemoryRecord:
        """Create a procedure or failure pattern grounded in source memories."""
        operation_key = promotion_operation_key(idempotency_key)
        if kind not in _LEARNING_TYPES:
            raise ValidationError("kind must be procedure|failure_pattern")
        ids = list(dict.fromkeys(value.strip() for value in memory_ids if value.strip()))
        if not ids:
            raise ValidationError("at least one source memory is required")
        sources = []
        for memory_id in ids:
            record = self.get(memory_id)
            if record is None:
                raise NotFoundError(f"memory not found: {memory_id}")
            sources.append(record)
        for record in sources:
            stats = _stats(dict(record.extra or {}))
            total = int(stats.get("total") or 0)
            successes = int(stats.get("successes") or 0)
            failures = int(stats.get("failures") or 0)
            utility = float(stats.get("utility") or 0.0)
            qualifies = (
                successes >= 2 and utility >= 0.75
                if kind == "procedure"
                else failures >= 2 and total > 0 and failures / total >= 0.5
            )
            if not qualifies:
                raise ValidationError(f"memory {record.id[:12]} lacks outcome evidence for {kind}")
        source_event_ids, promoted_at = _source_outcome_evidence(self, ids)
        body = (content or "").strip()
        if not body:
            parts = [f"## {record.title}\n\n{str(record.body or '').strip()}" for record in sources]
            body = "\n\n".join(parts)
        body = body[: self.cfg.max_content_chars]
        trust_tier = (
            TrustTier.EXTERNAL_UNTRUSTED.value
            if any(
                (record.extra or {}).get("trust_tier") == TrustTier.EXTERNAL_UNTRUSTED.value
                for record in sources
            )
            else TrustTier.AGENT_INFERRED.value
        )
        extra = {
            "trust_tier": trust_tier,
            "provenance": {
                "actor_id": actor_id,
                "route_reason": "procedural_promotion",
                "evidence_uris": [f"memo://memoria/{memory_id}" for memory_id in ids],
                "source_event_ids": list(source_event_ids),
            },
            "learning": {
                "kind": kind,
                "source_memory_ids": ids,
                "reason": reason,
                "promoted_at": promoted_at,
            },
            "priority": "high",
        }
        save_kwargs: dict[str, object] = {
            "content": body,
            "title": title,
            "type_": kind,
            "type": None,
            "tags": ["procedural", "outcome-backed"],
            "extra": extra,
            "auto_derive": False,
            "auto_project": True,
            "cwd": None,
            "created": None,
            "defer_embed": False,
            "enforce_write_policy": True,
            "allow_conflict_override": False,
            "override_reason": "",
            "topic_key": f"{kind}/{title}",
            "normalized_hash": None,
            "valid_at": None,
            "invalid_at": None,
            "actor": None,
        }
        capability = self._capabilities.get("durable_outbox")
        if isinstance(capability, DurableOutboxWorker):
            intent = capability.enqueue(
                idempotency_key=idempotency_key,
                save_kwargs=save_kwargs,
                source_event_ids=source_event_ids,
                created_at=utc_now_iso(),
            )
            return capability.reconcile(intent)
        request_hash = canonical_save_request_hash(save_kwargs)
        return self.save_operation(
            operation_key=operation_key,
            request_hash=request_hash,
            save_kwargs=save_kwargs,
        )


__all__ = ["_OutcomeFeedbackOpsMixin"]
