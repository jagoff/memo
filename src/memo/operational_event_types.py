"""Closed, versioned registry with type-specific payload validation."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping

from memo.errors import OperationalError, OperationalErrorCode

PayloadValidator = Callable[[Mapping[str, object]], None]
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

FOCUS_SET = "memo.operational.focus.set.v1"
FOCUS_CLEARED = "memo.operational.focus.cleared.v1"
HANDOFF_CREATED = "memo.operational.handoff.created.v1"
HANDOFF_CONSUMED = "memo.operational.handoff.consumed.v1"
ATTENTION_ADDED = "memo.operational.attention.added.v1"
ATTENTION_ACKNOWLEDGED = "memo.operational.attention.acknowledged.v1"
CONFLICT_OPENED = "memo.operational.conflict.opened.v1"
CONFLICT_RESOLVED = "memo.operational.conflict.resolved.v1"
OUTCOME_RECORDED = "memo.operational.outcome.recorded.v1"
SESSION_CHECKPOINTED = "memo.operational.session.checkpointed.v1"
SESSION_STATUS_CHANGED = "memo.operational.session.status_changed.v1"
COORDINATION_CREATED = "memo.operational.coordination.created.v1"
COORDINATION_CLAIMED = "memo.operational.coordination.claimed.v1"
COORDINATION_COMPLETED = "memo.operational.coordination.completed.v1"
DELIVERY_ENQUEUED = "memo.operational.delivery.enqueued.v1"
DELIVERY_ACKNOWLEDGED = "memo.operational.delivery.acknowledged.v1"
CURSOR_ADVANCED = "memo.operational.cursor.advanced.v1"
PRESENCE_UPDATED = "memo.operational.presence.updated.v1"
PRESENCE_EXPIRED = "memo.operational.presence.expired.v1"
TERMINAL_COMMAND_STARTED = "memo.operational.terminal.command_started.v1"
TERMINAL_COMMAND_FINISHED = "memo.operational.terminal.command_finished.v1"
HEALTH_REPORTED = "memo.operational.health.reported.v1"
ROSTER_UPDATED = "memo.operational.roster.updated.v1"
COMPACTION_COMPLETED = "memo.operational.compaction.completed.v1"
DURABLE_PROMOTION_ENQUEUED = "memo.operational.durable_promotion.enqueued.v1"
DURABLE_PROMOTION_COMPLETED = "memo.operational.durable_promotion.completed.v1"


def _invalid(message: str) -> OperationalError:
    return OperationalError(
        OperationalErrorCode.INVALID_EVENT,
        message,
        retryable=False,
    )


def _string(payload: Mapping[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise _invalid(f"payload field {field} must be a non-empty string")
    return value


def _string_allow_empty(payload: Mapping[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str):
        raise _invalid(f"payload field {field} must be a string")
    return value


def _mapping(payload: Mapping[str, object], field: str) -> Mapping[object, object]:
    value = payload.get(field)
    if not isinstance(value, Mapping):
        raise _invalid(f"payload field {field} must be a mapping")
    return value


def _integer(payload: Mapping[str, object], field: str, *, minimum: int = 0) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _invalid(f"payload field {field} must be an integer >= {minimum}")
    return value


def _boolean(payload: Mapping[str, object], field: str) -> bool:
    value = payload.get(field)
    if not isinstance(value, bool):
        raise _invalid(f"payload field {field} must be a boolean")
    return value


def _enum(
    payload: Mapping[str, object],
    field: str,
    values: frozenset[str],
) -> str:
    value = _string(payload, field)
    if value not in values:
        raise _invalid(f"payload field {field} has unsupported value {value!r}")
    return value


def _sha256(payload: Mapping[str, object], field: str) -> str:
    value = _string(payload, field)
    if not _SHA256_RE.fullmatch(value):
        raise _invalid(f"payload field {field} must be a lowercase SHA-256")
    return value


def _string_list(payload: Mapping[str, object], field: str) -> tuple[str, ...]:
    value = payload.get(field)
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise _invalid(f"payload field {field} must be a list of non-empty strings")
    return tuple(value)


def _focus_set(payload: Mapping[str, object]) -> None:
    _string(payload, "project")
    _string(payload, "summary")


def _focus_cleared(payload: Mapping[str, object]) -> None:
    _string(payload, "project")


def _handoff_created(payload: Mapping[str, object]) -> None:
    for field in ("id", "project", "summary", "from_actor"):
        _string(payload, field)
    _string_allow_empty(payload, "to_actor")
    if "created_at" in payload:
        _string(payload, "created_at")
    if "consumed_at" in payload:
        _string_allow_empty(payload, "consumed_at")
    if "metadata" in payload:
        _mapping(payload, "metadata")


def _handoff_consumed(payload: Mapping[str, object]) -> None:
    _string(payload, "id")
    if "consumed_at" in payload:
        _string(payload, "consumed_at")
    elif "actor_id" in payload:
        _string(payload, "actor_id")
    else:
        raise _invalid("handoff consumption requires consumed_at")


def _attention_added(payload: Mapping[str, object]) -> None:
    for field in ("id", "project", "summary"):
        _string(payload, field)
    _enum(payload, "severity", frozenset({"low", "medium", "high", "critical"}))


def _attention_acknowledged(payload: Mapping[str, object]) -> None:
    _string(payload, "id")
    if "acknowledged_at" in payload:
        _string(payload, "acknowledged_at")
    elif "actor_id" in payload:
        _string(payload, "actor_id")
    else:
        raise _invalid("attention acknowledgement requires acknowledged_at")


def _conflict_opened(payload: Mapping[str, object]) -> None:
    for field in ("id", "topic", "summary"):
        _string(payload, field)
    _boolean(payload, "freeze_write")
    _string_list(payload, "evidence_uris")


def _conflict_resolved(payload: Mapping[str, object]) -> None:
    for field in ("id", "resolution"):
        _string(payload, field)
    if "resolved_at" in payload:
        _string(payload, "resolved_at")
    elif "actor_id" in payload:
        _string(payload, "actor_id")
    else:
        raise _invalid("conflict resolution requires resolved_at")


def _outcome_recorded(payload: Mapping[str, object]) -> None:
    _string(payload, "task_id")
    _enum(payload, "status", frozenset({"success", "failure", "partial"}))
    _string_list(payload, "memory_ids")
    _string_list(payload, "artifacts")
    _mapping(payload, "environment")
    _string(payload, "actor_id")
    _string_allow_empty(payload, "idempotency_key")
    _string(payload, "recorded_at")


def _session_checkpointed(payload: Mapping[str, object]) -> None:
    for field in ("session_id", "principal_id", "checkpointed_at"):
        _string(payload, field)
    _enum(payload, "status", frozenset({"active", "recoverable", "terminated"}))


def _session_status_changed(payload: Mapping[str, object]) -> None:
    _string(payload, "session_id")
    _enum(payload, "status", frozenset({"active", "recoverable", "terminated"}))


def _coordination_created(payload: Mapping[str, object]) -> None:
    _string(payload, "task_id")
    _string(payload, "summary")
    _enum(payload, "status", frozenset({"pending"}))


def _coordination_claimed(payload: Mapping[str, object]) -> None:
    _string(payload, "task_id")
    _string(payload, "principal_id")


def _coordination_completed(payload: Mapping[str, object]) -> None:
    _string(payload, "task_id")
    _enum(payload, "status", frozenset({"completed", "failed", "cancelled"}))


def _delivery_enqueued(payload: Mapping[str, object]) -> None:
    _string(payload, "message_id")
    _string(payload, "recipient")
    _sha256(payload, "payload_sha256")


def _delivery_acknowledged(payload: Mapping[str, object]) -> None:
    _string(payload, "message_id")
    _string(payload, "recipient")


def _cursor_advanced(payload: Mapping[str, object]) -> None:
    _string(payload, "cursor")
    _integer(payload, "position")


def _presence_updated(payload: Mapping[str, object]) -> None:
    _string(payload, "principal_id")
    _enum(payload, "status", frozenset({"online", "away", "busy"}))
    _string(payload, "expires_at")


def _presence_expired(payload: Mapping[str, object]) -> None:
    _string(payload, "principal_id")


def _terminal_started(payload: Mapping[str, object]) -> None:
    _string(payload, "command_id")
    _sha256(payload, "command_sha256")


def _terminal_finished(payload: Mapping[str, object]) -> None:
    _string(payload, "command_id")
    _integer(payload, "exit_code", minimum=-255)


def _health_reported(payload: Mapping[str, object]) -> None:
    _string(payload, "component")
    _enum(payload, "status", frozenset({"healthy", "degraded", "unhealthy"}))


def _roster_updated(payload: Mapping[str, object]) -> None:
    _integer(payload, "version", minimum=1)
    _sha256(payload, "roster_hash")


def _compaction_completed(payload: Mapping[str, object]) -> None:
    _string(payload, "origin_device")
    _integer(payload, "through_sequence")
    _sha256(payload, "anchor_hash")


def _promotion_enqueued(payload: Mapping[str, object]) -> None:
    _string(payload, "promotion_id")
    _string(payload, "operation_key")


def _promotion_completed(payload: Mapping[str, object]) -> None:
    _string(payload, "promotion_id")
    _string(payload, "memory_id")


EVENT_TYPES: dict[str, PayloadValidator] = {
    FOCUS_SET: _focus_set,
    FOCUS_CLEARED: _focus_cleared,
    HANDOFF_CREATED: _handoff_created,
    HANDOFF_CONSUMED: _handoff_consumed,
    ATTENTION_ADDED: _attention_added,
    ATTENTION_ACKNOWLEDGED: _attention_acknowledged,
    CONFLICT_OPENED: _conflict_opened,
    CONFLICT_RESOLVED: _conflict_resolved,
    OUTCOME_RECORDED: _outcome_recorded,
    SESSION_CHECKPOINTED: _session_checkpointed,
    SESSION_STATUS_CHANGED: _session_status_changed,
    COORDINATION_CREATED: _coordination_created,
    COORDINATION_CLAIMED: _coordination_claimed,
    COORDINATION_COMPLETED: _coordination_completed,
    DELIVERY_ENQUEUED: _delivery_enqueued,
    DELIVERY_ACKNOWLEDGED: _delivery_acknowledged,
    CURSOR_ADVANCED: _cursor_advanced,
    PRESENCE_UPDATED: _presence_updated,
    PRESENCE_EXPIRED: _presence_expired,
    TERMINAL_COMMAND_STARTED: _terminal_started,
    TERMINAL_COMMAND_FINISHED: _terminal_finished,
    HEALTH_REPORTED: _health_reported,
    ROSTER_UPDATED: _roster_updated,
    COMPACTION_COMPLETED: _compaction_completed,
    DURABLE_PROMOTION_ENQUEUED: _promotion_enqueued,
    DURABLE_PROMOTION_COMPLETED: _promotion_completed,
}


def validate_event_payload(event_type: str, payload: Mapping[str, object]) -> None:
    validator = EVENT_TYPES.get(event_type)
    if validator is None:
        raise _invalid(f"unknown operational event type: {event_type}")
    if not isinstance(payload, Mapping):
        raise _invalid("operational event payload must be a mapping")
    validator(payload)


__all__ = [
    "ATTENTION_ACKNOWLEDGED",
    "ATTENTION_ADDED",
    "COMPACTION_COMPLETED",
    "CONFLICT_OPENED",
    "CONFLICT_RESOLVED",
    "COORDINATION_CLAIMED",
    "COORDINATION_COMPLETED",
    "COORDINATION_CREATED",
    "CURSOR_ADVANCED",
    "DELIVERY_ACKNOWLEDGED",
    "DELIVERY_ENQUEUED",
    "DURABLE_PROMOTION_COMPLETED",
    "DURABLE_PROMOTION_ENQUEUED",
    "EVENT_TYPES",
    "FOCUS_CLEARED",
    "FOCUS_SET",
    "HANDOFF_CONSUMED",
    "HANDOFF_CREATED",
    "HEALTH_REPORTED",
    "OUTCOME_RECORDED",
    "PRESENCE_EXPIRED",
    "PRESENCE_UPDATED",
    "ROSTER_UPDATED",
    "SESSION_CHECKPOINTED",
    "SESSION_STATUS_CHANGED",
    "TERMINAL_COMMAND_FINISHED",
    "TERMINAL_COMMAND_STARTED",
    "validate_event_payload",
]
