"""Closed, versioned registry with type-specific payload validation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from datetime import datetime

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
SESSION_RECOVERABLE = "memo.operational.session.recoverable.v1"
SESSION_TERMINATED = "memo.operational.session.terminated.v1"
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
DURABLE_PROMOTION_REQUESTED = "memo.operational.durable.promotion.requested.v1"
DURABLE_PROMOTION_RETRY_SCHEDULED = "memo.operational.durable.promotion.retry_scheduled.v1"
DURABLE_PROMOTION_COMPLETED = "memo.operational.durable.promotion.completed.v1"
DURABLE_PROMOTION_REJECTED = "memo.operational.durable.promotion.rejected.v1"
CHANNEL_OPENED = "memo.operational.coord.channel.opened.v1"
MESSAGE_SENT = "memo.operational.coord.message.sent.v1"
MESSAGE_SUPERSEDED = "memo.operational.coord.message.superseded.v1"
TOPIC_TERMINATED = "memo.operational.coord.topic.terminated.v1"
COORD_HANDOFF_CREATED = "memo.operational.coord.handoff.created.v1"
COORD_HANDOFF_CONSUMED = "memo.operational.coord.handoff.consumed.v1"
TASK_CREATED = "memo.operational.coord.task.created.v1"
TASK_ASSIGNED = "memo.operational.coord.task.assigned.v1"
TASK_COMPLETED = "memo.operational.coord.task.completed.v1"
TASK_CANCELLED = "memo.operational.coord.task.cancelled.v1"
TASK_EXPIRED = "memo.operational.coord.task.expired.v1"
DELIVERY_RESERVED = "memo.operational.delivery.reserved.v1"
DELIVERY_PRESENTED = "memo.operational.delivery.presented.v1"
DELIVERY_KNOWN_FAILED = "memo.operational.delivery.known_failed.v1"
DELIVERY_UNCERTAIN = "memo.operational.delivery.uncertain.v1"
DELIVERY_EXPIRED = "memo.operational.delivery.expired.v1"
DELIVERY_CURSOR_ADVANCED = "memo.operational.delivery.cursor.advanced.v1"
DELIVERY_ACK_RECORDED = "memo.operational.delivery.ack.recorded.v1"
PRESENCE_ANNOUNCED = "memo.operational.presence.announced.v1"
PRESENCE_RENEWED = "memo.operational.presence.renewed.v1"
PRESENCE_LEASE_EXPIRED = "memo.operational.presence.lease.expired.v1"


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


def _unique_string_list(payload: Mapping[str, object], field: str) -> tuple[str, ...]:
    values = _string_list(payload, field)
    if len(set(values)) != len(values):
        raise _invalid(f"payload field {field} must not contain duplicates")
    if values != tuple(sorted(values)):
        raise _invalid(f"payload field {field} must use canonical sorted order")
    return values


def _timestamp(payload: Mapping[str, object], field: str) -> str:
    value = _string(payload, field)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise _invalid(f"payload field {field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise _invalid(f"payload field {field} must include a timezone")
    return value


def _json_value(value: object, *, field: str) -> None:
    if value is None or isinstance(value, (str, bool, int, float)):
        return
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise _invalid(f"payload field {field} must use string mapping keys")
        for key, item in value.items():
            _json_value(item, field=f"{field}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _json_value(item, field=f"{field}[]")
        return
    raise _invalid(f"payload field {field} must contain only JSON values")


def _exact_fields(
    payload: Mapping[str, object],
    fields: frozenset[str],
) -> None:
    actual = set(payload)
    if actual != fields:
        missing = sorted(fields.difference(actual))
        unknown = sorted(actual.difference(fields))
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if unknown:
            details.append(f"unknown {', '.join(unknown)}")
        raise _invalid(f"durable promotion payload fields are invalid: {'; '.join(details)}")


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
    _exact_fields(
        payload,
        frozenset(
            {
                "session_id",
                "principal_id",
                "project",
                "workspace",
                "status",
                "branch",
                "head",
                "summary",
                "checkpointed_at",
                "source_event_id",
            }
        ),
    )
    for field in (
        "session_id",
        "principal_id",
        "project",
        "workspace",
        "source_event_id",
    ):
        _string(payload, field)
    _enum(payload, "status", frozenset({"active"}))
    for field in ("branch", "head", "summary"):
        _string_allow_empty(payload, field)
    _timestamp(payload, "checkpointed_at")


def _session_recoverable(payload: Mapping[str, object]) -> None:
    _exact_fields(
        payload,
        frozenset({"session_id", "recoverable_at", "reason"}),
    )
    _string(payload, "session_id")
    _timestamp(payload, "recoverable_at")
    _string_allow_empty(payload, "reason")


def _session_terminated(payload: Mapping[str, object]) -> None:
    _exact_fields(
        payload,
        frozenset({"session_id", "terminated_at", "summary"}),
    )
    _string(payload, "session_id")
    _timestamp(payload, "terminated_at")
    _string_allow_empty(payload, "summary")


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


def _promotion_binding(payload: Mapping[str, object]) -> None:
    promotion_id = _sha256(payload, "promotion_id")
    operation_key = _string(payload, "operation_key")
    if operation_key != f"promotion/{promotion_id}":
        raise _invalid("payload field operation_key must match promotion_id")
    _sha256(payload, "request_hash")


def _promotion_requested(payload: Mapping[str, object]) -> None:
    _exact_fields(
        payload,
        frozenset(
            {
                "promotion_id",
                "idempotency_key",
                "operation_key",
                "request_hash",
                "save_kwargs",
                "source_event_ids",
                "created_at",
            }
        ),
    )
    _promotion_binding(payload)
    idempotency_key = _string(payload, "idempotency_key")
    if idempotency_key != idempotency_key.strip():
        raise _invalid("payload field idempotency_key must be normalized")
    expected_id = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()
    if payload.get("promotion_id") != expected_id:
        raise _invalid("payload field promotion_id must match idempotency_key")
    save_kwargs = _mapping(payload, "save_kwargs")
    _json_value(save_kwargs, field="save_kwargs")
    try:
        encoded = json.dumps(
            save_kwargs,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise _invalid("payload field save_kwargs must be canonical JSON") from exc
    if hashlib.sha256(encoded).hexdigest() != payload.get("request_hash"):
        raise _invalid("payload field request_hash does not match save_kwargs")
    source_event_ids = _unique_string_list(payload, "source_event_ids")
    if not source_event_ids:
        raise _invalid("payload field source_event_ids must not be empty")
    extra = save_kwargs.get("extra")
    provenance = extra.get("provenance") if isinstance(extra, Mapping) else None
    stored_source_ids = (
        provenance.get("source_event_ids") if isinstance(provenance, Mapping) else None
    )
    if stored_source_ids != list(source_event_ids):
        raise _invalid("save_kwargs provenance source_event_ids must match the requested intent")
    _timestamp(payload, "created_at")


def _promotion_retry_scheduled(payload: Mapping[str, object]) -> None:
    _exact_fields(
        payload,
        frozenset(
            {
                "promotion_id",
                "operation_key",
                "request_hash",
                "attempt_number",
                "failure_class",
                "retry_at",
            }
        ),
    )
    _promotion_binding(payload)
    _integer(payload, "attempt_number", minimum=1)
    _string(payload, "failure_class")
    _timestamp(payload, "retry_at")


def _promotion_completed(payload: Mapping[str, object]) -> None:
    _exact_fields(
        payload,
        frozenset(
            {
                "promotion_id",
                "operation_key",
                "request_hash",
                "memory_id",
            }
        ),
    )
    _promotion_binding(payload)
    _string(payload, "memory_id")


def _promotion_rejected(payload: Mapping[str, object]) -> None:
    _exact_fields(
        payload,
        frozenset(
            {
                "promotion_id",
                "operation_key",
                "request_hash",
                "failure_class",
                "reason",
            }
        ),
    )
    _promotion_binding(payload)
    _string(payload, "failure_class")
    _string(payload, "reason")


def _channel_opened(payload: Mapping[str, object]) -> None:
    _string(payload, "channel")
    _string_allow_empty(payload, "topic")


def _message_sent(payload: Mapping[str, object]) -> None:
    for field in ("message_id", "channel", "body", "actor_id", "created_at"):
        _string(payload, field)
    _unique_string_list(payload, "target_ids")
    _string_allow_empty(payload, "topic")
    _boolean(payload, "expects_ack")
    _string_list(payload, "evidence_uris")


def _message_superseded(payload: Mapping[str, object]) -> None:
    for field in ("channel", "message_id", "superseded_by_message_id"):
        _string(payload, field)


def _topic_terminated(payload: Mapping[str, object]) -> None:
    for field in ("channel", "terminated_at"):
        _string(payload, field)
    _string_allow_empty(payload, "topic")


def _coord_handoff_created(payload: Mapping[str, object]) -> None:
    for field in ("id", "message_id", "project", "summary", "from_actor", "created_at"):
        _string(payload, field)
    _string_allow_empty(payload, "to_actor")
    _string_list(payload, "evidence_uris")


def _coord_handoff_consumed(payload: Mapping[str, object]) -> None:
    for field in ("id", "consumed_at", "actor_id"):
        _string(payload, field)


def _task_created(payload: Mapping[str, object]) -> None:
    for field in ("id", "project", "title", "created_at"):
        _string(payload, field)
    _string_allow_empty(payload, "assignee_id")


def _task_assigned(payload: Mapping[str, object]) -> None:
    for field in ("id", "assignee_id", "assigned_at"):
        _string(payload, field)


def _task_completed(payload: Mapping[str, object]) -> None:
    for field in ("id", "result", "completed_at"):
        _string(payload, field)


def _task_terminal(payload: Mapping[str, object]) -> None:
    for field in ("id", "at"):
        _string(payload, field)


def _delivery_transition(payload: Mapping[str, object]) -> None:
    for field in ("delivery_id", "message_id", "target_id", "transitioned_at"):
        _string(payload, field)
    _integer(payload, "attempt_count")
    _string_allow_empty(payload, "terminal_id")
    _string_allow_empty(payload, "error_code")


def _delivery_ack(payload: Mapping[str, object]) -> None:
    for field in (
        "delivery_id",
        "message_id",
        "target_id",
        "ack_actor_id",
        "ack_event_id",
        "transitioned_at",
    ):
        _string(payload, field)


def _delivery_cursor(payload: Mapping[str, object]) -> None:
    for field in ("consumer_id", "channel", "logical_clock", "event_id"):
        _string(payload, field)


def _presence_lease(payload: Mapping[str, object]) -> None:
    for field in (
        "id",
        "actor_id",
        "device_id",
        "project",
        "workspace",
        "topic",
        "intent",
        "expires_at",
    ):
        _string(payload, field)
    _unique_string_list(payload, "files")
    _integer(payload, "ttl_seconds", minimum=5)


def _presence_expiry(payload: Mapping[str, object]) -> None:
    for field in ("id", "expired_at"):
        _string(payload, field)


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
    SESSION_RECOVERABLE: _session_recoverable,
    SESSION_TERMINATED: _session_terminated,
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
    DURABLE_PROMOTION_REQUESTED: _promotion_requested,
    DURABLE_PROMOTION_RETRY_SCHEDULED: _promotion_retry_scheduled,
    DURABLE_PROMOTION_COMPLETED: _promotion_completed,
    DURABLE_PROMOTION_REJECTED: _promotion_rejected,
    CHANNEL_OPENED: _channel_opened,
    MESSAGE_SENT: _message_sent,
    MESSAGE_SUPERSEDED: _message_superseded,
    TOPIC_TERMINATED: _topic_terminated,
    COORD_HANDOFF_CREATED: _coord_handoff_created,
    COORD_HANDOFF_CONSUMED: _coord_handoff_consumed,
    TASK_CREATED: _task_created,
    TASK_ASSIGNED: _task_assigned,
    TASK_COMPLETED: _task_completed,
    TASK_CANCELLED: _task_terminal,
    TASK_EXPIRED: _task_terminal,
    DELIVERY_RESERVED: _delivery_transition,
    DELIVERY_PRESENTED: _delivery_transition,
    DELIVERY_ACK_RECORDED: _delivery_ack,
    DELIVERY_KNOWN_FAILED: _delivery_transition,
    DELIVERY_UNCERTAIN: _delivery_transition,
    DELIVERY_EXPIRED: _delivery_transition,
    DELIVERY_CURSOR_ADVANCED: _delivery_cursor,
    PRESENCE_ANNOUNCED: _presence_lease,
    PRESENCE_RENEWED: _presence_lease,
    PRESENCE_LEASE_EXPIRED: _presence_expiry,
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
    "CHANNEL_OPENED",
    "COMPACTION_COMPLETED",
    "CONFLICT_OPENED",
    "CONFLICT_RESOLVED",
    "COORDINATION_CLAIMED",
    "COORDINATION_COMPLETED",
    "COORDINATION_CREATED",
    "COORD_HANDOFF_CONSUMED",
    "COORD_HANDOFF_CREATED",
    "CURSOR_ADVANCED",
    "DELIVERY_ACKNOWLEDGED",
    "DELIVERY_ACK_RECORDED",
    "DELIVERY_CURSOR_ADVANCED",
    "DELIVERY_ENQUEUED",
    "DELIVERY_EXPIRED",
    "DELIVERY_KNOWN_FAILED",
    "DELIVERY_PRESENTED",
    "DELIVERY_RESERVED",
    "DELIVERY_UNCERTAIN",
    "DURABLE_PROMOTION_COMPLETED",
    "DURABLE_PROMOTION_REJECTED",
    "DURABLE_PROMOTION_REQUESTED",
    "DURABLE_PROMOTION_RETRY_SCHEDULED",
    "EVENT_TYPES",
    "FOCUS_CLEARED",
    "FOCUS_SET",
    "HANDOFF_CONSUMED",
    "HANDOFF_CREATED",
    "HEALTH_REPORTED",
    "MESSAGE_SENT",
    "MESSAGE_SUPERSEDED",
    "OUTCOME_RECORDED",
    "PRESENCE_ANNOUNCED",
    "PRESENCE_EXPIRED",
    "PRESENCE_LEASE_EXPIRED",
    "PRESENCE_RENEWED",
    "PRESENCE_UPDATED",
    "ROSTER_UPDATED",
    "SESSION_CHECKPOINTED",
    "SESSION_RECOVERABLE",
    "SESSION_TERMINATED",
    "TASK_ASSIGNED",
    "TASK_CANCELLED",
    "TASK_COMPLETED",
    "TASK_CREATED",
    "TASK_EXPIRED",
    "TERMINAL_COMMAND_FINISHED",
    "TERMINAL_COMMAND_STARTED",
    "TOPIC_TERMINATED",
    "validate_event_payload",
]
