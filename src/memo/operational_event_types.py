"""Closed, versioned registry of operational event names."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from memo.errors import OperationalError, OperationalErrorCode

PayloadValidator = Callable[[Mapping[str, object]], None]

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


def _mapping_payload(_payload: Mapping[str, object]) -> None:
    return None


EVENT_TYPES: dict[str, PayloadValidator] = {
    name: _mapping_payload
    for name in (
        FOCUS_SET,
        FOCUS_CLEARED,
        HANDOFF_CREATED,
        HANDOFF_CONSUMED,
        ATTENTION_ADDED,
        ATTENTION_ACKNOWLEDGED,
        CONFLICT_OPENED,
        CONFLICT_RESOLVED,
        OUTCOME_RECORDED,
        SESSION_CHECKPOINTED,
        SESSION_STATUS_CHANGED,
        COORDINATION_CREATED,
        COORDINATION_CLAIMED,
        COORDINATION_COMPLETED,
        DELIVERY_ENQUEUED,
        DELIVERY_ACKNOWLEDGED,
        CURSOR_ADVANCED,
        PRESENCE_UPDATED,
        PRESENCE_EXPIRED,
        TERMINAL_COMMAND_STARTED,
        TERMINAL_COMMAND_FINISHED,
        HEALTH_REPORTED,
        ROSTER_UPDATED,
        COMPACTION_COMPLETED,
        DURABLE_PROMOTION_ENQUEUED,
        DURABLE_PROMOTION_COMPLETED,
    )
}


def validate_event_payload(event_type: str, payload: Mapping[str, object]) -> None:
    validator = EVENT_TYPES.get(event_type)
    if validator is None:
        raise OperationalError(
            OperationalErrorCode.INVALID_EVENT,
            f"unknown operational event type: {event_type}",
            retryable=False,
        )
    if not isinstance(payload, Mapping):
        raise OperationalError(
            OperationalErrorCode.INVALID_EVENT,
            "operational event payload must be a mapping",
            retryable=False,
        )
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
