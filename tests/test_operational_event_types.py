from __future__ import annotations

import pytest

from memo.error_contract import OperationalError, OperationalErrorCode
from memo.operational_event_types import (
    ATTENTION_ACKNOWLEDGED,
    ATTENTION_ADDED,
    COMPACTION_COMPLETED,
    CONFLICT_OPENED,
    CONFLICT_RESOLVED,
    COORDINATION_CLAIMED,
    COORDINATION_COMPLETED,
    COORDINATION_CREATED,
    CURSOR_ADVANCED,
    DELIVERY_ACKNOWLEDGED,
    DELIVERY_ENQUEUED,
    DURABLE_PROMOTION_COMPLETED,
    DURABLE_PROMOTION_ENQUEUED,
    EVENT_TYPES,
    FOCUS_CLEARED,
    FOCUS_SET,
    HANDOFF_CONSUMED,
    HANDOFF_CREATED,
    HEALTH_REPORTED,
    OUTCOME_RECORDED,
    PRESENCE_EXPIRED,
    PRESENCE_UPDATED,
    ROSTER_UPDATED,
    SESSION_CHECKPOINTED,
    SESSION_STATUS_CHANGED,
    TERMINAL_COMMAND_FINISHED,
    TERMINAL_COMMAND_STARTED,
    validate_event_payload,
)


def test_registry_is_closed_and_fully_qualified() -> None:
    assert FOCUS_SET in EVENT_TYPES
    assert DELIVERY_ENQUEUED in EVENT_TYPES
    assert PRESENCE_UPDATED in EVENT_TYPES
    assert all(name.startswith("memo.operational.") and name.endswith(".v1") for name in EVENT_TYPES)


def test_registry_validates_mapping_payload_and_rejects_unknown_names() -> None:
    validate_event_payload(FOCUS_SET, {"project": "memo", "summary": "ship"})
    with pytest.raises(OperationalError) as exc:
        validate_event_payload("focus.set", {})
    assert exc.value.code is OperationalErrorCode.INVALID_EVENT
    with pytest.raises(OperationalError):
        validate_event_payload(FOCUS_SET, ["not", "a", "mapping"])  # type: ignore[arg-type]


VALID_PAYLOADS = {
    FOCUS_SET: {"project": "memo", "summary": "ship"},
    FOCUS_CLEARED: {"project": "memo"},
    HANDOFF_CREATED: {
        "id": "h1",
        "project": "memo",
        "summary": "continue",
        "from_actor": "a",
        "to_actor": "b",
    },
    HANDOFF_CONSUMED: {"id": "h1", "actor_id": "b"},
    ATTENTION_ADDED: {
        "id": "a1",
        "project": "memo",
        "summary": "review",
        "severity": "high",
    },
    ATTENTION_ACKNOWLEDGED: {"id": "a1", "actor_id": "a"},
    CONFLICT_OPENED: {
        "id": "c1",
        "topic": "release",
        "summary": "mismatch",
        "freeze_write": True,
        "evidence_uris": ["memo://evidence/1"],
    },
    CONFLICT_RESOLVED: {"id": "c1", "resolution": "fixed", "actor_id": "a"},
    OUTCOME_RECORDED: {
        "id": "o1",
        "project": "memo",
        "summary": "released",
        "status": "success",
    },
    SESSION_CHECKPOINTED: {
        "session_id": "s1",
        "principal_id": "p1",
        "status": "active",
        "checkpointed_at": "2026-07-29T12:00:00Z",
    },
    SESSION_STATUS_CHANGED: {
        "session_id": "s1",
        "status": "recoverable",
    },
    COORDINATION_CREATED: {"task_id": "t1", "summary": "work", "status": "pending"},
    COORDINATION_CLAIMED: {"task_id": "t1", "principal_id": "p1"},
    COORDINATION_COMPLETED: {"task_id": "t1", "status": "completed"},
    DELIVERY_ENQUEUED: {
        "message_id": "m1",
        "recipient": "p1",
        "payload_sha256": "a" * 64,
    },
    DELIVERY_ACKNOWLEDGED: {"message_id": "m1", "recipient": "p1"},
    CURSOR_ADVANCED: {"cursor": "delivery", "position": 1},
    PRESENCE_UPDATED: {
        "principal_id": "p1",
        "status": "online",
        "expires_at": "2026-07-29T12:05:00Z",
    },
    PRESENCE_EXPIRED: {"principal_id": "p1"},
    TERMINAL_COMMAND_STARTED: {
        "command_id": "cmd1",
        "command_sha256": "b" * 64,
    },
    TERMINAL_COMMAND_FINISHED: {"command_id": "cmd1", "exit_code": 0},
    HEALTH_REPORTED: {"component": "daemon", "status": "healthy"},
    ROSTER_UPDATED: {"version": 2, "roster_hash": "c" * 64},
    COMPACTION_COMPLETED: {
        "origin_device": "device-a",
        "through_sequence": 4,
        "anchor_hash": "d" * 64,
    },
    DURABLE_PROMOTION_ENQUEUED: {
        "promotion_id": "pr1",
        "operation_key": "op1",
    },
    DURABLE_PROMOTION_COMPLETED: {
        "promotion_id": "pr1",
        "memory_id": "mem1",
    },
}


@pytest.mark.parametrize(("event_type", "payload"), VALID_PAYLOADS.items())
def test_each_registered_event_has_a_specific_valid_payload(
    event_type: str, payload: dict[str, object]
) -> None:
    validate_event_payload(event_type, payload)


@pytest.mark.parametrize("event_type", EVENT_TYPES)
def test_each_registered_event_rejects_empty_payload(event_type: str) -> None:
    with pytest.raises(OperationalError) as exc:
        validate_event_payload(event_type, {})
    assert exc.value.code is OperationalErrorCode.INVALID_EVENT


@pytest.mark.parametrize(
    ("event_type", "payload"),
    [
        (FOCUS_SET, {"project": "memo", "summary": 7}),
        (ATTENTION_ADDED, {**VALID_PAYLOADS[ATTENTION_ADDED], "severity": "urgent"}),
        (CONFLICT_OPENED, {**VALID_PAYLOADS[CONFLICT_OPENED], "freeze_write": "yes"}),
        (CURSOR_ADVANCED, {"cursor": "delivery", "position": -1}),
        (PRESENCE_UPDATED, {**VALID_PAYLOADS[PRESENCE_UPDATED], "status": "maybe"}),
        (TERMINAL_COMMAND_FINISHED, {"command_id": "cmd1", "exit_code": True}),
    ],
)
def test_payload_validators_reject_wrong_types_enums_and_invariants(
    event_type: str, payload: dict[str, object]
) -> None:
    with pytest.raises(OperationalError):
        validate_event_payload(event_type, payload)
