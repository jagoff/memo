from __future__ import annotations

import pytest

from memo.contracts import ActorIdentity
from memo.error_contract import OperationalError, OperationalErrorCode
from memo.operational import OperationalStore
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
    assert all(
        name.startswith("memo.operational.") and name.endswith(".v1") for name in EVENT_TYPES
    )


def test_registry_validates_mapping_payload_and_rejects_unknown_names() -> None:
    validate_event_payload(FOCUS_SET, {"project": "memo", "summary": "ship"})
    with pytest.raises(OperationalError) as exc:
        validate_event_payload("focus.set", {})
    assert exc.value.code is OperationalErrorCode.INVALID_EVENT
    with pytest.raises(OperationalError):
        validate_event_payload(FOCUS_SET, ["not", "a", "mapping"])  # type: ignore[arg-type]


# Golden payloads emitted by the v1 producers in memo.operational. These are
# deliberately complete producer shapes rather than a second hand-written
# schema for the migration seed path.
REAL_V1_PAYLOADS = {
    FOCUS_SET: {
        "id": "focus-1",
        "project": "memo",
        "summary": "ship",
        "updated_at": "2026-07-29T12:00:00Z",
        "actor_id": "memo",
        "metadata": {},
    },
    FOCUS_CLEARED: {
        "project": "memo",
        "cleared_at": "2026-07-29T12:01:00Z",
    },
    HANDOFF_CREATED: {
        "id": "h1",
        "project": "memo",
        "summary": "continue",
        "from_actor": "a",
        "to_actor": "",
        "created_at": "2026-07-29T12:02:00Z",
        "consumed_at": "",
        "metadata": {},
    },
    HANDOFF_CONSUMED: {
        "id": "h1",
        "consumed_at": "2026-07-29T12:03:00Z",
    },
    ATTENTION_ADDED: {
        "id": "a1",
        "project": "memo",
        "summary": "review",
        "severity": "high",
        "created_at": "2026-07-29T12:04:00Z",
        "acknowledged_at": "",
        "metadata": {},
    },
    ATTENTION_ACKNOWLEDGED: {
        "id": "a1",
        "acknowledged_at": "2026-07-29T12:05:00Z",
    },
    CONFLICT_OPENED: {
        "id": "c1",
        "topic": "release",
        "summary": "mismatch",
        "lifecycle_state": "detected",
        "freeze_write": True,
        "created_at": "2026-07-29T12:06:00Z",
        "resolved_at": "",
        "resolution": "",
        "evidence_uris": ["memo://evidence/1"],
        "metadata": {},
    },
    CONFLICT_RESOLVED: {
        "id": "c1",
        "resolved_at": "2026-07-29T12:07:00Z",
        "resolution": "fixed",
    },
    OUTCOME_RECORDED: {
        "task_id": "task-1",
        "status": "success",
        "memory_ids": ["mem-1"],
        "artifacts": ["memo://artifact/1"],
        "environment": {"branch": "main"},
        "actor_id": "memo",
        "idempotency_key": "outcome-1",
        "recorded_at": "2026-07-29T12:08:00Z",
    },
}


V2_NATIVE_PAYLOADS = {
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


@pytest.mark.parametrize(("event_type", "payload"), REAL_V1_PAYLOADS.items())
def test_real_v1_producer_payloads_are_accepted_for_migration_seed(
    event_type: str, payload: dict[str, object]
) -> None:
    validate_event_payload(event_type, payload)


def test_current_v1_producers_emit_migration_seed_payloads_accepted_by_v2(
    tmp_path,
) -> None:
    store = OperationalStore(tmp_path, device_id="device-a")
    store.set_focus(project="memo", summary="ship")
    store.clear_focus("memo")
    handoff = store.create_handoff(
        project="memo",
        summary="continue",
        from_actor="agent-a",
    )
    assert store.consume_handoff(handoff.id, actor_id="agent-b")
    attention = store.add_attention(
        project="memo",
        summary="review",
        severity="high",
    )
    assert store.acknowledge_attention(attention.id, actor_id="agent-a")
    conflict = store.open_conflict(
        topic="release",
        summary="mismatch",
        evidence_uris=["memo://evidence/1"],
    )
    assert store.resolve_conflict(
        conflict.id,
        resolution="fixed",
        actor=ActorIdentity(actor_id="human-a", actor_kind="human"),
    )
    store.record_outcome(
        task_id="task-1",
        status="success",
        memory_ids=["mem-1"],
    )

    event_types = {
        "focus.set": FOCUS_SET,
        "focus.clear": FOCUS_CLEARED,
        "handoff.create": HANDOFF_CREATED,
        "handoff.consume": HANDOFF_CONSUMED,
        "attention.add": ATTENTION_ADDED,
        "attention.ack": ATTENTION_ACKNOWLEDGED,
        "conflict.open": CONFLICT_OPENED,
        "conflict.resolve": CONFLICT_RESOLVED,
        "outcome.record": OUTCOME_RECORDED,
    }
    emitted = store.ledger.validated_events()
    assert {event.op for event in emitted} == set(event_types)
    for event in emitted:
        validate_event_payload(event_types[event.op], event.payload)


@pytest.mark.parametrize(("event_type", "payload"), V2_NATIVE_PAYLOADS.items())
def test_each_v2_native_event_has_a_specific_valid_payload(
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
        (ATTENTION_ADDED, {**REAL_V1_PAYLOADS[ATTENTION_ADDED], "severity": "urgent"}),
        (CONFLICT_OPENED, {**REAL_V1_PAYLOADS[CONFLICT_OPENED], "freeze_write": "yes"}),
        (CURSOR_ADVANCED, {"cursor": "delivery", "position": -1}),
        (PRESENCE_UPDATED, {**V2_NATIVE_PAYLOADS[PRESENCE_UPDATED], "status": "maybe"}),
        (TERMINAL_COMMAND_FINISHED, {"command_id": "cmd1", "exit_code": True}),
    ],
)
def test_payload_validators_reject_wrong_types_enums_and_invariants(
    event_type: str, payload: dict[str, object]
) -> None:
    with pytest.raises(OperationalError):
        validate_event_payload(event_type, payload)
