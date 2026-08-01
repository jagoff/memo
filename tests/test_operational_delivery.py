from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from memo.config import Config
from memo.errors import OperationalError
from memo.identity import PrincipalIdentity
from memo.operational_activation import activate_fresh_operational_v2
from memo.operational_coordination import CoordinationService
from memo.operational_delivery import DeliveryService
from tests.operational_authority import build_test_fresh_v2_authority


def test_cross_origin_transition_reduces_after_its_message_even_if_sorted_first() -> None:
    message_event_id = "event-from-device-b"
    delivery_id = hashlib.sha256(f"{message_event_id}\0agent-a".encode()).hexdigest()
    transition = SimpleNamespace(
        event_type="memo.operational.delivery.reserved.v1",
        event_id="event-from-device-a",
        expires_at=None,
        payload={
            "delivery_id": delivery_id,
            "message_id": "message-b",
            "target_id": "agent-a",
            "transitioned_at": "2026-07-31T12:00:01Z",
            "attempt_count": 1,
            "terminal_id": "terminal-a",
            "error_code": "",
        },
    )
    message = SimpleNamespace(
        event_type="memo.operational.coord.message.sent.v1",
        event_id=message_event_id,
        expires_at=None,
        payload={
            "message_id": "message-b",
            "target_ids": ("agent-a",),
            "created_at": "2026-07-31T12:00:00Z",
            "channel": "handoff",
        },
    )
    ledger = SimpleNamespace(validated_events=lambda: (transition, message))
    store = SimpleNamespace(backend_version=2, ledger=ledger)
    service = DeliveryService(store, context_factory=lambda _identity: None)  # type: ignore[arg-type]

    delivery = service.deliveries(message_id="message-b")[0]

    assert delivery.state == "reserved"
    assert delivery.terminal_id == "terminal-a"


@pytest.fixture
def delivery_runtime(tmp_path):
    cfg = Config(
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        device_id="device-a",
        reranker_enabled=False,
    )
    test_authority = build_test_fresh_v2_authority(
        cfg.operational_root,
        device_id=cfg.device_id,
    )
    authority = test_authority.runtime_authority()
    store = activate_fresh_operational_v2(cfg, authority=authority)
    stamp = json.loads(
        (cfg.operational_root / "operational-v2-activated.json").read_text()
    )
    instant = datetime(2026, 7, 31, 12, tzinfo=UTC)

    def identity(actor: str) -> PrincipalIdentity:
        return PrincipalIdentity(
            principal_id=f"principal:{actor}",
            actor_id=actor,
            kind="agent",
            device_id=cfg.device_id,
            session_id=f"session:{actor}",
            source_client="pytest",
        )

    def context(principal: PrincipalIdentity):
        return authority.fence.context(
            principal,
            request_epoch=stamp["authority_epoch"],
            request_control_oid=stamp["control_oid"],
        )

    sender = identity("agent-a")
    target = identity("agent-b")
    daemon = identity("memo-daemon")
    coordination = CoordinationService(store, context_factory=context, clock=lambda: instant)
    delivery = DeliveryService(store, context_factory=context, clock=lambda: instant)
    return coordination, delivery, sender, target, daemon, instant


def test_message_creates_one_pending_delivery_per_recipient_and_ack_is_idempotent(
    delivery_runtime,
) -> None:
    coordination, delivery, sender, target, _, _ = delivery_runtime
    message = coordination.send_message(
        identity=sender,
        channel="handoff",
        body="resume",
        target_ids=(target.actor_id, "agent-c"),
        expects_ack=True,
        idempotency_key="message-1",
    )

    pending = delivery.deliveries(message_id=message.message_id)
    assert [item.target_id for item in pending] == ["agent-b", "agent-c"]
    assert {item.state for item in pending} == {"pending"}

    first = delivery.acknowledge(
        identity=target,
        message_id=message.message_id,
        idempotency_key="ack-1",
    )
    second = delivery.acknowledge(
        identity=target,
        message_id=message.message_id,
        idempotency_key="ack-1",
    )
    assert first.state == "acknowledged"
    assert second == first


def test_delivery_retry_and_terminal_states_are_monotonic(delivery_runtime) -> None:
    coordination, delivery, sender, target, daemon, instant = delivery_runtime
    message = coordination.send_message(
        identity=sender,
        channel="ops",
        body="present",
        target_ids=(target.actor_id,),
        idempotency_key="message-retry",
    )
    pending = delivery.deliveries(message_id=message.message_id)[0]
    reserved = delivery.reserve_due(identity=daemon, now=instant)[0]
    assert reserved.state == "reserved"
    failed = delivery.transition(
        identity=daemon,
        delivery_id=pending.id,
        state="known_failed",
        error_code="tty_busy",
        idempotency_key="failed-1",
        at=instant,
    )
    assert failed.next_attempt_at is not None
    assert delivery.reserve_due(identity=daemon, now=instant) == []
    retried = delivery.reserve_due(
        identity=daemon,
        now=instant + timedelta(seconds=1),
    )[0]
    presented = delivery.transition(
        identity=daemon,
        delivery_id=retried.id,
        state="presented",
        terminal_id="term-1",
        idempotency_key="presented-2",
        at=instant + timedelta(seconds=1),
    )
    acknowledged = delivery.acknowledge(
        identity=target,
        message_id=message.message_id,
        idempotency_key="ack-retry",
    )

    assert retried.attempt_count == 2
    assert presented.state == "presented"
    assert acknowledged.state == "acknowledged"
    with pytest.raises(OperationalError, match="not allowed"):
        delivery.transition(
            identity=daemon,
            delivery_id=retried.id,
            state="presented",
            idempotency_key="regress",
        )


def test_cursor_cannot_regress_and_unread_is_event_derived(delivery_runtime) -> None:
    coordination, delivery, sender, target, _, _ = delivery_runtime
    first = coordination.send_message(
        identity=sender,
        channel="handoff",
        body="one",
        target_ids=(target.actor_id,),
        idempotency_key="cursor-message-1",
    )
    coordination.send_message(
        identity=sender,
        channel="handoff",
        body="two",
        target_ids=(target.actor_id,),
        idempotency_key="cursor-message-2",
    )
    assert delivery.unread_count(identity=target, channel="handoff") == 2
    cursor = delivery.advance_cursor(
        identity=target,
        channel="handoff",
        logical_clock="10-0-device-a",
        event_id=first.event_id,
        idempotency_key="cursor-10",
    )
    assert cursor.event_id == first.event_id
    assert delivery.unread_count(identity=target, channel="handoff") == 1
    with pytest.raises(OperationalError, match="regress"):
        delivery.advance_cursor(
            identity=target,
            channel="handoff",
            logical_clock="9-0-device-a",
            event_id="old",
            idempotency_key="cursor-9",
        )
