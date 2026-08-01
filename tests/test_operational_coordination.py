from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from memo.config import Config
from memo.errors import OperationalError
from memo.identity import PrincipalIdentity
from memo.operational_activation import activate_fresh_operational_v2
from memo.operational_coordination import CoordinationService
from tests.operational_authority import build_test_fresh_v2_authority


@pytest.fixture
def coordination(tmp_path):
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

    service = CoordinationService(
        store,
        context_factory=context,
        clock=lambda: datetime(2026, 7, 31, 12, tzinfo=UTC),
    )
    return service, identity("agent-a"), identity("agent-b")


def test_send_handoff_and_consume_are_idempotent(coordination) -> None:
    service, actor, target = coordination
    kwargs = {
        "identity": actor,
        "channel": "handoff",
        "body": "resume here",
        "target_ids": (target.actor_id,),
        "topic": "absorption",
        "evidence_uris": ("commit:abc",),
        "expects_ack": True,
        "idempotency_key": "send-1",
    }
    first = service.send_message(**kwargs)
    second = service.send_message(**kwargs)

    assert second == first
    assert service.messages(channel="handoff") == [first]

    handoff = service.create_handoff(
        identity=actor,
        message_id=first.message_id,
        project="memo",
        summary="continue absorption",
        to_actor=target.actor_id,
        evidence_uris=("commit:abc",),
        idempotency_key="handoff-1",
    )
    consumed = service.consume_handoff(
        identity=target,
        handoff_id=handoff.id,
        idempotency_key="consume-1",
    )
    replay = service.consume_handoff(
        identity=target,
        handoff_id=handoff.id,
        idempotency_key="consume-1",
    )

    assert consumed.status == "consumed"
    assert replay == consumed
    assert consumed.evidence_uris == ("commit:abc",)


def test_handoff_target_and_task_lifecycle_are_monotonic(coordination) -> None:
    service, actor, target = coordination
    message = service.send_message(
        identity=actor,
        channel="handoff",
        body="targeted",
        target_ids=(target.actor_id,),
        idempotency_key="send-target",
    )
    handoff = service.create_handoff(
        identity=actor,
        message_id=message.message_id,
        project="memo",
        summary="targeted",
        to_actor=target.actor_id,
        idempotency_key="handoff-target",
    )
    with pytest.raises(OperationalError, match="target"):
        service.consume_handoff(
            identity=actor,
            handoff_id=handoff.id,
            idempotency_key="wrong-target",
        )

    task = service.create_task(
        identity=actor,
        project="memo",
        title="prove integration",
        assignee_id=target.actor_id,
        caused_by=message.event_id,
        idempotency_key="task-1",
    )
    completed = service.complete_task(
        identity=target,
        task_id=task.id,
        result="green",
        idempotency_key="task-complete-1",
    )
    replay = service.complete_task(
        identity=target,
        task_id=task.id,
        result="green",
        idempotency_key="task-complete-1",
    )

    assert completed.status == "completed"
    assert completed.result == "green"
    assert replay == completed
    with pytest.raises(OperationalError, match="terminal"):
        service.cancel_task(
            identity=actor,
            task_id=task.id,
            idempotency_key="task-cancel-late",
        )


def test_supersede_and_topic_termination_are_derived(coordination) -> None:
    service, actor, _ = coordination
    first = service.send_message(
        identity=actor,
        channel="ops",
        body="old",
        topic="cutover",
        idempotency_key="old",
    )
    replacement = service.send_message(
        identity=actor,
        channel="ops",
        body="new",
        topic="cutover",
        idempotency_key="new",
    )
    superseded = service.supersede_message(
        identity=actor,
        message_id=first.message_id,
        superseded_by_message_id=replacement.message_id,
        idempotency_key="supersede",
    )
    channel = service.terminate_topic(
        identity=actor,
        channel="ops",
        topic="cutover",
        idempotency_key="terminate",
    )

    assert superseded.superseded_by_message_id == replacement.message_id
    assert channel.status == "terminated"
