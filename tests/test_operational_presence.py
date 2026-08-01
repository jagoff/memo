from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from memo.config import Config
from memo.contracts import Visibility
from memo.errors import OperationalError
from memo.identity import PrincipalIdentity
from memo.operational_activation import activate_fresh_operational_v2
from memo.operational_event import OperationalCommand
from memo.operational_event_types import PRESENCE_LEASE_EXPIRED
from memo.operational_presence import PresenceService
from tests.operational_authority import build_test_fresh_v2_authority


class Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 7, 31, 12, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: int) -> None:
        self.value += timedelta(seconds=seconds)


@pytest.fixture
def presence_runtime(tmp_path):
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
    stamp = json.loads((cfg.operational_root / "operational-v2-activated.json").read_text())
    clock = Clock()

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

    return PresenceService(store, context_factory=context, clock=clock), identity, clock


def test_announce_clamps_ttl_and_renew_keeps_lease(presence_runtime) -> None:
    service, identity, _ = presence_runtime
    actor = identity("agent-a")
    lease = service.announce(
        identity=actor,
        project="memo",
        workspace="/work/memo",
        topic="ledger",
        intent="editing",
        files=("src/memo/a.py",),
        ttl_seconds=1,
        idempotency_key="presence-1",
    )
    renewed = service.renew(
        identity=actor,
        lease_id=lease.id,
        ttl_seconds=60,
        idempotency_key="presence-2",
    )

    assert lease.ttl_seconds == 5
    assert renewed.id == lease.id
    assert renewed.ttl_seconds == 60
    assert renewed.heartbeat_interval() == 15.0


def test_conflicts_ignore_self_and_expired_leases(presence_runtime) -> None:
    service, identity, clock = presence_runtime
    a = identity("agent-a")
    b = identity("agent-b")
    service.announce(
        identity=a,
        project="memo",
        workspace="repo:main",
        topic="ledger",
        intent="editing",
        files=("src/memo/a.py",),
        ttl_seconds=60,
        idempotency_key="a-1",
    )
    service.announce(
        identity=b,
        project="memo",
        workspace="repo:main",
        topic="ledger",
        intent="editing",
        files=("src/memo/a.py",),
        ttl_seconds=60,
        idempotency_key="b-1",
    )

    assert (
        len(
            service.conflicts(
                project="memo",
                files=("src/memo/a.py",),
                now=clock(),
            )
        )
        == 1
    )
    clock.advance(61)
    assert service.active(project="memo", now=clock()) == []


def test_owner_and_path_validation_fail_closed(presence_runtime) -> None:
    service, identity, _ = presence_runtime
    a = identity("agent-a")
    b = identity("agent-b")
    lease = service.announce(
        identity=a,
        project="memo",
        workspace="repo:main",
        topic="ledger",
        intent="editing",
        files=("src/memo/a.py",),
        idempotency_key="owned",
    )
    with pytest.raises(OperationalError, match="owner"):
        service.renew(
            identity=b,
            lease_id=lease.id,
            idempotency_key="foreign-renew",
        )
    with pytest.raises(OperationalError, match="owner"):
        service.expire(
            identity=b,
            lease_id=lease.id,
            idempotency_key="foreign-expire",
        )
    with pytest.raises(ValueError, match="relative"):
        service.announce(
            identity=a,
            project="memo",
            workspace="repo:main",
            topic="bad",
            intent="editing",
            files=("../secret",),
            idempotency_key="bad-path",
        )


def test_imported_foreign_actor_cannot_expire_presence(presence_runtime) -> None:
    service, identity, clock = presence_runtime
    owner = identity("agent-a")
    attacker = identity("agent-b")
    lease = service.announce(
        identity=owner,
        project="memo",
        workspace="repo:main",
        topic="ledger",
        intent="editing",
        files=("src/memo/a.py",),
        idempotency_key="owned-import",
    )
    service.store.commit(
        OperationalCommand(
            event_type=PRESENCE_LEASE_EXPIRED,
            actor=attacker,
            target_id=lease.id,
            project=lease.project,
            workspace=lease.workspace,
            expires_at=None,
            visibility=Visibility.SHARED.value,
            idempotency_key="imported-foreign-expire",
            caused_by=(),
            subject_uri=f"memo://presence/{lease.id}",
            trace_id="",
            payload={
                "id": lease.id,
                "expired_at": clock().isoformat().replace("+00:00", "Z"),
            },
        ),
        context=service.context_factory(attacker),
    )

    assert [row.id for row in service.active(project="memo")] == [lease.id]
