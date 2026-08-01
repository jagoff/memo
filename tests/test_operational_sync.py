from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from memo.definitive_integration import (
    sign_integration_receipt,
    verify_integration_receipt,
)
from memo.errors import OperationalError
from memo.git_transport import GitTransport
from memo.identity import PrincipalIdentity
from memo.operation_ledger_v2 import OperationLedgerV2
from memo.operation_views import OperationalViewStore
from memo.operational import OperationalStore
from memo.operational_continuity import ContinuityComposer
from memo.operational_coordination import CoordinationService
from memo.operational_delivery import DeliveryService
from memo.operational_epoch import CommitContext, EpochFence
from memo.operational_event import (
    EpochMarkerAuthorization,
    canonical_json_bytes,
    canonical_signed_bytes,
)
from memo.operational_key_store import (
    AuthorityPinStore,
    DeviceKeyStore,
    InMemoryAuthorityPinProvider,
)
from memo.operational_presence import PresenceService
from memo.operational_roster import VerificationRoster
from memo.operational_signing import OperationalSigner, OperationalVerifier
from memo.operational_sync import OperationalSync
from memo.terminal_bridge import (
    PresenterOutcome,
    TerminalBridge,
    TerminalPresentRequest,
    TerminalRegistration,
)


@dataclass(frozen=True)
class _Peer:
    identity: PrincipalIdentity
    context: CommitContext
    store: OperationalStore
    coordination: CoordinationService
    sync: OperationalSync
    signer: OperationalSigner
    key_id: str
    roster: VerificationRoster


def _pin_store(root: Path) -> AuthorityPinStore:
    return AuthorityPinStore._for_test(
        root,
        provider=InMemoryAuthorityPinProvider(),
    )


def _bootstrap_epoch(
    *,
    device_id: str,
    key_id: str,
    signer: OperationalSigner,
    fence: EpochFence,
) -> None:
    unsigned = EpochMarkerAuthorization(
        schema="memo.operational_epoch_authorization.v1",
        attempt_id=f"bootstrap-{device_id}",
        device_id=device_id,
        epoch=0,
        control_oid="control-0",
        artifact_digests={
            "bootstrap_roster": "a" * 64,
            "empty_anchor": "b" * 64,
        },
        roster_version=1,
        key_id=key_id,
        signature=None,  # type: ignore[arg-type]
    )
    authorization = replace(
        unsigned,
        signature=signer.sign(
            domain="memo.operational_epoch_authorization.v1",
            payload=canonical_signed_bytes(unsigned),
            key_id=key_id,
        ),
    )
    fence.bootstrap(
        authorization=authorization,
        observed_artifact_digests=authorization.artifact_digests,
    )


def _peer(
    root: Path,
    *,
    device_id: str,
    actor_id: str,
    keys: DeviceKeyStore,
    roster: VerificationRoster,
    pins: AuthorityPinStore,
    fence: EpochFence,
    transport: GitTransport,
) -> _Peer:
    identity = PrincipalIdentity(
        principal_id=f"{device_id}:session",
        actor_id=actor_id,
        kind="agent",
        device_id=device_id,
        session_id="session",
        source_client="pytest",
    )
    context = fence.context(
        identity,
        request_epoch=0,
        request_control_oid="control-0",
    )
    signer = OperationalSigner(keys, roster_version=2)
    ledger = OperationLedgerV2(
        root / "state",
        device_id=device_id,
        clock=lambda: datetime(2026, 7, 31, 12, tzinfo=UTC),
        signer=signer,
        verifier=OperationalVerifier(),
        roster=roster,
        roster_root=root / "authority",
        pin_store=pins,
        epoch_fence=fence,
    )
    views = OperationalViewStore(root / "views.sqlite")
    store = OperationalStore.for_v2(
        ledger=ledger,
        views=views,
        epoch_fence=fence,
        transaction_root=root / "transactions",
        context_provider=lambda: context,
    )
    coordination = CoordinationService(
        store,
        context_factory=lambda _identity: context,
        clock=lambda: datetime(2026, 7, 31, 12, tzinfo=UTC),
    )
    sync = OperationalSync(
        store,
        transport=transport,
        device_id=device_id,
        context_factory=lambda: context,
        clock=lambda: datetime(2026, 7, 31, 12, tzinfo=UTC),
    )
    return _Peer(
        identity,
        context,
        store,
        coordination,
        sync,
        signer,
        roster.local_key_id,
        roster,
    )


@pytest.fixture
def two_peer_runtime(tmp_path: Path) -> tuple[_Peer, _Peer, GitTransport]:
    keys = DeviceKeyStore.in_memory()
    key_a = keys.generate(device_id="device-a", roles=("origin",))
    key_b = keys.generate(device_id="device-b", roles=("origin",))
    transport = GitTransport(tmp_path / "remote")

    authority_a = tmp_path / "a" / "authority"
    pins_a = _pin_store(authority_a)
    roster_a_v1 = VerificationRoster.bootstrap(
        device_id="device-a",
        key=key_a,
        root=authority_a,
        pin_store=pins_a,
    )
    signer_a_v1 = OperationalSigner(keys, roster_version=1)
    fence_a = EpochFence(
        authority_a,
        roster=roster_a_v1,
        verifier=OperationalVerifier(),
        pin_store=pins_a,
    )
    _bootstrap_epoch(
        device_id="device-a",
        key_id=key_a.key_id,
        signer=signer_a_v1,
        fence=fence_a,
    )
    roster_a = roster_a_v1.with_keys(
        version=2,
        peers=("device-a", "device-b"),
        keys=(key_a, key_b),
        signer=signer_a_v1,
        root=authority_a,
        pin_store=pins_a,
    )

    authority_b = tmp_path / "b" / "authority"
    pins_b = _pin_store(authority_b)
    roster_b_v1 = VerificationRoster.bootstrap(
        device_id="device-b",
        key=key_b,
        root=authority_b,
        pin_store=pins_b,
    )
    signer_b_v1 = OperationalSigner(keys, roster_version=1)
    fence_b = EpochFence(
        authority_b,
        roster=roster_b_v1,
        verifier=OperationalVerifier(),
        pin_store=pins_b,
    )
    _bootstrap_epoch(
        device_id="device-b",
        key_id=key_b.key_id,
        signer=signer_b_v1,
        fence=fence_b,
    )
    roster_b = roster_b_v1.with_keys(
        version=2,
        peers=("device-a", "device-b"),
        keys=(key_b, key_a),
        signer=signer_b_v1,
        root=authority_b,
        pin_store=pins_b,
    )

    peer_a = _peer(
        tmp_path / "a",
        device_id="device-a",
        actor_id="agent-a",
        keys=keys,
        roster=roster_a,
        pins=pins_a,
        fence=fence_a,
        transport=transport,
    )
    peer_b = _peer(
        tmp_path / "b",
        device_id="device-b",
        actor_id="agent-b",
        keys=keys,
        roster=roster_b,
        pins=pins_b,
        fence=fence_b,
        transport=transport,
    )
    return peer_a, peer_b, transport


def _send(peer: _Peer, index: int, *, target: str) -> None:
    peer.coordination.send_message(
        identity=peer.identity,
        channel="handoff",
        body=f"message {index}",
        target_ids=(target,),
        idempotency_key=f"message-{peer.identity.device_id}-{index}",
    )


def test_single_writer_publish_and_incremental_ingest(two_peer_runtime) -> None:
    a, b, transport = two_peer_runtime
    _send(a, 1, target="agent-b")

    published = a.sync.publish()
    ingested = b.sync.ingest()

    assert published.published_events == 1
    assert ingested.ingested_events == 1
    assert b.store.views.state() == a.store.views.state()
    assert transport.read_head("device-a").sequence == 1  # type: ignore[union-attr]
    assert (transport.root / ".git").is_dir()

    duplicate = b.sync.ingest()
    assert duplicate.ingested_events == 0
    assert duplicate.duplicates == 1


def test_bidirectional_peers_converge_on_signed_events(two_peer_runtime) -> None:
    a, b, _transport = two_peer_runtime
    _send(a, 1, target="agent-b")
    a.sync.publish()
    b.sync.ingest()

    _send(b, 1, target="agent-a")
    b.sync.publish()
    a.sync.ingest()
    b.sync.ingest()

    assert a.store.views.state() == b.store.views.state()
    assert len(a.coordination.messages(channel="handoff")) == 2
    assert a.sync.status().health == b.sync.status().health == "healthy"


def test_gap_blocks_reduce_until_recovered(two_peer_runtime) -> None:
    a, b, transport = two_peer_runtime
    for index in range(1, 4):
        _send(a, index, target="agent-b")
    a.sync.publish()
    head = transport.read_head("device-a")
    assert head is not None
    missing = transport.segment_path(head, 2)
    saved = missing.read_bytes()
    missing.unlink()

    report = b.sync.ingest()
    assert report.gaps == {"device-a": 2}
    assert len(b.coordination.messages(channel="handoff")) == 1

    missing.write_bytes(saved)
    recovered = b.sync.recover_gap(device_id="device-a", expected_sequence=2)
    assert recovered.recovered_events == 2
    assert recovered.remaining_gap is None
    assert b.store.views.state() == a.store.views.state()


def test_tampered_remote_segment_fails_closed(two_peer_runtime) -> None:
    a, b, transport = two_peer_runtime
    _send(a, 1, target="agent-b")
    a.sync.publish()
    head = transport.read_head("device-a")
    assert head is not None
    segment = transport.segment_path(head, 1)
    value = json.loads(segment.read_text(encoding="utf-8"))
    value["payload"]["body"] = "attacker changed the message"
    segment.write_bytes(canonical_json_bytes(value) + b"\n")

    with pytest.raises(OperationalError, match=r"hash|signature|event"):
        b.sync.ingest()


def test_tampered_signed_head_fails_closed(two_peer_runtime) -> None:
    a, b, transport = two_peer_runtime
    _send(a, 1, target="agent-b")
    a.sync.publish()
    head_path = transport.root / "heads" / "device-a.json"
    value = json.loads(head_path.read_text(encoding="utf-8"))
    value["event_hash"] = "f" * 64
    head_path.write_bytes(canonical_json_bytes(value))

    with pytest.raises(OperationalError, match=r"head|chain"):
        b.sync.ingest()


class _Presenter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def present(self, *, terminal_id: str, mode: str, payload: str) -> PresenterOutcome:
        del terminal_id, mode
        self.calls.append(payload)
        return PresenterOutcome(state="presented")


class _NoSessions:
    def latest_recoverable(self, *, project, workspace):
        del project, workspace
        return None


def test_send_sync_present_ack_presence_continuity_and_signed_receipt(
    two_peer_runtime,
    tmp_path: Path,
) -> None:
    a, b, transport = two_peer_runtime
    now = datetime(2026, 7, 31, 12, tzinfo=UTC)
    sent = a.coordination.send_message(
        identity=a.identity,
        channel="handoff",
        body="continue the signed integration",
        target_ids=("agent-b",),
        expects_ack=True,
        idempotency_key="vertical-message-1",
    )
    a.sync.publish()
    b.sync.ingest()

    delivery_b = DeliveryService(
        b.store,
        context_factory=lambda _identity: b.context,
        clock=lambda: now,
    )
    reserved = delivery_b.reserve_due(identity=b.identity, now=now)
    assert len(reserved) == 1
    presenter = _Presenter()
    terminal = TerminalBridge(
        tmp_path / "terminal.sqlite",
        presenter=presenter,
        clock=lambda: now,
    )
    terminal.register(
        TerminalRegistration(
            id="terminal-b",
            principal_id=b.identity.principal_id,
            session_id=b.identity.session_id,
            uid=os.getuid(),
            capabilities=("notify",),
            issued_at=now.isoformat(),
            expires_at=(now + timedelta(minutes=5)).isoformat(),
            nonce="vertical-nonce",
            signature="vertical-registration-signature",
        ),
        peer_uid=os.getuid(),
    )
    terminal_receipt = terminal.present(
        TerminalPresentRequest(
            event_id=sent.event_id,
            message_id=sent.message_id,
            delivery_id=reserved[0].id,
            terminal_id="terminal-b",
            mode="notify",
            payload=sent.body,
            sanitized_payload_hash="",
            deadline=(now + timedelta(minutes=1)).isoformat(),
            idempotency_key="vertical-present-1",
            principal_id=b.identity.principal_id,
            session_id=b.identity.session_id,
        ),
        peer_uid=os.getuid(),
    )
    presented = delivery_b.transition(
        identity=b.identity,
        delivery_id=reserved[0].id,
        state="presented",
        terminal_id="terminal-b",
        idempotency_key="vertical-delivery-presented-1",
        at=now,
    )
    acknowledged = delivery_b.acknowledge(
        identity=b.identity,
        message_id=sent.message_id,
        idempotency_key="vertical-ack-1",
    )
    presence_b = PresenceService(
        b.store,
        context_factory=lambda _identity: b.context,
        clock=lambda: now,
    )
    lease = presence_b.announce(
        identity=b.identity,
        project="memo",
        workspace="/work/memo",
        topic="integration",
        intent="verifying",
        files=("src/memo/operational_sync.py",),
        ttl_seconds=60,
        idempotency_key="vertical-presence-1",
    )

    b.sync.publish()
    a.sync.ingest()
    delivery_a = DeliveryService(
        a.store,
        context_factory=lambda _identity: a.context,
        clock=lambda: now,
    )
    presence_a = PresenceService(
        a.store,
        context_factory=lambda _identity: a.context,
        clock=lambda: now,
    )
    continuity = ContinuityComposer(
        durable_briefing=lambda **_: "Memo is the sole integrated runtime.",
        coordination=a.coordination,
        delivery=delivery_a,
        presence=presence_a,
        sessions=_NoSessions(),
        health=lambda: asdict(a.sync.status()),
        clock=lambda: now,
    ).compose(cwd="/work/memo")
    before_rebuild = a.store.views.state()
    a.store.rebuild()
    after_rebuild = a.store.views.state()

    checks = {
        "two_peer_convergence": a.store.views.state() == b.store.views.state(),
        "terminal_exactly_once": presenter.calls == [sent.body],
        "delivery_presented": presented.state == "presented",
        "ack_roundtrip": delivery_a.status(acknowledged.id).state == "acknowledged",
        "presence_roundtrip": presence_a.active(project="memo", now=now) == [lease],
        "continuity_composed": (
            continuity.durable_available
            and continuity.operational_available
            and "Memo is the sole integrated runtime." in continuity.text
        ),
        "view_rebuild_stable": before_rebuild == after_rebuild,
    }
    assert all(checks.values()), checks
    evidence = {
        "git_head_a": transport.read_head("device-a").event_hash,  # type: ignore[union-attr]
        "git_head_b": transport.read_head("device-b").event_hash,  # type: ignore[union-attr]
        "terminal_receipt": terminal_receipt.receipt_hash,
        "continuity_sha256": hashlib.sha256(continuity.text.encode()).hexdigest(),
        "state_sha256": hashlib.sha256(canonical_json_bytes(after_rebuild)).hexdigest(),
    }
    receipt = sign_integration_receipt(
        attempt_id="hermetic-two-peer-vertical-1",
        checks=checks,
        evidence=evidence,
        signer_device_id="device-a",
        signer_key_id=a.key_id,
        roster_version=a.roster.version,
        created_at=now.isoformat(),
        signer=a.signer,
    )
    verify_integration_receipt(receipt, roster=a.roster)

    assert all(receipt.checks.values())
    assert receipt.signature is not None
    with pytest.raises(OperationalError, match="digest"):
        verify_integration_receipt(
            replace(receipt, evidence={**receipt.evidence, "state_sha256": "0" * 64}),
            roster=a.roster,
        )
