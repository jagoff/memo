"""Reproducible hermetic two-peer integration proof for Memo operations."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from memo.atomic_io import atomic_write_text
from memo.definitive_integration import (
    sign_integration_receipt,
    verify_integration_receipt,
    write_integration_receipt,
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
    root: Path
    identity: PrincipalIdentity
    context: CommitContext
    store: OperationalStore
    coordination: CoordinationService
    sync: OperationalSync
    signer: OperationalSigner
    key_id: str
    roster: VerificationRoster
    keys: DeviceKeyStore
    pins: AuthorityPinStore
    fence: EpochFence


class _Presenter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def present(self, *, terminal_id: str, mode: str, payload: str) -> PresenterOutcome:
        del terminal_id, mode
        self.calls.append(payload)
        return PresenterOutcome(state="presented")


class _NoSessions:
    def latest_recoverable(self, *, project: str | None, workspace: str | None) -> None:
        del project, workspace


def _pins(root: Path) -> AuthorityPinStore:
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
        attempt_id=f"hermetic-bootstrap-{device_id}",
        device_id=device_id,
        epoch=0,
        control_oid="hermetic-control-0",
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
    now: datetime,
) -> _Peer:
    identity = PrincipalIdentity(
        principal_id=f"{device_id}:hermetic-session",
        actor_id=actor_id,
        kind="agent",
        device_id=device_id,
        session_id="hermetic-session",
        source_client="memo-definitive",
    )
    context = fence.context(
        identity,
        request_epoch=0,
        request_control_oid="hermetic-control-0",
    )
    signer = OperationalSigner(keys, roster_version=roster.version)
    ledger = OperationLedgerV2(
        root / "state",
        device_id=device_id,
        clock=lambda: now,
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
        clock=lambda: now,
    )
    sync = OperationalSync(
        store,
        transport=transport,
        device_id=device_id,
        context_factory=lambda: context,
        clock=lambda: now,
    )
    return _Peer(
        root=root,
        identity=identity,
        context=context,
        store=store,
        coordination=coordination,
        sync=sync,
        signer=signer,
        key_id=roster.local_key_id,
        roster=roster,
        keys=keys,
        pins=pins,
        fence=fence,
    )


def _build_two_peers(root: Path, *, now: datetime) -> tuple[_Peer, _Peer, GitTransport]:
    keys = DeviceKeyStore.in_memory()
    key_a = keys.generate(device_id="device-a", roles=("origin",))
    key_b = keys.generate(device_id="device-b", roles=("origin",))
    transport = GitTransport(root / "remote")

    authority_a = root / "a" / "authority"
    pins_a = _pins(authority_a)
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

    authority_b = root / "b" / "authority"
    pins_b = _pins(authority_b)
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
    return (
        _peer(
            root / "a",
            device_id="device-a",
            actor_id="agent-a",
            keys=keys,
            roster=roster_a,
            pins=pins_a,
            fence=fence_a,
            transport=transport,
            now=now,
        ),
        _peer(
            root / "b",
            device_id="device-b",
            actor_id="agent-b",
            keys=keys,
            roster=roster_b,
            pins=pins_b,
            fence=fence_b,
            transport=transport,
            now=now,
        ),
        transport,
    )


def _restarted_state(peer: _Peer, *, now: datetime) -> dict[str, object]:
    ledger = OperationLedgerV2(
        peer.root / "state",
        device_id=peer.identity.device_id,
        clock=lambda: now,
        signer=peer.signer,
        verifier=OperationalVerifier(),
        roster=peer.roster,
        roster_root=peer.root / "authority",
        pin_store=peer.pins,
        epoch_fence=peer.fence,
    )
    verification = ledger.verify()
    if not verification.ok:
        raise OperationalError(
            code="invalid_event",  # type: ignore[arg-type]
            message="restarted operational ledger verification failed",
            retryable=False,
        )
    views = OperationalViewStore(peer.root / "views-restarted.sqlite")
    views.catch_up(ledger)
    return views.state()


def run_hermetic_integration_proof(*, receipt_path: Path) -> dict[str, object]:
    """Run the full two-peer path and persist its signed empirical receipt."""

    started = time.perf_counter()
    now = datetime(2026, 7, 31, 12, tzinfo=UTC)
    destination = Path(receipt_path).resolve()
    with tempfile.TemporaryDirectory(prefix="memo-definitive-integration-") as temporary:
        root = Path(temporary).resolve()
        a, b, transport = _build_two_peers(root, now=now)
        sent = a.coordination.send_message(
            identity=a.identity,
            channel="handoff",
            body="Memo native integration proof",
            target_ids=("agent-b",),
            expects_ack=True,
            idempotency_key="proof-message-1",
        )
        a.sync.publish()
        head_a = transport.read_head("device-a")
        assert head_a is not None
        first_segment = transport.segment_path(head_a, 1)
        original_segment = first_segment.read_bytes()
        first_segment.unlink()
        gap_report = b.sync.ingest()
        first_segment.write_bytes(original_segment)
        recovery = b.sync.recover_gap(device_id="device-a", expected_sequence=1)

        tampered = json.loads(original_segment.decode("utf-8"))
        tampered["payload"]["body"] = "tampered"
        first_segment.write_bytes(canonical_json_bytes(tampered) + b"\n")
        tamper_rejected = False
        try:
            b.sync.ingest()
        except OperationalError:
            tamper_rejected = True
        finally:
            first_segment.write_bytes(original_segment)

        delivery_b = DeliveryService(
            b.store,
            context_factory=lambda _identity: b.context,
            clock=lambda: now,
        )
        reserved = delivery_b.reserve_due(identity=b.identity, now=now)
        presenter_b = _Presenter()
        terminal_b = TerminalBridge(
            root / "terminal-b.sqlite",
            presenter=presenter_b,
            clock=lambda: now,
        )
        terminal_b.register(
            TerminalRegistration(
                id="terminal-b",
                principal_id=b.identity.principal_id,
                session_id=b.identity.session_id,
                uid=os.getuid(),
                capabilities=("notify",),
                issued_at=now.isoformat(),
                expires_at=(now + timedelta(minutes=5)).isoformat(),
                nonce="proof-terminal-nonce",
                signature="proof-terminal-registration",
            ),
            peer_uid=os.getuid(),
        )
        request = TerminalPresentRequest(
            event_id=sent.event_id,
            message_id=sent.message_id,
            delivery_id=reserved[0].id,
            terminal_id="terminal-b",
            mode="notify",
            payload=sent.body,
            sanitized_payload_hash="",
            deadline=(now + timedelta(minutes=1)).isoformat(),
            idempotency_key="proof-present-1",
            principal_id=b.identity.principal_id,
            session_id=b.identity.session_id,
        )
        terminal_receipt_b = terminal_b.present(request, peer_uid=os.getuid())
        duplicate_receipt_b = terminal_b.present(request, peer_uid=os.getuid())
        delivery_b.transition(
            identity=b.identity,
            delivery_id=reserved[0].id,
            state="presented",
            terminal_id="terminal-b",
            idempotency_key="proof-delivery-presented-1",
            at=now,
        )
        acknowledged = delivery_b.acknowledge(
            identity=b.identity,
            message_id=sent.message_id,
            idempotency_key="proof-ack-1",
        )
        presence_b = PresenceService(
            b.store,
            context_factory=lambda _identity: b.context,
            clock=lambda: now,
        )
        lease_b = presence_b.announce(
            identity=b.identity,
            project="memo",
            workspace="/work/memo",
            topic="integration",
            intent="verifying",
            files=("src/memo/operational_sync.py",),
            ttl_seconds=60,
            idempotency_key="proof-presence-1",
        )
        response = b.coordination.send_message(
            identity=b.identity,
            channel="handoff",
            body="Terminal B observed A and replied through Memo",
            target_ids=("agent-a",),
            expects_ack=True,
            idempotency_key="proof-response-1",
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
        reserved_a = delivery_a.reserve_due(identity=a.identity, now=now)
        assert len(reserved_a) == 1
        presenter_a = _Presenter()
        terminal_a = TerminalBridge(
            root / "terminal-a.sqlite",
            presenter=presenter_a,
            clock=lambda: now,
        )
        terminal_a.register(
            TerminalRegistration(
                id="terminal-a",
                principal_id=a.identity.principal_id,
                session_id=a.identity.session_id,
                uid=os.getuid(),
                capabilities=("notify",),
                issued_at=now.isoformat(),
                expires_at=(now + timedelta(minutes=5)).isoformat(),
                nonce="proof-terminal-a-nonce",
                signature="proof-terminal-a-registration",
            ),
            peer_uid=os.getuid(),
        )
        request_a = TerminalPresentRequest(
            event_id=response.event_id,
            message_id=response.message_id,
            delivery_id=reserved_a[0].id,
            terminal_id="terminal-a",
            mode="notify",
            payload=response.body,
            sanitized_payload_hash="",
            deadline=(now + timedelta(minutes=1)).isoformat(),
            idempotency_key="proof-present-a-1",
            principal_id=a.identity.principal_id,
            session_id=a.identity.session_id,
        )
        terminal_receipt_a = terminal_a.present(request_a, peer_uid=os.getuid())
        duplicate_receipt_a = terminal_a.present(request_a, peer_uid=os.getuid())
        delivery_a.transition(
            identity=a.identity,
            delivery_id=reserved_a[0].id,
            state="presented",
            terminal_id="terminal-a",
            idempotency_key="proof-delivery-a-presented-1",
            at=now,
        )
        response_ack = delivery_a.acknowledge(
            identity=a.identity,
            message_id=response.message_id,
            idempotency_key="proof-response-ack-1",
        )
        lease_a = presence_a.announce(
            identity=a.identity,
            project="memo",
            workspace="/work/memo-a",
            topic="integration",
            intent="observing-terminal-b",
            files=("src/memo/terminal_bridge.py",),
            ttl_seconds=60,
            idempotency_key="proof-presence-a-1",
        )
        a.sync.publish()
        b.sync.ingest()
        a.sync.ingest()

        messages_a = a.coordination.messages(channel="handoff")
        messages_b = b.coordination.messages(channel="handoff")
        active_a = presence_a.active(project="memo", now=now)
        active_b = presence_b.active(project="memo", now=now)
        continuity = ContinuityComposer(
            durable_briefing=lambda **_: "Memo is the sole integrated runtime.",
            coordination=a.coordination,
            delivery=delivery_a,
            presence=presence_a,
            sessions=_NoSessions(),
            health=lambda: asdict(a.sync.status()),
            clock=lambda: now,
        ).compose(cwd="/work/memo")
        live_state = a.store.views.state()
        restarted_state = _restarted_state(a, now=now)
        head_a = transport.read_head("device-a")
        head_b = transport.read_head("device-b")
        assert head_a is not None and head_b is not None
        checks = {
            "gap_detected": gap_report.gaps == {"device-a": 1},
            "gap_recovered": recovery.recovered_events == 1
            and recovery.remaining_gap is None,
            "tamper_rejected": tamper_rejected,
            "two_peer_convergence": a.store.views.state() == b.store.views.state(),
            "terminal_mesh_bidirectional": presenter_b.calls == [sent.body]
            and presenter_a.calls == [response.body],
            "terminal_exactly_once": duplicate_receipt_b == terminal_receipt_b
            and duplicate_receipt_a == terminal_receipt_a,
            "terminal_a_observes_b": any(
                message.actor_id == "agent-b" and message.body == response.body
                for message in messages_a
            )
            and lease_b in active_a,
            "terminal_b_observes_a": any(
                message.actor_id == "agent-a" and message.body == sent.body
                for message in messages_b
            )
            and lease_a in active_b,
            "ack_roundtrip": delivery_a.status(acknowledged.id).state == "acknowledged",
            "acks_bidirectional": delivery_b.status(response_ack.id).state
            == "acknowledged",
            "presence_roundtrip": set(active_a) == {lease_a, lease_b}
            and set(active_b) == {lease_a, lease_b},
            "continuity_composed": continuity.durable_available
            and continuity.operational_available,
            "restart_rebuild": live_state == restarted_state,
        }
        evidence = {
            "git_head_a": head_a.event_hash,
            "git_head_b": head_b.event_hash,
            "terminal_receipt_a": terminal_receipt_a.receipt_hash,
            "terminal_receipt_b": terminal_receipt_b.receipt_hash,
            "continuity_sha256": hashlib.sha256(continuity.text.encode()).hexdigest(),
            "state_sha256": hashlib.sha256(canonical_json_bytes(live_state)).hexdigest(),
        }
        receipt = sign_integration_receipt(
            attempt_id=f"hermetic-{uuid.uuid4().hex}",
            checks=checks,
            evidence=evidence,
            signer_device_id="device-a",
            signer_key_id=a.key_id,
            roster_version=a.roster.version,
            created_at=now.isoformat(),
            signer=a.signer,
        )
        verify_integration_receipt(receipt, roster=a.roster)
        write_integration_receipt(receipt, destination)
        roster_path = destination.with_name(f"{destination.name}.roster.json")
        atomic_write_text(
            roster_path,
            json.dumps(a.roster.to_dict(), sort_keys=True, indent=2) + "\n",
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        return {
            "schema": "memo.definitive_integration_report.v1",
            "ok": all(checks.values()),
            "checks": checks,
            "evidence": evidence,
            "receipt_sha256": receipt.receipt_sha256,
            "receipt_path": str(destination),
            "roster_path": str(roster_path),
            "elapsed_ms": elapsed_ms,
            "scope": "hermetic two-peer process restart; physical Mac reboot excluded",
        }


__all__ = ["run_hermetic_integration_proof"]
