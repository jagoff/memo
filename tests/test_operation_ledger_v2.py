from __future__ import annotations

import base64
import hashlib
import json
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import pytest

from memo.errors import OperationalError
from memo.identity import PrincipalIdentity
from memo.operation_ledger_v1 import LegacyOperationLedger
from memo.operation_ledger_v2 import OperationLedgerV2
from memo.operational_epoch import CommitContext, EpochFence
from memo.operational_event import (
    EMPTY_REDUCER_STATE_BYTES,
    ChainAnchor,
    EpochMarkerAuthorization,
    LedgerImportReport,
    MigrationOrigin,
    OperationalCommand,
    OperationalEventV2,
    OriginBundle,
    SourceProof,
    canonical_anchor_hash,
    canonical_event_hash,
    canonical_json_bytes,
    canonical_signed_bytes,
)
from memo.operational_event_types import FOCUS_SET
from memo.operational_key_store import (
    AuthorityPinStore,
    DeviceKeyStore,
    InMemoryAuthorityPinProvider,
    PublicKeyRecord,
)
from memo.operational_roster import VerificationRoster
from memo.operational_signing import OperationalSigner, OperationalVerifier

_AUTHORITY_PINS = InMemoryAuthorityPinProvider()
_STAMP = "2026-07-29T12:00:00Z"


def _clock() -> str:
    return _STAMP


def _pin_store(root: Path) -> AuthorityPinStore:
    return AuthorityPinStore._for_test(root, provider=_AUTHORITY_PINS)


def _authorization(
    signer: OperationalSigner,
    *,
    device_id: str,
    key_id: str,
    epoch: int,
    control_oid: str,
    digests: dict[str, str],
) -> EpochMarkerAuthorization:
    unsigned = EpochMarkerAuthorization(
        schema="memo.operational_epoch_authorization.v1",
        attempt_id=f"attempt-{epoch}",
        device_id=device_id,
        epoch=epoch,
        control_oid=control_oid,
        artifact_digests=digests,
        roster_version=signer.roster_version,
        key_id=key_id,
        signature=None,  # type: ignore[arg-type]
    )
    envelope = signer.sign(
        domain="memo.operational_epoch_authorization.v1",
        payload=canonical_signed_bytes(unsigned),
        key_id=key_id,
    )
    return replace(unsigned, signature=envelope)


@dataclass
class _Authority:
    root: Path
    device_id: str
    keys: DeviceKeyStore
    origin_key: PublicKeyRecord
    roster: VerificationRoster
    signer: OperationalSigner
    verifier: OperationalVerifier
    pin_store: AuthorityPinStore
    fence: EpochFence

    def context(self) -> CommitContext:
        return self.fence.context(
            _identity(self.device_id),
            request_epoch=0,
            request_control_oid="control-0",
        )


def _authority(
    root: Path,
    *,
    device_id: str = "device-a",
    remote_devices: tuple[str, ...] = (),
) -> _Authority:
    keys = DeviceKeyStore.in_memory()
    origin_key = keys.generate(device_id=device_id, roles=("origin",))
    pins = _pin_store(root)
    roster = VerificationRoster.bootstrap(
        device_id=device_id,
        key=origin_key,
        root=root,
        pin_store=pins,
    )
    signer = OperationalSigner(keys, roster_version=1)
    verifier = OperationalVerifier()
    fence = EpochFence(
        root,
        roster=roster,
        verifier=verifier,
        pin_store=pins,
    )
    bootstrap = _authorization(
        signer,
        device_id=device_id,
        key_id=origin_key.key_id,
        epoch=0,
        control_oid="control-0",
        digests={"bootstrap_roster": "a" * 64, "empty_anchor": "b" * 64},
    )
    fence.bootstrap(
        authorization=bootstrap,
        observed_artifact_digests=bootstrap.artifact_digests,
    )
    if remote_devices:
        remote_keys = tuple(
            keys.generate(
                device_id=remote,
                roles=("origin",),
                enrollment_sequence=1,
            )
            for remote in remote_devices
        )
        peers = tuple(sorted((device_id, *remote_devices)))
        roster = roster.with_keys(
            version=2,
            peers=peers,
            keys=(origin_key, *remote_keys),
            signer=signer,
            root=root,
            pin_store=pins,
        )
        signer = OperationalSigner(keys, roster_version=2)
    return _Authority(
        root=root,
        device_id=device_id,
        keys=keys,
        origin_key=origin_key,
        roster=roster,
        signer=signer,
        verifier=verifier,
        pin_store=pins,
        fence=fence,
    )


def _ledger(
    root: Path,
    authority: _Authority,
    *,
    signer: OperationalSigner | None = None,
) -> OperationLedgerV2:
    return OperationLedgerV2(
        root,
        device_id=authority.device_id,
        clock=_clock,
        signer=signer or authority.signer,
        verifier=authority.verifier,
        roster=authority.roster,
        roster_root=authority.root,
        pin_store=authority.pin_store,
        epoch_fence=authority.fence,
    )


def _identity(device_id: str = "device-a") -> PrincipalIdentity:
    return PrincipalIdentity(
        principal_id=f"{device_id}:session-a",
        actor_id="agent-a",
        kind="agent",
        device_id=device_id,
        session_id="session-a",
        source_client="codex",
    )


def _command(
    *,
    device_id: str = "device-a",
    idempotency_key: str = "idem-1",
    summary: str = "Current focus",
    source_proof: SourceProof | None = None,
) -> OperationalCommand:
    return OperationalCommand(
        event_type=FOCUS_SET,
        actor=_identity(device_id),
        target_id=None,
        project="demo",
        workspace="/tmp/demo",
        expires_at=None,
        visibility="owner",
        idempotency_key=idempotency_key,
        caused_by=(),
        subject_uri="memo://focus/demo",
        trace_id="trace-1",
        payload={"project": "demo", "summary": summary},
        source_proof=source_proof,
    )


def _key_for(authority: _Authority, device_id: str, role: str = "origin") -> PublicKeyRecord:
    return next(
        key for key in authority.roster.keys if key.device_id == device_id and role in key.roles
    )


def _signed_anchor(
    authority: _Authority,
    *,
    origin: str,
    kind: str = "empty",
    checkpoint: bytes = EMPTY_REDUCER_STATE_BYTES,
    base_sequence: int = 0,
    base_hash: str = "",
    previous_anchor_hash: str = "",
    signer_role: str = "origin",
    key: PublicKeyRecord | None = None,
    signer: OperationalSigner | None = None,
    source_manifest_sha256: str = "",
) -> ChainAnchor:
    key = key or _key_for(authority, origin, signer_role)
    signer = signer or authority.signer
    digest = hashlib.sha256(checkpoint).hexdigest()
    unsigned = ChainAnchor(
        schema="memo.operational_anchor.v1",
        anchor_id=(
            f"anchor-{origin}-{kind}-{base_sequence}-{hashlib.sha256(checkpoint).hexdigest()[:12]}"
        ),
        origin_device=origin,
        ledger_epoch=0,
        reducer_version=1,
        kind=cast(object, kind),  # type: ignore[arg-type]
        base_sequence=base_sequence,
        base_event_hash=base_hash,
        final_sequence=base_sequence,
        final_event_hash=base_hash,
        previous_anchor_hash=previous_anchor_hash,
        source_manifest_sha256=source_manifest_sha256,
        state_sha256=digest,
        checkpoint_id=f"checkpoint-{origin}-{kind}-{base_sequence}",
        checkpoint_sha256=digest,
        checkpoint_size=len(checkpoint),
        created_at=_STAMP,
        anchor_hash="",
        roster_version=authority.roster.version,
        signer_role=cast(object, signer_role),  # type: ignore[arg-type]
        attested_origin=origin,
        key_id=key.key_id,
        signature="",
    )
    hashed = replace(unsigned, anchor_hash=canonical_anchor_hash(unsigned))
    envelope = signer.sign(
        domain="memo.operational.anchor.v1",
        payload=canonical_signed_bytes(hashed),
        key_id=key.key_id,
    )
    return replace(hashed, signature=envelope.signature)


def _signed_event(
    authority: _Authority,
    *,
    origin: str,
    sequence: int,
    previous_hash: str,
    summary: str | None = None,
    source_proof: SourceProof | None = None,
) -> OperationalEventV2:
    key = _key_for(authority, origin)
    payload = {
        "project": "demo",
        "summary": summary or f"focus-{sequence}",
    }
    unsigned = OperationalEventV2(
        schema="memo.operational_event.v2",
        schema_version=2,
        event_id=f"event-{origin}-{sequence}-{hashlib.sha256(canonical_json_bytes(payload)).hexdigest()[:8]}",
        event_type=FOCUS_SET,
        actor=_identity(origin),
        target_id=None,
        project="demo",
        workspace="/tmp/demo",
        origin_device=origin,
        origin_sequence=sequence,
        logical_clock=str(sequence),
        authority_epoch=0,
        control_oid="control-0",
        created_at=_STAMP,
        expires_at=None,
        visibility="owner",
        idempotency_key=f"idem-{origin}-{sequence}",
        caused_by=(),
        subject_uri="memo://focus/demo",
        trace_id=f"trace-{sequence}",
        payload=payload,
        content_hash=hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
        previous_hash=previous_hash,
        event_hash="",
        source_proof=source_proof,
        roster_version=authority.roster.version,
        key_id=key.key_id,
        signature="",
    )
    hashed = replace(unsigned, event_hash=canonical_event_hash(unsigned))
    envelope = authority.signer.sign(
        domain="memo.operational.event.v2",
        payload=canonical_signed_bytes(hashed),
        key_id=key.key_id,
    )
    return replace(hashed, signature=envelope.signature)


def _bundle(
    authority: _Authority,
    *,
    origin: str,
    sequences: tuple[int, ...] = (1, 2),
    checkpoint: bytes = EMPTY_REDUCER_STATE_BYTES,
) -> OriginBundle:
    anchor = _signed_anchor(authority, origin=origin, checkpoint=checkpoint)
    previous = anchor.base_event_hash
    events: list[OperationalEventV2] = []
    for sequence in sequences:
        event = _signed_event(
            authority,
            origin=origin,
            sequence=sequence,
            previous_hash=previous,
        )
        events.append(event)
        previous = event.event_hash
    return OriginBundle(
        anchor=anchor,
        checkpoint=checkpoint,
        events=tuple(events),
        head_sequence=events[-1].origin_sequence if events else anchor.base_sequence,
        head_hash=events[-1].event_hash if events else anchor.base_event_hash,
    )


def _add_attestor(authority: _Authority) -> PublicKeyRecord:
    attestor = authority.keys.generate(
        device_id=authority.device_id,
        roles=("migration_attestor",),
        enrollment_sequence=2,
    )
    authority.roster = authority.roster.with_keys(
        version=2,
        peers=authority.roster.peers,
        keys=(*authority.roster.keys, attestor),
        signer=authority.signer,
        root=authority.root,
        pin_store=authority.pin_store,
    )
    authority.signer = OperationalSigner(authority.keys, roster_version=2)
    return attestor


def test_constructor_is_usable_without_authority_but_sensitive_operations_fail_closed(
    tmp_path: Path,
) -> None:
    ledger = OperationLedgerV2(
        tmp_path / "operational",
        device_id="device-a",
        clock=_clock,
    )

    with pytest.raises(OperationalError, match="authority"):
        ledger.ensure_anchor()
    with pytest.raises(OperationalError, match="authority"):
        ledger.append(_command(), context=None)  # type: ignore[arg-type]
    assert not (tmp_path / "operational" / "journal").exists()


def test_empty_anchor_uses_exact_filesystem_and_checkpoint_contract(tmp_path: Path) -> None:
    authority = _authority(tmp_path / "authority")
    operational_root = tmp_path / "operational"
    ledger = _ledger(operational_root, authority)

    anchor = ledger.ensure_anchor()

    assert anchor.kind == "empty"
    assert anchor.base_sequence == 0
    assert anchor.base_event_hash == ""
    assert anchor.signer_role == "origin"
    assert ledger.anchor("device-a") == anchor
    assert ledger.position("device-a").sequence == 0
    assert (
        operational_root / "journal" / "checkpoints" / "device-a" / f"{anchor.anchor_id}.json"
    ).read_bytes() == b"{}"
    assert (
        json.loads((operational_root / "journal" / "heads" / "device-a.json").read_bytes())[
            "anchor_hash"
        ]
        == anchor.anchor_hash
    )
    assert ledger.verify().ok is True


def test_memo_v1_builder_attests_verified_bytes_and_source_head(tmp_path: Path) -> None:
    authority = _authority(tmp_path / "authority")
    attestor = _add_attestor(authority)
    legacy = LegacyOperationLedger(tmp_path / "legacy", device_id="device-a")
    source = legacy.append(
        "focus_set",
        subject_uri="memo://focus/demo",
        payload={"project": "demo"},
        content_hash="c" * 64,
        ts=_STAMP,
    )
    ledger = _ledger(tmp_path / "operational", authority)

    anchor = ledger.ensure_anchor_from_v1(
        legacy,
        source_head_hash=source.event_hash,
        migration_attestor=authority.signer,
        attestor_key_id=attestor.key_id,
        checkpoint=b"{}",
    )

    assert anchor.kind == "memo_v1"
    assert anchor.base_sequence == source.sequence
    assert anchor.base_event_hash == source.event_hash
    assert anchor.source_manifest_sha256 == ledger.legacy_manifest_sha256(legacy)
    assert anchor.attested_origin == "device-a"
    assert anchor.signer_role == "migration_attestor"
    assert ledger.position("device-a").sequence == source.sequence
    with pytest.raises(OperationalError, match="head"):
        _ledger(tmp_path / "other-operational", authority).ensure_anchor_from_v1(
            legacy,
            source_head_hash="d" * 64,
            migration_attestor=authority.signer,
            attestor_key_id=attestor.key_id,
        )


def test_compaction_anchor_advances_only_from_current_anchor_and_keeps_raw_checkpoint(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path / "authority")
    ledger = _ledger(tmp_path / "operational", authority)
    first = ledger.ensure_anchor()
    first_event = ledger.append(_command(), context=authority.context())
    checkpoint = canonical_json_bytes({"focus": {"demo": "compacted"}})
    compacted = _signed_anchor(
        authority,
        origin="device-a",
        kind="compaction",
        checkpoint=checkpoint,
        base_sequence=first_event.origin_sequence,
        base_hash=first_event.event_hash,
        previous_anchor_hash=first.anchor_hash,
    )

    assert ledger.ensure_anchor(compacted, checkpoint=checkpoint) == compacted
    checkpoint_path = ledger.checkpoints_dir / "device-a" / f"{compacted.anchor_id}.json"
    assert checkpoint_path.read_bytes() == checkpoint
    with pytest.raises(OperationalError, match="anchor"):
        ledger.ensure_anchor(first, checkpoint=b"{}")
    assert ledger.anchor().anchor_hash == compacted.anchor_hash
    second_event = ledger.append(
        _command(idempotency_key="idem-after-compaction"),
        context=authority.context(),
    )
    assert second_event.origin_sequence == 2
    assert second_event.previous_hash == first_event.event_hash
    assert ledger.iter_events() == [second_event]
    assert ledger.verify().ok is True


def test_anchor_requires_matching_checkpoint_role_signature_and_enrolled_origin(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path / "authority", remote_devices=("device-b",))
    ledger = _ledger(tmp_path / "operational", authority)
    anchor = _signed_anchor(authority, origin="device-b")

    with pytest.raises(OperationalError):
        ledger.ensure_anchor(anchor, checkpoint=b'{"wrong":true}')
    with pytest.raises(OperationalError):
        ledger.ensure_anchor(
            replace(anchor, signer_role="migration_attestor"),
            checkpoint=b"{}",
        )
    with pytest.raises(OperationalError):
        ledger.ensure_anchor(replace(anchor, signature="bad"), checkpoint=b"{}")
    unsafe = replace(anchor, origin_device="../escape")
    with pytest.raises(OperationalError, match="origin"):
        ledger.ensure_anchor(unsafe, checkpoint=b"{}")
    assert not ledger.anchors_dir.exists()


def test_append_fsyncs_event_before_head_and_advances_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _authority(tmp_path / "authority")
    ledger = _ledger(tmp_path / "operational", authority)
    anchor = ledger.ensure_anchor()
    calls: list[int] = []
    real_fsync = __import__("os").fsync

    def record_fsync(descriptor: int) -> None:
        calls.append(descriptor)
        real_fsync(descriptor)

    monkeypatch.setattr("memo.operation_ledger_v2.os.fsync", record_fsync)
    event = ledger.append(_command(), context=authority.context())

    assert event.origin_sequence == 1
    assert event.previous_hash == anchor.base_event_hash
    assert ledger.position().sequence == 1
    assert len(calls) >= 2
    segment = next((ledger.events_dir / "device-a").glob("*.jsonl"))
    encoded = segment.read_bytes().splitlines()
    assert encoded == [canonical_json_bytes(json.loads(encoded[0]))]
    assert ledger.verify().ok is True


def test_append_before_head_crash_is_repaired_from_validated_segment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _authority(tmp_path / "authority")
    ledger = _ledger(tmp_path / "operational", authority)
    ledger.ensure_anchor()
    real_write_head = ledger._write_head_atomic

    def crash_after_event(_event: OperationalEventV2) -> None:
        raise OSError("simulated crash before head")

    monkeypatch.setattr(ledger, "_write_head_atomic", crash_after_event)
    with pytest.raises(OSError, match="simulated crash"):
        ledger.append(_command(), context=authority.context())

    monkeypatch.setattr(ledger, "_write_head_atomic", real_write_head)
    second = ledger.append(
        _command(idempotency_key="idem-2"),
        context=authority.context(),
    )

    assert second.origin_sequence == 2
    assert len(ledger.iter_events(origin_device="device-a")) == 2
    assert ledger.validated_events() == ledger.iter_events()
    assert ledger.verify().ok is True


def test_append_holds_epoch_fence_through_event_and_head_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _authority(tmp_path / "authority")
    ledger = _ledger(tmp_path / "operational", authority)
    ledger.ensure_anchor()
    context = authority.context()
    epoch_one = _authorization(
        authority.signer,
        device_id="device-a",
        key_id=authority.origin_key.key_id,
        epoch=1,
        control_oid="control-1",
        digests={"memo_generation": "d" * 64},
    )
    event_durable = threading.Event()
    release_head = threading.Event()
    activation_started = threading.Event()
    activation_finished = threading.Event()
    failures: list[BaseException] = []
    real_write_head = ledger._write_head_atomic

    def blocked_head(event: OperationalEventV2) -> None:
        event_durable.set()
        assert release_head.wait(timeout=2)
        real_write_head(event)

    def append_event() -> None:
        try:
            ledger.append(_command(), context=context)
        except BaseException as exc:  # pragma: no cover - diagnostic capture
            failures.append(exc)

    def activate_epoch() -> None:
        try:
            activation_started.set()
            authority.fence.activate(
                authorization=epoch_one,
                observed_artifact_digests=epoch_one.artifact_digests,
            )
            activation_finished.set()
        except BaseException as exc:  # pragma: no cover - diagnostic capture
            failures.append(exc)

    monkeypatch.setattr(ledger, "_write_head_atomic", blocked_head)
    appender = threading.Thread(target=append_event)
    activator = threading.Thread(target=activate_epoch)
    appender.start()
    assert event_durable.wait(timeout=2)
    activator.start()
    assert activation_started.wait(timeout=2)
    assert not activation_finished.wait(timeout=0.2)
    release_head.set()
    appender.join(timeout=2)
    activator.join(timeout=2)

    assert not appender.is_alive()
    assert not activator.is_alive()
    assert activation_finished.is_set()
    assert failures == []
    assert ledger.verify().ok is True
    assert ledger.iter_events()[0].authority_epoch == 0


def test_tamper_gap_and_head_fork_are_reported_without_repair_in_verify(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path / "authority")
    ledger = _ledger(tmp_path / "operational", authority)
    ledger.ensure_anchor()
    ledger.append(_command(), context=authority.context())
    segment = next((ledger.events_dir / "device-a").glob("*.jsonl"))
    body = json.loads(segment.read_text(encoding="utf-8"))
    body["payload"]["summary"] = "tampered"
    segment.write_bytes(canonical_json_bytes(body) + b"\n")

    report = ledger.verify()

    assert report.ok is False
    assert any("hash" in error for error in report.errors)
    with pytest.raises(OperationalError, match="hash"):
        ledger.position()


def test_source_proof_is_bound_to_sealed_v1_head_and_signed_migration_origin(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path / "authority")
    attestor = _add_attestor(authority)
    legacy = LegacyOperationLedger(tmp_path / "legacy", device_id="device-a")
    source = legacy.append(
        "focus_set",
        subject_uri="memo://focus/demo",
        payload={"project": "demo"},
        content_hash="e" * 64,
        ts=_STAMP,
    )
    ledger = _ledger(tmp_path / "operational", authority)
    anchor = ledger.ensure_anchor_from_v1(
        legacy,
        source_head_hash=source.event_hash,
        migration_attestor=authority.signer,
        attestor_key_id=attestor.key_id,
    )
    proof = SourceProof(
        source_system="memo_v1",
        source_event_id=source.event_id,
        source_schema=source.schema,
        source_origin=source.device_id,
        source_sequence=source.sequence,
        source_previous_hash=source.previous_hash,
        source_event_hash=source.event_hash,
        source_content_hash=source.content_hash,
        source_actor=source.actor.to_dict(),
        source_subject_uri=source.subject_uri,
    )
    unsigned_origin = MigrationOrigin(
        schema="memo.operational_migration_origin.v1",
        attempt_id="attempt-migration",
        migration_device_id="device-a",
        source_manifest_sha256=anchor.source_manifest_sha256,
        capability_manifest_sha256="f" * 64,
        attestor_device_id="device-a",
        attestor_key_id=attestor.key_id,
        roster_version=2,
        issued_at="2026-07-29T11:00:00Z",
        expires_at="2026-07-29T13:00:00Z",
        signature="",
    )
    envelope = authority.signer.sign(
        domain="memo.operational.migration_origin.v1",
        payload=canonical_signed_bytes(unsigned_origin),
        key_id=attestor.key_id,
    )
    migration_origin = replace(unsigned_origin, signature=envelope.signature)
    context = replace(authority.context(), migration_origin=migration_origin)

    event = ledger.append(
        _command(source_proof=proof),
        context=context,
    )

    assert event.origin_sequence == anchor.base_sequence + 1
    assert event.previous_hash == source.event_hash
    assert event.source_proof == proof
    bad = replace(proof, source_event_hash="0" * 64)
    with pytest.raises(OperationalError, match="source proof"):
        ledger.append(
            _command(idempotency_key="idem-bad", source_proof=bad),
            context=context,
        )
    assert ledger.position().sequence == event.origin_sequence


def test_bundle_import_validates_signatures_continuity_and_exact_replay(
    tmp_path: Path,
) -> None:
    authority = _authority(
        tmp_path / "authority",
        device_id="device-b",
        remote_devices=("device-a",),
    )
    incoming = _bundle(authority, origin="device-a")
    ledger = _ledger(tmp_path / "operational", authority)

    report = ledger.import_bundles([incoming])
    replay = ledger.import_bundles([incoming])

    assert isinstance(report, LedgerImportReport)
    assert report.events_inserted == 2
    assert replay.events_inserted == 0
    assert replay.events_replayed == 2
    assert ledger.position("device-a").sequence == 2
    assert ledger.export_bundles(origins=("device-a",)) == (incoming,)
    assert ledger.verify().ok is True


def test_import_rejects_gap_and_quarantines_stably(tmp_path: Path) -> None:
    authority = _authority(
        tmp_path / "authority",
        device_id="device-b",
        remote_devices=("device-a",),
    )
    gap = _bundle(authority, origin="device-a", sequences=(1, 3))
    ledger = _ledger(tmp_path / "operational", authority)

    for _ in range(2):
        with pytest.raises(OperationalError, match="sequence"):
            ledger.import_bundles([gap])

    quarantined = list(ledger.quarantine_dir.iterdir())
    assert len(quarantined) == 1
    assert not ledger.anchors_dir.exists()
    assert not ledger.events_dir.exists()


def test_import_rejects_unknown_schema_tamper_and_same_position_fork(
    tmp_path: Path,
) -> None:
    authority = _authority(
        tmp_path / "authority",
        device_id="device-b",
        remote_devices=("device-a",),
    )
    ledger = _ledger(tmp_path / "operational", authority)
    incoming = _bundle(authority, origin="device-a")
    unknown = replace(
        incoming,
        anchor=replace(incoming.anchor, schema="memo.operational_anchor.v9"),
    )
    with pytest.raises(OperationalError, match="schema"):
        ledger.import_bundles([unknown])
    tampered = replace(
        incoming,
        events=(replace(incoming.events[0], signature="bad"), *incoming.events[1:]),
    )
    with pytest.raises(OperationalError):
        ledger.import_bundles([tampered])

    ledger.import_bundles([incoming])
    forked_second = _signed_event(
        authority,
        origin="device-a",
        sequence=2,
        previous_hash=incoming.events[0].event_hash,
        summary="fork",
    )
    fork = replace(
        incoming,
        events=(incoming.events[0], forked_second),
        head_sequence=2,
        head_hash=forked_second.event_hash,
    )
    with pytest.raises(OperationalError, match="fork"):
        ledger.import_bundles([fork])
    assert ledger.position("device-a").event_hash == incoming.head_hash
    assert len(list(ledger.quarantine_dir.iterdir())) == 3


def test_all_bundles_validate_before_any_journal_write(tmp_path: Path) -> None:
    authority = _authority(
        tmp_path / "authority",
        device_id="device-b",
        remote_devices=("device-a", "device-c"),
    )
    ledger = _ledger(tmp_path / "operational", authority)
    valid = _bundle(authority, origin="device-a")
    invalid = _bundle(
        authority,
        origin="device-c",
        sequences=(1, 3),
    )

    with pytest.raises(OperationalError):
        ledger.import_bundles([valid, invalid])

    assert not ledger.anchors_dir.exists()
    assert not ledger.events_dir.exists()
    assert ledger.positions() == ()


def test_import_write_failure_rolls_back_every_bundle_before_quarantine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _authority(
        tmp_path / "authority",
        device_id="device-b",
        remote_devices=("device-a", "device-c"),
    )
    ledger = _ledger(tmp_path / "operational", authority)
    bundles = (
        _bundle(authority, origin="device-a"),
        _bundle(authority, origin="device-c"),
    )
    real_append = ledger._append_event_fsync
    calls = 0

    def fail_during_second_bundle(event: OperationalEventV2) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OperationalError(
                code="storage_unavailable",
                message="simulated import write failure",
                retryable=False,
            )
        real_append(event)

    monkeypatch.setattr(ledger, "_append_event_fsync", fail_during_second_bundle)
    with pytest.raises(OperationalError, match="simulated import write failure"):
        ledger.import_bundles(bundles)

    assert ledger.positions() == ()
    assert not ledger.anchors_dir.exists()
    assert not ledger.events_dir.exists()
    assert not ledger.heads_dir.exists()
    assert len(list(ledger.quarantine_dir.iterdir())) == 2


def test_import_requires_exact_checkpoint_and_reducer_version(tmp_path: Path) -> None:
    authority = _authority(
        tmp_path / "authority",
        device_id="device-b",
        remote_devices=("device-a",),
    )
    ledger = _ledger(tmp_path / "operational", authority)
    incoming = _bundle(authority, origin="device-a")

    with pytest.raises(OperationalError, match="checkpoint"):
        ledger.import_bundles([replace(incoming, checkpoint=b'{"wrong":true}')])
    changed_anchor = replace(incoming.anchor, reducer_version=2)
    changed_anchor = replace(
        changed_anchor,
        anchor_hash=canonical_anchor_hash(changed_anchor),
    )
    key = _key_for(authority, "device-a")
    envelope = authority.signer.sign(
        domain="memo.operational.anchor.v1",
        payload=canonical_signed_bytes(changed_anchor),
        key_id=key.key_id,
    )
    changed_anchor = replace(changed_anchor, signature=envelope.signature)
    with pytest.raises(OperationalError, match="reducer"):
        ledger.import_bundles([replace(incoming, anchor=changed_anchor)])
    assert ledger.positions() == ()


def test_symlinks_and_path_traversal_fail_closed(tmp_path: Path) -> None:
    authority = _authority(tmp_path / "authority")
    operational = tmp_path / "operational"
    journal = operational / "journal"
    outside = tmp_path / "outside"
    journal.mkdir(parents=True)
    outside.mkdir()
    (journal / "anchors").symlink_to(outside, target_is_directory=True)
    ledger = _ledger(operational, authority)

    with pytest.raises(OperationalError, match="symlink"):
        ledger.ensure_anchor()
    assert list(outside.iterdir()) == []
    with pytest.raises(OperationalError, match="device"):
        OperationLedgerV2(
            tmp_path / "other",
            device_id="../escape",
            clock=_clock,
        )


def test_missing_or_tampered_checkpoint_makes_verify_and_export_fail(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path / "authority")
    ledger = _ledger(tmp_path / "operational", authority)
    anchor = ledger.ensure_anchor()
    checkpoint = ledger.checkpoints_dir / "device-a" / f"{anchor.anchor_id}.json"
    checkpoint.write_bytes(b'{"tampered":true}')

    assert ledger.verify().ok is False
    with pytest.raises(OperationalError, match="checkpoint"):
        ledger.export_bundles()


def test_roster_refresh_uses_historical_rosters_and_rejects_revoked_signer(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path / "authority")
    operational = tmp_path / "operational"
    first_ledger = _ledger(operational, authority)
    first_ledger.ensure_anchor()
    first = first_ledger.append(_command(), context=authority.context())
    old_signer = authority.signer
    old_key = authority.origin_key
    new_key = authority.keys.generate(
        device_id="device-a",
        roles=("origin",),
        enrollment_sequence=2,
    )
    revoked = replace(
        old_key,
        revocation_sequence=2,
        proof_of_possession="",
    )
    revoked = replace(
        revoked,
        proof_of_possession=base64.urlsafe_b64encode(
            authority.keys.sign(
                key_id=revoked.key_id,
                payload=revoked.proof_payload(),
            )
        )
        .rstrip(b"=")
        .decode("ascii"),
    )
    authority.roster = authority.roster.with_keys(
        version=2,
        peers=("device-a",),
        keys=(revoked, new_key),
        signer=old_signer,
        root=authority.root,
        pin_store=authority.pin_store,
    )
    authority.signer = OperationalSigner(authority.keys, roster_version=2)
    refreshed = _ledger(operational, authority)

    second = refreshed.append(
        _command(idempotency_key="idem-2"),
        context=authority.context(),
    )

    assert first.roster_version == 1
    assert second.roster_version == 2
    assert refreshed.verify().ok is True
    stale = _ledger(operational, authority, signer=old_signer)
    with pytest.raises(OperationalError):
        stale.append(
            _command(idempotency_key="idem-3"),
            context=authority.context(),
        )
    assert refreshed.position().sequence == 2


def test_epoch_refresh_rejects_stale_context_before_any_append(tmp_path: Path) -> None:
    authority = _authority(tmp_path / "authority")
    ledger = _ledger(tmp_path / "operational", authority)
    ledger.ensure_anchor()
    stale = authority.context()
    epoch_one = _authorization(
        authority.signer,
        device_id="device-a",
        key_id=authority.origin_key.key_id,
        epoch=1,
        control_oid="control-1",
        digests={"memo_generation": "c" * 64},
    )
    authority.fence.activate(
        authorization=epoch_one,
        observed_artifact_digests=epoch_one.artifact_digests,
    )

    with pytest.raises(OperationalError, match="epoch"):
        ledger.append(_command(), context=stale)
    fresh = authority.fence.context(
        _identity(),
        request_epoch=1,
        request_control_oid="control-1",
    )
    event = ledger.append(_command(), context=fresh)
    assert event.origin_sequence == 1
    assert ledger.iter_events() == [event]


def test_command_actor_and_context_origin_must_match_before_write(tmp_path: Path) -> None:
    authority = _authority(tmp_path / "authority")
    ledger = _ledger(tmp_path / "operational", authority)
    ledger.ensure_anchor()
    mismatched = replace(authority.context(), identity=_identity("device-b"))

    with pytest.raises(OperationalError, match="principal"):
        ledger.append(_command(), context=mismatched)
    with pytest.raises(OperationalError, match="actor"):
        ledger.append(
            _command(device_id="device-b"),
            context=authority.context(),
        )
    assert ledger.iter_events() == []
