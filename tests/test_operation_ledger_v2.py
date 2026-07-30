from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import pytest

import memo.operation_ledger_v2 as operation_ledger_v2
from memo.atomic_io import authority_admission_lock
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
    StateCheckpoint,
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


class _CrashLedger(OperationLedgerV2):
    crash_at: str

    def _transaction_failpoint(self, label: str) -> None:
        if label == self.crash_at:
            os._exit(73)


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
                enrollment_sequence=2,
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


def _crash_ledger(
    root: Path,
    authority: _Authority,
    *,
    crash_at: str,
) -> _CrashLedger:
    ledger = _CrashLedger(
        root,
        device_id=authority.device_id,
        clock=_clock,
        signer=authority.signer,
        verifier=authority.verifier,
        roster=authority.roster,
        roster_root=authority.root,
        pin_store=authority.pin_store,
        epoch_fence=authority.fence,
    )
    ledger.crash_at = crash_at
    return ledger


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
    ledger_epoch: int = 0,
    state_sha256: str | None = None,
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
        ledger_epoch=ledger_epoch,
        reducer_version=1,
        kind=cast(object, kind),  # type: ignore[arg-type]
        base_sequence=base_sequence,
        base_event_hash=base_hash,
        final_sequence=base_sequence,
        final_event_hash=base_hash,
        previous_anchor_hash=previous_anchor_hash,
        source_manifest_sha256=source_manifest_sha256,
        state_sha256=state_sha256 or digest,
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


def _state_checkpoint_bytes(
    *,
    origin: str,
    through_sequence: int,
    through_hash: str,
    state: bytes,
    checkpoint_id: str | None = None,
    reducer_version: int = 1,
) -> bytes:
    checkpoint = StateCheckpoint(
        schema="memo.operational_checkpoint.v1",
        checkpoint_id=(checkpoint_id or f"checkpoint-{origin}-compaction-{through_sequence}"),
        reducer_version=reducer_version,
        origin_device=origin,
        through_sequence=through_sequence,
        through_event_hash=through_hash,
        state_bytes=state,
        state_sha256=hashlib.sha256(state).hexdigest(),
        created_at=_STAMP,
    )
    return canonical_json_bytes(checkpoint)


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
    next_version = authority.roster.version + 1
    attestor = authority.keys.generate(
        device_id=authority.device_id,
        roles=("migration_attestor",),
        enrollment_sequence=next_version,
    )
    authority.roster = authority.roster.with_keys(
        version=next_version,
        peers=authority.roster.peers,
        keys=(*authority.roster.keys, attestor),
        signer=authority.signer,
        root=authority.root,
        pin_store=authority.pin_store,
    )
    authority.signer = OperationalSigner(
        authority.keys,
        roster_version=next_version,
    )
    return attestor


def _authenticate_proofs(
    proofs: tuple[SourceProof, ...],
    *,
    manifest: str,
) -> tuple[str, tuple[SourceProof, ...]]:
    import memo.operational_event as event_module

    authenticate = getattr(event_module, "authenticate_source_proofs", None)
    assert authenticate is not None, "authenticated source-proof builder is required"
    return authenticate(proofs, source_manifest_sha256=manifest)


def _signed_migration_origin(
    authority: _Authority,
    *,
    attestor: PublicKeyRecord,
    migration_device: str,
    source_manifest: str,
    proof_root: str,
    proof_count: int,
    attempt_id: str = "attempt-migration",
) -> MigrationOrigin:
    unsigned = MigrationOrigin(
        schema="memo.operational_migration_origin.v1",
        attempt_id=attempt_id,
        migration_device_id=migration_device,
        source_manifest_sha256=source_manifest,
        capability_manifest_sha256="f" * 64,
        attestor_device_id=authority.device_id,
        attestor_key_id=attestor.key_id,
        roster_version=authority.roster.version,
        issued_at="2026-07-29T11:00:00Z",
        expires_at="2026-07-29T13:00:00Z",
        signature="",
        source_proof_root_sha256=proof_root,
        source_proof_count=proof_count,
    )
    envelope = authority.signer.sign(
        domain="memo.operational.migration_origin.v1",
        payload=canonical_signed_bytes(unsigned),
        key_id=attestor.key_id,
    )
    return replace(unsigned, signature=envelope.signature)


def _resign_event(
    authority: _Authority,
    event: OperationalEventV2,
    **changes: object,
) -> OperationalEventV2:
    unsigned = replace(event, **changes, event_hash="", signature="")
    hashed = replace(unsigned, event_hash=canonical_event_hash(unsigned))
    envelope = authority.signer.sign(
        domain="memo.operational.event.v2",
        payload=canonical_signed_bytes(hashed),
        key_id=hashed.key_id,
    )
    return replace(hashed, signature=envelope.signature)


def _with_revocation_proof(
    authority: _Authority,
    key: PublicKeyRecord,
    *,
    revocation_version: int,
) -> PublicKeyRecord:
    revoked = replace(
        key,
        revocation_sequence=revocation_version,
        proof_of_possession="",
    )
    return replace(
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


def _rotate_local_origin(authority: _Authority) -> PublicKeyRecord:
    previous = authority.origin_key
    old_signer = authority.signer
    next_version = authority.roster.version + 1
    new_key = authority.keys.generate(
        device_id=authority.device_id,
        roles=("origin",),
        enrollment_sequence=next_version,
    )
    revoked = _with_revocation_proof(
        authority,
        previous,
        revocation_version=next_version,
    )
    remaining = tuple(key for key in authority.roster.keys if key.key_id != previous.key_id)
    authority.roster = authority.roster.with_keys(
        version=next_version,
        peers=authority.roster.peers,
        keys=(revoked, new_key, *remaining),
        signer=old_signer,
        root=authority.root,
        pin_store=authority.pin_store,
    )
    authority.origin_key = new_key
    authority.signer = OperationalSigner(
        authority.keys,
        roster_version=next_version,
    )
    return new_key


def _fork_process_loss(
    operation: Callable[[], None],
    *,
    expected_exit: int = 73,
) -> None:
    if not hasattr(os, "fork"):
        pytest.skip("process-loss regression requires os.fork")
    pid = os.fork()
    if pid == 0:
        try:
            operation()
        except BaseException:
            os._exit(99)
        os._exit(expected_exit)
    waited, status = os.waitpid(pid, 0)
    assert waited == pid
    assert os.WIFEXITED(status)
    assert os.WEXITSTATUS(status) == expected_exit


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


def test_constructor_rejects_crossed_roster_and_epoch_authority_roots(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path / "authority")
    crossed_root = tmp_path / "crossed-authority"
    crossed_root.mkdir()

    with pytest.raises(ValueError, match="authority root"):
        OperationLedgerV2(
            tmp_path / "operational",
            device_id=authority.device_id,
            clock=_clock,
            signer=authority.signer,
            verifier=authority.verifier,
            roster=authority.roster,
            roster_root=crossed_root,
            pin_store=authority.pin_store,
            epoch_fence=authority.fence,
        )


def test_roster_and_epoch_reads_reuse_retained_authority_after_path_swap(
    tmp_path: Path,
) -> None:
    root = tmp_path / "authority"
    authority = _authority(root)
    retained_path = tmp_path / "retained-authority"
    outside = tmp_path / "outside"
    outside.mkdir()

    with authority_admission_lock(root):
        root.rename(retained_path)
        root.symlink_to(outside, target_is_directory=True)

        assert VerificationRoster.load(
            root,
            pin_store=authority.pin_store,
        ) == authority.roster
        context = authority.context()
        authority.fence.verify(context)

    assert (retained_path / "verification-roster.json").exists()
    assert (retained_path / "authority-epoch.json").exists()
    assert list(outside.iterdir()) == []


def test_unpinned_unsigned_roster_cannot_admit_an_anchor(tmp_path: Path) -> None:
    keys = DeviceKeyStore.in_memory()
    key = keys.generate(device_id="device-a", roles=("origin",))
    raw_roster = VerificationRoster(
        version=1,
        peers=("device-a",),
        keys=(key,),
        local_device_id="device-a",
    )
    ledger = OperationLedgerV2(
        tmp_path / "operational",
        device_id="device-a",
        clock=_clock,
        signer=OperationalSigner(keys, roster_version=1),
        verifier=OperationalVerifier(),
        roster=raw_roster,
    )

    with pytest.raises(OperationalError, match=r"pinned|authority"):
        ledger.ensure_anchor()


def test_operational_root_with_symlinked_ancestor_never_escapes(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path / "authority")
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)
    ledger = _ledger(linked / "operational", authority)

    with pytest.raises(OperationalError, match=r"symlink|unsafe|storage"):
        ledger.ensure_anchor()

    assert list(outside.iterdir()) == []


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
    state = canonical_json_bytes({"focus": {"demo": "compacted"}})
    checkpoint = _state_checkpoint_bytes(
        origin="device-a",
        through_sequence=first_event.origin_sequence,
        through_hash=first_event.event_hash,
        state=state,
    )
    compacted = _signed_anchor(
        authority,
        origin="device-a",
        kind="compaction",
        checkpoint=checkpoint,
        base_sequence=first_event.origin_sequence,
        base_hash=first_event.event_hash,
        previous_anchor_hash=first.anchor_hash,
        state_sha256=hashlib.sha256(state).hexdigest(),
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


def test_compaction_checkpoint_validates_structured_envelope_and_inner_state(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path / "authority")
    ledger = _ledger(tmp_path / "operational", authority)
    genesis = ledger.ensure_anchor()
    event = ledger.append(_command(), context=authority.context())
    state = canonical_json_bytes({"focus": {"demo": "compacted"}})
    checkpoint = _state_checkpoint_bytes(
        origin="device-a",
        through_sequence=event.origin_sequence,
        through_hash=event.event_hash,
        state=state,
    )
    anchor = _signed_anchor(
        authority,
        origin="device-a",
        kind="compaction",
        checkpoint=checkpoint,
        base_sequence=event.origin_sequence,
        base_hash=event.event_hash,
        previous_anchor_hash=genesis.anchor_hash,
        state_sha256=hashlib.sha256(state).hexdigest(),
    )

    assert ledger.ensure_anchor(anchor, checkpoint=checkpoint) == anchor

    body = json.loads(checkpoint)
    invalid_bodies = (
        {**body, "schema": "memo.operational_checkpoint.v9"},
        {**body, "checkpoint_id": "wrong-checkpoint"},
        {**body, "origin_device": "device-b"},
        {**body, "reducer_version": 2},
        {**body, "through_sequence": 999},
        {**body, "through_event_hash": "f" * 64},
        {**body, "state_sha256": "0" * 64},
        {**body, "state_bytes": "e30"},
    )
    for invalid_body in invalid_bodies:
        invalid = canonical_json_bytes(invalid_body)
        changed = _signed_anchor(
            authority,
            origin="device-a",
            kind="compaction",
            checkpoint=invalid,
            base_sequence=event.origin_sequence,
            base_hash=event.event_hash,
            previous_anchor_hash=genesis.anchor_hash,
            state_sha256=hashlib.sha256(state).hexdigest(),
        )
        with pytest.raises(OperationalError, match=r"checkpoint|state|reducer"):
            ledger._validate_checkpoint(changed, invalid)


def test_imported_compaction_rejects_ledger_epoch_regression(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path / "authority")
    ledger = _ledger(tmp_path / "operational", authority)
    genesis = ledger.ensure_anchor()
    event = ledger.append(_command(), context=authority.context())
    state_v5 = canonical_json_bytes({"epoch": 5})
    checkpoint_v5 = _state_checkpoint_bytes(
        origin="device-a",
        through_sequence=event.origin_sequence,
        through_hash=event.event_hash,
        state=state_v5,
    )
    anchor_v5 = _signed_anchor(
        authority,
        origin="device-a",
        kind="compaction",
        checkpoint=checkpoint_v5,
        base_sequence=event.origin_sequence,
        base_hash=event.event_hash,
        previous_anchor_hash=genesis.anchor_hash,
        ledger_epoch=5,
        state_sha256=hashlib.sha256(state_v5).hexdigest(),
    )
    ledger.ensure_anchor(anchor_v5, checkpoint=checkpoint_v5)

    state_v4 = canonical_json_bytes({"epoch": 4})
    checkpoint_v4 = _state_checkpoint_bytes(
        origin="device-a",
        through_sequence=event.origin_sequence,
        through_hash=event.event_hash,
        state=state_v4,
    )
    anchor_v4 = _signed_anchor(
        authority,
        origin="device-a",
        kind="compaction",
        checkpoint=checkpoint_v4,
        base_sequence=event.origin_sequence,
        base_hash=event.event_hash,
        previous_anchor_hash=anchor_v5.anchor_hash,
        ledger_epoch=4,
        state_sha256=hashlib.sha256(state_v4).hexdigest(),
    )
    regressed = OriginBundle(
        anchor=anchor_v4,
        checkpoint=checkpoint_v4,
        events=(),
        head_sequence=event.origin_sequence,
        head_hash=event.event_hash,
    )

    with pytest.raises(OperationalError, match=r"epoch|regression"):
        ledger.import_bundles(
            [regressed],
            context=authority.context(),
        )

    assert ledger.anchor().ledger_epoch == 5


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


def test_torn_jsonl_tail_recovers_at_multiple_byte_boundaries_after_restart(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path / "authority")
    operational = tmp_path / "operational"
    ledger = _ledger(operational, authority)
    ledger.ensure_anchor()
    first = ledger.append(_command(), context=authority.context())
    segment = next((ledger.events_dir / "device-a").glob("*.jsonl"))
    durable_prefix = segment.read_bytes()
    second = _signed_event(
        authority,
        origin="device-a",
        sequence=2,
        previous_hash=first.event_hash,
    )
    encoded = canonical_json_bytes(second) + b"\n"

    for boundary in (1, len(encoded) // 2, len(encoded) - 1):
        segment.write_bytes(durable_prefix)

        def lose_power(boundary: int = boundary) -> None:
            descriptor = os.open(segment, os.O_WRONLY | os.O_APPEND)
            try:
                os.write(descriptor, encoded[:boundary])
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

        _fork_process_loss(lose_power)
        restarted = _ledger(operational, authority)
        report = restarted.recover()
        assert report.repaired_tails == ("device-a",)
        assert restarted.position().sequence == first.origin_sequence
        assert segment.read_bytes() == durable_prefix
        assert list((restarted.root / "recovery").glob("*.json"))


def test_complete_event_before_stale_head_is_repaired_after_process_loss(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path / "authority")
    operational = tmp_path / "operational"
    ledger = _ledger(operational, authority)
    ledger.ensure_anchor()
    first = ledger.append(_command(), context=authority.context())
    second = _signed_event(
        authority,
        origin="device-a",
        sequence=2,
        previous_hash=first.event_hash,
    )
    segment = next((ledger.events_dir / "device-a").glob("*.jsonl"))

    def lose_power() -> None:
        descriptor = os.open(segment, os.O_WRONLY | os.O_APPEND)
        try:
            os.write(descriptor, canonical_json_bytes(second) + b"\n")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    _fork_process_loss(lose_power)
    restarted = _ledger(operational, authority)
    report = restarted.recover()

    assert report.repaired_heads == ("device-a",)
    assert restarted.position().sequence == 2
    assert restarted.iter_events() == [first, second]


def test_torn_tail_recovery_fails_closed_for_invalid_complete_rows_and_advanced_head(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path / "authority")

    invalid = _ledger(tmp_path / "invalid", authority)
    invalid.ensure_anchor()
    invalid.append(_command(), context=authority.context())
    invalid_segment = next((invalid.events_dir / "device-a").glob("*.jsonl"))
    with invalid_segment.open("ab") as handle:
        handle.write(b'{"complete":"but-invalid"}\n')
        handle.flush()
        os.fsync(handle.fileno())
    with pytest.raises(OperationalError):
        invalid.recover()

    advanced = _ledger(tmp_path / "advanced", authority)
    advanced.ensure_anchor()
    first = advanced.append(_command(), context=authority.context())
    advanced_segment = next((advanced.events_dir / "device-a").glob("*.jsonl"))
    with advanced_segment.open("ab") as handle:
        handle.write(b'{"torn":')
        handle.flush()
        os.fsync(handle.fileno())
    advanced._write_head(
        origin="device-a",
        sequence=99,
        event_hash="f" * 64,
        anchor_hash=advanced.anchor().anchor_hash,
    )
    with pytest.raises(OperationalError, match=r"head|fork|advanced"):
        advanced.recover()
    assert advanced_segment.read_bytes().endswith(b'{"torn":')
    assert first.origin_sequence == 1


def test_legacy_compaction_crash_after_anchor_before_head_recovers_on_restart(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path / "authority")
    operational = tmp_path / "operational"
    ledger = _ledger(operational, authority)
    genesis = ledger.ensure_anchor()
    event = ledger.append(_command(), context=authority.context())
    state = canonical_json_bytes({"focus": "compacted"})
    checkpoint = _state_checkpoint_bytes(
        origin="device-a",
        through_sequence=event.origin_sequence,
        through_hash=event.event_hash,
        state=state,
    )
    compacted = _signed_anchor(
        authority,
        origin="device-a",
        kind="compaction",
        checkpoint=checkpoint,
        base_sequence=event.origin_sequence,
        base_hash=event.event_hash,
        previous_anchor_hash=genesis.anchor_hash,
        ledger_epoch=1,
        state_sha256=hashlib.sha256(state).hexdigest(),
    )

    def lose_power() -> None:
        ledger._atomic_write_bytes(ledger._checkpoint_path(compacted), checkpoint)
        ledger._atomic_write_json(ledger._anchor_path("device-a"), compacted)

    _fork_process_loss(lose_power)
    restarted = _ledger(operational, authority)
    report = restarted.recover()

    assert report.recovered_compactions == ("device-a",)
    assert restarted.anchor() == compacted
    assert restarted.position().sequence == event.origin_sequence
    assert restarted.verify().ok is True


def test_legacy_compaction_recovery_rejects_regressing_ledger_epoch(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path / "authority")
    operational = tmp_path / "operational"
    ledger = _ledger(operational, authority)
    genesis = ledger.ensure_anchor()
    event = ledger.append(_command(), context=authority.context())
    state_v5 = canonical_json_bytes({"epoch": 5})
    checkpoint_v5 = _state_checkpoint_bytes(
        origin="device-a",
        through_sequence=event.origin_sequence,
        through_hash=event.event_hash,
        state=state_v5,
    )
    anchor_v5 = _signed_anchor(
        authority,
        origin="device-a",
        kind="compaction",
        checkpoint=checkpoint_v5,
        base_sequence=event.origin_sequence,
        base_hash=event.event_hash,
        previous_anchor_hash=genesis.anchor_hash,
        ledger_epoch=5,
        state_sha256=hashlib.sha256(state_v5).hexdigest(),
    )
    ledger.ensure_anchor(anchor_v5, checkpoint=checkpoint_v5)

    state_v4 = canonical_json_bytes({"epoch": 4})
    checkpoint_v4 = _state_checkpoint_bytes(
        origin="device-a",
        through_sequence=event.origin_sequence,
        through_hash=event.event_hash,
        state=state_v4,
    )
    anchor_v4 = _signed_anchor(
        authority,
        origin="device-a",
        kind="compaction",
        checkpoint=checkpoint_v4,
        base_sequence=event.origin_sequence,
        base_hash=event.event_hash,
        previous_anchor_hash=anchor_v5.anchor_hash,
        ledger_epoch=4,
        state_sha256=hashlib.sha256(state_v4).hexdigest(),
    )
    ledger._atomic_write_bytes(ledger._checkpoint_path(anchor_v4), checkpoint_v4)
    ledger._atomic_write_json(ledger._anchor_path("device-a"), anchor_v4)

    with pytest.raises(OperationalError, match=r"epoch|regression"):
        _ledger(operational, authority).recover()


def test_legacy_compaction_recovery_requires_predecessor_anchor_history(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path / "authority")
    operational = tmp_path / "operational"
    ledger = _ledger(operational, authority)
    genesis = ledger.ensure_anchor()
    event = ledger.append(_command(), context=authority.context())
    state = canonical_json_bytes({"epoch": 1})
    checkpoint = _state_checkpoint_bytes(
        origin="device-a",
        through_sequence=event.origin_sequence,
        through_hash=event.event_hash,
        state=state,
    )
    compacted = _signed_anchor(
        authority,
        origin="device-a",
        kind="compaction",
        checkpoint=checkpoint,
        base_sequence=event.origin_sequence,
        base_hash=event.event_hash,
        previous_anchor_hash=genesis.anchor_hash,
        ledger_epoch=1,
        state_sha256=hashlib.sha256(state).hexdigest(),
    )
    predecessor = ledger.anchor_history_dir / "device-a" / f"{genesis.anchor_hash}.json"
    predecessor.unlink()
    ledger._atomic_write_bytes(ledger._checkpoint_path(compacted), checkpoint)
    ledger._atomic_write_json(ledger._anchor_path("device-a"), compacted)

    with pytest.raises(OperationalError, match=r"history|predecessor"):
        _ledger(operational, authority).recover()


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
    raw_proof = SourceProof(
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
    proof_root, authenticated = _authenticate_proofs(
        (raw_proof,),
        manifest=anchor.source_manifest_sha256,
    )
    proof = authenticated[0]
    migration_origin = _signed_migration_origin(
        authority,
        attestor=attestor,
        migration_device="device-a",
        source_manifest=anchor.source_manifest_sha256,
        proof_root=proof_root,
        proof_count=1,
    )
    context = replace(authority.context(), migration_origin=migration_origin)

    event = ledger.append(
        _command(source_proof=proof),
        context=context,
    )

    assert event.origin_sequence == anchor.base_sequence + 1
    assert event.previous_hash == source.event_hash
    assert event.source_proof == proof
    assert event.migration_origin == migration_origin
    assert (
        event.migration_origin_sha256
        == hashlib.sha256(canonical_json_bytes(migration_origin)).hexdigest()
    )
    bad = replace(proof, source_event_hash="0" * 64)
    with pytest.raises(OperationalError, match="source proof"):
        ledger.append(
            _command(idempotency_key="idem-bad", source_proof=bad),
            context=context,
        )
    assert ledger.position().sequence == event.origin_sequence


def test_source_proof_without_migration_authority_and_forged_pre_head_fail(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path / "authority")
    attestor = _add_attestor(authority)
    legacy = LegacyOperationLedger(tmp_path / "legacy", device_id="device-a")
    first = legacy.append(
        "focus_set",
        subject_uri="memo://focus/demo",
        payload={"project": "demo", "summary": "first"},
        content_hash="1" * 64,
        ts=_STAMP,
    )
    second = legacy.append(
        "focus_set",
        subject_uri="memo://focus/demo",
        payload={"project": "demo", "summary": "second"},
        content_hash="2" * 64,
        ts=_STAMP,
    )
    ledger = _ledger(tmp_path / "operational", authority)
    anchor = ledger.ensure_anchor_from_v1(
        legacy,
        source_head_hash=second.event_hash,
        migration_attestor=authority.signer,
        attestor_key_id=attestor.key_id,
    )

    raw_proofs = tuple(
        SourceProof(
            source_system="memo_v1",
            source_event_id=event.event_id,
            source_schema=event.schema,
            source_origin=event.device_id,
            source_sequence=event.sequence,
            source_previous_hash=event.previous_hash,
            source_event_hash=event.event_hash,
            source_content_hash=event.content_hash,
            source_actor=event.actor.to_dict(),
            source_subject_uri=event.subject_uri,
        )
        for event in (first, second)
    )
    root, proofs = _authenticate_proofs(
        raw_proofs,
        manifest=anchor.source_manifest_sha256,
    )
    migration = _signed_migration_origin(
        authority,
        attestor=attestor,
        migration_device="device-a",
        source_manifest=anchor.source_manifest_sha256,
        proof_root=root,
        proof_count=2,
    )

    with pytest.raises(OperationalError, match=r"migration|authority"):
        ledger.append(
            _command(source_proof=proofs[-1]),
            context=authority.context(),
        )

    forged = replace(
        proofs[0],
        source_event_id="invented-pre-head",
        source_event_hash="9" * 64,
        source_actor={"actor_id": "forged"},
    )
    with pytest.raises(OperationalError, match=r"inclusion|proof"):
        ledger.append(
            _command(idempotency_key="forged", source_proof=forged),
            context=replace(authority.context(), migration_origin=migration),
        )
    assert ledger.position().sequence == anchor.base_sequence


def test_plan04_shaped_proof_keeps_source_origin_distinct_from_migration_origin(
    tmp_path: Path,
) -> None:
    authority = _authority(
        tmp_path / "authority",
        device_id="migration-device",
        remote_devices=("source-device",),
    )
    attestor = _add_attestor(authority)
    ledger = _ledger(tmp_path / "operational", authority)
    anchor = ledger.ensure_anchor()
    raw = SourceProof(
        source_system="memflow_active_state",
        source_event_id="memflow-record-1",
        source_schema="memflow.active_state.v1",
        source_origin="source-device",
        source_sequence=1,
        source_previous_hash="",
        source_event_hash="a" * 64,
        source_content_hash="b" * 64,
        source_actor={"actor_id": "memflow-migration"},
        source_subject_uri="memo://memflow/active/1",
    )
    manifest = "c" * 64
    root, proofs = _authenticate_proofs((raw,), manifest=manifest)
    migration = _signed_migration_origin(
        authority,
        attestor=attestor,
        migration_device="migration-device",
        source_manifest=manifest,
        proof_root=root,
        proof_count=1,
    )

    event = ledger.append(
        _command(
            device_id="migration-device",
            source_proof=proofs[0],
        ),
        context=replace(authority.context(), migration_origin=migration),
    )

    assert anchor.kind == "empty"
    assert event.origin_device == "migration-device"
    assert event.source_proof is not None
    assert event.source_proof.source_origin == "source-device"
    assert event.migration_origin == migration


def test_persisted_migration_origin_digest_and_inclusion_path_are_verified(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path / "authority")
    attestor = _add_attestor(authority)
    ledger = _ledger(tmp_path / "operational", authority)
    ledger.ensure_anchor()
    manifest = "d" * 64
    raw = SourceProof(
        source_system="memflow_active_state",
        source_event_id="source-1",
        source_schema="memflow.active_state.v1",
        source_origin="legacy-device",
        source_sequence=1,
        source_previous_hash="",
        source_event_hash="1" * 64,
        source_content_hash="2" * 64,
        source_actor={"actor_id": "migration"},
        source_subject_uri="memo://source/1",
    )
    root, proofs = _authenticate_proofs((raw,), manifest=manifest)
    migration = _signed_migration_origin(
        authority,
        attestor=attestor,
        migration_device="device-a",
        source_manifest=manifest,
        proof_root=root,
        proof_count=1,
    )
    event = ledger.append(
        _command(source_proof=proofs[0]),
        context=replace(authority.context(), migration_origin=migration),
    )
    bundle = ledger.export_bundles()
    assert len(bundle) == 1
    original = bundle[0]

    changed_origin = _signed_migration_origin(
        authority,
        attestor=attestor,
        migration_device="device-a",
        source_manifest=manifest,
        proof_root=root,
        proof_count=1,
        attempt_id="changed-attempt",
    )
    origin_tamper = _resign_event(
        authority,
        event,
        migration_origin=changed_origin,
    )
    digest_tamper = _resign_event(
        authority,
        event,
        migration_origin_sha256="0" * 64,
    )
    assert event.source_proof is not None
    assert event.source_proof.authentication is not None
    path_tamper = replace(
        event.source_proof.authentication,
        merkle_path=("f" * 64,),
    )
    proof_tamper = _resign_event(
        authority,
        event,
        source_proof=replace(
            event.source_proof,
            authentication=path_tamper,
        ),
    )

    for changed in (origin_tamper, digest_tamper, proof_tamper):
        invalid = replace(
            original,
            events=(changed,),
            head_sequence=changed.origin_sequence,
            head_hash=changed.event_hash,
        )
        with pytest.raises(
            OperationalError,
            match=r"migration|digest|inclusion|proof",
        ):
            ledger._validate_bundle(invalid)


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

    report = ledger.import_bundles([incoming], context=authority.context())
    replay = ledger.import_bundles([incoming], context=authority.context())

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
            ledger.import_bundles([gap], context=authority.context())

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
        ledger.import_bundles([unknown], context=authority.context())
    tampered = replace(
        incoming,
        events=(replace(incoming.events[0], signature="bad"), *incoming.events[1:]),
    )
    with pytest.raises(OperationalError):
        ledger.import_bundles([tampered], context=authority.context())

    ledger.import_bundles([incoming], context=authority.context())
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
        ledger.import_bundles([fork], context=authority.context())
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
        ledger.import_bundles(
            [valid, invalid],
            context=authority.context(),
        )

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
        ledger.import_bundles(bundles, context=authority.context())

    assert ledger.positions() == ()
    assert not ledger.anchors_dir.exists()
    assert not ledger.events_dir.exists()
    assert not ledger.heads_dir.exists()
    assert len(list(ledger.quarantine_dir.iterdir())) == 2


@pytest.mark.parametrize(
    "crash_at",
    (
        "before_commit",
        "after_commit",
        *(f"after_target:{index}" for index in range(8)),
    ),
)
def test_multi_bundle_transaction_recovers_process_loss_at_every_publish_boundary(
    tmp_path: Path,
    crash_at: str,
) -> None:
    authority = _authority(
        tmp_path / "authority",
        device_id="device-b",
        remote_devices=("device-a", "device-c"),
    )
    operational = tmp_path / "operational"
    bundles = (
        _bundle(authority, origin="device-a"),
        _bundle(authority, origin="device-c"),
    )
    context = authority.context()

    def lose_power() -> None:
        crashing = _crash_ledger(
            operational,
            authority,
            crash_at=crash_at,
        )
        crashing.import_bundles(bundles, context=context)
        raise AssertionError(f"transaction failpoint was not reached: {crash_at}")

    _fork_process_loss(lose_power)
    restarted = _ledger(operational, authority)

    if crash_at == "before_commit":
        recovery = restarted.recover()
        assert recovery.discarded_transactions
        assert restarted.positions() == ()
        first_retry = restarted.import_bundles(
            bundles,
            context=authority.context(),
        )
        assert first_retry.events_inserted == 4
    else:
        assert restarted.position("device-a").sequence == 2
        assert tuple(position.origin_device for position in restarted.positions()) == (
            "device-a",
            "device-c",
        )
        first_retry = restarted.import_bundles(
            bundles,
            context=authority.context(),
        )
        assert first_retry.events_inserted == 0
        assert first_retry.events_replayed == 4

    immediate_retry = restarted.import_bundles(
        bundles,
        context=authority.context(),
    )
    assert immediate_retry.events_inserted == 0
    assert immediate_retry.events_replayed == 4
    assert restarted.verify().ok is True
    for transaction in (restarted.root / "transactions").glob("*"):
        if (transaction / "COMMITTED.json").exists():
            assert (transaction / "APPLIED.json").exists()
            manifest = json.loads((transaction / "manifest.json").read_bytes())
            targets = manifest["targets"]
            first_head = next(
                (
                    index
                    for index, target in enumerate(targets)
                    if str(target["relative_target"]).startswith("heads/")
                ),
                len(targets),
            )
            assert all(
                not str(target["relative_target"]).startswith("heads/")
                for target in targets[:first_head]
            )
            assert all(
                str(target["relative_target"]).startswith("heads/")
                for target in targets[first_head:]
            )


def test_transaction_marker_phase_cannot_be_replayed_at_applied_path(
    tmp_path: Path,
) -> None:
    authority = _authority(
        tmp_path / "authority",
        device_id="device-b",
        remote_devices=("device-a",),
    )
    operational = tmp_path / "operational"
    bundle = _bundle(authority, origin="device-a")

    def lose_power() -> None:
        crashing = _crash_ledger(operational, authority, crash_at="after_commit")
        crashing.import_bundles((bundle,), context=authority.context())
        raise AssertionError("after_commit failpoint was not reached")

    _fork_process_loss(lose_power)
    transaction = next((operational / "journal" / "transactions").iterdir())
    (transaction / "APPLIED.json").write_bytes((transaction / "COMMITTED.json").read_bytes())

    with pytest.raises(OperationalError, match="phase"):
        _ledger(operational, authority).recover()


def test_applied_transaction_verifies_every_target_before_finalization(
    tmp_path: Path,
) -> None:
    authority = _authority(
        tmp_path / "authority",
        device_id="device-b",
        remote_devices=("device-a",),
    )
    operational = tmp_path / "operational"
    bundle = _bundle(authority, origin="device-a")

    def lose_power() -> None:
        crashing = _crash_ledger(operational, authority, crash_at="after_applied")
        crashing.import_bundles((bundle,), context=authority.context())
        raise AssertionError("after_applied failpoint was not reached")

    _fork_process_loss(lose_power)
    transaction = next((operational / "journal" / "transactions").iterdir())
    manifest = json.loads((transaction / "manifest.json").read_bytes())
    first_target = operational / "journal" / manifest["targets"][0]["relative_target"]
    first_target.write_bytes(b"tampered after applied")

    with pytest.raises(OperationalError, match=r"target|digest|verification"):
        _ledger(operational, authority).recover()


def test_successful_transaction_retires_stage_blobs_to_bounded_receipt(
    tmp_path: Path,
) -> None:
    authority = _authority(
        tmp_path / "authority",
        device_id="device-b",
        remote_devices=("device-a",),
    )
    ledger = _ledger(tmp_path / "operational", authority)

    report = ledger.import_bundles(
        (_bundle(authority, origin="device-a"),),
        context=authority.context(),
    )

    assert report.events_inserted == 2
    assert list(ledger.transactions_dir.iterdir()) == []
    receipts = list((ledger.recovery_dir / "transactions").glob("*.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_bytes())
    assert receipt["schema"] == "memo.operational_transaction_receipt.v1"
    assert receipt["origins"] == ["device-a"]
    assert not list(ledger.root.rglob("stage/*.bin"))


def test_transaction_cleanup_restarts_idempotently_after_receipt(
    tmp_path: Path,
) -> None:
    authority = _authority(
        tmp_path / "authority",
        device_id="device-b",
        remote_devices=("device-a",),
    )
    operational = tmp_path / "operational"
    bundle = _bundle(authority, origin="device-a")

    def lose_power() -> None:
        crashing = _crash_ledger(operational, authority, crash_at="after_receipt")
        crashing.import_bundles((bundle,), context=authority.context())
        raise AssertionError("after_receipt failpoint was not reached")

    _fork_process_loss(lose_power)
    transaction = next((operational / "journal" / "transactions").iterdir())
    receipt = operational / "journal" / "recovery" / "transactions" / (
        f"{transaction.name}.json"
    )
    assert receipt.exists()

    restarted = _ledger(operational, authority)
    restarted.recover()

    assert list(restarted.transactions_dir.iterdir()) == []
    assert receipt.exists()
    assert restarted.position("device-a").sequence == 2
    assert restarted.verify().ok is True


def test_transaction_receipts_and_stages_remain_bounded_across_incremental_imports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(operation_ledger_v2, "_MAX_TRANSACTION_RECEIPTS", 2)
    authority = _authority(
        tmp_path / "authority",
        device_id="device-b",
        remote_devices=("device-a",),
    )
    ledger = _ledger(tmp_path / "operational", authority)

    for sequences in ((1,), (1, 2), (1, 2, 3)):
        report = ledger.import_bundles(
            (_bundle(authority, origin="device-a", sequences=sequences),),
            context=authority.context(),
        )
        assert report.events_inserted == 1

    assert len(list(ledger.transaction_receipts_dir.glob("*.json"))) == 2
    assert list(ledger.transactions_dir.iterdir()) == []
    assert not list(ledger.root.rglob("stage/*.bin"))


def test_compaction_transaction_stages_exact_predecessor_anchor_history(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path / "authority")
    operational = tmp_path / "operational"
    ledger = _ledger(operational, authority)
    genesis = ledger.ensure_anchor()
    event = ledger.append(_command(), context=authority.context())
    state = canonical_json_bytes({"focus": "compacted"})
    checkpoint = _state_checkpoint_bytes(
        origin="device-a",
        through_sequence=event.origin_sequence,
        through_hash=event.event_hash,
        state=state,
    )
    compacted = _signed_anchor(
        authority,
        origin="device-a",
        kind="compaction",
        checkpoint=checkpoint,
        base_sequence=event.origin_sequence,
        base_hash=event.event_hash,
        previous_anchor_hash=genesis.anchor_hash,
        ledger_epoch=1,
        state_sha256=hashlib.sha256(state).hexdigest(),
    )

    def lose_power() -> None:
        crashing = _crash_ledger(operational, authority, crash_at="after_commit")
        crashing.ensure_anchor(compacted, checkpoint=checkpoint)
        raise AssertionError("after_commit failpoint was not reached")

    _fork_process_loss(lose_power)
    transaction = next(ledger.transactions_dir.iterdir())
    manifest = json.loads((transaction / "manifest.json").read_bytes())
    predecessor_relative = (
        f"anchor-history/device-a/{genesis.anchor_hash}.json"
    )
    predecessor_target = next(
        target
        for target in manifest["targets"]
        if target["relative_target"] == predecessor_relative
    )
    assert predecessor_target["after_sha256"] == hashlib.sha256(
        canonical_json_bytes(genesis)
    ).hexdigest()

    restarted = _ledger(operational, authority)
    restarted.recover()
    assert restarted.anchor().anchor_hash == compacted.anchor_hash


def test_import_requires_exact_checkpoint_and_reducer_version(tmp_path: Path) -> None:
    authority = _authority(
        tmp_path / "authority",
        device_id="device-b",
        remote_devices=("device-a",),
    )
    ledger = _ledger(tmp_path / "operational", authority)
    incoming = _bundle(authority, origin="device-a")

    with pytest.raises(OperationalError, match="checkpoint"):
        ledger.import_bundles(
            [replace(incoming, checkpoint=b'{"wrong":true}')],
            context=authority.context(),
        )
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
        ledger.import_bundles(
            [replace(incoming, anchor=changed_anchor)],
            context=authority.context(),
        )
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


def test_historical_roster_bundle_cannot_introduce_bytes_after_revocation(
    tmp_path: Path,
) -> None:
    authority = _authority(
        tmp_path / "authority",
        device_id="device-b",
        remote_devices=("device-a",),
    )
    historical = _bundle(authority, origin="device-a")
    remote = _key_for(authority, "device-a")
    revoked = _with_revocation_proof(
        authority,
        remote,
        revocation_version=3,
    )
    authority.roster = authority.roster.with_keys(
        version=3,
        peers=authority.roster.peers,
        keys=tuple(
            revoked if key.key_id == remote.key_id else key for key in authority.roster.keys
        ),
        signer=authority.signer,
        root=authority.root,
        pin_store=authority.pin_store,
    )
    authority.signer = OperationalSigner(authority.keys, roster_version=3)
    ledger = _ledger(tmp_path / "operational", authority)

    with pytest.raises(
        OperationalError,
        match=r"latest|historical|revoked|roster",
    ):
        ledger.import_bundles(
            [historical],
            context=authority.context(),
        )

    assert ledger.positions() == ()


def test_bundle_import_requires_authenticated_commit_context(tmp_path: Path) -> None:
    authority = _authority(
        tmp_path / "authority",
        device_id="device-b",
        remote_devices=("device-a",),
    )
    ledger = _ledger(tmp_path / "operational", authority)
    incoming = _bundle(authority, origin="device-a")

    with pytest.raises(
        (OperationalError, TypeError),
        match=r"context|authority",
    ):
        ledger.import_bundles([incoming])

    assert ledger.positions() == ()


def test_append_and_revocation_are_serialized_through_event_and_head_durability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _authority(tmp_path / "authority")
    ledger = _ledger(tmp_path / "operational", authority)
    ledger.ensure_anchor()
    context = authority.context()
    event_durable = threading.Event()
    release_head = threading.Event()
    update_finished = threading.Event()
    failures: list[BaseException] = []
    real_write_head = ledger._write_head_atomic

    def blocked_head(event: OperationalEventV2) -> None:
        event_durable.set()
        assert release_head.wait(timeout=3)
        real_write_head(event)

    def append_event() -> None:
        try:
            ledger.append(_command(), context=context)
        except BaseException as exc:
            failures.append(exc)

    def revoke_origin() -> None:
        try:
            _rotate_local_origin(authority)
            update_finished.set()
        except BaseException as exc:
            failures.append(exc)

    monkeypatch.setattr(ledger, "_write_head_atomic", blocked_head)
    appender = threading.Thread(target=append_event)
    revoker = threading.Thread(target=revoke_origin)
    appender.start()
    assert event_durable.wait(timeout=3)
    revoker.start()
    revoked_before_commit = update_finished.wait(timeout=0.25)
    release_head.set()
    appender.join(timeout=3)
    revoker.join(timeout=3)

    assert not revoked_before_commit
    assert update_finished.is_set()
    assert failures == []
    restarted = _ledger(tmp_path / "operational", authority)
    assert restarted.verify().ok is True
    assert restarted.iter_events()[0].roster_version == 1


def test_revocation_wins_before_append_and_historical_durable_event_still_verifies(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path / "authority")
    operational = tmp_path / "operational"
    old_signer = authority.signer
    ledger = _ledger(operational, authority)
    ledger.ensure_anchor()
    durable = ledger.append(_command(), context=authority.context())
    _rotate_local_origin(authority)

    stale = _ledger(operational, authority, signer=old_signer)
    with pytest.raises(OperationalError, match=r"stale|roster|revoked"):
        stale.append(
            _command(idempotency_key="stale-after-revocation"),
            context=authority.context(),
        )

    restarted = _ledger(operational, authority)
    assert restarted.verify().ok is True
    assert restarted.iter_events() == [durable]


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
