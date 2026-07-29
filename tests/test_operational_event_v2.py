from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

import memo.operational_event as operational_event
from memo.error_contract import OperationalError, OperationalErrorCode
from memo.identity import PrincipalIdentity
from memo.operational_event import (
    ChainAnchor,
    MigrationOrigin,
    OperationalEventV2,
    canonical_anchor_hash,
    canonical_event_hash,
    canonical_json_bytes,
    canonical_signed_bytes,
    validate_anchor,
    validate_event,
)
from memo.operational_event_types import FOCUS_SET
from memo.operational_key_store import DeviceKeyStore
from memo.operational_roster import VerificationRoster
from memo.operational_signing import OperationalSigner, OperationalVerifier


def _event(**changes: object) -> OperationalEventV2:
    identity = PrincipalIdentity(
        principal_id="device-a:session-a",
        actor_id="agent-a",
        kind="agent",
        device_id="device-a",
        session_id="session-a",
        source_client="codex",
    )
    event = OperationalEventV2(
        schema="memo.operational_event.v2",
        schema_version=2,
        event_id="event-1",
        event_type=FOCUS_SET,
        actor=identity,
        target_id=None,
        project="demo",
        workspace="/tmp/demo",
        origin_device="device-a",
        origin_sequence=1,
        logical_clock="1",
        authority_epoch=0,
        control_oid="control-0",
        created_at="2026-07-29T12:00:00Z",
        expires_at=None,
        visibility="owner",
        idempotency_key="idem-1",
        caused_by=(),
        subject_uri="memo://focus/demo",
        trace_id="trace-1",
        payload={"summary": "Current focus", "project": "demo"},
        content_hash="content",
        previous_hash="",
        event_hash="",
        source_proof=None,
        roster_version=1,
        key_id="key-1",
        signature="sig",
    )
    event = replace(event, **changes)
    return replace(event, event_hash=canonical_event_hash(event))


def test_v2_event_hash_is_canonical_and_tamper_evident() -> None:
    first = _event(payload={"summary": "Current focus", "project": "demo"})
    same = _event(payload={"project": "demo", "summary": "Current focus"})
    assert first.event_hash == same.event_hash
    changed = replace(first, payload={"project": "demo", "summary": "Changed"})
    assert canonical_event_hash(changed) != first.event_hash
    validate_event(first)


def test_v2_validation_rejects_schema_sequence_and_hash() -> None:
    with pytest.raises(OperationalError) as exc:
        validate_event(_event(schema="memo.operational_event.v9"))
    assert exc.value.code is OperationalErrorCode.UNKNOWN_SCHEMA
    with pytest.raises(OperationalError):
        validate_event(_event(origin_sequence=0))
    with pytest.raises(OperationalError):
        validate_event(replace(_event(), event_hash="bad"))


def test_migration_origin_uses_normative_migration_device_id() -> None:
    origin = MigrationOrigin(
        schema="memo.operational_migration_origin.v1",
        attempt_id="attempt-1",
        migration_device_id="migration-device",
        source_manifest_sha256="a" * 64,
        capability_manifest_sha256="b" * 64,
        attestor_device_id="device-a",
        attestor_key_id="key-1",
        roster_version=1,
        issued_at="2026-07-29T12:00:00Z",
        expires_at="2026-07-29T13:00:00Z",
        signature="sig",
    )
    assert origin.migration_device_id == "migration-device"
    assert not hasattr(origin, "device_id")


def test_canonical_json_rejects_nan_and_non_string_mapping_keys() -> None:
    with pytest.raises((TypeError, ValueError)):
        canonical_json_bytes({"value": float("nan")})
    with pytest.raises((TypeError, ValueError)):
        canonical_json_bytes({1: "integer", "1": "string"})


def test_signed_migration_origin_binds_exclusive_attestor_and_validity(tmp_path) -> None:
    keys = DeviceKeyStore.in_memory()
    origin_key = keys.generate(device_id="device-a", roles=("origin",))
    roster = VerificationRoster.bootstrap(
        device_id="device-a", key=origin_key, root=tmp_path
    )
    attestor_key = keys.generate(
        device_id="device-a",
        roles=("migration_attestor",),
        enrollment_sequence=2,
    )
    roster = roster.with_keys(
        version=2,
        peers=("device-a",),
        keys=(origin_key, attestor_key),
        signer=OperationalSigner(keys, roster_version=1),
        root=tmp_path,
    )
    signer = OperationalSigner(keys, roster_version=2)
    unsigned = MigrationOrigin(
        schema="memo.operational_migration_origin.v1",
        attempt_id="attempt-1",
        migration_device_id="migration-device",
        source_manifest_sha256="a" * 64,
        capability_manifest_sha256="b" * 64,
        attestor_device_id="device-a",
        attestor_key_id=attestor_key.key_id,
        roster_version=2,
        issued_at="2026-07-29T12:00:00Z",
        expires_at="2026-07-29T13:00:00Z",
        signature="",
    )
    envelope = signer.sign(
        domain="memo.operational.migration_origin.v1",
        payload=canonical_signed_bytes(unsigned),
        key_id=attestor_key.key_id,
    )
    signed = replace(unsigned, signature=envelope.signature)
    operational_event.validate_migration_origin(
        signed,
        roster=roster,
        verifier=OperationalVerifier(),
        at_time="2026-07-29T12:30:00Z",
    )
    with pytest.raises(OperationalError):
        operational_event.validate_migration_origin(
            replace(signed, expires_at="2026-07-29T12:20:00Z"),
            roster=roster,
            verifier=OperationalVerifier(),
            at_time="2026-07-29T12:30:00Z",
        )
    with pytest.raises(OperationalError):
        operational_event.validate_migration_origin(
            replace(signed, source_manifest_sha256="not-a-digest"),
            roster=roster,
            verifier=OperationalVerifier(),
            at_time="2026-07-29T12:30:00Z",
        )


def _signed_anchor(
    *,
    signer: OperationalSigner,
    key_id: str,
    kind: str,
    signer_role: str,
    checkpoint: bytes,
) -> ChainAnchor:
    unsigned = ChainAnchor(
        schema="memo.operational_anchor.v1",
        anchor_id=f"anchor-{kind}",
        origin_device="legacy-device" if kind == "memo_v1" else "device-a",
        ledger_epoch=0,
        reducer_version=1,
        kind=kind,  # type: ignore[arg-type]
        base_sequence=0,
        base_event_hash="",
        final_sequence=0,
        final_event_hash="",
        previous_anchor_hash="",
        source_manifest_sha256="a" * 64 if kind == "memo_v1" else "",
        state_sha256="b" * 64,
        checkpoint_id="checkpoint-1",
        checkpoint_sha256=hashlib.sha256(checkpoint).hexdigest(),
        checkpoint_size=len(checkpoint),
        created_at="2026-07-29T12:00:00Z",
        anchor_hash="",
        roster_version=1,
        signer_role=signer_role,  # type: ignore[arg-type]
        attested_origin="legacy-device" if kind == "memo_v1" else "",
        key_id=key_id,
        signature="",
    )
    hashed = replace(unsigned, anchor_hash=canonical_anchor_hash(unsigned))
    envelope = signer.sign(
        domain="memo.operational.anchor.v1",
        payload=canonical_signed_bytes(hashed),
        key_id=key_id,
    )
    return replace(hashed, signature=envelope.signature)


def test_anchor_kind_authorization_checkpoint_and_signature(tmp_path) -> None:
    keys = DeviceKeyStore.in_memory()
    origin_key = keys.generate(device_id="device-a", roles=("origin",))
    roster = VerificationRoster.bootstrap(
        device_id="device-a", key=origin_key, root=tmp_path
    )
    signer = OperationalSigner(keys, roster_version=1)
    checkpoint = b"{}"
    anchor = _signed_anchor(
        signer=signer,
        key_id=origin_key.key_id,
        kind="empty",
        signer_role="origin",
        checkpoint=checkpoint,
    )
    validate_anchor(
        anchor,
        checkpoint=checkpoint,
        roster=roster,
        verifier=OperationalVerifier(),
    )
    with pytest.raises(OperationalError):
        validate_anchor(
            replace(anchor, checkpoint_size=999),
            checkpoint=checkpoint,
            roster=roster,
            verifier=OperationalVerifier(),
        )
    with pytest.raises(OperationalError):
        validate_anchor(
            replace(anchor, signer_role="migration_attestor"),
            checkpoint=checkpoint,
            roster=roster,
            verifier=OperationalVerifier(),
        )
    nonempty = _signed_anchor(
        signer=signer,
        key_id=origin_key.key_id,
        kind="empty",
        signer_role="origin",
        checkpoint=b'{"prepopulated":true}',
    )
    with pytest.raises(OperationalError):
        validate_anchor(
            nonempty,
            roster=roster,
            verifier=OperationalVerifier(),
        )


def test_memo_v1_anchor_rejects_prepopulated_checkpoint_and_dual_role_attestor(
    tmp_path,
) -> None:
    keys = DeviceKeyStore.in_memory()
    key = keys.generate(device_id="device-a")
    roster = VerificationRoster.bootstrap(device_id="device-a", key=key, root=tmp_path)
    signer = OperationalSigner(keys, roster_version=1)
    checkpoint = canonical_json_bytes({"focus": {"memo": "already populated"}})
    anchor = _signed_anchor(
        signer=signer,
        key_id=key.key_id,
        kind="memo_v1",
        signer_role="migration_attestor",
        checkpoint=checkpoint,
    )
    with pytest.raises(OperationalError):
        validate_anchor(
            anchor,
            checkpoint=checkpoint,
            roster=roster,
            verifier=OperationalVerifier(),
        )
