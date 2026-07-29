from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from memo.errors import KeyRevokedError
from memo.operational_event import canonical_json_bytes
from memo.operational_key_store import (
    AuthorityPinStore,
    DeviceKeyStore,
    InMemoryAuthorityPinProvider,
)
from memo.operational_roster import VerificationRoster
from memo.operational_signing import (
    OperationalSigner,
    OperationalVerifier,
    SignatureError,
)

_AUTHORITY_PINS = InMemoryAuthorityPinProvider()


def _pin_store(root: Path) -> AuthorityPinStore:
    return AuthorityPinStore._for_test(root, provider=_AUTHORITY_PINS)


def test_signature_is_domain_separated_and_roster_bound(tmp_path) -> None:
    keys = DeviceKeyStore.in_memory()
    public_key = keys.generate(device_id="device-a")
    roster = VerificationRoster.bootstrap(
        device_id="device-a", key=public_key, root=tmp_path, pin_store=_pin_store(tmp_path)
    )
    signer = OperationalSigner(keys, roster_version=roster.version)
    verifier = OperationalVerifier()
    envelope = signer.sign(
        domain="memo.operational.event.v2",
        payload=b'{"event_id":"e1"}',
        key_id=roster.local_key_id,
    )
    verifier.verify(
        domain="memo.operational.event.v2",
        payload=b'{"event_id":"e1"}',
        envelope=envelope,
        roster=roster,
    )
    with pytest.raises(SignatureError):
        verifier.verify(
            domain="memo.cutover.vote.v1",
            payload=b'{"event_id":"e1"}',
            envelope=envelope,
            roster=roster,
        )
    with pytest.raises(SignatureError):
        verifier.verify(
            domain="memo.operational.event.v2",
            payload=b'{"event_id":"e2"}',
            envelope=envelope,
            roster=roster,
        )
    with pytest.raises(SignatureError):
        verifier.verify(
            domain="memo.operational.event.v2",
            payload=b'{"event_id":"e1"}',
            envelope=replace(envelope, roster_version=roster.version + 1),
            roster=roster,
        )


def test_signer_rejects_unknown_domain_and_key() -> None:
    keys = DeviceKeyStore.in_memory()
    record = keys.generate(device_id="device-a")
    signer = OperationalSigner(keys, roster_version=1)
    with pytest.raises(SignatureError):
        signer.sign(domain="unknown", payload=b"{}", key_id=record.key_id)
    with pytest.raises(SignatureError):
        signer.sign(domain="memo.operational.event.v2", payload=b"{}", key_id="missing")


def test_migration_signature_binds_declared_key_roster_and_exclusive_role(
    tmp_path,
) -> None:
    keys = DeviceKeyStore.in_memory()
    dual_role = keys.generate(device_id="device-a")
    roster = VerificationRoster.bootstrap(
        device_id="device-a", key=dual_role, root=tmp_path, pin_store=_pin_store(tmp_path)
    )
    signer = OperationalSigner(keys, roster_version=1)
    verifier = OperationalVerifier()
    body = {
        "schema": "memo.operational_migration_origin.v1",
        "attempt_id": "attempt-1",
        "migration_device_id": "migration-a",
        "source_manifest_sha256": "a" * 64,
        "capability_manifest_sha256": "b" * 64,
        "attestor_device_id": "device-a",
        "attestor_key_id": dual_role.key_id,
        "roster_version": 1,
        "issued_at": "2026-07-29T12:00:00Z",
        "expires_at": "2026-07-29T13:00:00Z",
        "signature": "",
    }
    payload = canonical_json_bytes(body)
    envelope = signer.sign(
        domain="memo.operational.migration_origin.v1",
        payload=payload,
        key_id=dual_role.key_id,
    )
    with pytest.raises(SignatureError):
        verifier.verify(
            domain="memo.operational.migration_origin.v1",
            payload=payload,
            envelope=envelope,
            roster=roster,
        )

    for field, value in (
        ("attestor_key_id", "declared-other-key"),
        ("roster_version", 999),
    ):
        changed = {**body, field: value}
        changed_payload = canonical_json_bytes(changed)
        changed_envelope = signer.sign(
            domain="memo.operational.migration_origin.v1",
            payload=changed_payload,
            key_id=dual_role.key_id,
        )
        with pytest.raises(SignatureError):
            verifier.verify(
                domain="memo.operational.migration_origin.v1",
                payload=changed_payload,
                envelope=changed_envelope,
                roster=roster,
            )


def test_event_signatures_enforce_device_enrollment_and_revocation(tmp_path) -> None:
    keys = DeviceKeyStore.in_memory()
    public = keys.generate(device_id="device-a")
    roster = VerificationRoster.bootstrap(
        device_id="device-a", key=public, root=tmp_path, pin_store=_pin_store(tmp_path)
    )
    signer = OperationalSigner(keys, roster_version=1)
    verifier = OperationalVerifier()

    def verify_with(record: object, sequence: int) -> None:
        body = {
            "schema": "memo.operational_event.v2",
            "origin_device": "device-a",
            "origin_sequence": sequence,
            "key_id": public.key_id,
            "roster_version": 1,
            "signature": "",
        }
        payload = canonical_json_bytes(body)
        envelope = signer.sign(
            domain="memo.operational.event.v2",
            payload=payload,
            key_id=public.key_id,
        )
        custom = replace(roster, keys=(record,))  # type: ignore[arg-type]
        verifier.verify(
            domain="memo.operational.event.v2",
            payload=payload,
            envelope=envelope,
            roster=custom,
        )

    with pytest.raises(SignatureError):
        verify_with(replace(public, enrollment_sequence=2), 1)
    with pytest.raises(KeyRevokedError):
        verify_with(replace(public, revocation_sequence=2), 2)
    with pytest.raises(SignatureError):
        body = {
            "schema": "memo.operational_event.v2",
            "origin_device": "device-b",
            "origin_sequence": 1,
            "key_id": public.key_id,
            "roster_version": 1,
            "signature": "",
        }
        payload = canonical_json_bytes(body)
        verifier.verify(
            domain="memo.operational.event.v2",
            payload=payload,
            envelope=signer.sign(
                domain="memo.operational.event.v2",
                payload=payload,
                key_id=public.key_id,
            ),
            roster=roster,
        )


def test_anchor_signature_binds_embedded_key_and_roster(tmp_path) -> None:
    keys = DeviceKeyStore.in_memory()
    public = keys.generate(device_id="device-a")
    roster = VerificationRoster.bootstrap(
        device_id="device-a", key=public, root=tmp_path, pin_store=_pin_store(tmp_path)
    )
    signer = OperationalSigner(keys, roster_version=1)
    verifier = OperationalVerifier()
    body = {
        "schema": "memo.operational_anchor.v1",
        "origin_device": "device-a",
        "final_sequence": 1,
        "signer_role": "origin",
        "key_id": "declared-other-key",
        "roster_version": 1,
        "signature": "",
    }
    payload = canonical_json_bytes(body)
    envelope = signer.sign(
        domain="memo.operational.anchor.v1",
        payload=payload,
        key_id=public.key_id,
    )
    with pytest.raises(SignatureError):
        verifier.verify(
            domain="memo.operational.anchor.v1",
            payload=payload,
            envelope=envelope,
            roster=roster,
        )


def test_system_capability_signature_binds_key_device_roster_and_internal_role(
    tmp_path,
) -> None:
    keys = DeviceKeyStore.in_memory()
    public = keys.generate(device_id="device-a", roles=("origin",))
    roster = VerificationRoster.bootstrap(
        device_id="device-a",
        key=public,
        root=tmp_path,
        pin_store=_pin_store(tmp_path),
    )
    signer = OperationalSigner(keys, roster_version=1)
    verifier = OperationalVerifier()
    body = {
        "schema": "memo.operational_system_capability.v1",
        "authority_id": "4b443953-c1bb-4eaf-aa67-b7608f82f98b",
        "authority_root_sha256": "a" * 64,
        "process_nonce": "b" * 64,
        "fence_nonce": "c" * 64,
        "system_role": "daemon",
        "device_id": "device-a",
        "roster_version": 1,
        "roster_hash": roster.roster_hash,
        "key_id": public.key_id,
    }
    payload = canonical_json_bytes(body)
    envelope = signer.sign(
        domain="memo.operational.system_capability.v1",
        payload=payload,
        key_id=public.key_id,
    )
    verifier.verify(
        domain="memo.operational.system_capability.v1",
        payload=payload,
        envelope=envelope,
        roster=roster,
    )

    for field, value in (
        ("key_id", "declared-other-key"),
        ("device_id", "device-b"),
        ("roster_version", 999),
        ("system_role", "adapter"),
    ):
        changed_payload = canonical_json_bytes({**body, field: value})
        changed_envelope = signer.sign(
            domain="memo.operational.system_capability.v1",
            payload=changed_payload,
            key_id=public.key_id,
        )
        with pytest.raises(SignatureError):
            verifier.verify(
                domain="memo.operational.system_capability.v1",
                payload=changed_payload,
                envelope=changed_envelope,
                roster=roster,
            )
