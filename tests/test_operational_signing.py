from __future__ import annotations

from dataclasses import replace

import pytest

from memo.operational_key_store import DeviceKeyStore
from memo.operational_roster import VerificationRoster
from memo.operational_signing import (
    OperationalSigner,
    OperationalVerifier,
    SignatureError,
)


def test_signature_is_domain_separated_and_roster_bound(tmp_path) -> None:
    keys = DeviceKeyStore.in_memory()
    public_key = keys.generate(device_id="device-a")
    roster = VerificationRoster.bootstrap(
        device_id="device-a", key=public_key, root=tmp_path
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
        signer.sign(
            domain="memo.operational.event.v2", payload=b"{}", key_id="missing"
        )
