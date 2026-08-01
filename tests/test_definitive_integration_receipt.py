from __future__ import annotations

from pathlib import Path

import pytest

from memo.definitive_integration import sign_integration_receipt, verify_integration_receipt
from memo.errors import OperationalError
from tests.operational_authority import build_test_fresh_v2_authority


def _receipt(authority, *, attempt_id: str):
    return sign_integration_receipt(
        attempt_id=attempt_id,
        checks={"two_peer": True},
        evidence={"state_sha256": "a" * 64},
        signer_device_id=authority.roster.local_device_id,
        signer_key_id=authority.key_id,
        roster_version=authority.roster.version,
        roster_hash=authority.roster.roster_hash,
        created_at="2026-07-31T12:00:00+00:00",
        signer=authority.signer,
    )


def test_receipt_and_roster_substitution_cannot_replace_external_trust_root(
    tmp_path: Path,
) -> None:
    trusted = build_test_fresh_v2_authority(tmp_path / "trusted", device_id="device-a")
    attacker = build_test_fresh_v2_authority(tmp_path / "attacker", device_id="device-x")

    signed = _receipt(trusted, attempt_id="trusted-proof")
    verify_integration_receipt(
        signed,
        roster=trusted.roster,
        trusted_roster_hash=trusted.roster.roster_hash,
    )

    substituted = _receipt(attacker, attempt_id="forged-proof")
    with pytest.raises(OperationalError, match="trusted roster"):
        verify_integration_receipt(
            substituted,
            roster=attacker.roster,
            trusted_roster_hash=trusted.roster.roster_hash,
        )
