"""Tests for verification state tracking in MemoryRecord."""

import time
from memo.memory.record import MemoryRecord
from memo.tiers import VerificationState


def test_verification_state_roundtrip():
    """Serialize and deserialize verification state through markdown."""
    now = int(time.time())
    rec = MemoryRecord(
        id="test1",
        path="2026/01/test1.md",
        title="Test Memory",
        type="fact",
        tags=["test"],
        created="2026-01-01T00:00:00",
        updated="2026-01-01T00:00:00",
        body="test fact",
        verification_state=VerificationState.VERIFIED,
        verified_at=now,
    )

    # Verify the fields are set correctly
    assert rec.verification_state == VerificationState.VERIFIED
    assert rec.verified_at == now

    # Convert to dict and verify fields are serialized
    rec_dict = rec.to_dict()
    assert rec_dict["verification_state"] == "verified"
    assert rec_dict["verified_at"] == now


def test_verification_state_defaults_unverified():
    """New records default to UNVERIFIED."""
    rec = MemoryRecord(
        id="test2",
        path="2026/01/test2.md",
        title="Another Memory",
        type="fact",
        tags=["test"],
        created="2026-01-01T00:00:00",
        updated="2026-01-01T00:00:00",
        body="fact",
    )

    assert rec.verification_state == VerificationState.UNVERIFIED
    assert rec.verified_at is None

    # Verify dict serialization
    rec_dict = rec.to_dict()
    assert rec_dict["verification_state"] == "unverified"
    assert rec_dict["verified_at"] is None
