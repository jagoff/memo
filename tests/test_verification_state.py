"""Tests for verification state tracking in MemoryRecord and state transitions."""

import time

from memo.memory import Memory
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


def test_stale_transition(mock_memory: Memory):
    """VERIFIED memory transitions to STALE after stale_age_days."""
    # Create a memory with VERIFIED state from 35 days ago
    old_timestamp = int(time.time()) - (35 * 86400)  # 35 days ago
    old_rec = MemoryRecord(
        id="old_verified",
        path="2026/01/old_verified.md",
        title="Old Verified Memory",
        type="fact",
        tags=["test"],
        created="2026-01-01T00:00:00",
        updated="2026-01-01T00:00:00",
        body="old fact",
        verification_state=VerificationState.VERIFIED,
        verified_at=old_timestamp,
    )

    # Set up memory_map for the transition logic
    mock_memory.memory_map = {"old_verified": old_rec}

    # Run transition with stale_age_days=30 (35 days old should trigger)
    transitioned = mock_memory._transition_stale_memories(stale_age_days=30, unverify_age_days=60)

    assert transitioned == 1
    # Verify the memory was transitioned to STALE
    updated_mem = mock_memory.memory_map.get("old_verified")
    assert updated_mem is not None
    assert updated_mem.verification_state == VerificationState.STALE
    assert updated_mem.verified_at == old_timestamp  # verified_at is preserved


def test_stale_to_unverified_transition(mock_memory: Memory):
    """STALE memory transitions to UNVERIFIED after unverify_age_days."""
    # Create a memory with STALE state from 65 days ago
    old_timestamp = int(time.time()) - (65 * 86400)  # 65 days ago
    stale_rec = MemoryRecord(
        id="stale_memory",
        path="2026/01/stale_memory.md",
        title="Stale Memory",
        type="fact",
        tags=["test"],
        created="2026-01-01T00:00:00",
        updated="2026-01-01T00:00:00",
        body="stale fact",
        verification_state=VerificationState.STALE,
        verified_at=old_timestamp,
    )

    # Set up memory_map for the transition logic
    mock_memory.memory_map = {"stale_memory": stale_rec}

    # Run transition with unverify_age_days=60 (65 days old should trigger)
    transitioned = mock_memory._transition_stale_memories(stale_age_days=30, unverify_age_days=60)

    assert transitioned == 1
    # Verify the memory was transitioned to UNVERIFIED and verified_at was cleared
    updated_mem = mock_memory.memory_map.get("stale_memory")
    assert updated_mem is not None
    assert updated_mem.verification_state == VerificationState.UNVERIFIED
    assert updated_mem.verified_at is None  # verified_at is cleared


def test_no_transition_when_too_recent(mock_memory: Memory):
    """Recent VERIFIED memories do not transition."""
    # Create a memory with VERIFIED state from 10 days ago (within 30-day threshold)
    recent_timestamp = int(time.time()) - (10 * 86400)  # 10 days ago
    recent_rec = MemoryRecord(
        id="recent_verified",
        path="2026/01/recent_verified.md",
        title="Recent Verified Memory",
        type="fact",
        tags=["test"],
        created="2026-01-01T00:00:00",
        updated="2026-01-01T00:00:00",
        body="recent fact",
        verification_state=VerificationState.VERIFIED,
        verified_at=recent_timestamp,
    )

    # Set up memory_map for the transition logic
    mock_memory.memory_map = {"recent_verified": recent_rec}

    # Run transition with stale_age_days=30
    transitioned = mock_memory._transition_stale_memories(stale_age_days=30, unverify_age_days=60)

    assert transitioned == 0
    # Verify the memory was NOT transitioned
    updated_mem = mock_memory.memory_map.get("recent_verified")
    assert updated_mem is not None
    assert updated_mem.verification_state == VerificationState.VERIFIED
