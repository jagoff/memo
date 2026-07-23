"""Tests for verification state tracking in MemoryRecord and state transitions.

The transition tests drive the REAL store (`mock_memory` is a real `Memory`
with a stubbed embedder): `_transition_stale_memories` loads its candidates from
`store.verification_candidates()` and persists via `store.update_verification`,
so these tests seed state through the store and read it back through it.
"""

import time

from memo.memory import Memory
from memo.memory.record import MemoryRecord
from memo.tiers import VerificationState


def _seed(
    mem: Memory,
    *,
    title: str,
    state: VerificationState,
    verified_at: int | None,
    review_after: str | None = None,
) -> str:
    """Save a fact and set its verification state directly in the store."""
    rec = mem.save(content=f"a durable fact body for {title}", title=title, type_="fact")
    mem.store.update_review_state(
        id_=rec.id,
        review_after=review_after,
        verification_state=state.value,
        verified_at=verified_at,
    )
    return rec.id


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

    assert rec.verification_state == VerificationState.VERIFIED
    assert rec.verified_at == now

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

    rec_dict = rec.to_dict()
    assert rec_dict["verification_state"] == "unverified"
    assert rec_dict["verified_at"] is None


def test_stale_transition(mock_memory: Memory):
    """VERIFIED memory transitions to STALE after the stale-days threshold."""
    old_ts = int(time.time()) - (35 * 86400)  # 35 days ago (> default 30)
    mid = _seed(
        mock_memory,
        title="Old Verified",
        state=VerificationState.VERIFIED,
        verified_at=old_ts,
        review_after="2020-01-01T00:00:00+00:00",
    )

    transitioned = mock_memory._transition_stale_memories()

    assert transitioned == 1
    updated = mock_memory.get(mid)
    assert updated is not None
    assert updated.verification_state == VerificationState.STALE
    assert updated.verified_at == old_ts  # preserved: STALE clock keeps counting


def test_stale_remains_stale_until_reviewed_or_invalidated(mock_memory: Memory):
    """STALE no longer decays into semantically never-verified state."""
    old_ts = int(time.time()) - (65 * 86400)  # 65 days ago (> default 60)
    mid = _seed(mock_memory, title="Stale One", state=VerificationState.STALE, verified_at=old_ts)

    transitioned = mock_memory._transition_stale_memories()

    assert transitioned == 0
    updated = mock_memory.get(mid)
    assert updated is not None
    assert updated.verification_state == VerificationState.STALE
    assert updated.verified_at == old_ts


def test_no_transition_when_too_recent(mock_memory: Memory):
    """A recently-verified memory does not transition."""
    recent_ts = int(time.time()) - (10 * 86400)  # within the 30-day threshold
    mid = _seed(
        mock_memory,
        title="Recent Verified",
        state=VerificationState.VERIFIED,
        verified_at=recent_ts,
    )

    transitioned = mock_memory._transition_stale_memories()

    assert transitioned == 0
    updated = mock_memory.get(mid)
    assert updated is not None
    assert updated.verification_state == VerificationState.VERIFIED


def test_dry_run_reports_but_does_not_persist(mock_memory: Memory):
    """dry_run counts would-be transitions without mutating the store."""
    old_ts = int(time.time()) - (35 * 86400)
    mid = _seed(
        mock_memory,
        title="DryRun Verified",
        state=VerificationState.VERIFIED,
        verified_at=old_ts,
        review_after="2020-01-01T00:00:00+00:00",
    )

    would = mock_memory._transition_stale_memories(dry_run=True)

    assert would == 1
    assert mock_memory.get(mid).verification_state == VerificationState.VERIFIED  # unchanged


def test_unverified_memories_are_not_candidates(mock_memory: Memory):
    """UNVERIFIED memories (the default majority) never enter the candidate set."""
    _seed(mock_memory, title="Plain", state=VerificationState.UNVERIFIED, verified_at=None)

    assert mock_memory.store.verification_candidates() == []
    assert mock_memory._transition_stale_memories() == 0
