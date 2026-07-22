"""Tests for the verification-state recall penalty (`_apply_verification_decay`).

This is the live search-scoring pass (wired in search_ops behind
MEMO_VERIFICATION_STATE_TRACKING): it multiplies each hit's score by its
verification-state decay factor. verification_state / verified_at ride on the
MemoryRecord, so the pass is pure — no memory_map, no store lookup.
"""

from __future__ import annotations

import time

from memo.memory import Memory
from memo.memory.record import MemoryRecord
from memo.tiers import VerificationState


def _rec(
    id_: str, *, state: VerificationState, verified_at: int | None, score: float
) -> MemoryRecord:
    return MemoryRecord(
        id=id_,
        path=f"test/{id_}.md",
        title=id_,
        type="fact",
        tags=[],
        created="2026-01-01T00:00:00",
        updated="2026-01-01T00:00:00",
        body=f"body for {id_}",
        verification_state=state,
        verified_at=verified_at,
        score=score,
    )


def test_verified_outranks_unverified_at_equal_input_score(mock_memory: Memory):
    now = int(time.time())
    out = mock_memory._apply_verification_decay(
        [
            _rec("verified1", state=VerificationState.VERIFIED, verified_at=now, score=0.9),
            _rec("unverified1", state=VerificationState.UNVERIFIED, verified_at=None, score=0.9),
        ]
    )
    scores = {r.id: r.score for r in out}
    assert scores["verified1"] > scores["unverified1"]  # 0.9×1.0 > 0.9×0.8


def test_stale_is_penalized_below_verified(mock_memory: Memory):
    now = int(time.time())
    out = mock_memory._apply_verification_decay(
        [
            _rec("verified1", state=VerificationState.VERIFIED, verified_at=now, score=0.9),
            _rec("stale1", state=VerificationState.STALE, verified_at=now, score=0.9),
        ]
    )
    scores = {r.id: r.score for r in out}
    assert scores["verified1"] > scores["stale1"]  # 0.9×1.0 > 0.9×0.7


def test_decay_reorders_a_higher_scored_stale_below_verified(mock_memory: Memory):
    """A STALE hit that led on raw score falls behind a fresh VERIFIED one."""
    now = int(time.time())
    out = mock_memory._apply_verification_decay(
        [
            _rec(
                "stale_lead", state=VerificationState.STALE, verified_at=now, score=0.90
            ),  # ×0.7=0.63
            _rec(
                "verified_tail", state=VerificationState.VERIFIED, verified_at=now, score=0.80
            ),  # ×1.0=0.80
        ]
    )
    assert out[0].id == "verified_tail"  # re-sorted ahead after decay
    assert out[1].id == "stale_lead"


def test_all_verified_fresh_is_a_noop(mock_memory: Memory):
    """Factor 1.0 for every hit → scores and order unchanged (identity)."""
    now = int(time.time())
    hits = [
        _rec("a", state=VerificationState.VERIFIED, verified_at=now, score=0.9),
        _rec("b", state=VerificationState.VERIFIED, verified_at=now, score=0.8),
    ]
    out = mock_memory._apply_verification_decay(hits)
    assert [(r.id, r.score) for r in out] == [("a", 0.9), ("b", 0.8)]


def test_verified_but_old_gets_mild_penalty(mock_memory: Memory):
    """VERIFIED older than 7 days decays to 0.95, not the full-fresh 1.0."""
    old = int(time.time()) - (10 * 86400)
    out = mock_memory._apply_verification_decay(
        [_rec("v", state=VerificationState.VERIFIED, verified_at=old, score=1.0)]
    )
    assert out[0].score == 0.95
