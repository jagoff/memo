"""Tests for verification state decay in rerank."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from memo.config import Config
from memo.memory.facade import Memory
from memo.memory.record import MemoryRecord
from memo.tiers import VerificationState


@pytest.fixture
def tmp_cfg(tmp_path):
    """Isolated test config."""
    data_dir = tmp_path / "data"
    state_dir = tmp_path / "state"
    data_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    return Config(data_dir=str(data_dir), state_dir=str(state_dir))


def test_rerank_prioritizes_verified(tmp_cfg):
    """VERIFIED memories score higher than UNVERIFIED when state tracking enabled."""
    memory = Memory(tmp_cfg)

    now = int(time.time())
    verified_rec = MemoryRecord(
        id="verified1",
        path="test/verified.md",
        title="verified fact",
        type="fact",
        tags=[],
        created="2026-01-01T00:00:00",
        updated="2026-01-01T00:00:00",
        body="This is a verified fact",
        verification_state=VerificationState.VERIFIED,
        verified_at=now,
    )
    unverified_rec = MemoryRecord(
        id="unverified1",
        path="test/unverified.md",
        title="unverified fact",
        type="fact",
        tags=[],
        created="2026-01-01T00:00:00",
        updated="2026-01-01T00:00:00",
        body="This is an unverified fact",
        verification_state=VerificationState.UNVERIFIED,
        verified_at=None,
    )

    # Populate memory_map (normally populated by search)
    memory.memory_map = {"verified1": verified_rec, "unverified1": unverified_rec}

    # Input hits with equal scores
    hits = [
        {"id": "verified1", "score": 0.9},
        {"id": "unverified1", "score": 0.9},
    ]

    # Apply rerank logic with state tracking enabled
    with patch("memo.memory.rerank_ops.flag_bool") as mock_flag_bool:

        def flag_bool_side_effect(flag_name):
            if flag_name == "MEMO_VERIFICATION_STATE_TRACKING":
                return True
            if flag_name == "MEMO_GRAPH_DISTANCE_DECAY":
                return False
            return False

        mock_flag_bool.side_effect = flag_bool_side_effect
        result = memory._rerank_logic(
            hits=hits,
            query="test",
            rerank_candidates=2,
        )

    # Extract scores
    verified_score = next((h["score"] for h in result if h["id"] == "verified1"), None)
    unverified_score = next((h["score"] for h in result if h["id"] == "unverified1"), None)

    # VERIFIED (1.0 decay) should be higher than UNVERIFIED (0.8 decay)
    assert verified_score is not None, "verified1 should be in result"
    assert unverified_score is not None, "unverified1 should be in result"
    assert verified_score > unverified_score, (
        f"VERIFIED {verified_score:.3f} should > UNVERIFIED {unverified_score:.3f}"
    )


def test_rerank_stale_decays(tmp_cfg):
    """STALE memories score lower than VERIFIED."""
    memory = Memory(tmp_cfg)

    now = int(time.time())
    verified_rec = MemoryRecord(
        id="verified1",
        path="test/verified.md",
        title="verified fact",
        type="fact",
        tags=[],
        created="2026-01-01T00:00:00",
        updated="2026-01-01T00:00:00",
        body="This is a verified fact",
        verification_state=VerificationState.VERIFIED,
        verified_at=now,
    )
    stale_rec = MemoryRecord(
        id="stale1",
        path="test/stale.md",
        title="stale fact",
        type="fact",
        tags=[],
        created="2026-01-01T00:00:00",
        updated="2026-01-01T00:00:00",
        body="This is a stale fact",
        verification_state=VerificationState.STALE,
        verified_at=now,
    )

    memory.memory_map = {"verified1": verified_rec, "stale1": stale_rec}

    hits = [
        {"id": "verified1", "score": 0.9},
        {"id": "stale1", "score": 0.9},
    ]

    with patch("memo.memory.rerank_ops.flag_bool") as mock_flag_bool:

        def flag_bool_side_effect(flag_name):
            if flag_name == "MEMO_VERIFICATION_STATE_TRACKING":
                return True
            if flag_name == "MEMO_GRAPH_DISTANCE_DECAY":
                return False
            return False

        mock_flag_bool.side_effect = flag_bool_side_effect
        result = memory._rerank_logic(
            hits=hits,
            query="test",
            rerank_candidates=2,
        )

    verified_score = next((h["score"] for h in result if h["id"] == "verified1"), None)
    stale_score = next((h["score"] for h in result if h["id"] == "stale1"), None)

    # VERIFIED (1.0 decay) should be higher than STALE (0.7 decay)
    assert verified_score is not None
    assert stale_score is not None
    assert verified_score > stale_score, (
        f"VERIFIED {verified_score:.3f} should > STALE {stale_score:.3f}"
    )


def test_rerank_no_state_decay_when_disabled(tmp_cfg):
    """When state tracking is disabled, scores remain unchanged."""
    memory = Memory(tmp_cfg)

    now = int(time.time())
    verified_rec = MemoryRecord(
        id="verified1",
        path="test/verified.md",
        title="verified fact",
        type="fact",
        tags=[],
        created="2026-01-01T00:00:00",
        updated="2026-01-01T00:00:00",
        body="This is a verified fact",
        verification_state=VerificationState.VERIFIED,
        verified_at=now,
    )
    unverified_rec = MemoryRecord(
        id="unverified1",
        path="test/unverified.md",
        title="unverified fact",
        type="fact",
        tags=[],
        created="2026-01-01T00:00:00",
        updated="2026-01-01T00:00:00",
        body="This is an unverified fact",
        verification_state=VerificationState.UNVERIFIED,
        verified_at=None,
    )

    memory.memory_map = {"verified1": verified_rec, "unverified1": unverified_rec}

    hits = [
        {"id": "verified1", "score": 0.8},
        {"id": "unverified1", "score": 0.9},
    ]

    with patch("memo.memory.rerank_ops.flag_bool") as mock_flag_bool:

        def flag_bool_side_effect(flag_name):
            # State tracking disabled
            if flag_name == "MEMO_VERIFICATION_STATE_TRACKING":
                return False
            if flag_name == "MEMO_GRAPH_DISTANCE_DECAY":
                return False
            return False

        mock_flag_bool.side_effect = flag_bool_side_effect
        result = memory._rerank_logic(
            hits=hits,
            query="test",
            rerank_candidates=2,
        )

    # Without state decay, hits should be ordered by original score (0.9 > 0.8)
    # so unverified1 should be first
    assert result[0]["id"] == "unverified1", "Without state decay, should sort by original score"
    assert result[1]["id"] == "verified1"
