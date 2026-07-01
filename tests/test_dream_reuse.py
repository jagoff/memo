"""Unit tests for dream_reuse.consolidated_reuse (F4 metric).

MLX-free: uses a fake memory object and writes grounding.log directly.
Isolated under tmp_path — never touches the developer's real store.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from memo import dream_reuse
from memo.dashboard_logs import (
    append_grounding_log,
)

# ---------------------------------------------------------------------------
# Minimal fakes
# ---------------------------------------------------------------------------


class _FakeStore:
    """Stub that returns a fixed set of synthesis records."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records

    def list_recent(
        self,
        limit: int = 1000,
        type_: str | None = None,
    ) -> list[dict[str, Any]]:
        if type_ == "synthesis" or type_ is None:
            return self._records[:limit]
        return []


class _FakeCfg:
    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir


class _FakeMem:
    def __init__(self, state_dir: Path, records: list[dict[str, Any]]) -> None:
        self.store = _FakeStore(records)
        self.cfg = _FakeCfg(state_dir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SYNTHESIS_RECORDS = [
    # id "aabbccdd" — cross_session, will be grounded
    {"id": "aabbccdd1111", "extra": {"synthesis_kind": "cross_session"}},
    # id "eeff0011" — cross_session, will be grounded
    {"id": "eeff001122aa", "extra": {"synthesis_kind": "cross_session"}},
    # id "44556677" — plain synthesis, NOT grounded
    {"id": "44556677889a", "extra": {}},
]


def _seed_grounding(state_dir: Path, *, recall_id: str, used_score: float) -> None:
    """Write one entry to grounding.log that is considered 'used' when score ≥ 0.8."""
    append_grounding_log(
        state_dir,
        session_id="test-session",
        turn=1,
        recall_id=recall_id,
        used_score=used_score,
        method="cosine",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_consolidated_reuse_basic(tmp_path: Path) -> None:
    """2 of 3 synthesis ids grounded → reuse_fraction = 2/3."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    mem = _FakeMem(state_dir, _SYNTHESIS_RECORDS)

    # Ground the first two synthesis ids (used_score=0.9 > USED_SCORE_STRONG 0.8).
    _seed_grounding(state_dir, recall_id="aabbccdd", used_score=0.9)
    _seed_grounding(state_dir, recall_id="eeff0011", used_score=0.9)
    # Third id ("44556677") is not in grounding.log at all.

    result = dream_reuse.consolidated_reuse(mem)

    assert result["n_consolidated"] == 3
    assert result["n_reused"] == 2
    # reuse_fraction is rounded to 4 decimal places → 0.6667
    assert abs(result["reuse_fraction"] - 2 / 3) < 1e-4
    # Both reused ids are cross_session
    assert result["cross_session"] == 2


def test_no_synthesis_memories(tmp_path: Path) -> None:
    """Empty store → zeros, no errors."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    mem = _FakeMem(state_dir, [])
    result = dream_reuse.consolidated_reuse(mem)

    assert result == {
        "n_consolidated": 0,
        "n_reused": 0,
        "reuse_fraction": 0.0,
        "cross_session": 0,
    }


def test_none_reused(tmp_path: Path) -> None:
    """Synthesis memories exist but none appear in grounding.log."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    mem = _FakeMem(state_dir, _SYNTHESIS_RECORDS)
    # No grounding entries at all.
    result = dream_reuse.consolidated_reuse(mem)

    assert result["n_consolidated"] == 3
    assert result["n_reused"] == 0
    assert result["reuse_fraction"] == 0.0
    assert result["cross_session"] == 0


def test_low_score_not_counted_as_reused(tmp_path: Path) -> None:
    """A grounding row with used_score below USED_SCORE_STRONG (0.8) and no
    specific_score or downstream_action is NOT counted as reused."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    mem = _FakeMem(state_dir, _SYNTHESIS_RECORDS)
    # Seed with a score that does NOT satisfy grounding_used (< 0.8, no margin).
    _seed_grounding(state_dir, recall_id="aabbccdd", used_score=0.5)

    result = dream_reuse.consolidated_reuse(mem)

    assert result["n_reused"] == 0
    assert result["reuse_fraction"] == 0.0


def test_cross_session_count_only_reused(tmp_path: Path) -> None:
    """cross_session counts only the reused cross-session memories, not all of them."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    # Two cross_session memories; only the second is grounded.
    records = [
        {"id": "aabbccdd1111", "extra": {"synthesis_kind": "cross_session"}},
        {"id": "eeff001122aa", "extra": {"synthesis_kind": "cross_session"}},
    ]
    mem = _FakeMem(state_dir, records)
    _seed_grounding(state_dir, recall_id="eeff0011", used_score=0.9)

    result = dream_reuse.consolidated_reuse(mem)

    assert result["n_reused"] == 1
    assert result["cross_session"] == 1
