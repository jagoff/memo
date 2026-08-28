from __future__ import annotations

import tempfile
from pathlib import Path

from memo.kernel.belief_network import BeliefNetwork
from memo.kernel.projector import ZeroSearchProjector
from memo.kernel.world_model import WorldModel


def test_belief_network_auto_invalidation() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        wm = WorldModel(Path(tmpdir), project_name="test")
        bn = BeliefNetwork(wm)

        wm.upsert_belief("b1", "database", "We use MongoDB database for storage", confidence=1.0)
        assert len(wm.get_active_beliefs()) == 1

        # Statement indicating replacement/deprecated
        invalidated = bn.auto_invalidate_conflicts(
            "We use PostgreSQL instead of MongoDB database", "database"
        )

        assert "b1" in invalidated
        assert len(wm.get_active_beliefs()) == 0


def test_zero_search_projector() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        wm = WorldModel(Path(tmpdir), project_name="memo_proj")
        wm.state.active_task = "Refactor storage layer"
        wm.upsert_belief("b1", "arch", "Use SQLite WAL mode", confidence=0.95)

        projector = ZeroSearchProjector(wm)
        ctx = projector.project_context()

        assert "<memo-world-kernel>" in ctx
        assert "memo_proj" in ctx
        assert "Refactor storage layer" in ctx
        assert "Use SQLite WAL mode" in ctx


def test_projector_emits_nothing_when_it_has_nothing_to_project() -> None:
    """An empty world model must project an empty string, not naked scaffolding.

    Regression: with no active task, code summary, or beliefs, project_context
    still returned the `<memo-world-kernel>` wrapper plus a bare project-name
    header — 74 chars of pure structure prepended to every balanced/context
    recall injection, carrying zero information.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        wm = WorldModel(Path(tmpdir), project_name="default")

        assert ZeroSearchProjector(wm).project_context() == ""


def test_projector_still_emits_when_it_has_content() -> None:
    """The empty-projection guard must not suppress a projection with content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        wm = WorldModel(Path(tmpdir), project_name="memo")
        wm.state.active_task = "ship the briefing budget fix"

        out = ZeroSearchProjector(wm).project_context()

        assert "<memo-world-kernel>" in out
        assert "ship the briefing budget fix" in out
