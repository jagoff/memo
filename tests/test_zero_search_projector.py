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
