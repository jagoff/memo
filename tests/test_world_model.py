from __future__ import annotations

import tempfile
from pathlib import Path

from memo.kernel.world_model import WorldModel


def test_world_model_crud() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        state_dir = Path(tmpdir)
        wm = WorldModel(state_dir, project_name="test_proj")

        # Upsert belief
        item = wm.upsert_belief(
            belief_id="b1",
            topic="database",
            statement="We use PostgreSQL for storage",
            confidence=0.95,
            source="memory:b1",
        )
        assert item.id == "b1"
        assert item.status == "active"

        # Query active beliefs
        active = wm.get_active_beliefs()
        assert len(active) == 1
        assert active[0].statement == "We use PostgreSQL for storage"

        # Invalidate belief
        assert wm.invalidate_belief("b1", reason="test")
        assert len(wm.get_active_beliefs()) == 0

        # Reload from disk
        wm2 = WorldModel(state_dir, project_name="test_proj")
        assert len(wm2.get_active_beliefs()) == 0
        assert "b1" in wm2.state.beliefs
        assert wm2.state.beliefs["b1"].status == "invalidated"
