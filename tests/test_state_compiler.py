from __future__ import annotations

import tempfile
from pathlib import Path

from memo.kernel.state_compiler import StateCompiler
from memo.kernel.world_model import WorldModel


def test_state_compiler() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        state_dir = Path(tmpdir)
        wm = WorldModel(state_dir, project_name="test_proj")
        compiler = StateCompiler(wm)

        memories = [
            {
                "id": "mem_1001",
                "title": "Use Redis for caching",
                "body": "Fast in-memory cache",
                "type": "architecture",
            },
            {
                "id": "mem_1002",
                "title": "Postgres main DB",
                "body": "ACID compliant",
                "type": "config",
            },
        ]

        count = compiler.compile_from_memories(memories)
        assert count == 2

        active = wm.get_active_beliefs()
        assert len(active) == 2

        compiler.update_active_task("Implement Redis caching layer")
        assert wm.state.active_task == "Implement Redis caching layer"
