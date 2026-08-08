"""Continuous State Compiler for memo Active World Model."""

from __future__ import annotations

import logging
from typing import Any

from memo.kernel.world_model import WorldModel

_logger = logging.getLogger(__name__)


class StateCompiler:
    """Aggregates Markdown memories, Git diffs, and session context into WorldModel."""

    def __init__(self, world_model: WorldModel) -> None:
        self.world_model = world_model

    def compile_from_memories(self, memories: list[dict[str, Any]]) -> int:
        """Compile a list of memory dicts into active beliefs in the WorldModel."""
        compiled_count = 0
        for mem in memories:
            mid = str(mem.get("id") or "")[:8]
            title = str(mem.get("title") or "").strip()
            body = str(mem.get("body") or "").strip()
            type_ = str(mem.get("type") or "note").strip()

            if not mid or not title:
                continue

            statement = f"{title}: {body[:200]}" if body else title
            confidence = float(mem.get("confidence", 1.0))

            self.world_model.upsert_belief(
                belief_id=mid,
                topic=type_,
                statement=statement,
                confidence=confidence,
                source=f"memory:{mid}",
            )
            compiled_count += 1

        return compiled_count

    def update_active_task(self, task_description: str) -> None:
        """Update active task state."""
        self.world_model.state.active_task = task_description.strip()
        self.world_model.save()

    def update_code_summary(self, code_summary: str) -> None:
        """Update code context summary state."""
        self.world_model.state.code_summary = code_summary.strip()
        self.world_model.save()
