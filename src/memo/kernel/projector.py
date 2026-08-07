"""Zero-Search Context Projector for memo Active World Model."""

from __future__ import annotations

import logging

from memo.kernel.world_model import WorldModel

_logger = logging.getLogger(__name__)


class ZeroSearchProjector:
    """Projects WorldModel active state into hyper-dense cognitive context (<5ms)."""

    def __init__(self, world_model: WorldModel) -> None:
        self.world_model = world_model

    def project_context(self, prompt: str = "", max_beliefs: int = 5) -> str:
        """Project active state into hyper-dense XML/Markdown context."""
        state = self.world_model.state
        active_beliefs = self.world_model.get_active_beliefs()[:max_beliefs]

        lines = [
            "<memo-world-kernel>",
            f"## Active Project State: {state.project_name}",
        ]

        if state.active_task:
            lines.append(f"**Task**: {state.active_task}")

        if state.code_summary:
            lines.append(f"**Code Context**: {state.code_summary}")

        if active_beliefs:
            lines.append("### Key Beliefs & Decisions:")
            for b in active_beliefs:
                lines.append(f"- [{b.id}] ({b.topic}) {b.statement}")

        lines.append("</memo-world-kernel>")
        return "\n".join(lines)
