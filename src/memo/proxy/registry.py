"""Constructs the transform list. Separate from plan.py so transforms may import
the protocol without a cycle back through the planner."""

from __future__ import annotations

from memo.proxy.plan import Transform
from memo.proxy.transforms.toolschemas import ToolSchemas


def build_registry() -> list[Transform]:
    """Every enabled transform, in application order. Populated by later tasks."""
    return [ToolSchemas()]
