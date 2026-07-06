"""MCP surface: memo_around — timeline context around one memory."""

from __future__ import annotations

import logging
from typing import Any

from memo.memory import Memory
from memo.server_annotations import READ_ONLY, annotated_tool

_log = logging.getLogger(__name__)


def register(server: Any, memory: Memory) -> None:
    @annotated_tool(server, **READ_ONLY)
    def memo_around(id: str, before: int = 2, after: int = 2) -> dict[str, Any]:
        """Timeline context around one memory: seq-adjacent sibling chunks for
        reference-tier chunks (WhatsApp/vault docs), created-time neighbours
        for durable memories. Turns a search hit into narrative context.
        Read-only; window clamped to ±10."""
        before = max(0, min(int(before), 10))
        after = max(0, min(int(after), 10))
        return memory.around(id, before=before, after=after)
