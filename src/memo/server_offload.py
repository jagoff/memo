"""MCP tools for context offloading — `memo_offload` (memo_get is the drill-down)."""

from __future__ import annotations

from typing import Any

from memo.memory import Memory
from memo.server_annotations import WRITE_IDEMPOTENT, annotated_tool


def register(server: Any, memory: Memory) -> None:
    @annotated_tool(server, **WRITE_IDEMPOTENT)
    def memo_offload(content: str, title: str | None = None) -> dict[str, Any]:
        """Offload a bulky payload (tool output, log, dump) out of the context
        window: memo stores it content-addressed as a reference-tier memory
        and returns `{id, sha256, kind, synopsis, deduplicated, drill_down}`.

        The synopsis is deterministic (no LLM): JSON keys, CSV headers, code
        symbols, or compressed text. Reference tier is excluded from
        auto-recall, so offloaded blobs never appear in the recall hook.
        Fetch the full payload later with `memo_get(id)`.
        """
        from memo.offload import offload

        return offload(memory, content, title=title)
