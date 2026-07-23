"""MCP tools for explicit review, verification, and truth validity."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from memo.memory import Memory
from memo.server_annotations import DESTRUCTIVE, READ_ONLY, WRITE_IDEMPOTENT, annotated_tool


def register(server: FastMCP, memory: Memory) -> None:
    @annotated_tool(server, **READ_ONLY)
    def memo_review_due(project: str | None = None, limit: int = 50) -> dict[str, Any]:
        """List records whose explicit review date passed or have an open conflict."""
        rows = memory.list_due_reviews(project=project, limit=max(1, min(limit, 200)))
        return {"due": rows, "count": len(rows)}

    @annotated_tool(server, **WRITE_IDEMPOTENT)
    def memo_mark_reviewed(
        id: str,
        evidence: str | None = None,
        actor: str | None = None,
    ) -> dict[str, Any]:
        """Record review evidence, verify the memory, and schedule its next review."""
        return memory.mark_reviewed(id, evidence=evidence, actor=actor).to_dict()

    @annotated_tool(server, **DESTRUCTIVE)
    def memo_invalidate(id: str, reason: str, at: str | None = None) -> dict[str, Any]:
        """Close one memory's truth-validity interval without deleting it."""
        return memory.invalidate(id, reason=reason, at=at).to_dict()

    @annotated_tool(server, **DESTRUCTIVE)
    def memo_supersede(old_id: str, new_id: str, reason: str) -> dict[str, Any]:
        """Close the old memory at the successor's validity start."""
        return memory.supersede(old_id, new_id, reason=reason).to_dict()
