"""MCP tools — feedback domain (split from server.py).

Registered by `build_server()` via `register(server, memory)`. Tool names,
signatures, defaults, docstrings and bodies are identical to the originals;
only the enclosing function and indentation changed.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from memo.memory import Memory
from memo.server_annotations import DESTRUCTIVE, READ_ONLY, WRITE, annotated_tool


def register(server: FastMCP, memory: Memory) -> None:
    @annotated_tool(server, **WRITE)
    def memo_feedback_record(
        source_id: str,
        query: str,
        rating: str,
    ) -> dict[str, Any]:
        """Record a feedback signal on a memory for a given query text.

        Supported signal values for `rating`:
          "thumbs_up" / "up"  — explicit positive vote; boosts score ~0.15 per vote.
          "click"              — implicit positive (user used/viewed this result); ~0.08 boost.
          "thumbs_down" / "down" — explicit rejection; hard-excludes from future similar queries.
          "ignore"             — implicit negative (user skipped); soft 0.7× score penalty.

        Idempotent on (source_id, query, rating). Recording a different signal
        for the same (source, query) pair replaces the prior vote.

        Args:
            source_id: meta.id (full or unique prefix >= 4 chars).
            query: query text the feedback applies to.
            rating: "thumbs_up", "click", "thumbs_down", "ignore", "up", or "down".
        """
        return memory.feedback_record(source_id, query_text=query, rating=rating)

    @annotated_tool(server, **READ_ONLY)
    def memo_feedback_list(
        source_id: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """List recorded source-level feedback, newest first.

        Args:
            source_id: optional filter (full id or unique prefix).
            limit: max rows.
        """
        rows = memory.feedback_list(source_id=source_id, limit=limit)
        return {"rows": rows, "count": len(rows)}

    @annotated_tool(server, **DESTRUCTIVE)
    def memo_feedback_clear(source_id: str) -> dict[str, Any]:
        """Drop all feedback rows for `source_id`. Returns count deleted."""
        n = memory.feedback_clear(source_id)
        return {"source_id": source_id, "deleted": n}
