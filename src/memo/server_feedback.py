"""MCP tools — feedback domain (split from server.py).

Registered by `build_server()` via `register(server, memory)`. Tool names,
signatures, defaults, docstrings and bodies are identical to the originals;
only the enclosing function and indentation changed.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from memo.memory import Memory

def register(server: FastMCP, memory: Memory) -> None:
    @server.tool()
    def memory_feedback_record(
        source_id: str,
        query: str,
        rating: str,
    ) -> dict[str, Any]:
        """Record a 👍 / 👎 vote on a memoria for a given query text.

        Negative votes (rating="down") exclude the source for any future
        query whose embedding cosine-similarity with `query` is >= 0.85.
        Positive votes boost score by ~0.15 per vote (capped). Idempotent
        on (source_id, query, rating). Recording the opposite rating for
        the same (source, query) pair replaces the prior vote.

        Args:
            source_id: meta.id (full or unique prefix >= 4 chars).
            query: query text the feedback applies to. Embedded with the
                asymmetric retrieval prefix so future queries compare
                fairly.
            rating: "up" or "down".
        """
        return memory.feedback_record(source_id, query_text=query, rating=rating)

    @server.tool()
    def memory_feedback_list(
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

    @server.tool()
    def memory_feedback_clear(source_id: str) -> dict[str, Any]:
        """Drop all feedback rows for `source_id`. Returns count deleted."""
        n = memory.feedback_clear(source_id)
        return {"source_id": source_id, "deleted": n}
