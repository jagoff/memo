"""MCP tools — contextual-search domain (split from server.py).

Registered by `build_server()` via `register(server, memory)`. Tool names,
signatures, defaults, docstrings and bodies are identical to the originals;
only the enclosing function and indentation changed.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from fastmcp import FastMCP

from memo.memory import Memory
from memo.server_annotations import READ_ONLY, WRITE, annotated_tool


def register(server: FastMCP, memory: Memory) -> None:
    @annotated_tool(server, **READ_ONLY)
    def memo_contextual_search(
        query: str,
        limit: int = 10,
        mode: str = "hybrid",
    ) -> list[dict[str, Any]]:
        """Search with contextual re-ranking based on conversation history.

        Uses conversation context and learned user preferences to re-rank
        search results. Boosts memories that overlap with recent context
        and aligns with user preferences.

        Args:
            query: Search query.
            limit: Max results.
            mode: Search mode (vec, bm25, hybrid).
        """
        results = memory.contextual.search_with_context(
            query=query,
            limit=limit,
            mode=mode,
        )
        return [dataclasses.asdict(r) for r in results]

    @annotated_tool(server, **WRITE)
    def memo_contextual_record_search(
        query: str,
        memory_ids: list[str],
    ) -> dict[str, Any]:
        """Record a search in the conversation history for learning.

        Use this after each search to build context for future searches.
        The system learns from which memories are recalled to improve
        future contextual ranking.

        Args:
            query: The search query that was used.
            memory_ids: List of memory IDs that were recalled.
        """
        memory.contextual.record_search(query, memory_ids)
        return {"status": "recorded", "count": len(memory_ids)}

    @annotated_tool(server, **WRITE)
    def memo_contextual_record_click(
        memory_id: str,
    ) -> dict[str, Any]:
        """Record that the user clicked/viewed a memory (for preference learning).

        Use this when the user explicitly selects a memory from search results.
        This teaches the system which memory types and entities the user prefers.

        Args:
            memory_id: The memory ID that was clicked/viewed.
        """
        memory.contextual.record_click(memory_id)
        return {"status": "recorded", "memory_id": memory_id}

    @annotated_tool(server, **READ_ONLY)
    def memo_contextual_preferences() -> dict[str, Any]:
        """Show learned user preferences for memory recall.

        Returns the current preference scores for memory types, entities,
        and recency/diversity weights. Useful for understanding what the
        system has learned about the user's preferences.
        """
        prefs = memory.contextual.context.get_preferences()
        return dataclasses.asdict(prefs)

    @annotated_tool(server, **READ_ONLY)
    def memo_contextual_history(
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Show recent conversation history used for contextual recall.

        Returns the N most recent prompts with their recalled memories.
        This history is used to build context for search re-ranking.

        Args:
            limit: Number of recent prompts to return.
        """
        history = memory.contextual.context.get_recent_context(n=limit)
        return [dataclasses.asdict(c) for c in history]
