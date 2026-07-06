"""MCP tools — temporal-reasoning domain (split from server.py).

Registered by `build_server()` via `register(server, memory)`. Tool names,
signatures, defaults, docstrings and bodies are identical to the originals;
only the enclosing function and indentation changed.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from memo.memory import Memory
from memo.server_annotations import READ_ONLY, annotated_tool


def register(server: FastMCP, memory: Memory) -> None:
    @annotated_tool(server, **READ_ONLY)
    def memo_temporal_contradictions(
        entity: str,
        entity_type: str | None = None,
        confidence_threshold: float = 0.7,
        max_pairs: int = 20,
    ) -> list[dict[str, Any]]:
        """Detect contradictions among memories mentioning a specific entity.

        Uses the helper LLM to classify pairs of memories as contradiction,
        evolution, consistent, or unrelated. Returns only contradictions and
        evolutions above the confidence threshold.

        Args:
            entity: The entity name to analyze (e.g. "ollama", "mlx").
            entity_type: Optional entity type filter from graph.
            confidence_threshold: Minimum confidence (0-1). Default 0.7.
            max_pairs: Maximum number of pairs to analyze (LLM is expensive).
        """
        contradictions = memory.temporal.detect_entity_contradictions(
            entity_name=entity,
            entity_type=entity_type,
            confidence_threshold=confidence_threshold,
            max_pairs=max_pairs,
        )
        return [c.__dict__ for c in contradictions]

    @annotated_tool(server, **READ_ONLY)
    def memo_temporal_timeline(
        entity: str,
        entity_type: str | None = None,
    ) -> dict[str, Any] | None:
        """Build a chronological timeline of all memories mentioning an entity.

        Returns a timeline with events ordered by date, including first/last
        seen timestamps. Useful for tracking evolution of decisions or
        opinions over time.

        Args:
            entity: The entity name to analyze.
            entity_type: Optional entity type filter from graph.
        """
        timeline = memory.temporal.build_entity_timeline(
            entity_name=entity,
            entity_type=entity_type,
        )
        if timeline is None:
            return None
        return {
            "entity_name": timeline.entity_name,
            "entity_type": timeline.entity_type,
            "first_seen": timeline.first_seen,
            "last_seen": timeline.last_seen,
            "events": [e.__dict__ for e in timeline.events],
        }

    @annotated_tool(server, **READ_ONLY)
    def memo_temporal_stale(
        days_threshold: int = 180,
        min_access_count: int = 0,
    ) -> list[dict[str, Any]]:
        """Find memories that may be stale based on age and lack of access.

        Returns potentially stale memories with metadata including days since
        update and access count. Useful for identifying outdated information
        that may need review.

        Args:
            days_threshold: Days since last update to consider stale.
            min_access_count: Minimum access count to exclude (frequently-accessed
                old memories may still be relevant).
        """
        return memory.temporal.detect_stale_memories(
            days_threshold=days_threshold,
            min_access_count=min_access_count,
        )

    @annotated_tool(server, **READ_ONLY)
    def memo_temporal_patterns() -> dict[str, Any]:
        """Analyze high-level temporal patterns across the entire corpus.

        Returns metrics including:
        - memories_per_month: histogram of creation activity
        - type_distribution_over_time: how memory types change over time
        - most_active_entities: entities with most temporal churn
        """
        return memory.temporal.detect_temporal_patterns()
