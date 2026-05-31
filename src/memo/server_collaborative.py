"""MCP tools — collaborative domain (split from server.py).

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
    def memory_collaborative_share_connection(
        user_id: str,
        entity_a: str,
        entity_b: str,
        relationship: str,
        confidence: float = 0.7,
    ) -> dict[str, str]:
        """Share a discovered connection with the community.

        Args:
            user_id: User ID who discovered the connection.
            entity_a: First entity.
            entity_b: Second entity.
            relationship: Type of relationship.
            confidence: Confidence score.

        Returns:
            Shared connection data.
        """
        conn = memory.collaborative.share_connection(
            user_id=user_id,
            entity_a=entity_a,
            entity_b=entity_b,
            relationship=relationship,
            confidence=confidence,
        )
        return conn.__dict__

    @server.tool()
    def memory_collaborative_connections(
        entity: str,
    ) -> list[dict[str, Any]]:
        """Get shared connections for an entity.

        Args:
            entity: Entity of interest.

        Returns:
            List of SharedConnection objects.
        """
        connections = memory.collaborative.get_shared_connections(entity)
        return [c.__dict__ for c in connections]

    @server.tool()
    def memory_collaborative_recommend(
        entity: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get recommended connections based on collective patterns.

        Args:
            entity: Entity of interest.
            limit: Max results.

        Returns:
            List of recommended SharedConnection objects.
        """
        recommendations = memory.collaborative.get_recommended_connections(entity, limit=limit)
        return [r.__dict__ for r in recommendations]

    @server.tool()
    def memory_collaborative_share_insight(
        user_id: str,
        content: str,
    ) -> dict[str, str]:
        """Share an insight with the community.

        Args:
            user_id: User ID.
            content: Insight content.

        Returns:
            Shared insight data.
        """
        insight = memory.collaborative.share_insight(user_id, content)
        return insight.__dict__

    @server.tool()
    def memory_collaborative_insights(
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get top voted insights from the community.

        Args:
            limit: Max results.

        Returns:
            List of CollectiveInsight objects.
        """
        insights = memory.collaborative.get_top_insights(limit=limit)
        return [i.__dict__ for i in insights]
