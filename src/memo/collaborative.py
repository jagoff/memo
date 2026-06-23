"""EXPERIMENTAL — not covered by the test suite, not exposed via MCP. API may change without notice.

Collaborative Social Memory with a Shared Knowledge Graph.

A neural network of shared memories across multiple users that learns
from everyone's connections.

## Gamechanger

- Knowledge grows collectively, not just individually
- Connections other users discover benefit you
- A shared knowledge graph that evolves
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

_log = logging.getLogger(__name__)


@dataclass
class SharedConnection:
    """A connection discovered by another user."""

    connection_id: str
    from_user: str
    entity_a: str
    entity_b: str
    relationship: str
    confidence: float
    discovered_at: str
    votes: int


@dataclass
class CollectiveInsight:
    """A collectively generated insight."""

    insight_id: str
    content: str
    contributors: list[str]
    upvotes: int
    downvotes: int
    created_at: str


@dataclass
class UserProfile:
    """User profile in the collaborative network."""

    user_id: str
    username: str
    reputation: int
    contributions_count: int
    joined_at: str


class CollaborativeGraph:
    """Shared knowledge graph across users.

    Args:
        state_dir: Directory to store the graph.
    """

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir
        self.graph_file = state_dir / "collaborative_graph.json"
        self._connections: dict[str, SharedConnection] = {}
        self._insights: dict[str, CollectiveInsight] = {}
        self._users: dict[str, UserProfile] = {}
        self._load()

    def _load(self) -> None:
        """Load the graph from disk."""
        if self.graph_file.is_file():
            try:
                data = json.loads(self.graph_file.read_text(encoding="utf-8"))

                # Load connections
                for cid, cdata in data.get("connections", {}).items():
                    self._connections[cid] = SharedConnection(**cdata)

                # Load insights
                for iid, idata in data.get("insights", {}).items():
                    self._insights[iid] = CollectiveInsight(**idata)

                # Load users
                for uid, udata in data.get("users", {}).items():
                    self._users[uid] = UserProfile(**udata)
            except Exception:
                self._connections = {}
                self._insights = {}
                self._users = {}

    def _save(self) -> None:
        """Save the graph to disk."""
        try:
            data = {
                "connections": {cid: c.__dict__ for cid, c in self._connections.items()},
                "insights": {iid: i.__dict__ for iid, i in self._insights.items()},
                "users": {uid: u.__dict__ for uid, u in self._users.items()},
            }
            self.graph_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except (OSError, TypeError, ValueError) as exc:
            _log.error("collaborative: failed to persist graph: %s", exc)

    def add_connection(
        self,
        from_user: str,
        entity_a: str,
        entity_b: str,
        relationship: str,
        confidence: float = 0.7,
    ) -> SharedConnection:
        """Add a connection discovered by a user.

        Args:
            from_user: User who discovered the connection.
            entity_a: First entity.
            entity_b: Second entity.
            relationship: Type of relationship.
            confidence: Confidence in the connection.

        Returns:
            The added SharedConnection.
        """
        import uuid

        cid = str(uuid.uuid4())
        conn = SharedConnection(
            connection_id=cid,
            from_user=from_user,
            entity_a=entity_a,
            entity_b=entity_b,
            relationship=relationship,
            confidence=confidence,
            discovered_at=datetime.now(UTC).isoformat(),
            votes=0,
        )
        self._connections[cid] = conn
        self._save()

        # Update user reputation
        if from_user not in self._users:
            self._users[from_user] = UserProfile(
                user_id=from_user,
                username=from_user,
                reputation=10,
                contributions_count=1,
                joined_at=datetime.now(UTC).isoformat(),
            )
        else:
            self._users[from_user].contributions_count += 1
            self._users[from_user].reputation += 5

        self._save()
        return conn

    def vote_connection(self, connection_id: str, upvote: bool = True) -> bool:
        """Vote on a connection (upvote/downvote).

        Args:
            connection_id: ID of the connection.
            upvote: True to upvote, False to downvote.

        Returns:
            True if successful.
        """
        if connection_id not in self._connections:
            return False

        conn = self._connections[connection_id]
        if upvote:
            conn.votes += 1
            # Reward the discoverer
            if conn.from_user in self._users:
                self._users[conn.from_user].reputation += 2
        else:
            conn.votes -= 1

        self._save()
        return True

    def get_connections_for_entity(self, entity: str) -> list[SharedConnection]:
        """Get all connections for an entity.

        Args:
            entity: Name of the entity.

        Returns:
            List of SharedConnection.
        """
        return [
            c for c in self._connections.values() if c.entity_a == entity or c.entity_b == entity
        ]

    def add_insight(
        self,
        user_id: str,
        content: str,
    ) -> CollectiveInsight:
        """Add a collective insight.

        Args:
            user_id: ID of the user.
            content: Content of the insight.

        Returns:
            The added CollectiveInsight.
        """
        import uuid

        iid = str(uuid.uuid4())
        insight = CollectiveInsight(
            insight_id=iid,
            content=content,
            contributors=[user_id],
            upvotes=0,
            downvotes=0,
            created_at=datetime.now(UTC).isoformat(),
        )
        self._insights[iid] = insight
        self._save()
        return insight

    def vote_insight(self, insight_id: str, upvote: bool = True) -> bool:
        """Vote on an insight.

        Args:
            insight_id: ID of the insight.
            upvote: True to upvote, False to downvote.

        Returns:
            True if successful.
        """
        if insight_id not in self._insights:
            return False

        insight = self._insights[insight_id]
        if upvote:
            insight.upvotes += 1
        else:
            insight.downvotes += 1

        self._save()
        return True

    def get_top_insights(self, limit: int = 10) -> list[CollectiveInsight]:
        """Get the most-voted insights.

        Args:
            limit: Maximum number of results.

        Returns:
            List of CollectiveInsight ordered by upvotes.
        """
        sorted_insights = sorted(
            self._insights.values(),
            key=lambda i: i.upvotes - i.downvotes,
            reverse=True,
        )
        return sorted_insights[:limit]

    def get_user_profile(self, user_id: str) -> UserProfile | None:
        """Get a user's profile.

        Args:
            user_id: ID of the user.

        Returns:
            UserProfile or None.
        """
        return self._users.get(user_id)


class CollaborativeFilter:
    """Recommendations based on collective patterns.

    Args:
        graph: CollaborativeGraph.
    """

    def __init__(self, graph: CollaborativeGraph) -> None:
        self.graph = graph

    def recommend_connections(
        self,
        entity: str,
        limit: int = 10,
    ) -> list[SharedConnection]:
        """Recommend connections based on collective patterns.

        Args:
            entity: Entity of interest.
            limit: Maximum number of results.

        Returns:
            List of recommended SharedConnection.
        """
        # Get existing connections for the entity
        existing = self.graph.get_connections_for_entity(entity)

        # Look for similar connections from other users
        recommendations = []
        for conn in self.graph._connections.values():
            if conn.entity_a == entity or conn.entity_b == entity:
                continue

            # If the connection shares an entity with existing ones, recommend it
            for existing_conn in existing:
                if (
                    conn.entity_a == existing_conn.entity_b
                    or conn.entity_b == existing_conn.entity_a
                ):
                    recommendations.append(conn)

        # Sort by votes and confidence
        recommendations.sort(
            key=lambda c: c.votes + c.confidence * 10,
            reverse=True,
        )

        return recommendations[:limit]

    def recommend_insights(
        self,
        user_interests: list[str],
        limit: int = 10,
    ) -> list[CollectiveInsight]:
        """Recommend insights based on the user's interests.

        Args:
            user_interests: List of the user's interests.
            limit: Maximum number of results.

        Returns:
            List of recommended CollectiveInsight.
        """
        # Find insights that mention the interests
        recommendations = []
        for insight in self.graph._insights.values():
            for interest in user_interests:
                if interest.lower() in insight.content.lower():
                    recommendations.append(insight)
                    break

        # Sort by upvotes
        recommendations.sort(key=lambda i: i.upvotes - i.downvotes, reverse=True)

        return recommendations[:limit]


class CollaborativeManager:
    """Manages collaborative functionality.

    Args:
        graph: CollaborativeGraph.
        filter: CollaborativeFilter.
    """

    def __init__(
        self,
        graph: CollaborativeGraph,
        filter: CollaborativeFilter,
    ) -> None:
        self.graph = graph
        self.filter = filter

    def share_connection(
        self,
        user_id: str,
        entity_a: str,
        entity_b: str,
        relationship: str,
        confidence: float = 0.7,
    ) -> SharedConnection:
        """Share a discovered connection with the community.

        Args:
            user_id: ID of the user.
            entity_a: First entity.
            entity_b: Second entity.
            relationship: Type of relationship.
            confidence: Confidence.

        Returns:
            The shared SharedConnection.
        """
        return self.graph.add_connection(
            from_user=user_id,
            entity_a=entity_a,
            entity_b=entity_b,
            relationship=relationship,
            confidence=confidence,
        )

    def get_shared_connections(self, entity: str) -> list[SharedConnection]:
        """Get shared connections for an entity.

        Args:
            entity: Entity of interest.

        Returns:
            List of SharedConnection.
        """
        return self.graph.get_connections_for_entity(entity)

    def get_recommended_connections(self, entity: str, limit: int = 10) -> list[SharedConnection]:
        """Get recommended connections.

        Args:
            entity: Entity of interest.
            limit: Maximum number of results.

        Returns:
            List of recommended SharedConnection.
        """
        return self.filter.recommend_connections(entity, limit)

    def share_insight(self, user_id: str, content: str) -> CollectiveInsight:
        """Share an insight with the community.

        Args:
            user_id: ID of the user.
            content: Content of the insight.

        Returns:
            The shared CollectiveInsight.
        """
        return self.graph.add_insight(user_id, content)

    def get_top_insights(self, limit: int = 10) -> list[CollectiveInsight]:
        """Get the most-voted insights.

        Args:
            limit: Maximum number of results.

        Returns:
            List of CollectiveInsight.
        """
        return self.graph.get_top_insights(limit)


__all__ = [
    "CollaborativeFilter",
    "CollaborativeGraph",
    "CollaborativeManager",
    "CollectiveInsight",
    "SharedConnection",
    "UserProfile",
]
