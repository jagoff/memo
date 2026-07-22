"""Tests for collaborative module."""

import pytest

from memo.collaborative import (
    CollaborativeFilter,
    CollaborativeGraph,
    CollaborativeManager,
    CollectiveInsight,
    SharedConnection,
    UserProfile,
)


@pytest.fixture
def collaborative_graph(tmp_cfg):
    """Fixture providing CollaborativeGraph instance."""
    return CollaborativeGraph(tmp_cfg.state_dir)


@pytest.fixture
def collaborative_filter(collaborative_graph):
    """Fixture providing CollaborativeFilter instance."""
    return CollaborativeFilter(collaborative_graph)


@pytest.fixture
def collaborative_manager(collaborative_graph, collaborative_filter):
    """Fixture providing CollaborativeManager instance."""
    return CollaborativeManager(collaborative_graph, collaborative_filter)


def test_collaborative_graph_init(collaborative_graph):
    """Test CollaborativeGraph initialization."""
    assert collaborative_graph.state_dir.is_dir()


def test_collaborative_graph_add_connection(collaborative_graph):
    """Test adding a connection."""
    conn = collaborative_graph.add_connection(
        from_user="user1",
        entity_a="MLX",
        entity_b="Apple",
        relationship="optimized for",
        confidence=0.8,
    )

    assert conn.connection_id
    assert conn.entity_a == "MLX"
    assert conn.entity_b == "Apple"
    assert conn.votes == 0


def test_collaborative_graph_vote_connection(collaborative_graph):
    """Test voting on a connection."""
    conn = collaborative_graph.add_connection(
        from_user="user1",
        entity_a="MLX",
        entity_b="Apple",
        relationship="optimized for",
    )

    success = collaborative_graph.vote_connection(conn.connection_id, upvote=True)

    assert success is True
    assert collaborative_graph.get_connections_for_entity("MLX")[0].votes == 1


def test_collaborative_graph_get_connections_for_entity(collaborative_graph):
    """Test getting connections for an entity."""
    collaborative_graph.add_connection(
        from_user="user1",
        entity_a="MLX",
        entity_b="Apple",
        relationship="optimized for",
    )

    connections = collaborative_graph.get_connections_for_entity("MLX")

    assert len(connections) >= 1
    assert connections[0].entity_a == "MLX"


def test_collaborative_graph_add_insight(collaborative_graph):
    """Test adding an insight."""
    insight = collaborative_graph.add_insight(
        user_id="user1",
        content="MLX is ideal for edge computing",
    )

    assert insight.insight_id
    assert insight.content == "MLX is ideal for edge computing"
    assert insight.upvotes == 0


def test_collaborative_graph_vote_insight(collaborative_graph):
    """Test voting on an insight."""
    insight = collaborative_graph.add_insight(
        user_id="user1",
        content="Test insight",
    )

    success = collaborative_graph.vote_insight(insight.insight_id, upvote=True)

    assert success is True


def test_collaborative_graph_get_top_insights(collaborative_graph):
    """Test getting top insights."""
    collaborative_graph.add_insight("user1", "Insight 1")
    collaborative_graph.add_insight("user2", "Insight 2")

    # Upvote the first insight
    insights = collaborative_graph.get_top_insights(limit=10)

    assert isinstance(insights, list)


def test_collaborative_graph_get_user_profile(collaborative_graph):
    """Test getting user profile."""
    collaborative_graph.add_connection(
        from_user="user1",
        entity_a="MLX",
        entity_b="Apple",
        relationship="optimized for",
    )

    profile = collaborative_graph.get_user_profile("user1")

    assert profile is not None
    assert profile.user_id == "user1"
    assert profile.reputation >= 10


def test_collaborative_graph_persistence(tmp_cfg):
    """Test graph persistence across instances."""
    state_dir = tmp_cfg.state_dir

    # Create first instance and add connection
    graph1 = CollaborativeGraph(state_dir)
    graph1.add_connection(
        from_user="user1",
        entity_a="MLX",
        entity_b="Apple",
        relationship="optimized for",
    )

    # Create second instance and verify persistence
    graph2 = CollaborativeGraph(state_dir)
    connections = graph2.get_connections_for_entity("MLX")

    assert len(connections) >= 1
    assert connections[0].entity_a == "MLX"


def test_collaborative_filter_init(collaborative_filter):
    """Test CollaborativeFilter initialization."""
    assert collaborative_filter.graph is not None


def test_collaborative_filter_recommend_connections(collaborative_graph, collaborative_filter):
    """Test recommending connections."""
    # Add some connections
    collaborative_graph.add_connection("user1", "MLX", "Apple", "optimized for")
    collaborative_graph.add_connection("user2", "Apple", "Swift", "written in")

    recommendations = collaborative_filter.recommend_connections("MLX")

    assert isinstance(recommendations, list)


def test_collaborative_manager_init(collaborative_manager):
    """Test CollaborativeManager initialization."""
    assert collaborative_manager.graph is not None
    assert collaborative_manager.filter is not None


def test_collaborative_manager_share_connection(collaborative_manager):
    """Test sharing a connection."""
    conn = collaborative_manager.share_connection(
        user_id="user1",
        entity_a="MLX",
        entity_b="Apple",
        relationship="optimized for",
    )

    assert conn.connection_id
    assert conn.from_user == "user1"


def test_collaborative_manager_get_shared_connections(collaborative_manager):
    """Test getting shared connections."""
    collaborative_manager.share_connection(
        user_id="user1",
        entity_a="MLX",
        entity_b="Apple",
        relationship="optimized for",
    )

    connections = collaborative_manager.get_shared_connections("MLX")

    assert len(connections) >= 1


def test_collaborative_manager_share_insight(collaborative_manager):
    """Test sharing an insight."""
    insight = collaborative_manager.share_insight(
        user_id="user1",
        content="Test insight",
    )

    assert insight.insight_id
    assert insight.content == "Test insight"


def test_collaborative_manager_get_top_insights(collaborative_manager):
    """Test getting top insights."""
    collaborative_manager.share_insight("user1", "Insight 1")
    collaborative_manager.share_insight("user2", "Insight 2")

    insights = collaborative_manager.get_top_insights(limit=10)

    assert isinstance(insights, list)


def test_collaborative_manager_vote_connection(collaborative_manager):
    """Votes cast through the Manager actually move the connection counter."""
    conn = collaborative_manager.share_connection(
        user_id="user1",
        entity_a="MLX",
        entity_b="Apple",
        relationship="optimized for",
    )

    assert collaborative_manager.vote_connection(conn.connection_id) is True
    assert collaborative_manager.get_shared_connections("MLX")[0].votes == 1
    # Unknown id is a no-op, reported as False.
    assert collaborative_manager.vote_connection("nonexistent") is False


def test_collaborative_manager_vote_insight(collaborative_manager):
    """Votes cast through the Manager actually move the insight counter."""
    insight = collaborative_manager.share_insight("user1", "MLX runs on-device")

    assert collaborative_manager.vote_insight(insight.insight_id) is True
    assert collaborative_manager.get_top_insights()[0].upvotes == 1
    assert collaborative_manager.vote_insight("nonexistent", upvote=False) is False


def test_shared_connection_dataclass():
    """Test SharedConnection dataclass structure."""
    conn = SharedConnection(
        connection_id="conn-1",
        from_user="user1",
        entity_a="MLX",
        entity_b="Apple",
        relationship="optimized for",
        confidence=0.8,
        discovered_at="2026-01-01T00:00:00Z",
        votes=5,
    )
    assert conn.entity_a == "MLX"
    assert conn.votes == 5


def test_collective_insight_dataclass():
    """Test CollectiveInsight dataclass structure."""
    insight = CollectiveInsight(
        insight_id="insight-1",
        content="Test insight",
        contributors=["user1", "user2"],
        upvotes=10,
        downvotes=2,
        created_at="2026-01-01T00:00:00Z",
    )
    assert insight.content == "Test insight"
    assert insight.upvotes == 10


def test_user_profile_dataclass():
    """Test UserProfile dataclass structure."""
    profile = UserProfile(
        user_id="user1",
        username="user1",
        reputation=100,
        contributions_count=10,
        joined_at="2026-01-01T00:00:00Z",
    )
    assert profile.reputation == 100
    assert profile.contributions_count == 10
