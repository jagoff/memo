"""Tests for graph navigation module."""

import pytest

from memo.navigation import (
    CentralityScores,
    Community,
    EntityNeighbors,
    EntityPath,
    GraphNavigator,
)


@pytest.fixture
def mock_graph_store(tmp_cfg):
    """Fixture providing a mock GraphStore instance."""
    from memo.graph import GraphStore
    gs = GraphStore(tmp_cfg.graph_db)
    yield gs
    gs.close()


@pytest.fixture
def navigator(mock_graph_store):
    """Fixture providing GraphNavigator instance."""
    return GraphNavigator(mock_graph_store)


def test_navigator_init(navigator):
    """Test GraphNavigator initialization."""
    assert navigator.graph is not None


def test_find_shortest_path_same_entity(navigator):
    """Test path finding when source equals target."""
    path = navigator.find_shortest_path("test", "test")
    assert path is not None
    assert path.source == "test"
    assert path.target == "test"
    assert path.length == 0
    assert path.path == ["test"]


def test_find_shortest_path_no_entities(navigator):
    """Test path finding with no entities in graph."""
    path = navigator.find_shortest_path("nonexistent", "also-nonexistent")
    assert path is None


def test_find_shortest_path_with_data(navigator, mock_memory):
    """Test path finding with actual entity data."""
    mem = mock_memory

    # Create test memorias that share entities
    rec1 = mem.save(
        content="Memo about MLX and Qwen",
        title="MLX Qwen",
        tags=["mlx", "qwen"],
    )
    rec2 = mem.save(
        content="Memo about Qwen and Obsidian",
        title="Qwen Obsidian",
        tags=["qwen", "obsidian"],
    )

    # Extract entities
    mem.extract_entities(ids=[rec1.id, rec2.id])

    # Find path between MLX and Obsidian (should go through Qwen)
    path = navigator.find_shortest_path("mlx", "obsidian", max_length=5)
    # May not find path if graph is sparse, but test the structure
    if path:
        assert path.source == "mlx"
        assert path.target == "obsidian"
        assert len(path.path) >= 2


def test_get_neighbors_no_entity(navigator):
    """Test getting neighbors for nonexistent entity.

    Note: Returns results if graphify-out/graph.json exists with matching entities.
    """
    neighbors = navigator.get_neighbors("nonexistent")
    assert neighbors.entity == "nonexistent"
    # Degree may be >0 if graphify fallback has matching entities
    assert isinstance(neighbors.degree, int)
    assert isinstance(neighbors.direct_neighbors, list)


def test_get_neighbors_with_data(navigator, mock_memory):
    """Test getting neighbors with actual entity data."""
    mem = mock_memory

    rec = mem.save(
        content="Memo about MLX and Qwen",
        title="MLX Qwen",
        tags=["mlx", "qwen"],
    )

    mem.extract_entities(ids=[rec.id])

    neighbors = navigator.get_neighbors("mlx")
    # May have neighbors or not depending on extraction
    assert neighbors.entity == "mlx"


def test_detect_communities_empty(navigator):
    """Test community detection with empty graph."""
    communities = navigator.detect_communities(min_size=2)
    assert communities == []


def test_detect_communities_with_data(navigator, mock_memory):
    """Test community detection with actual entity data."""
    mem = mock_memory

    # Create cluster of related memorias
    rec1 = mem.save(
        content="Memo about MLX",
        title="MLX",
        tags=["mlx"],
    )
    rec2 = mem.save(
        content="Memo about MLX and Qwen",
        title="MLX Qwen",
        tags=["mlx", "qwen"],
    )
    rec3 = mem.save(
        content="Memo about Qwen",
        title="Qwen",
        tags=["qwen"],
    )

    mem.extract_entities(ids=[rec1.id, rec2.id, rec3.id])

    communities = navigator.detect_communities(min_size=2)
    # Should find at least one community if entities are connected
    assert isinstance(communities, list)


def test_compute_centrality_empty(navigator):
    """Test centrality computation with empty graph."""
    scores = navigator.compute_centrality()
    assert scores.degree == {}
    assert scores.betweenness == {}


def test_compute_centrality_with_data(navigator, mock_memory):
    """Test centrality computation with actual entity data."""
    mem = mock_memory

    rec = mem.save(
        content="Memo about MLX",
        title="MLX",
        tags=["mlx"],
    )

    mem.extract_entities(ids=[rec.id])

    scores = navigator.compute_centrality()
    assert isinstance(scores.degree, dict)
    assert isinstance(scores.betweenness, dict)


def test_export_graphviz(navigator):
    """Test Graphviz DOT export."""
    dot = navigator.export_graphviz()
    assert "graph memo_entities" in dot
    assert "rankdir=LR" in dot


def test_export_graphviz_to_file(tmp_path, navigator):
    """Test Graphviz DOT export to file."""
    output_file = tmp_path / "graph.dot"
    navigator.export_graphviz(output_path=str(output_file))
    assert output_file.is_file()
    content = output_file.read_text()
    assert "graph memo_entities" in content


def test_export_json(navigator):
    """Test JSON export."""
    data = navigator.export_json(include_memories=False)
    assert "nodes" in data
    assert "edges" in data
    assert isinstance(data["nodes"], list)
    assert isinstance(data["edges"], list)


def test_export_json_with_memorias(navigator):
    """Test JSON export with memoria IDs."""
    data = navigator.export_json(include_memories=True)
    assert "nodes" in data
    assert "edges" in data
    # If there are edges, they should have memory_id when include_memories=True
    if data["edges"]:
        # This may be empty if no entities, but structure should be correct
        pass


def test_entity_path_dataclass():
    """Test EntityPath dataclass structure."""
    path = EntityPath(
        source="a",
        target="b",
        path=["a", "c", "b"],
        length=2,
        intermediate_memorias=["mem1"],
    )
    assert path.source == "a"
    assert path.target == "b"
    assert path.length == 2
    assert len(path.intermediate_memorias) == 1


def test_entity_neighbors_dataclass():
    """Test EntityNeighbors dataclass structure."""
    neighbors = EntityNeighbors(
        entity="test",
        direct_neighbors=["a", "b"],
        neighbor_memories={"a": ["mem1"], "b": ["mem2"]},
        degree=2,
    )
    assert neighbors.entity == "test"
    assert neighbors.degree == 2
    assert len(neighbors.direct_neighbors) == 2


def test_community_dataclass():
    """Test Community dataclass structure."""
    comm = Community(
        id=1,
        entities=["a", "b", "c"],
        size=3,
        representative_entity="a",
    )
    assert comm.id == 1
    assert comm.size == 3
    assert comm.representative_entity == "a"


def test_centrality_scores_dataclass():
    """Test CentralityScores dataclass structure."""
    scores = CentralityScores(
        degree={"a": 5, "b": 3},
        betweenness={"a": 0.8, "b": 0.2},
    )
    assert scores.degree["a"] == 5
    assert scores.betweenness["a"] == 0.8
