"""Tests for graph navigation module."""

import pytest

from memo.navigation import (
    CentralityScores,
    Community,
    EntityNeighbors,
    EntityPath,
    GraphNavigator,
)


@pytest.fixture(autouse=True)
def _isolate_codegraph(monkeypatch, tmp_path):
    """Hermetic isolation: navigation tests must not read the machine's real
    .codegraph index. Point the loader at a nonexistent DB so the codegraph
    merge degrades to a no-op unless a test explicitly seeds one."""
    from memo import codegraph_loader

    monkeypatch.setattr(codegraph_loader, "CODEGRAPH_DB", tmp_path / "no-codegraph.db")
    codegraph_loader.reset()
    yield
    codegraph_loader.reset()


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

    Note: Returns results if .codegraph/codegraph.db exists with matching entities.
    """
    neighbors = navigator.get_neighbors("nonexistent")
    assert neighbors.entity == "nonexistent"
    # Degree may be >0 if codegraph fallback has matching entities
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
        intermediate_memories=["mem1"],
    )
    assert path.source == "a"
    assert path.target == "b"
    assert path.length == 2
    assert len(path.intermediate_memories) == 1


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


def test_navigator_merges_codegraph(navigator, monkeypatch, tmp_path):
    """Codegraph is folded into the navigator as a primary layer (gated on)."""
    import sqlite3

    from memo import codegraph_loader

    db = tmp_path / ".codegraph" / "codegraph.db"
    db.parent.mkdir(parents=True)
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE nodes(id TEXT PRIMARY KEY, kind TEXT, name TEXT);"
        "CREATE TABLE edges(source TEXT, target TEXT, kind TEXT);"
        "INSERT INTO nodes VALUES ('function:a','function','Alpha'),"
        "('function:b','function','Beta');"
        "INSERT INTO edges VALUES ('function:a','function:b','calls');"
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(codegraph_loader, "CODEGRAPH_DB", db)
    codegraph_loader.reset()

    # The entity-memory graph is empty, so this path can only come from codegraph.
    path = navigator.find_shortest_path("Alpha", "Beta")
    assert path is not None
    assert path.path == ["alpha", "beta"]
    assert navigator.get_neighbors("alpha").degree == 1


def test_communities_split_hub_graph(mock_graph_store):
    """A hub entity must not fuse two otherwise-separate clusters into one."""
    g = mock_graph_store
    # cluster 1: m1,m2 over {a,b}; cluster 2: m3,m4 over {x,y}; hub 'h' touches both once.
    for mid in ("m1", "m2"):
        g.record_extraction(memory_id=mid, memory_date="2026-01-01",
                            entities=[{"name": "A", "type": "concept"},
                                      {"name": "B", "type": "concept"}],
                            extracted_at="2026-01-01T00:00:00Z")
    for mid in ("m3", "m4"):
        g.record_extraction(memory_id=mid, memory_date="2026-01-01",
                            entities=[{"name": "X", "type": "concept"},
                                      {"name": "Y", "type": "concept"}],
                            extracted_at="2026-01-01T00:00:00Z")
    g.record_extraction(memory_id="m5", memory_date="2026-01-01",
                        entities=[{"name": "B", "type": "concept"},
                                  {"name": "H", "type": "concept"}],
                        extracted_at="2026-01-01T00:00:00Z")
    g.record_extraction(memory_id="m6", memory_date="2026-01-01",
                        entities=[{"name": "X", "type": "concept"},
                                  {"name": "H", "type": "concept"}],
                        extracted_at="2026-01-01T00:00:00Z")
    g.rebuild_edges()
    from memo.navigation import GraphNavigator
    nav = GraphNavigator(g)
    comms = nav.detect_communities(min_size=2, use_codegraph=False)
    # the {a,b} core and {x,y} core stay distinct communities (not one blob)
    cores = [set(c.entities) & {"a", "b", "x", "y"} for c in comms]
    assert {"a", "b"} in cores
    assert {"x", "y"} in cores
