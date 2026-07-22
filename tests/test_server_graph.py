"""Tests for server_graph MCP tool registration (memo.server_graph domain)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from memo.memory import Memory
from memo.navigation import CentralityScores, Community, EntityNeighbors, EntityPath


def _make_server_and_tools() -> tuple[MagicMock, dict]:
    """Return a (server_mock, tools_dict) pair.

    ``server.tool()`` is wired so each ``@server.tool()`` decorated function is
    captured in ``tools`` by its ``__name__``, without going through FastMCP.
    """
    server = MagicMock()
    tools: dict = {}

    def tool_decorator():
        def wrapper(fn):
            tools[fn.__name__] = fn
            return fn

        return wrapper

    server.tool = tool_decorator
    return server, tools


def test_register_exposes_graph_tools(tmp_cfg) -> None:
    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    from memo.server_graph import register

    register(server, mem)

    expected = {
        "memo_graph_path",
        "memo_graph_neighbors",
        "memo_explore",
        "memo_graph_communities",
        "memo_graph_centrality",
        "memo_graph_export",
        "memo_graph_trace",
        "memo_graph_discover",
    }
    assert expected == set(tools), f"Tool mismatch: {set(tools)}"


def test_memo_graph_trace_routes_to_core_api(tmp_cfg) -> None:
    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.graph_trace.return_value = {"available": True, "code_refs": []}
    server, tools = _make_server_and_tools()
    from memo.server_graph import register

    register(server, mem)
    result = tools["memo_graph_trace"](memory_id="abc", limit=20)

    assert result["available"] is True
    mem.graph_trace.assert_called_once_with(memory_id="abc", code=None, limit=20)


def test_memo_graph_discover_routes_to_core_api(tmp_cfg) -> None:
    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.graph_discover.return_value = {"available": True, "communities": [], "bridges": []}
    server, tools = _make_server_and_tools()
    from memo.server_graph import register

    register(server, mem)
    result = tools["memo_graph_discover"](min_community_size=3, include_code=False)

    assert result["available"] is True
    mem.graph_discover.assert_called_once_with(
        min_community_size=3,
        min_bridge_side=2,
        max_communities=5,
        max_bridges=5,
        include_code=False,
    )


def test_memo_graph_path_found(tmp_cfg) -> None:
    """memo_graph_path returns path.__dict__ when a path exists."""
    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    entity_path = EntityPath(
        source="alpha",
        target="gamma",
        path=["alpha", "beta", "gamma"],
        length=2,
        intermediate_memories=["m1", "m2"],
    )
    mem.navigator.find_shortest_path.return_value = entity_path

    server, tools = _make_server_and_tools()
    from memo.server_graph import register

    register(server, mem)

    result = tools["memo_graph_path"](source="alpha", target="gamma")

    mem.navigator.find_shortest_path.assert_called_once_with("alpha", "gamma", max_length=5)
    assert result is not None
    assert result["source"] == "alpha"
    assert result["target"] == "gamma"
    assert result["path"] == ["alpha", "beta", "gamma"]
    assert result["length"] == 2
    assert result["intermediate_memories"] == ["m1", "m2"]


def test_memo_graph_path_custom_max_length(tmp_cfg) -> None:
    """memo_graph_path passes max_length through to the navigator."""
    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    mem.navigator.find_shortest_path.return_value = None

    server, tools = _make_server_and_tools()
    from memo.server_graph import register

    register(server, mem)

    result = tools["memo_graph_path"](source="alpha", target="omega", max_length=3)

    mem.navigator.find_shortest_path.assert_called_once_with("alpha", "omega", max_length=3)
    assert result is None


def test_memo_graph_neighbors(tmp_cfg) -> None:
    """memo_graph_neighbors returns neighbors.__dict__ with all fields."""
    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    neighbors = EntityNeighbors(
        entity="beta",
        direct_neighbors=["alpha", "gamma"],
        neighbor_memories={"alpha": ["m1"], "gamma": ["m2"]},
        degree=2,
    )
    mem.navigator.get_neighbors.return_value = neighbors

    server, tools = _make_server_and_tools()
    from memo.server_graph import register

    register(server, mem)

    result = tools["memo_graph_neighbors"](entity="beta")

    mem.navigator.get_neighbors.assert_called_once_with("beta", max_neighbors=50)
    assert result["entity"] == "beta"
    assert result["direct_neighbors"] == ["alpha", "gamma"]
    assert result["degree"] == 2
    assert result["neighbor_memories"] == {"alpha": ["m1"], "gamma": ["m2"]}


def test_memo_graph_neighbors_custom_limit(tmp_cfg) -> None:
    """memo_graph_neighbors passes max_neighbors through to the navigator."""
    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    neighbors = EntityNeighbors(
        entity="alpha",
        direct_neighbors=[],
        neighbor_memories={},
        degree=0,
    )
    mem.navigator.get_neighbors.return_value = neighbors

    server, tools = _make_server_and_tools()
    from memo.server_graph import register

    register(server, mem)

    tools["memo_graph_neighbors"](entity="alpha", max_neighbors=10)
    mem.navigator.get_neighbors.assert_called_once_with("alpha", max_neighbors=10)


def test_memo_explore_delegates_to_explore_entity(tmp_cfg) -> None:
    """memo_explore calls explore_entity and returns its result unchanged."""
    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    expected: dict = {
        "entity": "beta",
        "degree": 2,
        "neighbors": [{"name": "alpha", "shared": 1}],
        "memories": [{"id": "m1", "title": "Test memory"}],
    }

    server, tools = _make_server_and_tools()
    from memo.server_graph import register

    register(server, mem)

    with patch("memo.explore.explore_entity", return_value=expected) as mock_explore:
        result = tools["memo_explore"](entity="beta", max_neighbors=5, max_memories=3)

    mock_explore.assert_called_once_with(mem, "beta", max_neighbors=5, max_memories=3)
    assert result == expected
    assert result["entity"] == "beta"
    assert result["degree"] == 2


def test_memo_graph_communities_returns_list_of_dicts(tmp_cfg) -> None:
    """memo_graph_communities converts each Community to a dict."""
    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    comms = [
        Community(id=0, entities=["alpha", "beta"], size=2, representative_entity="alpha"),
        Community(
            id=1, entities=["gamma", "delta", "epsilon"], size=3, representative_entity="gamma"
        ),
    ]
    mem.navigator.detect_communities.return_value = comms

    server, tools = _make_server_and_tools()
    from memo.server_graph import register

    register(server, mem)

    result = tools["memo_graph_communities"](min_size=2)

    mem.navigator.detect_communities.assert_called_once_with(min_size=2)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["id"] == 0
    assert result[0]["entities"] == ["alpha", "beta"]
    assert result[0]["size"] == 2
    assert result[0]["representative_entity"] == "alpha"
    assert result[1]["id"] == 1
    assert result[1]["size"] == 3


def test_memo_graph_communities_empty_graph(tmp_cfg) -> None:
    """memo_graph_communities returns an empty list when no communities detected."""
    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    mem.navigator.detect_communities.return_value = []

    server, tools = _make_server_and_tools()
    from memo.server_graph import register

    register(server, mem)

    result = tools["memo_graph_communities"]()
    assert result == []


def test_memo_graph_centrality_sorted_by_degree(tmp_cfg) -> None:
    """memo_graph_centrality returns top N entities sorted by degree descending."""
    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    scores = CentralityScores(
        degree={"alpha": 3, "beta": 5, "gamma": 1},
        betweenness={"alpha": 0.2, "beta": 0.8, "gamma": 0.0},
    )
    mem.navigator.compute_centrality.return_value = scores

    server, tools = _make_server_and_tools()
    from memo.server_graph import register

    register(server, mem)

    result = tools["memo_graph_centrality"](top=2)

    mem.navigator.compute_centrality.assert_called_once_with()
    assert result["total_entities"] == 3
    top = result["top_entities"]
    assert len(top) == 2
    # Sorted by degree descending: beta(5) > alpha(3)
    assert top[0]["entity"] == "beta"
    assert top[0]["degree"] == 5
    assert top[0]["betweenness"] == 0.8
    assert top[1]["entity"] == "alpha"
    assert top[1]["degree"] == 3
    assert top[1]["betweenness"] == 0.2


def test_memo_graph_centrality_missing_betweenness_defaults_to_zero(tmp_cfg) -> None:
    """memo_graph_centrality uses 0.0 betweenness when not present in scores."""
    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    scores = CentralityScores(
        degree={"alpha": 2},
        betweenness={},  # alpha absent → should default to 0.0
    )
    mem.navigator.compute_centrality.return_value = scores

    server, tools = _make_server_and_tools()
    from memo.server_graph import register

    register(server, mem)

    result = tools["memo_graph_centrality"](top=5)

    assert result["top_entities"][0]["entity"] == "alpha"
    assert result["top_entities"][0]["betweenness"] == 0.0


def test_memo_graph_export_dot_format(tmp_cfg) -> None:
    """memo_graph_export with format='dot' calls export_graphviz and wraps result."""
    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    dot_content = 'digraph {\n  "alpha" -> "beta";\n}'
    mem.navigator.export_graphviz.return_value = dot_content

    server, tools = _make_server_and_tools()
    from memo.server_graph import register

    register(server, mem)

    result = tools["memo_graph_export"](format="dot")

    mem.navigator.export_graphviz.assert_called_once()
    mem.navigator.export_json.assert_not_called()
    assert result["format"] == "dot"
    assert result["content"] == dot_content


def test_memo_graph_export_json_format(tmp_cfg) -> None:
    """memo_graph_export with format='json' calls export_json and wraps result."""
    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    json_data = {
        "nodes": [{"id": "alpha"}, {"id": "beta"}],
        "edges": [{"source": "alpha", "target": "beta"}],
    }
    mem.navigator.export_json.return_value = json_data

    server, tools = _make_server_and_tools()
    from memo.server_graph import register

    register(server, mem)

    result = tools["memo_graph_export"](format="json", include_memories=True)

    mem.navigator.export_json.assert_called_once_with(include_memories=True)
    mem.navigator.export_graphviz.assert_not_called()
    assert result["format"] == "json"
    assert result["data"] == json_data


def test_memo_graph_export_json_default_no_memories(tmp_cfg) -> None:
    """memo_graph_export passes include_memories=False by default for json format."""
    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    mem.navigator.export_json.return_value = {"nodes": [], "edges": []}

    server, tools = _make_server_and_tools()
    from memo.server_graph import register

    register(server, mem)

    tools["memo_graph_export"](format="json")
    mem.navigator.export_json.assert_called_once_with(include_memories=False)
