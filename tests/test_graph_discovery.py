from __future__ import annotations

from memo.graph_discovery import discover_graph
from memo.graph_projection import GraphReadModel, ProjectedEdge, ProjectedNode


def _node(name: str) -> ProjectedNode:
    return ProjectedNode(
        uri=f"entity://concept/{name}",
        label=name,
        entity_type="concept",
        canonical_key=name,
        doc_freq=1,
        degree=1,
        quality=0.9,
        is_hub=False,
        idf=1.0,
    )


def _edge(a: str, b: str, memory_id: str) -> ProjectedEdge:
    return ProjectedEdge(
        source_uri=f"entity://concept/{a}",
        target_uri=f"entity://concept/{b}",
        relation="co_occurs",
        weight=1.0,
        confidence=0.9,
        evidence_ids=(f"memory://{memory_id}",),
        first_seen=None,
        last_seen=None,
    )


def _model() -> GraphReadModel:
    nodes = {name: _node(name) for name in ("a", "b", "x", "c", "d")}
    edges = (
        _edge("a", "b", "m1"),
        _edge("b", "x", "m2"),
        _edge("x", "c", "m3"),
        _edge("c", "d", "m4"),
    )
    by_uri: dict[str, list[ProjectedEdge]] = {node.uri: [] for node in nodes.values()}
    for edge in edges:
        by_uri[edge.source_uri].append(edge)
        by_uri[edge.target_uri].append(edge)
    return GraphReadModel(
        available=True,
        version="v1",
        nodes={node.uri: node for node in nodes.values()},
        memberships={
            "m1": (nodes["a"].uri, nodes["b"].uri),
            "m2": (nodes["b"].uri, nodes["x"].uri),
            "m3": (nodes["x"].uri, nodes["c"].uri),
            "m4": (nodes["c"].uri, nodes["d"].uri),
        },
        edges={uri: tuple(value) for uri, value in by_uri.items()},
    )


def test_discovery_splits_regions_around_bridge_with_exact_evidence() -> None:
    result = discover_graph(
        _model(),
        min_community_size=2,
        min_bridge_side=2,
        max_communities=5,
        max_bridges=5,
    )

    assert result["available"] is True
    assert [[node["label"] for node in item["nodes"]] for item in result["communities"]] == [
        ["a", "b"],
        ["c", "d"],
    ]
    bridge = result["bridges"][0]
    assert bridge["bridge"]["label"] == "x"
    assert {bridge["left_rep"]["label"], bridge["right_rep"]["label"]} == {"b", "c"}
    assert set(bridge["memory_ids"]) == {"m2", "m3"}
    assert {edge["evidence_ids"][0] for edge in bridge["edge_evidence"]} == {
        "memory://m2",
        "memory://m3",
    }


def test_discovery_never_falls_back_when_projection_unavailable() -> None:
    result = discover_graph(GraphReadModel.unavailable("projection_stale"))
    assert result == {
        "available": False,
        "reason": "projection_stale",
        "projection_version": None,
        "communities": [],
        "bridges": [],
    }


def test_memory_discovery_is_flag_gated_and_uses_projection(mem_with_stub, monkeypatch) -> None:
    disabled = mem_with_stub.graph_discover()
    assert disabled["reason"] == "disabled"

    monkeypatch.setenv("MEMO_GRAPH_DISCOVERY_ENABLED", "1")
    monkeypatch.setattr(mem_with_stub.graph.projection, "read_model", lambda _age: _model())
    active = mem_with_stub.graph_discover(min_community_size=2)

    assert active["available"] is True
    assert active["projection_version"] == "v1"
    assert len(active["bridges"]) == 1
