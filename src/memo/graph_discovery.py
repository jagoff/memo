"""Deterministic discovery packets over one curated graph projection."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from urllib.parse import unquote, urlsplit

from memo.graph_bridges import find_bridges
from memo.graph_projection import GraphReadModel, ProjectedEdge, ProjectedNode


def _memory_ids(edges: list[ProjectedEdge]) -> list[str]:
    ids: set[str] = set()
    for edge in edges:
        for evidence in edge.evidence_ids:
            parts = urlsplit(evidence)
            if parts.scheme == "memory":
                value = unquote(parts.netloc or parts.path.lstrip("/"))
                if value:
                    ids.add(value)
    return sorted(ids)


def _components(adjacency: dict[str, dict[str, float]], excluded: set[str]) -> list[list[str]]:
    seen = set(excluded)
    out: list[list[str]] = []
    for start in sorted(adjacency):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component: list[str] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency.get(current, {}), reverse=True):
                if neighbor not in seen and neighbor not in excluded:
                    seen.add(neighbor)
                    stack.append(neighbor)
        out.append(sorted(component))
    return out


def _representative(uris: list[str], adjacency: dict[str, dict[str, float]]) -> str:
    return min(uris, key=lambda uri: (-sum(adjacency.get(uri, {}).values()), uri))


def _edge_dict(edge: ProjectedEdge) -> dict[str, object]:
    return asdict(edge)


def _node_dict(node: ProjectedNode) -> dict[str, object]:
    return asdict(node)


def discover_graph(
    model: GraphReadModel,
    *,
    min_community_size: int = 4,
    min_bridge_side: int = 2,
    max_communities: int = 5,
    max_bridges: int = 5,
    max_region_size: int = 40,
    include_code: bool = True,
) -> dict[str, object]:
    """Return bounded communities and articulation bridges with exact evidence."""
    empty: dict[str, object] = {
        "projection_version": model.version,
        "communities": [],
        "bridges": [],
    }
    if not model.available:
        return {
            "available": False,
            "reason": model.skip_reason or "projection_unavailable",
            **empty,
        }
    nodes = {
        node.uri: node
        for node in model.all_nodes()
        if not node.is_hub and (include_code or not node.uri.startswith("codegraph://"))
    }
    edges = [
        edge for edge in model.all_edges() if edge.source_uri in nodes and edge.target_uri in nodes
    ]
    adjacency: dict[str, dict[str, float]] = {uri: {} for uri in nodes}
    for edge in edges:
        adjacency[edge.source_uri][edge.target_uri] = edge.weight
        adjacency[edge.target_uri][edge.source_uri] = edge.weight

    raw_bridges = find_bridges(
        adjacency,
        min_side=max(1, min_bridge_side),
        max_side=max(1, max_region_size),
    )
    raw_bridges.sort(key=lambda item: (-(len(item["left"]) + len(item["right"])), item["bridge"]))
    bridge_uris = {str(item["bridge"]) for item in raw_bridges}
    components = [
        component
        for component in _components(adjacency, bridge_uris)
        if min_community_size <= len(component) <= max_region_size
    ]
    components.sort(key=lambda component: (-len(component), component))

    communities: list[dict[str, object]] = []
    for component in components[: max(0, max_communities)]:
        uri_set = set(component)
        evidence = [
            edge for edge in edges if edge.source_uri in uri_set and edge.target_uri in uri_set
        ]
        representative = _representative(component, adjacency)
        digest = hashlib.sha256("|".join(component).encode()).hexdigest()[:16]
        communities.append(
            {
                "id": digest,
                "size": len(component),
                "representative": _node_dict(nodes[representative]),
                "nodes": [_node_dict(nodes[uri]) for uri in component],
                "memory_ids": _memory_ids(evidence),
                "edge_evidence": [_edge_dict(edge) for edge in evidence],
            }
        )

    bridges: list[dict[str, object]] = []
    for item in raw_bridges[: max(0, max_bridges)]:
        bridge_uri = str(item["bridge"])
        left = [str(uri) for uri in item["left"]]
        right = [str(uri) for uri in item["right"]]
        left_rep = _representative(left, adjacency)
        right_rep = _representative(right, adjacency)
        incident = [
            edge
            for edge in edges
            if {edge.source_uri, edge.target_uri}
            in ({bridge_uri, left_rep}, {bridge_uri, right_rep})
        ]
        bridges.append(
            {
                "bridge": _node_dict(nodes[bridge_uri]),
                "left": [_node_dict(nodes[uri]) for uri in left],
                "right": [_node_dict(nodes[uri]) for uri in right],
                "left_rep": _node_dict(nodes[left_rep]),
                "right_rep": _node_dict(nodes[right_rep]),
                "memory_ids": _memory_ids(incident),
                "edge_evidence": [_edge_dict(edge) for edge in incident],
            }
        )
    return {
        "available": True,
        "reason": None,
        "projection_version": model.version,
        "communities": communities,
        "bridges": bridges,
    }


__all__ = ["discover_graph"]
