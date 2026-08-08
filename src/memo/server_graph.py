"""MCP tools — knowledge-graph domain (split from server.py).

Registered by `build_server()` via `register(server, memory)`. Tool names,
signatures, defaults, docstrings and bodies are identical to the originals;
only the enclosing function and indentation changed.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastmcp import FastMCP
from pydantic import Field

from memo.mcp_budget import bounded_list
from memo.memory import Memory
from memo.server_annotations import READ_ONLY, annotated_tool

# Per-community entity cap. Measured 2026-08-06 against the developer's LIVE
# install (~11.3k memories with the codegraph layer merged in), NOT the
# synthetic conformance corpus -- pytest cannot reach `.codegraph/codegraph.db`
# and the conformance fixture seeds no code layer at all: `detect_communities`
# returned 2,278 communities, the largest holding 155 entities. The community
# COUNT dominated that payload, but one hub community can carry a long entity
# list on its own, so both dimensions are bounded. The conformance gate
# (`tests/conformance/test_mcp_response_budget.py`) seeds its own entity graph
# and holds the resulting payload under the cap; it does not reproduce these
# live numbers.
_MAX_COMMUNITY_ENTITIES = 50


def _bounded_dot(dot: str, *, max_edges: int) -> dict[str, Any]:
    """Trim the DOT export to `max_edges` edge lines, reporting the true count.

    The live graph renders 100,141 edge lines (3.6 MB) — a whole-graph dump no
    tool result can carry. Only edge lines are dropped; the header and the
    closing brace stay, so the trimmed DOT still parses.
    """
    lines = dot.split("\n")
    edge_lines = [i for i, line in enumerate(lines) if " -- " in line]
    if len(edge_lines) <= max_edges:
        return {
            "format": "dot",
            "content": dot,
            "edge_count": len(edge_lines),
            "truncated": False,
        }
    dropped = set(edge_lines[max_edges:])
    return {
        "format": "dot",
        "content": "\n".join(line for i, line in enumerate(lines) if i not in dropped),
        "edge_count": len(edge_lines),
        "truncated": True,
    }


def _bounded_json(data: dict[str, Any], *, max_edges: int) -> dict[str, Any]:
    """Trim the JSON export to `max_edges` edges, reporting the true sizes.

    The live graph exports 18,724 nodes and 82,517 edges (6.0 MB). Nodes are
    filtered down to the endpoints of the retained edges so the payload stays a
    drawable graph rather than edges pointing at absent nodes plus isolates.
    """
    nodes = data.get("nodes") or []
    edges = data.get("edges") or []
    if len(edges) <= max_edges:
        return {
            "format": "json",
            "data": data,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "truncated": False,
        }
    kept = edges[:max_edges]
    endpoints = {e.get("source") for e in kept} | {e.get("target") for e in kept}
    return {
        "format": "json",
        "data": {"nodes": [n for n in nodes if n.get("id") in endpoints], "edges": kept},
        "node_count": len(nodes),
        "edge_count": len(edges),
        "truncated": True,
    }


def _bounded_dot(dot: str, *, max_edges: int) -> dict[str, Any]:
    """Trim the DOT export to `max_edges` edge lines, reporting the true count.

    The live graph renders 100,141 edge lines (3.6 MB) — a whole-graph dump no
    tool result can carry. Only edge lines are dropped; the header and the
    closing brace stay, so the trimmed DOT still parses.
    """
    lines = dot.split("\n")
    edge_lines = [i for i, line in enumerate(lines) if " -- " in line]
    if len(edge_lines) <= max_edges:
        return {
            "format": "dot",
            "content": dot,
            "edge_count": len(edge_lines),
            "truncated": False,
        }
    dropped = set(edge_lines[max_edges:])
    return {
        "format": "dot",
        "content": "\n".join(line for i, line in enumerate(lines) if i not in dropped),
        "edge_count": len(edge_lines),
        "truncated": True,
    }


def _bounded_json(data: dict[str, Any], *, max_edges: int) -> dict[str, Any]:
    """Trim the JSON export to `max_edges` edges, reporting the true sizes.

    The live graph exports 18,724 nodes and 82,517 edges (6.0 MB). Nodes are
    filtered down to the endpoints of the retained edges so the payload stays a
    drawable graph rather than edges pointing at absent nodes plus isolates.
    """
    nodes = data.get("nodes") or []
    edges = data.get("edges") or []
    if len(edges) <= max_edges:
        return {
            "format": "json",
            "data": data,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "truncated": False,
        }
    kept = edges[:max_edges]
    endpoints = {e.get("source") for e in kept} | {e.get("target") for e in kept}
    return {
        "format": "json",
        "data": {"nodes": [n for n in nodes if n.get("id") in endpoints], "edges": kept},
        "node_count": len(nodes),
        "edge_count": len(edges),
        "truncated": True,
    }


def register(server: FastMCP, memory: Memory) -> None:
    @annotated_tool(server, **READ_ONLY)
    def memo_graph_path(
        source: str,
        target: str,
        max_length: int = 5,
    ) -> dict[str, Any] | None:
        """Find shortest path between two entities in the entity graph.

        Uses BFS to find the shortest path. Two entities are connected if
        they share a memory. Returns the path including intermediate entities.

        Args:
            source: Source entity name.
            target: Target entity name.
            max_length: Maximum path length to search.
        """
        path = memory.navigator.find_shortest_path(source, target, max_length=max_length)
        return path.__dict__ if path else None

    @annotated_tool(server, **READ_ONLY)
    def memo_graph_neighbors(
        entity: str,
        max_neighbors: int = 50,
    ) -> dict[str, Any]:
        """Get direct neighbors of an entity in the graph.

        Returns entities directly connected to the given entity, along with
        the memories that connect them.

        Args:
            entity: Entity name.
            max_neighbors: Maximum neighbors to return.
        """
        neighbors = memory.navigator.get_neighbors(entity, max_neighbors=max_neighbors)
        return neighbors.to_bounded_dict()

    @annotated_tool(server, **READ_ONLY)
    def memo_explore(
        entity: str,
        max_neighbors: int = 8,
        max_memories: int = 8,
    ) -> dict[str, Any]:
        """Zoom into one entity: its degree, the neighbours it connects to (with
        how many memories bridge each link), and the memories that mention it —
        one rich "what's around X" view over the merged entity + code graph.

        Args:
            entity: Entity or code-symbol name.
            max_neighbors: Max neighbours to return.
            max_memories: Max mentioning memories to return.
        """
        from memo.explore import explore_entity

        return explore_entity(
            memory, entity, max_neighbors=max_neighbors, max_memories=max_memories
        )

    @annotated_tool(server, **READ_ONLY)
    def memo_graph_communities(
        min_size: int = 2,
        limit: Annotated[
            int,
            Field(
                description="Maximum communities to return, largest first. A full "
                "page (exactly `limit` items) means more may exist — raise `limit` "
                "or `min_size` to see past it."
            ),
        ] = 20,
    ) -> list[dict[str, Any]]:
        """Detect communities (connected components) in the entity graph.

        Uses connected components to find clusters of related entities.
        Useful for discovering thematic clusters in the knowledge graph.

        Bounded on both dimensions that scale with the corpus: the list is
        capped at `limit` (largest first) and each community's `entities` at
        50. `size` is always the community's TRUE entity count, so
        `size > len(entities)` means the entity list was trimmed and
        `entities_truncated` says so outright.

        The return stays a LIST — memo's tool surface is a public contract
        (PyPI, MCP registries, the Claude store), and wrapping it in an
        object to carry a community COUNT would break every caller doing
        `for c in result` and change the generated output schema. The count
        of communities beyond the page is therefore not reported; a full
        page is the signal, per the `limit` field description.

        Args:
            min_size: Minimum community size to include.
            limit: Maximum communities to return, largest first.
        """
        communities = memory.navigator.detect_communities(min_size=min_size)
        kept, _ = bounded_list(communities, cap=max(0, limit), key=lambda c: -c.size)
        bounded: list[dict[str, Any]] = []
        for c in kept:
            entities, entity_meta = bounded_list(c.entities, cap=_MAX_COMMUNITY_ENTITIES)
            bounded.append(
                {
                    "id": c.id,
                    "representative_entity": c.representative_entity,
                    "size": c.size,
                    "entities": entities,
                    "entities_truncated": entity_meta["truncated"],
                }
            )
        return bounded

    @annotated_tool(server, **READ_ONLY)
    def memo_graph_trace(
        memory_id: Annotated[
            str | None,
            Field(
                description=(
                    "Memory id (or unique id prefix) whose linked code references to "
                    "return. Provide exactly one of memory_id or code."
                ),
            ),
        ] = None,
        code: Annotated[
            str | None,
            Field(
                description=(
                    "Code reference whose linked memories to return: a codegraph:// "
                    "URI (exact, case-insensitive) or a substring of a symbol label, "
                    "qualified name, file path, or stable symbol id. Provide exactly "
                    "one of memory_id or code."
                ),
            ),
        ] = None,
        limit: Annotated[
            int,
            Field(description="Maximum linked items to return (clamped to 1-200)."),
        ] = 50,
    ) -> dict[str, Any]:
        """Trace evidence from one memory to code, or one code reference to memories.

        Provide exactly one of ``memory_id`` or ``code``. Code accepts a stable
        codegraph URI, symbol id, qualified name, or file path.
        """
        return memory.graph_trace(memory_id=memory_id, code=code, limit=limit)

    @annotated_tool(server, **READ_ONLY)
    def memo_graph_discover(
        min_community_size: Annotated[
            int,
            Field(
                description=(
                    "Minimum node count for a connected component to be reported as "
                    "a community (components above the internal 40-node region cap "
                    "are also skipped)."
                ),
            ),
        ] = 4,
        min_bridge_side: Annotated[
            int,
            Field(
                description=(
                    "Minimum node count required on each side of an articulation "
                    "bridge (floored to 1)."
                ),
            ),
        ] = 2,
        max_communities: Annotated[
            int,
            Field(description="Maximum communities to return, largest first (floored to 0)."),
        ] = 5,
        max_bridges: Annotated[
            int,
            Field(
                description=(
                    "Maximum bridges to return, largest combined sides first (floored to 0)."
                ),
            ),
        ] = 5,
        include_code: Annotated[
            bool,
            Field(
                description=(
                    "If false, exclude codegraph:// code nodes from the graph before "
                    "detecting communities and bridges."
                ),
            ),
        ] = True,
    ) -> dict[str, Any]:
        """Discover bounded curated communities and bridges with exact evidence."""
        return memory.graph_discover(
            min_community_size=min_community_size,
            min_bridge_side=min_bridge_side,
            max_communities=max_communities,
            max_bridges=max_bridges,
            include_code=include_code,
        )

    @annotated_tool(server, **READ_ONLY)
    def memo_graph_centrality(
        top: int = 20,
    ) -> dict[str, Any]:
        """Compute centrality metrics for all entities.

        Returns degree centrality (number of connections) and betweenness
        centrality (how often entity lies on shortest paths). Useful for
        identifying hub entities in the graph.

        Args:
            top: Return top N entities by degree centrality.
        """
        scores = memory.navigator.compute_centrality()
        sorted_by_degree = sorted(scores.degree.items(), key=lambda x: x[1], reverse=True)[:top]
        return {
            "top_entities": [
                {"entity": e, "degree": d, "betweenness": scores.betweenness.get(e, 0.0)}
                for e, d in sorted_by_degree
            ],
            "total_entities": len(scores.degree),
        }

    @annotated_tool(server, **READ_ONLY)
    def memo_graph_export(
        format: str = "dot",
        include_memories: bool = False,
        max_edges: Annotated[
            int,
            Field(
                description=(
                    "Maximum edges to return, first seen first (floored to 0). The "
                    "true graph size is always reported in edge_count/node_count, "
                    "and truncated says whether edges were dropped. For the whole "
                    "graph use the CLI: `memo graph export -o <file>`."
                ),
            ),
        ] = 500,
    ) -> dict[str, Any]:
        """Export a bounded slice of the entity graph for visualization.

        Returns graph data in the specified format. Use with external tools
        like Graphviz (dot format) or web visualization libraries (JSON format).

        Args:
            format: Either "dot" for Graphviz DOT format or "json" for web UI.
            include_memories: If True and format is "json", include memory IDs in edge data.
            max_edges: Maximum edges to return; the true sizes come back alongside.
        """
        cap = max(0, max_edges)
        if format == "dot":
            return _bounded_dot(memory.navigator.export_graphviz(), max_edges=cap)
        else:
            data = memory.navigator.export_json(include_memories=include_memories)
            return _bounded_json(data, max_edges=cap)
