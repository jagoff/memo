"""MCP tools — knowledge-graph domain (split from server.py).

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
        return neighbors.__dict__

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
    ) -> list[dict[str, Any]]:
        """Detect communities (connected components) in the entity graph.

        Uses connected components to find clusters of related entities.
        Useful for discovering thematic clusters in the knowledge graph.

        Args:
            min_size: Minimum community size to include.
        """
        communities = memory.navigator.detect_communities(min_size=min_size)
        return [c.__dict__ for c in communities]

    @annotated_tool(server, **READ_ONLY)
    def memo_graph_trace(
        memory_id: str | None = None,
        code: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Trace evidence from one memory to code, or one code reference to memories.

        Provide exactly one of ``memory_id`` or ``code``. Code accepts a stable
        codegraph URI, symbol id, qualified name, or file path.
        """
        return memory.graph_trace(memory_id=memory_id, code=code, limit=limit)

    @annotated_tool(server, **READ_ONLY)
    def memo_graph_discover(
        min_community_size: int = 4,
        min_bridge_side: int = 2,
        max_communities: int = 5,
        max_bridges: int = 5,
        include_code: bool = True,
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
    ) -> dict[str, Any]:
        """Export the entity graph for visualization.

        Returns graph data in the specified format. Use with external tools
        like Graphviz (dot format) or web visualization libraries (JSON format).

        Args:
            format: Either "dot" for Graphviz DOT format or "json" for web UI.
            include_memories: If True and format is "json", include memory IDs in edge data.
        """
        if format == "dot":
            dot = memory.navigator.export_graphviz()
            return {"format": "dot", "content": dot}
        else:
            data = memory.navigator.export_json(include_memories=include_memories)
            return {"format": "json", "data": data}
