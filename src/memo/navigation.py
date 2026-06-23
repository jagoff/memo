"""EXPERIMENTAL — not covered by the test suite, not exposed via MCP. API may change without notice.

Graph-based memory navigation — path finding, community detection, visualization.

Extends the basic entity graph in graph.py with:
- Shortest path finding between entities (BFS)
- Community detection (connected components, centrality)
- Graph visualization export (Graphviz DOT, JSON for web UI)
- Entity relationship queries (neighbors, paths, clusters)

## Path Finding

Uses BFS to find shortest paths between entities in the entity-memory graph.
Two entities are connected if they share a memory. Path length = number of
intermediate entities.

## Community Detection

Uses simple connected components for now. Future: Louvain/Leiden for
more sophisticated community detection.

## Visualization

Exports to Graphviz DOT format for rendering with dot/neato, and to JSON
for consumption by web visualization libraries (D3.js, Cytoscape.js).
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EntityPath:
    """A path between two entities in the graph."""

    source: str
    target: str
    path: list[str]  # List of entity names including source and target
    length: int
    intermediate_memorias: list[str]  # Memory IDs that connect each step


@dataclass(frozen=True)
class EntityNeighbors:
    """Neighbors of an entity in the graph."""

    entity: str
    direct_neighbors: list[str]  # Entities directly connected
    neighbor_memorias: dict[str, list[str]]  # entity -> memory IDs that connect
    degree: int


@dataclass(frozen=True)
class Community:
    """A community (cluster) of connected entities."""

    id: int
    entities: list[str]
    size: int
    representative_entity: str  # Most central entity in the community


@dataclass(frozen=True)
class CentralityScores:
    """Centrality metrics for entities."""

    degree: dict[str, int]  # Number of connections
    betweenness: dict[str, float]  # How often entity lies on shortest paths


class GraphNavigator:
    """Navigator for the entity-memory graph.

    Args:
        graph_store: The GraphStore instance to query.
    """

    def __init__(self, graph_store: Any) -> None:
        self.graph = graph_store

    def find_shortest_path(
        self,
        source: str,
        target: str,
        max_length: int = 5,
    ) -> EntityPath | None:
        """Find shortest path between two entities using BFS.

        Args:
            source: Source entity name (lowercased).
            target: Target entity name (lowercased).
            max_length: Maximum path length to search.

        Returns:
            EntityPath if a path exists, None otherwise.
        """
        source = source.lower().strip()
        target = target.lower().strip()

        if source == target:
            return EntityPath(
                source=source,
                target=target,
                path=[source],
                length=0,
                intermediate_memorias=[],
            )

        # Build adjacency list: entity -> set of (neighbor_entity, memoria_id)
        adj = self._build_adjacency_list()

        if source not in adj or target not in adj:
            return None

        # BFS
        queue: deque[tuple[str, list[str], list[str]]] = deque(
            [(source, [source], [])],
        )  # (current, path, memoria_ids)
        visited = {source}

        while queue:
            current, path, memoria_ids = queue.popleft()

            if len(path) > max_length:
                continue

            if current == target:
                return EntityPath(
                    source=source,
                    target=target,
                    path=path,
                    length=len(path) - 1,
                    intermediate_memorias=memoria_ids,
                )

            for neighbor, mem_id in adj[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, [*path, neighbor], [*memoria_ids, mem_id]))

        return None

    def _build_adjacency_list(self) -> dict[str, set[tuple[str, str]]]:
        """Build adjacency list from entity-memory graph.

        Returns: entity -> set of (neighbor_entity, memoria_id)
        """
        adj: dict[str, set[tuple[str, str]]] = defaultdict(set)

        # For each memory, connect all entities that mention it
        # This is O(N * E) where N=memories, E=entities per memory
        # Acceptable for corpora with <10k memories
        all_entities = self.graph.top_entities(limit=10000)

        entity_to_memorias: dict[str, list[str]] = defaultdict(list)
        for ent in all_entities:
            name = ent["name"].lower()
            memoria_ids = self.graph.entity_memorias(name)
            for mid in memoria_ids:
                entity_to_memorias[name].append(mid)

        # Connect entities that share memories
        memoria_to_entities: dict[str, list[str]] = defaultdict(list)
        for ent_name, mem_ids in entity_to_memorias.items():
            for mid in mem_ids:
                memoria_to_entities[mid].append(ent_name)

        for mid, entities in memoria_to_entities.items():
            for i, e1 in enumerate(entities):
                for e2 in entities[i + 1 :]:
                    adj[e1].add((e2, mid))
                    adj[e2].add((e1, mid))

        return adj

    def get_neighbors(self, entity: str, max_neighbors: int = 50) -> EntityNeighbors:
        """Get direct neighbors of an entity.

        Args:
            entity: Entity name (lowercased).
            max_neighbors: Maximum neighbors to return.

        Returns:
            EntityNeighbors with direct connections and shared memories.
        """
        entity = entity.lower().strip()
        adj = self._build_adjacency_list()

        if entity not in adj:
            return EntityNeighbors(
                entity=entity,
                direct_neighbors=[],
                neighbor_memorias={},
                degree=0,
            )

        # Group neighbors by memory
        neighbor_mems: dict[str, list[str]] = defaultdict(list)
        for neighbor, mem_id in adj[entity]:
            neighbor_mems[neighbor].append(mem_id)

        # Sort by number of shared memories
        sorted_neighbors = sorted(
            neighbor_mems.items(),
            key=lambda x: len(x[1]),
            reverse=True,
        )[:max_neighbors]

        return EntityNeighbors(
            entity=entity,
            direct_neighbors=[n for n, _ in sorted_neighbors],
            neighbor_memorias={n: mems for n, mems in sorted_neighbors},
            degree=len(adj[entity]),
        )

    def detect_communities(self, min_size: int = 2) -> list[Community]:
        """Detect connected components as communities.

        Args:
            min_size: Minimum community size to include.

        Returns:
            List of communities sorted by size descending.
        """
        adj = self._build_adjacency_list()

        visited: set[str] = set()
        communities: list[Community] = []
        community_id = 0

        for entity in adj:
            if entity in visited:
                continue

            # BFS to find connected component
            component = []
            queue = deque([entity])
            visited.add(entity)

            while queue:
                current = queue.popleft()
                component.append(current)

                for neighbor, _ in adj[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            if len(component) >= min_size:
                # Find representative (most central = highest degree)
                degrees = {e: len(adj[e]) for e in component}
                representative = max(component, key=lambda e: degrees[e])

                communities.append(
                    Community(
                        id=community_id,
                        entities=sorted(component),
                        size=len(component),
                        representative_entity=representative,
                    )
                )
                community_id += 1

        communities.sort(key=lambda c: c.size, reverse=True)
        return communities

    def compute_centrality(self) -> CentralityScores:
        """Compute degree and betweenness centrality for all entities.

        Returns:
            CentralityScores with degree and betweenness metrics.
        """
        adj = self._build_adjacency_list()

        # Degree centrality
        degree = {e: len(neighbors) for e, neighbors in adj.items()}

        # Betweenness centrality (simplified: count of shortest paths through entity)
        # Full betweenness is O(V * E) which is expensive, so we use an approximation
        betweenness: dict[str, float] = defaultdict(float)

        # Sample pairs and count paths through each entity
        entities = list(adj.keys())
        sample_size = min(len(entities), 100)  # Limit for performance

        for i in range(sample_size):
            for j in range(i + 1, sample_size):
                source, target = entities[i], entities[j]
                path = self.find_shortest_path(source, target, max_length=4)
                if path and path.length > 1:
                    # Count intermediate nodes
                    for intermediate in path.path[1:-1]:
                        betweenness[intermediate] += 1.0

        # Normalize
        max_betweenness = max(betweenness.values()) if betweenness else 1.0
        normalized_betweenness = {e: b / max_betweenness for e, b in betweenness.items()}

        return CentralityScores(
            degree=degree,
            betweenness=normalized_betweenness,
        )

    def export_graphviz(self, output_path: str | None = None) -> str:
        """Export graph to Graphviz DOT format.

        Args:
            output_path: Optional file path to write the DOT file.

        Returns:
            DOT format string.
        """
        adj = self._build_adjacency_list()
        lines = ["graph memo_entities {"]
        lines.append("  rankdir=LR;")
        lines.append("  node [shape=box, style=rounded];")

        # Add edges
        for entity, neighbors in adj.items():
            for neighbor, _ in neighbors:
                # Avoid duplicate edges (undirected graph)
                if entity < neighbor:
                    lines.append(f'  "{entity}" -- "{neighbor}";')

        lines.append("}")
        dot = "\n".join(lines)

        if output_path:
            from pathlib import Path

            Path(output_path).write_text(dot, encoding="utf-8")

        return dot

    def export_json(self, include_memorias: bool = False) -> dict[str, Any]:
        """Export graph to JSON format for web visualization.

        Args:
            include_memorias: If True, include memory IDs in edge data.

        Returns:
            Dict with nodes and edges suitable for D3.js/Cytoscape.js.
        """
        adj = self._build_adjacency_list()

        # Nodes
        nodes = []
        for entity in adj:
            nodes.append({"id": entity, "label": entity})

        # Edges
        edges = []
        seen_edges = set()
        for entity, neighbors in adj.items():
            for neighbor, mem_id in neighbors:
                # Avoid duplicate edges
                edge_key = tuple(sorted([entity, neighbor]))
                if edge_key not in seen_edges:
                    edge_data = {"source": entity, "target": neighbor}
                    if include_memorias:
                        edge_data["memoria_id"] = mem_id
                    edges.append(edge_data)
                    seen_edges.add(edge_key)

        return {"nodes": nodes, "edges": edges}


__all__ = [
    "CentralityScores",
    "Community",
    "EntityNeighbors",
    "EntityPath",
    "GraphNavigator",
]
