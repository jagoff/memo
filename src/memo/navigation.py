"""Graph-based memory navigation — path finding, weighted community detection, visualization.

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

import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from memo import codegraph_loader
from memo.flags import flag_bool

_log = logging.getLogger(__name__)

# Exemplar bridging-memory ids kept per neighbour when serialising for MCP.
MAX_BRIDGE_IDS = 5


@dataclass(frozen=True)
class EntityPath:
    """A path between two entities in the graph."""

    source: str
    target: str
    path: list[str]  # List of entity names including source and target
    length: int
    intermediate_memories: list[str]  # Memory IDs that connect each step


@dataclass(frozen=True)
class EntityNeighbors:
    """Neighbors of an entity in the graph."""

    entity: str
    direct_neighbors: list[str]  # Entities directly connected
    neighbor_memories: dict[str, list[str]]  # entity -> memory IDs that connect
    degree: int

    def to_bounded_dict(self, *, max_bridge_ids: int = MAX_BRIDGE_IDS) -> dict[str, Any]:
        """JSON payload that keeps link strength without dumping every id.

        On a hub entity `neighbor_memories` holds hundreds of bridging ids per
        neighbour, so serialising it whole makes the "cheap" graph traversal
        cost more than a search. Callers get a few exemplar ids plus the true
        per-neighbour counts instead.
        """
        cap = max(0, max_bridge_ids)
        return {
            "entity": self.entity,
            "direct_neighbors": list(self.direct_neighbors),
            "neighbor_memories": {n: ids[:cap] for n, ids in self.neighbor_memories.items()},
            "neighbor_memory_counts": {n: len(ids) for n, ids in self.neighbor_memories.items()},
            "degree": self.degree,
        }


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
        *,
        use_codegraph: bool | None = None,
    ) -> EntityPath | None:
        """Find shortest path between two entities using BFS.

        Args:
            source: Source entity name (lowercased).
            target: Target entity name (lowercased).
            max_length: Maximum path length to search.
            use_codegraph: Override the codegraph-merge flag (None = read flag;
                False = entity-only memory graph).

        Returns:
            EntityPath if a path exists, None otherwise.
        """
        source = source.lower().strip()
        target = target.lower().strip()

        # Build the adjacency once, then BFS. compute_centrality reuses _bfs with
        # a pre-built adjacency so it does not rebuild the graph O(N^2) times.
        adj = self._build_adjacency_list(use_codegraph=use_codegraph)
        return self._bfs(adj, source, target, max_length)

    def _bfs(
        self,
        adj: dict[str, set[tuple[str, str]]],
        source: str,
        target: str,
        max_length: int,
    ) -> EntityPath | None:
        """Shortest path between two entities over a pre-built adjacency list."""
        if source == target:
            return EntityPath(
                source=source,
                target=target,
                path=[source],
                length=0,
                intermediate_memories=[],
            )

        if source not in adj or target not in adj:
            return None

        queue: deque[tuple[str, list[str], list[str]]] = deque(
            [(source, [source], [])],
        )  # (current, path, memory_ids)
        visited = {source}

        while queue:
            current, path, memory_ids = queue.popleft()

            if current == target:
                return EntityPath(
                    source=source,
                    target=target,
                    path=path,
                    length=len(path) - 1,
                    intermediate_memories=memory_ids,
                )

            if len(path) > max_length:
                continue

            for neighbor, mem_id in adj[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, [*path, neighbor], [*memory_ids, mem_id]))

        return None

    def _build_adjacency_list(
        self, *, use_codegraph: bool | None = None
    ) -> dict[str, set[tuple[str, str]]]:
        """Build adjacency list from entity-memory graph.

        Returns: entity -> set of (neighbor_entity, memory_id)

        ``use_codegraph`` overrides the MEMO_GRAPH_USE_CODEGRAPH flag per call
        (None = read the flag). Lets a caller (e.g. community synthesis) force the
        entity-only graph without mutating process-global env.
        """
        adj: dict[str, set[tuple[str, str]]] = defaultdict(set)

        # For each memory, connect all entities that mention it
        # This is O(N * E) where N=memories, E=entities per memory
        # Acceptable for corpora with <10k memories
        all_entities = self.graph.top_entities(limit=10000)

        entity_to_memories: dict[str, list[str]] = defaultdict(list)
        for ent in all_entities:
            name = ent["name"].lower()
            memory_ids = self.graph.entity_memories(name)
            for mid in memory_ids:
                entity_to_memories[name].append(mid)

        # Connect entities that share memories
        memory_to_entities: dict[str, list[str]] = defaultdict(list)
        for ent_name, mem_ids in entity_to_memories.items():
            for mid in mem_ids:
                memory_to_entities[mid].append(ent_name)

        for mid, entities in memory_to_entities.items():
            for i, e1 in enumerate(entities):
                for e2 in entities[i + 1 :]:
                    adj[e1].add((e2, mid))
                    adj[e2].add((e1, mid))

        # Fold the codegraph code graph in as a primary layer (gated). One merge
        # point lights up every navigator op — path, neighbors, communities,
        # centrality, export — so they leverage code structure, not just the
        # entity-memory graph. Degrades silently if the index is absent or off.
        _merge_cg = (
            flag_bool("MEMO_GRAPH_USE_CODEGRAPH") if use_codegraph is None else use_codegraph
        )
        if _merge_cg:
            try:
                cg_adj, _ = codegraph_loader.load()
                for node, neighbors in cg_adj.items():
                    for nb in neighbors:
                        adj[node].add((nb, "(codegraph)"))
            except Exception as e:
                _log.debug("codegraph merge skipped: %s", e)

        return adj

    def get_neighbors(
        self, entity: str, max_neighbors: int = 50, *, use_codegraph: bool | None = None
    ) -> EntityNeighbors:
        """Get direct neighbors of an entity.

        Args:
            entity: Entity name (lowercased).
            max_neighbors: Maximum neighbors to return.
            use_codegraph: Override the codegraph-merge flag (None = read flag;
                False = entity-only memory graph).

        Returns:
            EntityNeighbors with direct connections and shared memories.
        """
        entity = entity.lower().strip()
        adj = self._build_adjacency_list(use_codegraph=use_codegraph)

        if entity not in adj:
            return EntityNeighbors(
                entity=entity,
                direct_neighbors=[],
                neighbor_memories={},
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
            neighbor_memories={n: mems for n, mems in sorted_neighbors},
            degree=len(adj[entity]),
        )

    def _weighted_adjacency(
        self, *, use_codegraph: bool | None = None
    ) -> dict[str, dict[str, float]]:
        """entity -> {neighbor -> weight}, read from materialized entity_edges.

        Falls back to deriving unweighted (weight 1) from entity_memory when the
        edge table is empty (e.g. before the first rebuild). Optionally merges the
        codegraph layer at weight 1.
        """
        adj: dict[str, dict[str, float]] = defaultdict(dict)
        edges: list[tuple[str, str, float]] = []
        try:
            edges = self.graph.all_weighted_edges()
        except Exception as e:  # pragma: no cover - defensive
            _log.debug("all_weighted_edges failed: %s", e)
        if edges:
            for a, b, w in edges:
                adj[a][b] = w
                adj[b][a] = w
        else:
            base = self._build_adjacency_list(use_codegraph=False)
            for ent, nbrs in base.items():
                for nb, _mid in nbrs:
                    adj[ent][nb] = adj[ent].get(nb, 0.0) + 1.0

        _merge_cg = (
            flag_bool("MEMO_GRAPH_USE_CODEGRAPH") if use_codegraph is None else use_codegraph
        )
        if _merge_cg:
            try:
                cg_adj, _ = codegraph_loader.load()
                for node, neighbors in cg_adj.items():
                    for nb in neighbors:
                        adj[node][nb] = adj[node].get(nb, 0.0) + 1.0
            except Exception as e:
                _log.debug("codegraph merge skipped: %s", e)
        return adj

    def weighted_path(
        self, a: str, b: str, max_length: int = 5, *, use_codegraph: bool | None = None
    ) -> dict[str, Any] | None:
        """Shortest path between two entities as evidence, with per-edge weights.

        BFS for the fewest-hop path over the materialized weighted adjacency
        (``_weighted_adjacency``), then reads each traversed edge's weight.

        Returns ``{"path": [entity, ...], "edges": [{"from", "to", "weight"}, ...]}``
        or ``None`` when either endpoint is absent or no path exists within
        ``max_length`` hops.
        """
        a = a.lower().strip()
        b = b.lower().strip()

        adj = self._weighted_adjacency(use_codegraph=use_codegraph)
        if a not in adj or b not in adj:
            return None  # absent endpoint (incl. a self-query for a missing entity)
        if a == b:
            return {"path": [a], "edges": []}

        queue: deque[list[str]] = deque([[a]])
        visited = {a}
        while queue:
            path = queue.popleft()
            current = path[-1]
            if current == b:
                edges = [
                    {"from": path[i], "to": path[i + 1], "weight": adj[path[i]][path[i + 1]]}
                    for i in range(len(path) - 1)
                ]
                return {"path": path, "edges": edges}
            if len(path) - 1 >= max_length:
                continue
            for neighbor in adj[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append([*path, neighbor])

        return None

    def why_connected(
        self,
        source: str,
        target: str,
        max_length: int = 5,
        *,
        use_codegraph: bool | None = None,
    ) -> dict[str, Any]:
        """Explain an entity connection with weighted edges and memory evidence."""
        source_key = source.lower().strip()
        target_key = target.lower().strip()
        weighted = self.weighted_path(
            source_key,
            target_key,
            max_length=max_length,
            use_codegraph=use_codegraph,
        )
        if weighted is None:
            return {
                "source": source_key,
                "target": target_key,
                "path": [],
                "edges": [],
                "evidence_memory_ids": [],
            }

        path = self.find_shortest_path(
            source_key,
            target_key,
            max_length=max_length,
            use_codegraph=use_codegraph,
        )
        evidence: list[str] = []
        if path is not None:
            evidence = [mid for mid in path.intermediate_memories if mid and mid != "(codegraph)"]

        edges = []
        for idx, edge in enumerate(weighted["edges"]):
            memory_id = (
                path.intermediate_memories[idx]
                if path and idx < len(path.intermediate_memories)
                else ""
            )
            edges.append({**edge, "memory_id": memory_id})

        return {
            "source": source_key,
            "target": target_key,
            "path": weighted["path"],
            "length": max(0, len(weighted["path"]) - 1),
            "edges": edges,
            "evidence_memory_ids": evidence,
        }

    def detect_communities(
        self, min_size: int = 2, *, use_codegraph: bool | None = None
    ) -> list[Community]:
        """Detect communities via deterministic weighted label propagation.

        Args:
            min_size: Minimum community size to include.
            use_codegraph: Override the codegraph-merge flag (None = read flag).

        Returns:
            List of communities sorted by size descending.
        """
        from memo.graph_communities import degree_normalized, label_propagation

        adj = self._weighted_adjacency(use_codegraph=use_codegraph)
        # Down-weight hub votes so a ubiquitous entity does not fuse the graph
        # into one blob; representative selection below still uses raw degree.
        labels = label_propagation(degree_normalized(adj))

        groups: dict[int, list[str]] = defaultdict(list)
        for node, lb in labels.items():
            groups[lb].append(node)

        communities: list[Community] = []
        community_id = 0
        for members in groups.values():
            if len(members) < min_size:
                continue
            degrees = {e: sum((adj.get(e) or {}).values()) for e in members}
            representative = max(members, key=lambda e: degrees[e])
            communities.append(
                Community(
                    id=community_id,
                    entities=sorted(members),
                    size=len(members),
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
                path = self._bfs(adj, source, target, max_length=4)
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

    def export_json(self, include_memories: bool = False) -> dict[str, Any]:
        """Export graph to JSON format for web visualization.

        Args:
            include_memories: If True, include memory IDs in edge data.

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
                    if include_memories:
                        edge_data["memory_id"] = mem_id
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
