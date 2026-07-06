"""Memory-to-memory graph — topological distance and derived-from relationships.

This module provides a graph structure for computing distances between memories
in the knowledge graph, useful for reranking based on derivation distance.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memo.memory.record import MemoryRecord


class Graph:
    """Graph of memories with derived_from edges.

    Supports BFS distance computation to find the shortest path from any memory
    to the nearest FACT-type memory (T2 memory).
    """

    def __init__(self) -> None:
        """Initialize an empty graph."""
        self.edges: dict[str, list[tuple[str, float]]] = {}
        self.memory_map: dict[str, MemoryRecord] = {}

    def add_memory(self, memory_record: MemoryRecord) -> None:
        """Add a memory record to the graph.

        Args:
            memory_record: The MemoryRecord to add.
        """
        self.memory_map[memory_record.id] = memory_record
        if memory_record.id not in self.edges:
            self.edges[memory_record.id] = []

    def add_edge(self, from_id: str, to_id: str, weight: float) -> None:
        """Add a directed edge from one memory to another.

        Args:
            from_id: Source memory ID.
            to_id: Target memory ID.
            weight: Edge weight.
        """
        if from_id not in self.edges:
            self.edges[from_id] = []
        self.edges[from_id].append((to_id, weight))

    def distance_to_nearest_fact(self, memory_id: str) -> int:
        """Compute shortest path distance from memory_id to any T2 (FACT) memory.

        Uses BFS. Returns 0 if memory_id itself is T2 (type="fact").
        Returns 999 if unreachable.

        Args:
            memory_id: The source memory ID.

        Returns:
            Shortest path distance to nearest FACT memory, or 999 if unreachable.
        """
        # Check if the source memory exists and is itself a fact
        if memory_id not in self.memory_map:
            return 999

        mem = self.memory_map[memory_id]
        if mem.type == "fact":
            return 0

        # BFS
        queue: deque[tuple[str, int]] = deque([(memory_id, 0)])
        visited = {memory_id}
        max_depth = 20  # Prevent infinite loops in cyclic graphs

        while queue:
            current_id, dist = queue.popleft()

            if dist > max_depth:
                break

            # Get outgoing edges from current_id
            if current_id in self.edges:
                for target_id, _ in self.edges[current_id]:
                    if target_id not in visited:
                        visited.add(target_id)

                        # Check if target is a fact
                        if target_id in self.memory_map:
                            mem = self.memory_map[target_id]
                            if mem.type == "fact":
                                return dist + 1

                        queue.append((target_id, dist + 1))

        return 999


__all__ = ["Graph"]
