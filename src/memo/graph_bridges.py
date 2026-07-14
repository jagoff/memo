"""Deterministic articulation-bridge detection over a weighted entity graph.

A *bridge* is an articulation entity whose removal splits its neighbourhood
into two or more connected components, each of size >= ``min_side``. Removing
the candidate node and recomputing connected components, if the candidate's
neighbours fall into >=2 distinct components (each large enough), the node
*bridges* them. Pure — no DB, no memo imports, fully deterministic.
"""

from __future__ import annotations

from typing import Any

__all__ = ["find_bridges"]


def _symmetric_neighbors(
    adjacency: dict[str, dict[str, float]],
) -> dict[str, set[str]]:
    """Undirected neighbour sets over all nodes mentioned in ``adjacency``."""
    neighbors: dict[str, set[str]] = {n: set() for n in adjacency}
    for a, nbrs in adjacency.items():
        for b in nbrs:
            neighbors.setdefault(a, set()).add(b)
            neighbors.setdefault(b, set()).add(a)
    return neighbors


def _articulation_points(neighbors: dict[str, set[str]]) -> set[str]:
    """Articulation points via iterative Tarjan DFS — O(V+E), recursion-safe.

    A node is an articulation point iff removing it increases the number of
    connected components. Iterative (explicit stack) so a deep DFS over a large
    entity graph cannot blow Python's recursion limit.
    """
    disc: dict[str, int] = {}
    low: dict[str, int] = {}
    parent: dict[str, str | None] = {}
    aps: set[str] = set()
    timer = 0

    for root in sorted(neighbors):
        if root in disc:
            continue
        parent[root] = None
        disc[root] = low[root] = timer
        timer += 1
        root_children = 0
        stack: list[tuple[str, Any]] = [(root, iter(sorted(neighbors[root])))]
        while stack:
            u, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                stack.pop()
                p = parent[u]
                if p is not None:
                    low[p] = min(low[p], low[u])
                    if parent[p] is not None and low[u] >= disc[p]:
                        aps.add(p)
                continue
            v = nxt
            if v == parent[u]:
                continue
            if v in disc:
                low[u] = min(low[u], disc[v])
            else:
                parent[v] = u
                disc[v] = low[v] = timer
                timer += 1
                if u == root:
                    root_children += 1
                stack.append((v, iter(sorted(neighbors[v]))))
        if root_children >= 2:
            aps.add(root)
    return aps


def _components_without(neighbors: dict[str, set[str]], excluded: str) -> list[list[str]]:
    """Connected components of the graph with ``excluded`` removed (sorted)."""
    seen: set[str] = {excluded}
    components: list[list[str]] = []
    for start in sorted(neighbors):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp: list[str] = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in neighbors.get(cur, ()):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        components.append(sorted(comp))
    return components


def find_bridges(
    adjacency: dict[str, dict[str, float]], *, min_side: int = 2, max_side: int = 40
) -> list[dict[str, Any]]:
    """Find articulation bridges between two BOUNDED, meaningful clusters.

    Returns ``[{"bridge": str, "left": list[str], "right": list[str]}]`` — one
    entry per bridging entity, where ``left``/``right`` are the two largest
    qualifying components its neighbours fall into. A side qualifies only when
    its size is in ``[min_side, max_side]``: the upper bound EXCLUDES the global
    giant component, so a node that merely tethers a small cluster to the whole
    graph is not reported (that produced the degenerate "memo and X via Y"
    insights). Deterministic.

    Only articulation points are examined (Tarjan, O(V+E)), so the per-node
    component flood runs for the few cut vertices instead of every node.
    """
    neighbors = _symmetric_neighbors(adjacency)
    aps = _articulation_points(neighbors)
    out: list[dict[str, Any]] = []
    for node in sorted(aps):
        nbr_set = neighbors.get(node, set())
        if len(nbr_set) < 2:
            continue
        components = _components_without(neighbors, node)
        # A side must contain a neighbour of ``node`` and be a BOUNDED cluster
        # (>= min_side so it is meaningful, <= max_side so it is not the global
        # giant component masquerading as a "theme").
        qualifying = [
            c for c in components if min_side <= len(c) <= max_side and any(x in nbr_set for x in c)
        ]
        if len(qualifying) < 2:
            continue
        qualifying.sort(key=lambda c: (-len(c), c[0]))
        left, right = qualifying[0], qualifying[1]
        out.append({"bridge": node, "left": left, "right": right})
    return out
