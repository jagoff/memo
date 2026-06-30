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


def _components_without(
    neighbors: dict[str, set[str]], excluded: str
) -> list[list[str]]:
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
    adjacency: dict[str, dict[str, float]], *, min_side: int = 2
) -> list[dict[str, Any]]:
    """Find articulation bridges.

    Returns ``[{"bridge": str, "left": list[str], "right": list[str]}]`` — one
    entry per bridging entity, where ``left``/``right`` are the two largest
    qualifying components its neighbours fall into. Deterministic.
    """
    neighbors = _symmetric_neighbors(adjacency)
    out: list[dict[str, Any]] = []
    for node in sorted(neighbors):
        nbr_set = neighbors.get(node, set())
        if len(nbr_set) < 2:
            continue
        components = _components_without(neighbors, node)
        # Keep only components that contain a neighbour of ``node`` and are big
        # enough to be a meaningful side.
        qualifying = [
            c
            for c in components
            if len(c) >= min_side and any(x in nbr_set for x in c)
        ]
        if len(qualifying) < 2:
            continue
        qualifying.sort(key=lambda c: (-len(c), c[0]))
        left, right = qualifying[0], qualifying[1]
        out.append({"bridge": node, "left": left, "right": right})
    return out
