"""Deterministic weighted label propagation for entity-graph communities.

Replaces connected-components (which fuses a hub-heavy graph into one giant
blob). Each node starts in its own label; nodes adopt the highest-weight label
among neighbours; ties break to the smallest label id; iteration is in sorted
node order for full determinism. Pure — no DB, no memo imports.
"""

from __future__ import annotations

__all__ = ["degree_normalized", "label_propagation"]


def degree_normalized(
    adjacency: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Scale each neighbour's vote by 1/its weighted degree.

    A ubiquitous hub (e.g. an entity mentioned across the whole corpus) has a
    huge weighted degree, so each of its edges contributes negligibly — it no
    longer fuses tight local clusters into one giant community under label
    propagation. Tight, low-degree clusters dominate. Pure."""
    deg = {n: sum(nbrs.values()) for n, nbrs in adjacency.items()}
    return {
        n: {nb: w / (deg.get(nb) or 1.0) for nb, w in nbrs.items()} for n, nbrs in adjacency.items()
    }


def label_propagation(
    adjacency: dict[str, dict[str, float]],
    *,
    max_iters: int = 20,
) -> dict[str, int]:
    """Return {node: label_id}. Deterministic."""
    nodes = sorted(adjacency)
    labels: dict[str, int] = {n: i for i, n in enumerate(nodes)}

    for _ in range(max_iters):
        changed = False
        for n in nodes:
            nbrs = adjacency.get(n) or {}
            if not nbrs:
                continue
            tally: dict[int, float] = {}
            for nb, w in nbrs.items():
                lb = labels.get(nb)
                if lb is None:
                    continue
                tally[lb] = tally.get(lb, 0.0) + w
            if not tally:
                continue
            # Highest weight wins; tie -> smallest label id (deterministic).
            best = min((-v, lb) for lb, v in tally.items())[1]
            if labels[n] != best:
                labels[n] = best
                changed = True
        if not changed:
            break
    return labels
