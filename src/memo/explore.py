"""Knowledge exploration (spec 4) — zoom into one entity over the merged graph.

Given an entity (or code symbol) name, return a single rich view: how connected
it is, what it neighbours (entity + codegraph, with how many memories bridge
each link), and the memories that mention it. Reuses the GraphNavigator and the
entity-memory store — no new state.
"""

from __future__ import annotations

from typing import Any


def explore_entity(
    memory: Any,
    entity: str,
    *,
    max_neighbors: int = 8,
    max_memories: int = 8,
) -> dict[str, Any]:
    """Return a neighbourhood view of ``entity``.

    Shape::

        {
          "entity": str, "degree": int,
          "neighbors": [{"name": str, "shared": int}, ...],
          "memories": [{"id": str, "title": str}, ...],
        }
    """
    entity = entity.lower().strip()
    nav = memory.navigator
    nb = nav.get_neighbors(entity, max_neighbors=max_neighbors)

    # "shared" = how many MEMORIES bridge this neighbour. Code-graph edges are
    # stored with a "(codegraph)" placeholder instead of a memory id; exclude it
    # so a pure code-symbol link doesn't read as a shared memory.
    neighbors = [
        {
            "name": n,
            "shared": sum(1 for m in nb.neighbor_memories.get(n, []) if m != "(codegraph)"),
        }
        for n in nb.direct_neighbors[:max_neighbors]
    ]

    memories: list[dict[str, str]] = []
    try:
        for mid in memory.graph.entity_memories(entity)[:max_memories]:
            rec = memory.get(mid)
            if rec is not None:
                memories.append({"id": mid, "title": getattr(rec, "title", mid) or mid})
    except Exception:  # noqa: S110 — best-effort enrichment, never sink the view
        pass

    return {
        "entity": entity,
        "degree": nb.degree,
        "neighbors": neighbors,
        "memories": memories,
    }
