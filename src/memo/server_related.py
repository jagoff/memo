"""MCP tool: memo_related — on-demand associative recall with the connecting path."""

from __future__ import annotations

from typing import Any

from memo.associative import associate
from memo.flags import flag_float


def related_for(memory: Any, query_or_id: str, hops: int = 2, limit: int = 5) -> list[dict]:
    """Resolve seed(s) for `query_or_id`, run associative expansion, return hits.

    If `query_or_id` matches a memory id it is the single seed; otherwise it is
    treated as a search query whose top hits seed the expansion.
    """
    seed = memory.get(query_or_id)
    if seed is not None:
        seed_ids = [seed.id]
    else:
        seed_ids = [r.id for r in memory.search(query_or_id, limit=5)]
    if not seed_ids:
        return []
    try:
        from memo import codegraph_loader

        cg = codegraph_loader.load()[0]
    except Exception:
        cg = None
    hits = associate(
        seed_ids,
        store=memory.graph,
        codegraph_adj=cg,
        hops=hops,
        limit=limit,
        exclude_ids=frozenset(seed_ids),
        min_activation=flag_float("MEMO_ASSOCIATIVE_MIN_ACTIVATION") or 0.0,
    )
    out = []
    for h in hits:
        rec = memory.get(h.id)
        out.append({
            "id": h.id,
            "title": getattr(rec, "title", h.id) if rec else h.id,
            "via": h.via,
            "activation": round(h.activation, 3),
        })
    return out


def register(server: Any, memory: Any) -> None:
    @server.tool()
    def memo_related(query_or_id: str, hops: int = 2, limit: int = 5) -> list[dict]:
        """Memories structurally connected (via the entity/code graph) to a memory or query."""
        return related_for(memory, query_or_id, hops=hops, limit=limit)
