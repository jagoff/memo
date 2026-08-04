"""Graph-aware compaction of memo_context_pack hits: drop hits that share a
rare entity with a higher-ranked hit before they ever reach a pack bucket.
Thin adapter over the chat pipeline's already-tested, IDF-weighted
compact_by_entity_overlap — see
docs/superpowers/specs/2026-08-04-context-graph-compact-design.md for why
this wraps rather than duplicates that algorithm.
"""

from __future__ import annotations

from typing import Any

from memo.chat.graph_compact import compact_by_entity_overlap


def compact_hits_by_entity_overlap(
    hits: list[Any],
    memory: Any,
    *,
    min_idf_overlap: float,
    min_group_size: int = 2,
) -> list[Any]:
    """Drop `memory.search()` hit objects that are IDF-overlap duplicates of
    a higher-ranked hit, before they occupy a context-pack bucket slot.

    Adapts each hit to the dict shape `compact_by_entity_overlap` expects
    (carrying a synthetic `_hit_index` so surviving rows map back to the
    original hit objects without relying on `id` uniqueness), delegates the
    overlap/IDF grouping to it unchanged, then filters the original `hits`
    list — preserving its original rank order, not the delegate's internal
    resort.

    Fail-open: any exception (graph lookup, IDF lookup, or adaptation)
    returns `hits` unchanged — this must never block a tool response.
    """
    if len(hits) < min_group_size:
        return list(hits)
    try:
        dict_hits = []
        for i, h in enumerate(hits):
            score = getattr(h, "score", None)
            dict_hits.append(
                {
                    "id": str(getattr(h, "id", "") or ""),
                    "title": str(getattr(h, "title", "") or ""),
                    "score": 0.0 if score is None else score,
                    "_hit_index": i,
                }
            )
        compacted = compact_by_entity_overlap(
            dict_hits, memory, min_idf_overlap=min_idf_overlap, min_group_size=min_group_size
        )
        keep = {d["_hit_index"] for d in compacted}
    except Exception:
        return list(hits)
    return [h for i, h in enumerate(hits) if i in keep]


__all__ = ["compact_hits_by_entity_overlap"]
