"""Graph-aware source compaction for evidence_pack: collapse retrieval hits
that share a rare entity into one representative, citing the rest by id
instead of paying their full char/item cost. IDF-weighted so ubiquitous
entities never trigger a collapse — see
docs/superpowers/specs/2026-08-04-evidence-graph-compact-design.md. Adapts
memo.chat.graph_compact's algorithm to evidence_pack's attribute-based,
frozen-dataclass MemoryRecord hits (chat's `sources` are plain dicts); see
the spec's "dict-vs-object adaptation" section for why this is a standalone
module rather than a shared import.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any


def _score_of(hit: Any) -> float:
    value = getattr(hit, "score", None)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _entity_set(memory: Any, hit_id: str) -> set[str]:
    if not hit_id:
        return set()
    ents = memory.graph.memory_entities(hit_id)
    return {str(e.get("name") or "").strip().lower() for e in ents if e.get("name")}


def _idf_map(memory: Any, names: set[str]) -> dict[str, float]:
    if not names:
        return {}
    total = memory.graph.total_indexed_memories()
    if total <= 0:
        return {}
    dfs = memory.graph.entity_doc_freqs(sorted(names))
    return {name: max(0.0, math.log(total / df)) for name, df in dfs.items() if df > 0}


def _weighted_overlap(a: set[str], b: set[str], idf: dict[str, float]) -> float:
    union = a | b
    if not union:
        return 0.0
    union_weight = sum(idf.get(n, 0.0) for n in union)
    if union_weight <= 0:
        return 0.0
    shared_weight = sum(idf.get(n, 0.0) for n in (a & b))
    return shared_weight / union_weight


def _with_related_ids(hit: Any, related: list[tuple[str, str]]) -> Any:
    """Return a copy of `hit` with `related_ids` attached under
    `extra["provenance"]`, so `_build_items`'s `normalize_provenance`
    (memo.contracts) carries it into the final `EvidenceItem.provenance` with
    no change to the EvidenceItem/EvidencePack contract."""
    extra = dict(getattr(hit, "extra", None) or {})
    provenance = dict(extra.get("provenance") or {})
    provenance["related_ids"] = related
    extra["provenance"] = provenance
    return replace(hit, extra=extra)


def compact_by_entity_overlap(
    hits: list[Any],
    memory: Any,
    *,
    min_idf_overlap: float,
    min_group_size: int = 2,
) -> list[Any]:
    """Collapse hits whose IDF-weighted entity overlap clears
    ``min_idf_overlap`` into one representative per group (the
    highest-``score`` member), attaching ``related_ids`` (a list of
    ``(id, title)`` pairs for the absorbed hits) to the representative's
    ``extra["provenance"]``. Groups smaller than ``min_group_size`` are left
    as separate, unmodified hits.

    Fail-open: any exception (graph lookup, IDF computation, or the
    ``dataclasses.replace()`` used to attach ``related_ids``) returns
    ``hits`` unchanged — this must never block an ``evidence_pack`` response.
    """
    if len(hits) < min_group_size:
        return list(hits)
    try:
        entity_sets = {
            i: _entity_set(memory, str(getattr(h, "id", "") or "")) for i, h in enumerate(hits)
        }
        all_names: set[str] = set()
        for names in entity_sets.values():
            all_names.update(names)
        if not all_names:
            return list(hits)
        idf = _idf_map(memory, all_names)

        ordered = sorted(range(len(hits)), key=lambda i: _score_of(hits[i]), reverse=True)
        groups: list[list[int]] = []
        for idx in ordered:
            placed = False
            for group in groups:
                rep_idx = group[0]
                if (
                    _weighted_overlap(entity_sets[idx], entity_sets[rep_idx], idf)
                    >= min_idf_overlap
                ):
                    group.append(idx)
                    placed = True
                    break
            if not placed:
                groups.append([idx])

        out: list[Any] = []
        for group in groups:
            if len(group) >= min_group_size:
                related = [
                    (
                        str(getattr(hits[i], "id", "") or ""),
                        str(getattr(hits[i], "title", "") or ""),
                    )
                    for i in group[1:]
                ]
                out.append(_with_related_ids(hits[group[0]], related))
            else:
                out.extend(hits[i] for i in group)
        out.sort(key=_score_of, reverse=True)
        return out
    except Exception:
        return list(hits)


__all__ = ["compact_by_entity_overlap"]
