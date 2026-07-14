from __future__ import annotations

from typing import Any

from memo.graph_reason import format_graph_reason


def build_search_explanations(
    hits: list[Any], trace: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Build honest per-hit explanations from the current search trace.

    The existing trace is stage-oriented, not per-candidate. This helper does
    not guess hidden leg scores; it reports available stage evidence and marks
    unavailable leg-level details explicitly.
    """

    candidate = _stage(trace, "candidate_generation")
    stages = [str(row.get("stage") or "") for row in trace if isinstance(row, dict)]
    out: dict[str, dict[str, Any]] = {}
    mode = str(candidate.get("mode") or "")
    for rank, hit in enumerate(hits, 1):
        hit_id = str(getattr(hit, "id", "") or "")
        score = getattr(hit, "score", None)
        extra = getattr(hit, "extra", None)
        if not isinstance(extra, dict):
            extra = {}
        graph_reason = extra.get("graph_reason")
        why = _why(candidate, stages)
        legs: dict[str, Any] = {
            "vec": _leg(candidate, "vec_count", mode == "vec"),
            "bm25": _leg(candidate, "bm25_count", mode == "bm25"),
            "exact": _leg(candidate, "exact_count", mode == "exact"),
            "graph": _leg(candidate, "graph_count", False),
            "recency": {"present": "recency_decay" in stages, "detail_available": False},
            "quality": {"present": "quality_rerank" in stages, "detail_available": False},
        }
        if isinstance(graph_reason, dict):
            legs["graph_reason"] = graph_reason
            why.append(format_graph_reason(graph_reason))
        out[hit_id] = {
            "rank": rank,
            "id": hit_id,
            "final_score": score,
            "mode": mode,
            "legs": legs,
            "stages": stages,
            "why": why,
        }
    return out


def _stage(trace: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for row in trace:
        if isinstance(row, dict) and row.get("stage") == name:
            return row
    return {}


def _leg(candidate: dict[str, Any], count_key: str, mode_match: bool) -> dict[str, Any]:
    if count_key in candidate:
        count = int(candidate.get(count_key) or 0)
        return {"present": count > 0, "candidate_count": count, "detail_available": False}
    return {
        "present": bool(mode_match and candidate.get("output_count")),
        "detail_available": False,
    }


def _why(candidate: dict[str, Any], stages: list[str]) -> list[str]:
    why: list[str] = []
    if (candidate.get("vec_count") or 0) > 0 or candidate.get("mode") == "vec":
        why.append("semantic candidates were considered")
    if (candidate.get("bm25_count") or 0) > 0 or candidate.get("mode") == "bm25":
        why.append("lexical BM25 candidates were considered")
    if (candidate.get("exact_count") or 0) > 0 or candidate.get("mode") == "exact":
        why.append("exact lexical candidates were considered")
    if (candidate.get("graph_count") or 0) > 0:
        why.append("graph candidates were considered")
    if "rerank" in stages:
        why.append("cross-encoder reranking ran")
    if "quality_rerank" in stages:
        why.append("quality reranking ran")
    if "recency_decay" in stages:
        why.append("recency decay adjusted scores")
    if not why:
        why.append("ranked by the active search pipeline")
    return why
