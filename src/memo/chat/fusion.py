"""RRF fusion and per-group score normalization (ported lean from synapse)."""

from __future__ import annotations

from typing import Any

DEFAULT_RRF_K = 60
_COMPRESSION_RATIO = 0.15


def source_dedup_key(item: dict[str, Any]) -> str:
    if item.get("locator"):
        return f"loc::{item['locator']}"
    if item.get("id"):
        return f"id::{item['id']}"
    if item.get("path"):
        return f"path::{item['path']}"
    return f"title::{str(item.get('title', '')).strip().lower()}"


def rrf_fuse(
    rankings: list[list[dict[str, Any]]],
    *,
    k: int = DEFAULT_RRF_K,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    rankings = [r for r in (rankings or []) if r]
    if not rankings:
        return []
    score_by_key: dict[str, float] = {}
    origins_by_key: dict[str, list[int]] = {}
    canonical_by_key: dict[str, dict[str, Any]] = {}
    best_rank_by_key: dict[str, int] = {}
    for list_idx, ranking in enumerate(rankings):
        for rank, item in enumerate(ranking, start=1):
            if not isinstance(item, dict):
                continue
            key = source_dedup_key(item)
            score_by_key[key] = score_by_key.get(key, 0.0) + 1.0 / (k + rank)
            origins_by_key.setdefault(key, []).append(list_idx)
            if key not in canonical_by_key:
                canonical_by_key[key] = item
                best_rank_by_key[key] = rank
            elif rank < best_rank_by_key[key]:
                best_rank_by_key[key] = rank
    fused = []
    for key, score in score_by_key.items():
        canonical = dict(canonical_by_key[key])
        canonical["rrf_score"] = round(score, 6)
        canonical["rrf_origins"] = list(origins_by_key[key])
        fused.append((score, best_rank_by_key[key], canonical))
    fused.sort(key=lambda t: (-t[0], t[1]))
    out = [item for _, _, item in fused]
    return out[: max(0, limit)] if limit is not None else out


def _group_of(item: dict[str, Any]) -> str:
    source = str(item.get("source") or "")
    if source == "memory":
        return "memory"
    if source in {"vault", "repo"} or item.get("type") == "repo":
        return "vault"
    return "other"


def normalize_scores(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = [dict(s) for s in sources]
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(out):
        groups.setdefault(_group_of(s), []).append(i)
    for label, indices in groups.items():
        scores = [float(out[i].get("score") or 0.0) for i in indices]
        lo, hi = min(scores), max(scores)
        span = hi - lo
        if len(indices) == 1 or hi <= lo or span < hi * _COMPRESSION_RATIO:
            for i in indices:
                out[i]["normalized_score"] = 0.5
                out[i]["score_group"] = label
            continue
        for i, s in zip(indices, scores, strict=False):
            out[i]["normalized_score"] = round((s - lo) / span, 6)
            out[i]["score_group"] = label
    return out
