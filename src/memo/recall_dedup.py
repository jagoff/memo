"""Dedup / collapse / MMR primitives for the recall hot path.

Pure functions — no store access, no flag reads beyond what each docstring
says, no memo imports beyond typing. Extracted from `recall_logic.py` so the
5 s hook keeps a small, leaf module it can reason about in isolation; the
orchestrator re-exports these names unchanged (see `recall_logic.py`).
"""

from __future__ import annotations

from typing import Any


def _dedup_tokens(text: str) -> set[str]:
    import re

    return {t for t in re.findall(r"\w+", (text or "").lower()) if len(t) > 2}


def collapse_near_dups(relevant: list[Any], *, threshold: float) -> list[Any]:
    """Drop hits whose title+body token-Jaccard with a kept, higher-scored hit
    exceeds ``threshold``. Lexical only — safe for the 5s recall hook (no MLX)."""
    kept: list[Any] = []
    kept_sets: list[set[str]] = []
    for h in sorted(relevant, key=lambda x: x.score or 0.0, reverse=True):
        toks = _dedup_tokens(f"{h.title} {h.body or ''}")
        dup = False
        for ks in kept_sets:
            union = toks | ks
            if union and len(toks & ks) / len(union) >= threshold:
                dup = True
                break
        if not dup:
            kept.append(h)
            kept_sets.append(toks)
    # preserve the caller's original ordering among survivors
    survivors = {id(h) for h in kept}
    return [h for h in relevant if id(h) in survivors]


# ── Verified code citations (MEMO_RECALL_CODE_REFS_ENABLED, default OFF) ─────
_CODE_REFS_PER_MEMORY_CAP = 2  # max '↳ code' lines per rendered memory
_CODE_REFS_PER_RENDER_CAP = 4  # max '↳ code' lines per render (token budget wins)


def _mmr_token_set(hit: Any) -> frozenset[str]:
    text = f"{getattr(hit, 'title', '') or ''} {getattr(hit, 'body', '') or ''}"
    return frozenset(text.lower().split())


def _apply_mmr(
    hits: list[Any],
    lam: float,
    explain: dict[str, dict[str, Any]] | None = None,
) -> list[Any]:
    """Maximal-marginal-relevance re-ORDERING of the final gated pool.

    Greedy selection: score' = lam*relevance - (1-lam)*max_sim_to_already_
    selected, where relevance is the (boosted) hit score and similarity is
    token-set Jaccard over title+body — doc-doc vectors are not available
    here, and Jaccard needs no embed calls, no store round-trips: O(K^2)
    over the candidate pool only, hook-budget safe. Hit scores are NOT
    mutated — only the order changes. The first pick is always the
    max-relevance hit, so skip-below floors on the top hit are unaffected."""
    if len(hits) <= 1:
        return list(hits)
    tokens = [_mmr_token_set(h) for h in hits]

    def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    remaining = list(range(len(hits)))
    selected: list[int] = []
    while remaining:
        best_i = remaining[0]
        best_score = float("-inf")
        best_pen = 0.0
        for i in remaining:
            rel = hits[i].score or 0.0
            pen = max((_jaccard(tokens[i], tokens[j]) for j in selected), default=0.0)
            score = lam * rel - (1.0 - lam) * pen
            if score > best_score:
                best_i, best_score, best_pen = i, score, pen
        remaining.remove(best_i)
        selected.append(best_i)
        if explain is not None:
            entry = explain.get(getattr(hits[best_i], "id", ""))
            if entry is not None:
                entry["mmr"] = {
                    "mmr_score": round(best_score, 6),
                    "max_sim_to_selected": round(best_pen, 6),
                }
    return [hits[i] for i in selected]


def _dedup_key(hit: Any) -> str:
    title = " ".join((getattr(hit, "title", "") or "").lower().split())
    body = " ".join((getattr(hit, "body", "") or "").lower().split())[:120]
    return f"{title}|{body}"


def _deduplicate_synthesis(hits: list[Any]) -> list[Any]:
    """Remove source memories that are already covered by a synthesis hit.

    A synthesis hit has extra.synthesis_sources = [id1, id2, ...].
    If a synthesis hit appears alongside its source memories, the sources
    are redundant — remove them.
    """
    covered_ids: set[str] = set()
    for h in hits:
        if getattr(h, "type", "") == "synthesis":
            sources = (getattr(h, "extra", None) or {}).get("synthesis_sources") or []
            covered_ids.update(sources)
    if not covered_ids:
        return list(hits)
    return [h for h in hits if h.id not in covered_ids]


def dedup_hits(hits: list[Any]) -> list[Any]:
    seen_ids: set[str] = set()
    seen_keys: set[str] = set()
    out: list[Any] = []
    for h in hits:
        hid = getattr(h, "id", None)
        key = _dedup_key(h)
        if hid in seen_ids or key in seen_keys:
            continue
        if hid is not None:
            seen_ids.add(hid)
        seen_keys.add(key)
        out.append(h)
    return out
