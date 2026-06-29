"""Spreading-activation associative recall — pure, local, no I/O.

Given the recall seeds (the hybrid top-K), walk one or two hops over the
entity-memory graph and the codegraph symbol graph (joined by name) and return
the most-activated *other* memories. Stateless: all graph access is injected so
the engine is hermetically testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Activation weights per hop distance (index 0 = direct neighbor).
_HOP_WEIGHT = (1.0, 0.5, 0.25)
_FANOUT_TOKENS = 50  # cap neighbor tokens explored
_FANOUT_MEMS = 200  # cap candidate memories scored


@dataclass(frozen=True)
class AssociativeHit:
    id: str
    via: str
    activation: float


def _seed_tokens(seed_ids: list[str], store: Any) -> dict[str, float]:
    """Seed entity names -> base activation 1.0."""
    tokens: dict[str, float] = {}
    for mid in seed_ids:
        for ent in store.memory_entities(mid):
            name = (ent.get("name") or "").lower()
            if name:
                tokens[name] = 1.0
    return tokens


def _expand(tokens: dict[str, float], codegraph_adj: dict[str, set[str]] | None,
            hops: int) -> dict[str, tuple[float, str]]:
    """Expand seed tokens outward; return token -> (activation, via_seed_token)."""
    frontier = {t: (1.0, t) for t in tokens}
    seen: dict[str, tuple[float, str]] = dict(frontier)
    for depth in range(min(hops, len(_HOP_WEIGHT))):
        nxt: dict[str, tuple[float, str]] = {}
        for tok, (_act, via) in list(frontier.items())[:_FANOUT_TOKENS]:
            neighbors = (codegraph_adj or {}).get(tok, set())
            for nb in neighbors:
                w = _HOP_WEIGHT[depth]
                if nb not in seen or seen[nb][0] < w:
                    nxt[nb] = (w, via)
                    seen[nb] = (w, via)
        frontier = nxt
        if not frontier:
            break
    return seen


def associate(
    seed_ids: list[str],
    *,
    store: Any,
    codegraph_adj: dict[str, set[str]] | None,
    hops: int = 2,
    limit: int = 2,
    exclude_ids: frozenset[str] = frozenset(),
    min_activation: float = 0.0,
) -> list[AssociativeHit]:
    seed_tokens = _seed_tokens(seed_ids, store)
    if not seed_tokens:
        return []

    # Token activation map: seeds + code-graph expansion.
    tok_act = _expand(seed_tokens, codegraph_adj, hops)

    # Map activated tokens -> candidate memories, accumulating activation + via.
    cand: dict[str, tuple[float, str]] = {}
    for tok, (act, via) in tok_act.items():
        for mid in store.entity_memories(tok)[:_FANOUT_MEMS]:
            if mid in exclude_ids:
                continue
            prev = cand.get(mid)
            if prev is None or prev[0] < act:
                cand[mid] = (act, via)

    # Co-recall boost: memories historically recalled alongside the seeds.
    for seed in seed_ids:
        boosts = store.co_recall_counts(seed, list(cand.keys()))
        for mid, n in boosts.items():
            if mid in cand:
                act, via = cand[mid]
                cand[mid] = (act + 0.1 * float(n), via)

    hits = [
        AssociativeHit(id=mid, via=via, activation=act)
        for mid, (act, via) in cand.items()
        if act >= min_activation
    ]
    hits.sort(key=lambda h: h.activation, reverse=True)
    return hits[:limit]
