"""Spreading-activation associative recall — pure, local, no I/O.

Given the recall seeds (the hybrid top-K), walk one or two hops over the
entity-memory graph and the codegraph symbol graph (joined by name) and return
the most-activated *other* memories. Stateless: all graph access is injected so
the engine is hermetically testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# Activation weights per hop distance (index 0 = direct neighbor).
_HOP_WEIGHT = (1.0, 0.5, 0.25)
_FANOUT_TOKENS = 50  # cap neighbor tokens explored per hop
_FANOUT_MEMS = 200  # cap candidate memories scored per token
_ENTITY_HOP_MEMS = 30  # memories scanned when discovering entity neighbours
_HUB_DF = 50  # a token in more memories than this is a hub: no expansion, low signal


@dataclass(frozen=True)
class AssociativeHit:
    id: str
    via: str
    activation: float


def _rarity(df: int) -> float:
    """Inverse-document-frequency weight: hub tokens (large df) carry no signal."""
    return 1.0 if df <= 1 else min(1.0, 1.0 / math.log1p(df))


def _seed_tokens(seed_ids: list[str], store: Any) -> dict[str, float]:
    """Seed entity names -> base activation 1.0."""
    tokens: dict[str, float] = {}
    for mid in seed_ids:
        for ent in store.memory_entities(mid):
            name = (ent.get("name") or "").lower()
            if name:
                tokens[name] = 1.0
    return tokens


def _entity_neighbors(tok: str, store: Any) -> set[str]:
    """Entities that co-occur with `tok` in a memory (entity-graph 1-hop).

    Skipped for hub tokens (they connect to almost everything and add only
    noise), and bounded so a single hop stays cheap on the recall path.
    """
    mems = store.entity_memories(tok)
    if len(mems) > _HUB_DF:
        return set()
    out: set[str] = set()
    for mid in mems[:_ENTITY_HOP_MEMS]:
        for ent in store.memory_entities(mid):
            n = (ent.get("name") or "").lower()
            if n and n != tok:
                out.add(n)
                if len(out) >= _FANOUT_TOKENS:
                    return out
    return out


def _expand(
    seed_tokens: dict[str, float],
    store: Any,
    codegraph_adj: dict[str, set[str]] | None,
    hops: int,
) -> dict[str, float]:
    """Expand seed tokens outward over codegraph + entity graph.

    Returns token -> activation (max over the paths that reached it).
    """
    frontier = {t: 1.0 for t in seed_tokens}
    seen: dict[str, float] = dict(frontier)
    for depth in range(min(hops, len(_HOP_WEIGHT))):
        nxt: dict[str, float] = {}
        for tok in list(frontier)[:_FANOUT_TOKENS]:
            neighbors = (codegraph_adj or {}).get(tok, set()) | _entity_neighbors(tok, store)
            w = _HOP_WEIGHT[depth]
            for nb in neighbors:
                if nb not in seen or seen[nb] < w:
                    nxt[nb] = w
                    seen[nb] = w
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

    tok_act = _expand(seed_tokens, store, codegraph_adj, hops)

    # Map activated tokens -> candidate memories. Each token is weighted by its
    # rarity (IDF): a hub entity like "memo" in hundreds of memories carries
    # almost no signal; a rare entity or code symbol is a strong, specific link.
    # Scores ACCUMULATE across distinct connecting tokens (overlap), so a memory
    # reached through several specific links ranks above one reached through one.
    cand: dict[str, list[Any]] = {}  # mid -> [score, via, via_weight]
    for tok, act in tok_act.items():
        mems = store.entity_memories(tok)
        weighted = act * _rarity(len(mems))
        for mid in mems[:_FANOUT_MEMS]:
            if mid in exclude_ids:
                continue
            c = cand.get(mid)
            if c is None:
                cand[mid] = [weighted, tok, weighted]
            else:
                c[0] += weighted
                if weighted > c[2]:
                    c[1], c[2] = tok, weighted

    # Co-recall boost: memories historically recalled alongside the seeds.
    for seed in seed_ids:
        for mid, n in store.co_recall_counts(seed, list(cand)).items():
            if mid in cand:
                cand[mid][0] += 0.1 * float(n)

    # Gate on the strongest single connection (via_weight): a candidate must have
    # at least one rare/specific link, not merely many hub links that sum past the
    # floor. Rank by the accumulated overlap score.
    hits = [
        AssociativeHit(id=mid, via=via, activation=score)
        for mid, (score, via, via_w) in cand.items()
        if via_w >= min_activation
    ]
    hits.sort(key=lambda h: h.activation, reverse=True)
    return hits[:limit]
