"""Graph-proximity recall boost — Phase 2's ``graph_boost`` seam.

A pure, dependency-free reranker that nudges recall candidates up when their
entities sit one hop from the query's entities in the materialized entity graph
(``GraphStore.entity_edges``). It plugs into ``recall_logic.rank_hits`` via the
``graph_boost`` callable seam, runs BEFORE the similarity gate, and does **no**
embedding / MLX work — only cheap graph lookups — so it respects the recall-hook
5s budget. Default OFF (``MEMO_RECALL_GRAPH_PROXIMITY``); identity when off.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

__all__ = ["extract_query_entities", "graph_boost_factory"]

_TOKEN_RE = re.compile(r"[a-záéíóúüñ0-9]+", re.IGNORECASE)


def extract_query_entities(prompt: str, graph: Any) -> list[str]:
    """Query entities for the graph-proximity boost.

    Union of (a) the proper-noun/acronym/quoted regex (``extract_entities``) and
    (b) graph-vocabulary matches: lowercase unigrams + bigrams of the prompt that
    are actual entity names in the graph. The vocabulary match lets the boost
    fire on natural lowercase prompts the regex misses ("how does the recall hook
    budget work") while admitting ONLY terms that exist in the graph — so no
    stopword noise leaks in. Falls back to the regex alone if the graph yields no
    vocabulary.
    """
    from memo.entity_extractor import extract_entities

    out = list(extract_entities(prompt))
    seen = {e.lower() for e in out}

    names: set[str] = set()
    with contextlib.suppress(Exception):
        names = graph.entity_names()
    if not names:
        return out

    toks = [t for t in _TOKEN_RE.findall(prompt.lower()) if len(t) >= 3]
    cands = set(toks)
    for i in range(len(toks) - 1):
        cands.add(f"{toks[i]} {toks[i + 1]}")
    for c in cands:
        if c in names and c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _identity(hits: list[Any]) -> list[Any]:
    return hits


def graph_boost_factory(
    graph: Any,
    query_entities: Sequence[str],
    *,
    weight: float,
) -> Callable[[list[Any]], list[Any]]:
    """Build the ``graph_boost`` callable for ``rank_hits``.

    For each hit, ``proximity`` is the sum over the hit's entity names of the
    edge weight connecting that entity to *any* query entity (1 hop; the max
    edge weight when several query entities reach the same hit entity). The hit
    score is boosted by ``weight * proximity`` and the list re-sorted desc.

    Degrades to the identity transform (returns the list unchanged) when
    ``weight <= 0``, there are no query entities, or the graph yields no edges
    reachable from the query entities.
    """
    if weight <= 0 or not query_entities:
        return _identity

    # neighbor_name(lower) -> max edge weight reachable from any query entity (1 hop)
    proximity_weights: dict[str, float] = {}
    for qe in query_entities:
        neighbors: dict[str, float] = {}
        with contextlib.suppress(Exception):
            neighbors = graph.weighted_neighbors(qe)
        for nm, w in (neighbors or {}).items():
            key = str(nm).strip().lower()
            edge_w = float(w)
            if edge_w > proximity_weights.get(key, 0.0):
                proximity_weights[key] = edge_w

    if not proximity_weights:
        return _identity

    def _boost(hits: list[Any]) -> list[Any]:
        out: list[Any] = []
        for h in hits:
            if h.score is None:
                out.append(h)
                continue
            ents: list[dict[str, Any]] = []
            with contextlib.suppress(Exception):
                ents = graph.memory_entities(h.id)
            proximity = 0.0
            for e in ents or []:
                nm = str(e.get("name", "")).strip().lower()
                proximity += proximity_weights.get(nm, 0.0)
            if proximity > 0:
                out.append(replace(h, score=h.score + weight * proximity))
            else:
                out.append(h)
        out.sort(key=lambda h: h.score or 0.0, reverse=True)
        return out

    return _boost
