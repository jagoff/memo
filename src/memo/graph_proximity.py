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
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

__all__ = ["extract_query_entities", "graph_boost_factory"]

_TOKEN_RE = re.compile(r"[a-záéíóúüñ0-9]+", re.IGNORECASE)


def _idf(df: float, n: int) -> float:
    """Inverse document frequency of an entity: ``log(N / df)``, clamped ≥ 0.

    An entity in every memory (``df == N``) yields 0 — it carries no
    discriminating signal (e.g. "memo", "synapse" on this corpus). A rare
    entity yields a large value. This is what turns the graph boost from a
    raw co-occurrence count (which promotes generic-entity junk) into a
    relevance-weighted signal.
    """
    if n <= 0 or df <= 0:
        return 0.0
    return max(0.0, math.log(n / df))


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
    min_idf: float = 0.0,
) -> Callable[[list[Any]], list[Any]]:
    """Build the ``graph_boost`` callable for ``rank_hits``.

    For each hit, ``proximity`` is the sum over the hit's entity names of the
    edge weight connecting that entity to *any* query entity (1 hop; the max
    edge weight when several query entities reach the same hit entity), each
    contribution scaled by the neighbor entity's **IDF** (rarity). The hit
    score is boosted by ``weight * proximity`` and the list re-sorted desc.

    IDF weighting is the fix for raw entity-overlap noise: a hit sharing a
    ubiquitous entity ("memo", "synapse") gets ~0 boost, while one reaching a
    *rare* entity gets a strong one — so the boost stops promoting generic junk
    over the true answer. When the graph exposes no corpus size (a stub, or an
    empty index) the IDF term degrades to 1.0, recovering the raw-weight
    behavior.

    ``min_idf`` gates the whole boost: if no query entity is at least this
    discriminating, the transform is the identity (a query whose only entities
    are ubiquitous should not trigger a graph boost at all). Default 0.0 = no
    gate.

    Degrades to the identity transform (returns the list unchanged) when
    ``weight <= 0``, there are no query entities, the ``min_idf`` gate fails, or
    the graph yields no edges reachable from the query entities.
    """
    if weight <= 0 or not query_entities:
        return _identity

    # Corpus size for IDF; 0 when the graph can't report it (stub / empty) ->
    # _idf() returns 0 for every entity, so we fall back to raw edge weights.
    n_docs = 0
    with contextlib.suppress(Exception):
        n_docs = int(graph.total_indexed_memories())

    # Rare-entity gate: require at least one discriminating query entity.
    if min_idf > 0 and n_docs > 0:
        q_df: dict[str, float] = {}
        with contextlib.suppress(Exception):
            q_df = graph.entity_doc_freqs(list(query_entities))
        max_q_idf = max(
            (_idf(q_df.get(str(qe).strip().lower(), 0.0), n_docs) for qe in query_entities),
            default=0.0,
        )
        if max_q_idf < min_idf:
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

    # Scale each neighbor's edge weight by its IDF. When n_docs == 0 the IDF
    # factor is 1.0 (raw-weight fallback), so stubs and empty graphs behave as
    # before.
    if n_docs > 0:
        neigh_df: dict[str, float] = {}
        with contextlib.suppress(Exception):
            neigh_df = graph.entity_doc_freqs(list(proximity_weights))
        scaled = {
            nm: edge_w * _idf(neigh_df.get(nm, 0.0), n_docs)
            for nm, edge_w in proximity_weights.items()
        }
        # Keep only neighbors that carry discriminating signal.
        proximity_weights = {nm: v for nm, v in scaled.items() if v > 0.0}
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
