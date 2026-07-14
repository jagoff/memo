"""Unified graph signal collection for retrieval attribution.

This module centralizes the cheap graph work search/recall can safely do:
query entity extraction, IDF-based hub suppression, bounded proximity scoring,
and per-hit attribution. It owns no storage and degrades to an empty signal on
any graph failure.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from memo.flags import flag_bool, flag_float, flag_int
from memo.graph_proximity import extract_query_entities


@dataclass(frozen=True)
class GraphSignalConfig:
    enabled: bool = False
    hub_suppression: bool = True
    hub_max_doc_freq_ratio: float = 0.25
    min_entity_idf: float = 0.5
    weight: float = 0.05
    budget_ms: int = 150
    outcome_signal_enabled: bool = False
    outcome_weight: float = 0.05


@dataclass(frozen=True)
class GraphSignalTrace:
    mode: str
    query_entities: list[str]
    hit_entities: list[str]
    neighbor_edges: list[dict[str, Any]] = field(default_factory=list)
    outcome_score: float | None = None
    skipped: str | None = None


@dataclass(frozen=True)
class GraphSignal:
    enabled: bool
    query_entities: list[str]
    boosts: dict[str, float]
    traces: dict[str, GraphSignalTrace]
    skipped: str | None = None
    elapsed_ms: float = 0.0


def config_from_flags() -> GraphSignalConfig:
    return GraphSignalConfig(
        enabled=flag_bool("MEMO_GRAPH_SIGNAL_ENABLED"),
        hub_suppression=flag_bool("MEMO_GRAPH_HUB_SUPPRESSION"),
        hub_max_doc_freq_ratio=flag_float("MEMO_GRAPH_HUB_MAX_DOC_FREQ_RATIO") or 0.25,
        min_entity_idf=flag_float("MEMO_GRAPH_MIN_ENTITY_IDF") or 0.5,
        budget_ms=flag_int("MEMO_GRAPH_SIGNAL_BUDGET_MS") or 150,
        outcome_signal_enabled=flag_bool("MEMO_GRAPH_OUTCOME_SIGNAL_ENABLED"),
        outcome_weight=flag_float("MEMO_GRAPH_OUTCOME_WEIGHT") or 0.05,
    )


def entity_idf(df: float, n_docs: int) -> float:
    if n_docs <= 0 or df <= 0:
        return 0.0
    return max(0.0, math.log(n_docs / df))


def _elapsed_ms(started: float) -> float:
    return (time.monotonic() - started) * 1000.0


def collect_graph_signal(
    graph: Any,
    query: str,
    candidate_ids: Sequence[str],
    *,
    deadline: float | None = None,
    config: GraphSignalConfig | None = None,
    outcome_scores: Mapping[str, float] | None = None,
) -> GraphSignal:
    started = time.monotonic()
    cfg = config or config_from_flags()
    if not cfg.enabled:
        return GraphSignal(False, [], {}, {}, skipped="disabled")

    if deadline is None:
        deadline = started + (cfg.budget_ms / 1000.0)

    try:
        query_entities = [str(e).strip().lower() for e in extract_query_entities(query, graph)]
        query_entities = list(dict.fromkeys(e for e in query_entities if e))
        if not query_entities:
            return GraphSignal(
                True, [], {}, {}, skipped="no_query_entities", elapsed_ms=_elapsed_ms(started)
            )

        n_docs = int(graph.total_indexed_memories())
        q_df = graph.entity_doc_freqs(query_entities) if n_docs > 0 else {}
        allowed_query_entities = [
            ent
            for ent in query_entities
            if entity_idf(float(q_df.get(ent, 0.0)), n_docs) >= cfg.min_entity_idf
        ]
        if not allowed_query_entities:
            return GraphSignal(
                True,
                query_entities,
                {},
                {},
                skipped="query_entities_below_idf",
                elapsed_ms=_elapsed_ms(started),
            )

        proximity: dict[str, dict[str, Any]] = {}
        for ent in allowed_query_entities:
            if time.monotonic() > deadline:
                return GraphSignal(
                    True,
                    query_entities,
                    {},
                    {},
                    skipped="deadline",
                    elapsed_ms=_elapsed_ms(started),
                )
            for neighbor, weight in graph.weighted_neighbors(ent).items():
                key = str(neighbor).strip().lower()
                if not key:
                    continue
                existing = proximity.get(key)
                if existing is None or float(weight) > float(existing["weight"]):
                    proximity[key] = {"from": ent, "to": key, "weight": float(weight)}

        if not proximity:
            return GraphSignal(
                True,
                query_entities,
                {},
                {},
                skipped="no_neighbors",
                elapsed_ms=_elapsed_ms(started),
            )

        neigh_df = graph.entity_doc_freqs(list(proximity)) if n_docs > 0 else {}
        boosts: dict[str, float] = {}
        traces: dict[str, GraphSignalTrace] = {}
        for mid in candidate_ids:
            if time.monotonic() > deadline:
                break
            hit_entities = [
                str(e.get("name", "")).strip().lower()
                for e in graph.memory_entities(mid)
                if str(e.get("name", "")).strip()
            ]
            edges: list[dict[str, Any]] = []
            score = 0.0
            for ent in hit_entities:
                edge = proximity.get(ent)
                if edge is None:
                    continue
                df = float(neigh_df.get(ent, 0.0))
                if (
                    cfg.hub_suppression
                    and n_docs > 0
                    and (df / n_docs) > cfg.hub_max_doc_freq_ratio
                ):
                    continue
                idf = entity_idf(df, n_docs) if n_docs > 0 else 1.0
                if idf <= 0:
                    continue
                score += float(edge["weight"]) * idf * cfg.weight
                edges.append({**edge, "idf": idf})
            if score > 0:
                outcome_score: float | None = None
                if cfg.outcome_signal_enabled and outcome_scores is not None:
                    outcome_score = max(0.0, float(outcome_scores.get(mid, 1.0)))
                    score *= max(0.0, 1.0 + cfg.outcome_weight * (outcome_score - 1.0))
                boosts[mid] = round(score, 6)
                traces[mid] = GraphSignalTrace(
                    mode="proximity",
                    query_entities=allowed_query_entities,
                    hit_entities=hit_entities,
                    neighbor_edges=edges,
                    outcome_score=outcome_score,
                )
        return GraphSignal(
            True,
            query_entities,
            boosts,
            traces,
            elapsed_ms=_elapsed_ms(started),
        )
    except Exception as exc:
        return GraphSignal(
            True,
            [],
            {},
            {},
            skipped=f"error:{type(exc).__name__}",
            elapsed_ms=_elapsed_ms(started),
        )
