"""Bounded, evidence-aware graph ordering over already-eligible candidates."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

from memo.flags import flag_bool, flag_float, flag_int
from memo.graph_projection import GraphReadModel, ProjectedEdge, ProjectedNode


@dataclass(frozen=True)
class GraphSignalConfig:
    enabled: bool = False
    alpha: float = 0.15
    rrf_k: int = 60
    budget_ms: int = 150
    min_entity_idf: float = 0.5
    hub_suppression: bool = True
    max_age_hours: int = 36


@dataclass(frozen=True)
class GraphEdgeEvidence:
    query_uri: str
    hit_uri: str
    relation: str
    weight: float
    confidence: float
    query_idf: float
    hit_idf: float
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class GraphEvidenceTrace:
    projection_version: str
    mode: str
    query_nodes: tuple[str, ...]
    hit_nodes: tuple[str, ...]
    edges: tuple[GraphEdgeEvidence, ...]
    normalized_signal: float
    hub_suppressed: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraphSignalResult:
    enabled: bool
    signals: dict[str, float]
    traces: dict[str, GraphEvidenceTrace]
    ordered_ids: list[str]
    query_nodes: tuple[str, ...] = ()
    skipped: str | None = None
    elapsed_ms: float = 0.0


def config_from_flags() -> GraphSignalConfig:
    alpha = flag_float("MEMO_GRAPH_SIGNAL_ALPHA")
    min_idf = flag_float("MEMO_GRAPH_MIN_ENTITY_IDF")
    budget = flag_int("MEMO_GRAPH_SIGNAL_BUDGET_MS")
    max_age = flag_int("MEMO_GRAPH_PROJECTION_MAX_AGE_HOURS")
    return GraphSignalConfig(
        enabled=flag_bool("MEMO_GRAPH_SIGNAL_ENABLED"),
        alpha=0.15 if alpha is None else alpha,
        min_entity_idf=0.5 if min_idf is None else min_idf,
        budget_ms=150 if budget is None else budget,
        hub_suppression=flag_bool("MEMO_GRAPH_HUB_SUPPRESSION"),
        max_age_hours=36 if max_age is None else max_age,
    )


def _elapsed_ms(started: float) -> float:
    return (time.monotonic() - started) * 1000.0


def _other_node_uri(edge: ProjectedEdge, uri: str) -> str | None:
    if edge.source_uri == uri:
        return edge.target_uri
    if edge.target_uri == uri:
        return edge.source_uri
    return None


def _edge_signal(query: ProjectedNode, hit: ProjectedNode, edge: ProjectedEdge) -> float:
    if query.degree <= 0 or hit.degree <= 0:
        return 0.0
    return (
        math.log1p(max(0.0, edge.weight))
        * max(0.0, query.idf)
        * max(0.0, hit.idf)
        * max(0.0, min(1.0, edge.confidence))
        / math.sqrt(query.degree * hit.degree)
    )


def _fuse_order(
    candidate_ids: Sequence[str],
    signals: dict[str, float],
    *,
    alpha: float,
    rrf_k: int,
) -> list[str]:
    base = list(candidate_ids)
    if not signals or alpha <= 0:
        return base
    base_rank = {memory_id: rank for rank, memory_id in enumerate(base, 1)}
    graph_order = sorted(
        signals,
        key=lambda memory_id: (-signals[memory_id], base_rank.get(memory_id, len(base) + 1)),
    )
    graph_rank = {memory_id: rank for rank, memory_id in enumerate(graph_order, 1)}
    bounded_alpha = max(0.0, min(0.5, alpha))
    k = max(1, rrf_k)

    def fused(memory_id: str) -> float:
        value = 1.0 / (k + base_rank[memory_id])
        if memory_id in graph_rank:
            value += bounded_alpha / (k + graph_rank[memory_id])
        return value

    return sorted(base, key=lambda memory_id: (-fused(memory_id), base_rank[memory_id]))


def _identity(
    candidate_ids: Sequence[str],
    *,
    enabled: bool,
    skipped: str,
    started: float,
    query_nodes: tuple[str, ...] = (),
) -> GraphSignalResult:
    return GraphSignalResult(
        enabled=enabled,
        signals={},
        traces={},
        ordered_ids=list(candidate_ids),
        query_nodes=query_nodes,
        skipped=skipped,
        elapsed_ms=_elapsed_ms(started),
    )


def collect_graph_signal(
    read_model: GraphReadModel,
    query: str,
    candidate_ids: Sequence[str],
    *,
    config: GraphSignalConfig | None = None,
    deadline: float | None = None,
) -> GraphSignalResult:
    """Return an identity-safe graph ordering over the supplied candidate set."""
    started = time.monotonic()
    cfg = config or config_from_flags()
    if not cfg.enabled:
        return _identity(candidate_ids, enabled=False, skipped="disabled", started=started)
    if not read_model.available:
        return _identity(
            candidate_ids,
            enabled=True,
            skipped=read_model.skip_reason or "projection_unavailable",
            started=started,
        )
    if deadline is None:
        deadline = started + max(0, cfg.budget_ms) / 1000.0
    try:
        if time.monotonic() > deadline:
            return _identity(candidate_ids, enabled=True, skipped="deadline", started=started)
        resolved = read_model.resolve_query_entities(query)
        query_nodes = tuple(node.uri for node in resolved)
        if not resolved:
            return _identity(
                candidate_ids,
                enabled=True,
                skipped="no_query_entities",
                started=started,
            )
        allowed = tuple(
            node for node in resolved if node.idf >= cfg.min_entity_idf or node.is_hub
        )
        if not allowed:
            return _identity(
                candidate_ids,
                enabled=True,
                skipped="query_entities_below_idf",
                started=started,
                query_nodes=query_nodes,
            )

        raw_scores: dict[str, float] = {}
        raw_edges: dict[str, list[tuple[float, GraphEdgeEvidence]]] = {}
        hit_uris: dict[str, set[str]] = {}
        suppressed: dict[str, set[str]] = {}
        query_uri_set = {node.uri for node in allowed}
        for memory_id in candidate_ids:
            if time.monotonic() > deadline:
                return _identity(
                    candidate_ids,
                    enabled=True,
                    skipped="deadline",
                    started=started,
                    query_nodes=query_nodes,
                )
            candidate_nodes = {node.uri: node for node in read_model.memory_nodes(memory_id)}
            contributions: list[tuple[float, GraphEdgeEvidence]] = []
            touched: set[str] = set()
            suppressed_nodes: set[str] = set()
            for query_node in allowed:
                for edge in read_model.neighbors(query_node.uri):
                    hit_uri = _other_node_uri(edge, query_node.uri)
                    hit_node = candidate_nodes.get(hit_uri or "")
                    if hit_node is None:
                        continue
                    if (
                        cfg.hub_suppression
                        and hit_node.is_hub
                        and hit_node.uri not in query_uri_set
                    ):
                        suppressed_nodes.add(hit_node.uri)
                        continue
                    score = _edge_signal(query_node, hit_node, edge)
                    if score <= 0:
                        continue
                    touched.add(hit_node.uri)
                    contributions.append(
                        (
                            score,
                            GraphEdgeEvidence(
                                query_uri=query_node.uri,
                                hit_uri=hit_node.uri,
                                relation=edge.relation,
                                weight=edge.weight,
                                confidence=edge.confidence,
                                query_idf=query_node.idf,
                                hit_idf=hit_node.idf,
                                evidence_ids=edge.evidence_ids,
                            ),
                        )
                    )
            if contributions:
                strongest = sorted(
                    contributions,
                    key=lambda value: (-value[0], value[1].query_uri, value[1].hit_uri),
                )[:3]
                raw_scores[memory_id] = sum(value for value, _edge in strongest)
                raw_edges[memory_id] = strongest
                hit_uris[memory_id] = touched
            if suppressed_nodes:
                suppressed[memory_id] = suppressed_nodes

        if time.monotonic() > deadline:
            return _identity(
                candidate_ids,
                enabled=True,
                skipped="deadline",
                started=started,
                query_nodes=query_nodes,
            )
        if not raw_scores:
            return _identity(
                candidate_ids,
                enabled=True,
                skipped="no_candidate_connections",
                started=started,
                query_nodes=query_nodes,
            )
        maximum = max(raw_scores.values())
        signals = {
            memory_id: min(1.0, score / maximum)
            for memory_id, score in raw_scores.items()
            if maximum > 0
        }
        traces = {
            memory_id: GraphEvidenceTrace(
                projection_version=read_model.version or "unknown",
                mode="curated_proximity",
                query_nodes=tuple(node.uri for node in allowed),
                hit_nodes=tuple(sorted(hit_uris[memory_id])),
                edges=tuple(edge for _score, edge in raw_edges[memory_id]),
                normalized_signal=signals[memory_id],
                hub_suppressed=tuple(sorted(suppressed.get(memory_id, set()))),
            )
            for memory_id in signals
        }
        return GraphSignalResult(
            enabled=True,
            signals=signals,
            traces=traces,
            ordered_ids=_fuse_order(
                candidate_ids,
                signals,
                alpha=cfg.alpha,
                rrf_k=cfg.rrf_k,
            ),
            query_nodes=query_nodes,
            elapsed_ms=_elapsed_ms(started),
        )
    except Exception as exc:
        return _identity(
            candidate_ids,
            enabled=True,
            skipped=f"error:{type(exc).__name__}",
            started=started,
        )


__all__ = [
    "GraphEdgeEvidence",
    "GraphEvidenceTrace",
    "GraphSignalConfig",
    "GraphSignalResult",
    "collect_graph_signal",
    "config_from_flags",
]
