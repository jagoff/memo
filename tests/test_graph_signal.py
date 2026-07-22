from __future__ import annotations

import time

from memo.graph_projection import GraphReadModel, ProjectedEdge, ProjectedNode
from memo.graph_signal import GraphSignalConfig, collect_graph_signal


def _node(
    uri: str,
    label: str,
    *,
    degree: int = 1,
    idf: float = 2.0,
    is_hub: bool = False,
) -> ProjectedNode:
    return ProjectedNode(
        uri=uri,
        label=label,
        entity_type="technology",
        canonical_key=label.replace(" ", "").lower(),
        doc_freq=2,
        degree=degree,
        quality=0.9,
        is_hub=is_hub,
        idf=idf,
    )


def _projected_graph() -> GraphReadModel:
    mlx = _node("entity://technology/mlx", "mlx", degree=2, idf=3.0)
    daemon = _node("entity://project/daemon", "daemon", idf=2.5)
    hub = _node("entity://project/memo", "memo", degree=1, idf=0.1, is_hub=True)
    daemon_edge = ProjectedEdge(
        source_uri=mlx.uri,
        target_uri=daemon.uri,
        relation="co_occurs",
        weight=4.0,
        confidence=0.9,
        evidence_ids=("memory://m1", "memory://m2"),
        first_seen="2026-01-01",
        last_seen="2026-07-20",
    )
    hub_edge = ProjectedEdge(
        source_uri=mlx.uri,
        target_uri=hub.uri,
        relation="co_occurs",
        weight=20.0,
        confidence=0.9,
        evidence_ids=("memory://m3",),
        first_seen=None,
        last_seen=None,
    )
    return GraphReadModel(
        available=True,
        version="v1",
        built_at="2026-07-22T00:00:00+00:00",
        total_memories=100,
        nodes={node.uri: node for node in (mlx, daemon, hub)},
        memberships={"daemon": (daemon.uri,), "hub": (hub.uri,)},
        edges={
            mlx.uri: (daemon_edge, hub_edge),
            daemon.uri: (daemon_edge,),
            hub.uri: (hub_edge,),
        },
    )


def _config(**overrides: object) -> GraphSignalConfig:
    values = {
        "enabled": True,
        "alpha": 0.15,
        "rrf_k": 60,
        "budget_ms": 150,
        "min_entity_idf": 0.5,
        "hub_suppression": True,
        **overrides,
    }
    return GraphSignalConfig(**values)


def test_empty_signal_preserves_exact_candidate_order() -> None:
    result = collect_graph_signal(
        _projected_graph(),
        "unknown words",
        ["m2", "m1"],
        config=_config(),
    )

    assert result.ordered_ids == ["m2", "m1"]
    assert result.skipped == "no_query_entities"


def test_curated_signal_reorders_without_adding_candidates() -> None:
    result = collect_graph_signal(
        _projected_graph(),
        "mlx",
        ["generic", "daemon"],
        config=_config(alpha=0.15),
    )

    assert result.ordered_ids == ["daemon", "generic"]
    assert set(result.ordered_ids) == {"generic", "daemon"}
    assert 0.0 < result.signals["daemon"] <= 1.0
    assert "hub" not in result.signals


def test_deadline_discards_partial_signal() -> None:
    result = collect_graph_signal(
        _projected_graph(),
        "mlx",
        ["generic", "daemon"],
        config=_config(),
        deadline=time.monotonic() - 1,
    )

    assert result.ordered_ids == ["generic", "daemon"]
    assert result.signals == {}
    assert result.skipped == "deadline"


def test_trace_contains_only_stored_evidence() -> None:
    result = collect_graph_signal(
        _projected_graph(),
        "mlx",
        ["daemon"],
        config=_config(),
    )

    trace = result.traces["daemon"]
    assert trace.projection_version == "v1"
    assert trace.edges[0].evidence_ids == ("memory://m1", "memory://m2")
    assert trace.mode == "curated_proximity"


def test_unavailable_projection_is_identity() -> None:
    result = collect_graph_signal(
        GraphReadModel.unavailable("projection_stale"),
        "mlx",
        ["a", "b"],
        config=_config(),
    )

    assert result.ordered_ids == ["a", "b"]
    assert result.skipped == "projection_stale"


def test_disabled_signal_is_identity() -> None:
    result = collect_graph_signal(
        _projected_graph(),
        "mlx",
        ["a", "b"],
        config=_config(enabled=False),
    )

    assert result.enabled is False
    assert result.ordered_ids == ["a", "b"]
    assert result.skipped == "disabled"
