from __future__ import annotations

from memo.graph_reason import build_graph_reason, format_graph_reason
from memo.graph_signal import GraphEdgeEvidence, GraphEvidenceTrace


def _trace() -> GraphEvidenceTrace:
    return GraphEvidenceTrace(
        projection_version="v1",
        mode="curated_proximity",
        query_nodes=("entity://technology/mlx",),
        hit_nodes=("entity://project/daemon",),
        edges=(
            GraphEdgeEvidence(
                query_uri="entity://technology/mlx",
                hit_uri="entity://project/daemon",
                relation="co_occurs",
                weight=4.0,
                confidence=0.9,
                query_idf=3.0,
                hit_idf=2.5,
                evidence_ids=("memory://m1", "memory://m2"),
            ),
        ),
        normalized_signal=1.0,
    )


def test_build_graph_reason_is_honest_and_compact() -> None:
    reason = build_graph_reason("abc123", _trace())

    assert reason["memory_id"] == "abc123"
    assert reason["projection_version"] == "v1"
    assert reason["mode"] == "curated_proximity"
    assert reason["query_nodes"] == ["entity://technology/mlx"]
    assert reason["hit_nodes"] == ["entity://project/daemon"]
    assert reason["edges"][0]["evidence_ids"] == ["memory://m1", "memory://m2"]
    assert reason["confidence"] == "derived"
    assert "path" not in reason


def test_format_graph_reason_does_not_claim_verification() -> None:
    text = format_graph_reason(build_graph_reason("abc123", _trace()))

    assert "related via graph" in text
    assert "verified" not in text.lower()


def test_graph_reason_includes_relations_when_supplied() -> None:
    reason = build_graph_reason(
        "mem-a",
        _trace(),
        relations=[{"relation": "supersedes", "target_id": "mem-b", "confidence": 1.0}],
    )

    assert reason["relations"][0]["relation"] == "supersedes"
