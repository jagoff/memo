from __future__ import annotations

from memo.graph_reason import build_graph_reason, format_graph_reason
from memo.graph_signal import GraphSignalTrace


def test_build_graph_reason_is_honest_and_compact() -> None:
    trace = GraphSignalTrace(
        mode="proximity",
        query_entities=["mlx"],
        hit_entities=["daemon"],
        neighbor_edges=[{"from": "mlx", "to": "daemon", "weight": 4.0, "idf": 3.1}],
    )

    reason = build_graph_reason("abc123", trace)

    assert reason["memory_id"] == "abc123"
    assert reason["mode"] == "proximity"
    assert reason["query_entities"] == ["mlx"]
    assert reason["hit_entities"] == ["daemon"]
    assert reason["neighbor_edges"][0]["to"] == "daemon"
    assert reason["confidence"] == "derived"
    assert "path" not in reason


def test_format_graph_reason_does_not_claim_verification() -> None:
    trace = GraphSignalTrace(
        mode="proximity",
        query_entities=["mlx"],
        hit_entities=["daemon"],
        neighbor_edges=[{"from": "mlx", "to": "daemon", "weight": 4.0, "idf": 3.1}],
    )
    text = format_graph_reason(build_graph_reason("abc123", trace))

    assert "related via graph" in text
    assert "verified" not in text.lower()


def test_graph_reason_includes_relations_when_supplied() -> None:
    trace = GraphSignalTrace(mode="proximity", query_entities=["a"], hit_entities=["b"])
    reason = build_graph_reason(
        "mem-a",
        trace,
        relations=[{"relation": "supersedes", "target_id": "mem-b", "confidence": 1.0}],
    )

    assert reason["relations"][0]["relation"] == "supersedes"
