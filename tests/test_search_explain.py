from __future__ import annotations

from types import SimpleNamespace

from memo.search_explain import build_search_explanations


def test_build_search_explanations_includes_graph_reason() -> None:
    hit = SimpleNamespace(
        id="abc123",
        score=0.9,
        extra={
            "graph_reason": {
                "mode": "proximity",
                "query_entities": ["mlx"],
                "hit_entities": ["daemon"],
                "neighbor_edges": [{"from": "mlx", "to": "daemon", "idf": 2.0}],
            }
        },
    )

    explanations = build_search_explanations([hit], [{"stage": "graph_signal"}])

    exp = explanations["abc123"]
    assert exp["legs"]["graph_reason"]["mode"] == "proximity"
    assert "related via graph (proximity): mlx -> daemon" in exp["why"]
