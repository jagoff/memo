from __future__ import annotations

from memo.graph_signal import GraphSignalConfig, collect_graph_signal


class _Graph:
    def __init__(self) -> None:
        self.entities_by_memory = {
            "rare-doc": [{"name": "daemon", "type": "technology", "mention_count": 2}],
            "hub-doc": [{"name": "memo", "type": "project", "mention_count": 100}],
        }

    def total_indexed_memories(self) -> int:
        return 100

    def entity_doc_freqs(self, names: list[str]) -> dict[str, float]:
        values = {"mlx": 2.0, "daemon": 3.0, "memo": 90.0}
        return {n: values[n] for n in names if n in values}

    def weighted_neighbors(self, name: str) -> dict[str, float]:
        if name == "mlx":
            return {"daemon": 4.0, "memo": 20.0}
        return {}

    def memory_entities(self, memory_id: str) -> list[dict[str, object]]:
        return self.entities_by_memory.get(memory_id, [])

    def entity_names(self) -> set[str]:
        return {"mlx", "daemon", "memo"}


def test_collect_graph_signal_boosts_rare_neighbor_and_suppresses_hub() -> None:
    signal = collect_graph_signal(
        _Graph(),
        "mlx",
        ["rare-doc", "hub-doc"],
        config=GraphSignalConfig(enabled=True, hub_suppression=True, min_entity_idf=0.5),
    )

    assert signal.enabled is True
    assert signal.query_entities == ["mlx"]
    assert signal.boosts["rare-doc"] > 0
    assert "hub-doc" not in signal.boosts
    assert signal.traces["rare-doc"].mode == "proximity"


def test_collect_graph_signal_returns_disabled_state() -> None:
    signal = collect_graph_signal(
        _Graph(),
        "mlx",
        ["rare-doc"],
        config=GraphSignalConfig(enabled=False),
    )

    assert signal.enabled is False
    assert signal.skipped == "disabled"
    assert signal.boosts == {}


def test_collect_graph_signal_modulates_by_outcome_score() -> None:
    low = collect_graph_signal(
        _Graph(),
        "mlx",
        ["rare-doc"],
        outcome_scores={"rare-doc": 0.5},
        config=GraphSignalConfig(
            enabled=True,
            hub_suppression=True,
            min_entity_idf=0.5,
            outcome_signal_enabled=True,
            outcome_weight=0.5,
        ),
    )
    high = collect_graph_signal(
        _Graph(),
        "mlx",
        ["rare-doc"],
        outcome_scores={"rare-doc": 2.0},
        config=GraphSignalConfig(
            enabled=True,
            hub_suppression=True,
            min_entity_idf=0.5,
            outcome_signal_enabled=True,
            outcome_weight=0.5,
        ),
    )

    assert high.boosts["rare-doc"] > low.boosts["rare-doc"]
    assert high.traces["rare-doc"].outcome_score == 2.0
