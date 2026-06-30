"""Unit tests for graph_proximity.graph_boost_factory (pure, stub graph)."""

from __future__ import annotations

from dataclasses import dataclass

from memo.graph_proximity import graph_boost_factory


@dataclass(frozen=True)
class _Hit:
    id: str
    score: float | None


class _StubGraph:
    """Minimal GraphStore stand-in: query entity 'fastapi' neighbors 'pydantic'
    (edge weight 2.0). Memory 'a' mentions pydantic; memory 'b' mentions django."""

    def __init__(self, neighbors=None, mem_entities=None):
        self._neighbors = neighbors if neighbors is not None else {"fastapi": {"pydantic": 2.0}}
        self._mem_entities = (
            mem_entities
            if mem_entities is not None
            else {
                "a": [{"name": "pydantic", "type": "tech", "mention_count": 3}],
                "b": [{"name": "django", "type": "tech", "mention_count": 1}],
            }
        )

    def weighted_neighbors(self, name):
        return dict(self._neighbors.get(name.strip().lower(), {}))

    def memory_entities(self, memory_id):
        return list(self._mem_entities.get(memory_id, []))


def test_graph_proximal_hit_is_boosted_above_non_proximal():
    g = _StubGraph()
    boost = graph_boost_factory(g, ["FastAPI"], weight=0.1)
    # b starts higher than a; a is graph-proximal so it should overtake b.
    hits = [_Hit("a", 0.5), _Hit("b", 0.6)]
    out = boost(hits)
    assert out[0].id == "a"
    assert out[0].score == 0.5 + 0.1 * 2.0  # weight * proximity (edge weight)
    assert out[1].id == "b"
    assert out[1].score == 0.6  # untouched


def test_weight_zero_is_noop():
    g = _StubGraph()
    boost = graph_boost_factory(g, ["FastAPI"], weight=0.0)
    hits = [_Hit("a", 0.5), _Hit("b", 0.6)]
    out = boost(hits)
    assert out == hits  # identity, same order and scores


def test_no_query_entities_is_noop():
    g = _StubGraph()
    boost = graph_boost_factory(g, [], weight=0.1)
    hits = [_Hit("a", 0.5), _Hit("b", 0.6)]
    assert boost(hits) == hits


def test_empty_graph_is_noop():
    g = _StubGraph(neighbors={})  # no edges reachable from any query entity
    boost = graph_boost_factory(g, ["FastAPI"], weight=0.1)
    hits = [_Hit("a", 0.5), _Hit("b", 0.6)]
    assert boost(hits) == hits


def test_none_score_hit_is_preserved():
    g = _StubGraph()
    boost = graph_boost_factory(g, ["FastAPI"], weight=0.1)
    hits = [_Hit("a", None), _Hit("b", 0.6)]
    out = boost(hits)
    ids = {h.id: h.score for h in out}
    assert ids["a"] is None  # no boost applied to a scoreless hit
