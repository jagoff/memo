"""Tests for knowledge exploration (spec 4)."""

from __future__ import annotations

from memo.explore import explore_entity
from memo.navigation import EntityNeighbors


def test_explore_entity_view():
    class _Nav:
        def get_neighbors(self, entity, max_neighbors=50, *, use_codegraph=None):
            return EntityNeighbors(
                entity=entity,
                direct_neighbors=["a", "b"],
                neighbor_memories={"a": ["m1", "m2"], "b": ["m1"]},
                degree=2,
            )

    class _Graph:
        def entity_memories(self, name, type_=None):
            return ["m1", "m2"]

    class _Rec:
        def __init__(self, id):
            self.id = id
            self.title = f"t-{id}"

    class _Mem:
        navigator = _Nav()
        graph = _Graph()

        def get(self, mid):
            return _Rec(mid)

    v = explore_entity(_Mem(), "VecStore")
    assert v["entity"] == "vecstore"  # lowercased
    assert v["degree"] == 2
    assert {n["name"] for n in v["neighbors"]} == {"a", "b"}
    assert next(n for n in v["neighbors"] if n["name"] == "a")["shared"] == 2
    assert {m["id"] for m in v["memories"]} == {"m1", "m2"}
    assert all(m["title"].startswith("t-") for m in v["memories"])


def test_explore_entity_empty():
    class _Nav:
        def get_neighbors(self, entity, max_neighbors=50, *, use_codegraph=None):
            return EntityNeighbors(
                entity=entity, direct_neighbors=[], neighbor_memories={}, degree=0
            )

    class _Graph:
        def entity_memories(self, name, type_=None):
            return []

    class _Mem:
        navigator = _Nav()
        graph = _Graph()

        def get(self, mid):
            return None

    v = explore_entity(_Mem(), "nothing")
    assert v["degree"] == 0
    assert v["neighbors"] == []
    assert v["memories"] == []


def test_explore_excludes_codegraph_placeholder_from_shared():
    class _Nav:
        def get_neighbors(self, entity, max_neighbors=50, *, use_codegraph=None):
            return EntityNeighbors(
                entity=entity,
                direct_neighbors=["codesym", "realnbr"],
                neighbor_memories={"codesym": ["(codegraph)"], "realnbr": ["m1", "(codegraph)"]},
                degree=3,
            )

    class _Graph:
        def entity_memories(self, name, type_=None):
            return []

    class _Mem:
        navigator = _Nav()
        graph = _Graph()

        def get(self, mid):
            return None

    v = explore_entity(_Mem(), "x")
    by = {n["name"]: n["shared"] for n in v["neighbors"]}
    assert by["codesym"] == 0  # pure code link bridges 0 memories
    assert by["realnbr"] == 1  # one real memory; the codegraph placeholder excluded
