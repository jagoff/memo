"""Tests for the entity-centric knowledge-map briefing section (spec 2)."""

from __future__ import annotations

from memo.briefing import entity_graph_lines


class _Graph:
    def top_entities(self, *, limit=50, type_=None):
        return [
            {"name": "memo", "mention_count": 10},
            {"name": "rag", "mention_count": 5},
            {"name": "lonely", "mention_count": 1},
        ]

    def entity_memories(self, name, type_=None):
        return {"memo": ["m1", "m2"], "rag": ["m1"], "lonely": ["m3"]}.get(name, [])

    def memory_entities(self, mid):
        return {
            "m1": [{"name": "memo"}, {"name": "rag"}, {"name": "chunk"}],
            "m2": [{"name": "memo"}, {"name": "aws"}],
            "m3": [{"name": "lonely"}],
        }.get(mid, [])


class _Mem:
    graph = _Graph()


def test_entity_graph_lines_shows_hubs_and_clusters():
    lines = entity_graph_lines(_Mem(), top=2)
    text = "\n".join(lines)
    assert "Knowledge map" in text
    assert "**memo** (10)" in text
    # memo co-occurs with rag / chunk / aws — at least one cluster member shown
    assert any(x in text for x in ("rag", "chunk", "aws"))


def test_entity_graph_lines_isolated_entity_has_no_cluster():
    lines = entity_graph_lines(_Mem(), top=3)
    text = "\n".join(lines)
    # 'lonely' shares no memory with another entity -> em dash placeholder
    assert "**lonely** (1) → —" in text


def test_entity_graph_lines_empty_graph_returns_nothing():
    class _EmptyGraph:
        def top_entities(self, *, limit=50, type_=None):
            return []

        def entity_memories(self, name, type_=None):
            return []

        def memory_entities(self, mid):
            return []

    class _EmptyMem:
        graph = _EmptyGraph()

    assert entity_graph_lines(_EmptyMem()) == []


def test_memory_relations_name_the_memories_they_relate():
    """Two hex prefixes and a verb are not a relation a reader can act on.

    Live briefing, 2026-08-31: `- `492e8d82` related `715ed835``, three of
    them, 31 tokens per session, naming nothing. The row already carries the
    ids; the titles are one batch lookup away, and without them the section is
    the only graph output in the briefing that a human cannot read.
    """
    from memo import briefing

    class _Store:
        def list_relations(self, *, status, limit):
            return [{"source_id": "492e8d82aa", "target_id": "715ed835bb", "relation": "related"}]

        def get_batch(self, ids):
            return [
                {"id": "492e8d82aa", "title": "MLX embedder choice"},
                {"id": "715ed835bb", "title": "sqlite-vec dims guard"},
            ]

    lines = briefing.relation_lines(_Store())
    body = "\n".join(lines)

    assert "MLX embedder choice" in body
    assert "sqlite-vec dims guard" in body
