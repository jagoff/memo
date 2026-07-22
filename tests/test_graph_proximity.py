"""Compatibility tests for raw graph query-entity matching."""

from __future__ import annotations

from memo.graph_proximity import extract_query_entities


class _VocabGraph:
    def __init__(self, names: set[str]) -> None:
        self._names = names

    def entity_names(self) -> set[str]:
        return self._names


def test_extract_query_entities_matches_graph_vocabulary_lowercase() -> None:
    graph = _VocabGraph({"recall hook", "synapse", "memflow", "budget"})

    entities = [
        entity.lower()
        for entity in extract_query_entities("how does the recall hook budget work", graph)
    ]

    assert "recall hook" in entities
    assert "budget" in entities


def test_extract_query_entities_keeps_regex_proper_nouns() -> None:
    entities = {
        entity.lower()
        for entity in extract_query_entities(
            "How does Synapse connect to MLX?",
            _VocabGraph(set()),
        )
    }

    assert "mlx" in entities or "synapse" in entities


def test_extract_query_entities_no_stopword_noise() -> None:
    graph = _VocabGraph({"synapse"})

    assert extract_query_entities("the quick brown fox jumps over", graph) == []
