from __future__ import annotations

from memo.graph import GraphStore


def test_semantic_relation_upsert_is_idempotent(tmp_path) -> None:
    graph = GraphStore(tmp_path / "graph.db")

    graph.upsert_semantic_relation(
        source_kind="memory",
        source_id="a",
        target_kind="memory",
        target_id="b",
        relation="supports",
        weight=0.8,
        confidence=0.9,
        evidence_id="fact-1",
        derived_from="test",
    )
    graph.upsert_semantic_relation(
        source_kind="memory",
        source_id="a",
        target_kind="memory",
        target_id="b",
        relation="supports",
        weight=0.8,
        confidence=0.9,
        evidence_id="fact-1",
        derived_from="test",
    )

    rows = graph.semantic_relations_for(source_id="a")
    assert len(rows) == 1
    assert rows[0]["relation"] == "supports"
    assert rows[0]["target_id"] == "b"
    assert rows[0]["confidence"] == 0.9


def test_delete_semantic_relations_for_source(tmp_path) -> None:
    graph = GraphStore(tmp_path / "graph.db")
    graph.upsert_semantic_relation(
        source_kind="memory",
        source_id="a",
        target_kind="memory",
        target_id="b",
        relation="supersedes",
        derived_from="test",
    )

    assert graph.delete_semantic_relations_for_source("a") == 1
    assert graph.semantic_relations_for(source_id="a") == []
