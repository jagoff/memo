from __future__ import annotations

from memo.graph import GraphStore
from memo.semantic_relations import (
    DETERMINISTIC_DERIVED_FROM,
    extract_relations_batch,
    store_relations,
)


def test_extract_relations_batch_detects_explicit_relation() -> None:
    source = {
        "id": "new",
        "title": "New graph plan",
        "body": "This supersedes Old graph plan and requires the graph signal work.",
    }
    target = {"id": "old", "title": "Old graph plan", "body": "previous graph signal work"}

    rels = extract_relations_batch([(source, target)])

    assert len(rels) == 1
    assert rels[0].source_id == "new"
    assert rels[0].target_id == "old"
    assert rels[0].relation_type == "supersedes"
    assert rels[0].derived_from == DETERMINISTIC_DERIVED_FROM


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


def test_drop_for_memoria_cascades_to_semantic_relations(tmp_path) -> None:
    graph = GraphStore(tmp_path / "graph.db")
    graph.upsert_semantic_relation(
        source_kind="memory",
        source_id="a",
        target_kind="memory",
        target_id="b",
        relation="supersedes",
        derived_from="test",
    )
    assert graph.semantic_relations_for(source_id="a") != []

    graph.drop_for_memoria("a")

    assert graph.semantic_relations_for(source_id="a") == []


def test_drop_for_memoria_cascades_to_target_semantic_relations(tmp_path) -> None:
    # Symmetric to the source-side cascade: deleting the memory that is the
    # TARGET of a relation must also drop the edge. Previously drop_for_memoria
    # only deleted WHERE source_id = ?, leaving a dangling target-side row.
    graph = GraphStore(tmp_path / "graph.db")
    graph.upsert_semantic_relation(
        source_kind="memory",
        source_id="a",
        target_kind="memory",
        target_id="b",
        relation="supersedes",
        derived_from="test",
    )
    assert graph.semantic_relations_for(target_id="b") != []

    graph.drop_for_memoria("b")

    assert graph.semantic_relations_for(target_id="b") == []
    assert graph.semantic_relations_for(source_id="a") == []


def test_store_relations_and_delete_by_derived_from(tmp_path) -> None:
    graph = GraphStore(tmp_path / "graph.db")
    rel = extract_relations_batch(
        [
            (
                {"id": "a", "title": "A", "body": "A supports B memory."},
                {"id": "b", "title": "B memory", "body": ""},
            )
        ]
    )

    assert store_relations(graph, rel) == 1
    assert graph.semantic_relations_for(source_id="a")[0]["relation"] == "supports"
    assert graph.delete_semantic_relations_by_derived_from(DETERMINISTIC_DERIVED_FROM) == 1
    assert graph.semantic_relations_for(source_id="a") == []
