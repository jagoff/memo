"""Tests for GraphStore (entity ↔ memoria links).

NB: GraphStore lowercases entity names and only keeps types in
VALID_ENTITY_TYPES (person/project/technology/file/org/concept).
"""

from __future__ import annotations

from pathlib import Path

from memo.graph import GraphStore


def _store(tmp_path: Path) -> GraphStore:
    return GraphStore(tmp_path / "graph.db")


def test_record_and_query_entities(tmp_path: Path) -> None:
    g = _store(tmp_path)
    n = g.record_extraction(
        memoria_id="m1", memoria_date="2026-01-01",
        entities=[{"name": "Synapse", "type": "project"},
                  {"name": "Fernando", "type": "person"}],
        extracted_at="2026-01-01T00:00:00Z",
    )
    assert n == 2
    # names are stored lowercased
    names = {e["name"] for e in g.memoria_entities("m1")}
    assert names == {"synapse", "fernando"}
    # lookup is case-insensitive
    assert "m1" in g.entity_memorias("Synapse")
    assert g.count_entities() == 2


def test_invalid_entity_type_is_skipped(tmp_path: Path) -> None:
    g = _store(tmp_path)
    n = g.record_extraction(
        memoria_id="m1", memoria_date="2026-01-01",
        entities=[{"name": "Thing", "type": "not-a-valid-type"}],
        extracted_at="2026-01-01T00:00:00Z",
    )
    assert n == 0
    assert g.memoria_entities("m1") == []


def test_record_extraction_is_idempotent(tmp_path: Path) -> None:
    g = _store(tmp_path)
    ents = [{"name": "Memo", "type": "project"}]
    g.record_extraction(memoria_id="m1", memoria_date="2026-01-01",
                        entities=ents, extracted_at="2026-01-01T00:00:00Z")
    g.record_extraction(memoria_id="m1", memoria_date="2026-01-01",
                        entities=ents, extracted_at="2026-01-02T00:00:00Z")
    assert len(g.memoria_entities("m1")) == 1


def test_drop_for_memoria(tmp_path: Path) -> None:
    g = _store(tmp_path)
    g.record_extraction(memoria_id="m1", memoria_date="2026-01-01",
                        entities=[{"name": "Widget", "type": "concept"}],
                        extracted_at="2026-01-01T00:00:00Z")
    g.drop_for_memoria("m1")
    assert g.memoria_entities("m1") == []
    assert g.entity_memorias("Widget") == []


def test_two_memorias_share_entity(tmp_path: Path) -> None:
    g = _store(tmp_path)
    for mid in ("m1", "m2"):
        g.record_extraction(memoria_id=mid, memoria_date="2026-01-01",
                            entities=[{"name": "Shared", "type": "concept"}],
                            extracted_at="2026-01-01T00:00:00Z")
    assert set(g.entity_memorias("Shared")) == {"m1", "m2"}
