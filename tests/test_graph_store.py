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
        memory_id="m1", memory_date="2026-01-01",
        entities=[{"name": "Synapse", "type": "project"},
                  {"name": "Fernando", "type": "person"}],
        extracted_at="2026-01-01T00:00:00Z",
    )
    assert n == 2
    # names are stored lowercased
    names = {e["name"] for e in g.memory_entities("m1")}
    assert names == {"synapse", "fernando"}
    # lookup is case-insensitive
    assert "m1" in g.entity_memories("Synapse")
    assert g.count_entities() == 2


def test_invalid_entity_type_is_skipped(tmp_path: Path) -> None:
    g = _store(tmp_path)
    n = g.record_extraction(
        memory_id="m1", memory_date="2026-01-01",
        entities=[{"name": "Thing", "type": "not-a-valid-type"}],
        extracted_at="2026-01-01T00:00:00Z",
    )
    assert n == 0
    assert g.memory_entities("m1") == []


def test_record_extraction_is_idempotent(tmp_path: Path) -> None:
    g = _store(tmp_path)
    ents = [{"name": "Memo", "type": "project"}]
    g.record_extraction(memory_id="m1", memory_date="2026-01-01",
                        entities=ents, extracted_at="2026-01-01T00:00:00Z")
    g.record_extraction(memory_id="m1", memory_date="2026-01-01",
                        entities=ents, extracted_at="2026-01-02T00:00:00Z")
    assert len(g.memory_entities("m1")) == 1


def test_drop_for_memoria(tmp_path: Path) -> None:
    g = _store(tmp_path)
    g.record_extraction(memory_id="m1", memory_date="2026-01-01",
                        entities=[{"name": "Widget", "type": "concept"}],
                        extracted_at="2026-01-01T00:00:00Z")
    g.drop_for_memoria("m1")
    assert g.memory_entities("m1") == []
    assert g.entity_memories("Widget") == []


def test_two_memorias_share_entity(tmp_path: Path) -> None:
    g = _store(tmp_path)
    for mid in ("m1", "m2"):
        g.record_extraction(memory_id=mid, memory_date="2026-01-01",
                            entities=[{"name": "Shared", "type": "concept"}],
                            extracted_at="2026-01-01T00:00:00Z")
    assert set(g.entity_memories("Shared")) == {"m1", "m2"}


def test_schema_has_edge_and_alias_tables(tmp_path: Path) -> None:
    g = _store(tmp_path)
    tables = {
        r[0]
        for r in g._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert "entity_edges" in tables
    assert "entity_aliases" in tables


def test_canonicalize_merges_fragmented_entities(tmp_path: Path) -> None:
    g = _store(tmp_path)
    # Two spellings + a cross-type duplicate, all the same real entity.
    g.record_extraction(memory_id="m1", memory_date="2026-01-01",
                        entities=[{"name": "fast api", "type": "concept"}],
                        extracted_at="2026-01-01T00:00:00Z")
    g.record_extraction(memory_id="m2", memory_date="2026-01-02",
                        entities=[{"name": "FastAPI", "type": "technology"}],
                        extracted_at="2026-01-02T00:00:00Z")
    assert g.count_entities() == 2  # fragmented before

    merged = g.canonicalize_existing()
    assert merged == 1
    assert g.count_entities() == 1
    # both memories now resolve through the single canonical entity
    name = g.memory_entities("m1")[0]["name"]
    assert "m1" in g.entity_memories(name)
    assert "m2" in g.entity_memories(name)
    # cross-type fold prefers the non-"concept" type
    assert g.memory_entities("m2")[0]["type"] == "technology"
    # idempotent
    assert g.canonicalize_existing() == 0


def test_rebuild_edges_weights_by_shared_memories(tmp_path: Path) -> None:
    g = _store(tmp_path)
    # A & B share two memories; A & C share one.
    for mid in ("m1", "m2"):
        g.record_extraction(memory_id=mid, memory_date="2026-01-01",
                            entities=[{"name": "A", "type": "concept"},
                                      {"name": "B", "type": "concept"}],
                            extracted_at="2026-01-01T00:00:00Z")
    g.record_extraction(memory_id="m3", memory_date="2026-01-01",
                        entities=[{"name": "A", "type": "concept"},
                                  {"name": "C", "type": "concept"}],
                        extracted_at="2026-01-01T00:00:00Z")
    n = g.rebuild_edges()
    assert n == 2  # (A,B) and (A,C)
    nbrs = g.weighted_neighbors("A")
    assert nbrs == {"b": 2.0, "c": 1.0}
    edges = {(a, b): w for a, b, w in g.all_weighted_edges()}
    assert edges[("a", "b")] == 2.0
    # rebuild is idempotent on count
    assert g.rebuild_edges() == 2


def test_decay_weight_halves_after_one_half_life() -> None:
    from memo.graph import decay_weight
    w = decay_weight(8.0, "2025-07-01", now_iso="2025-12-28", half_life_days=180.0)
    assert 3.8 < w < 4.2  # ~one half-life elapsed
    # no date -> undecayed
    assert decay_weight(8.0, None, now_iso="2025-12-28") == 8.0


def test_edge_stats_reports_counts_and_weight_distribution(tmp_path: Path) -> None:
    g = _store(tmp_path)
    for mid in ("m1", "m2"):  # A-B co-occur twice -> weight 2
        g.record_extraction(memory_id=mid, memory_date="2026-01-01",
                            entities=[{"name": "A", "type": "concept"},
                                      {"name": "B", "type": "concept"}],
                            extracted_at="2026-01-01T00:00:00Z")
    g.record_extraction(memory_id="m3", memory_date="2026-01-01",  # A-C once -> weight 1
                        entities=[{"name": "A", "type": "concept"},
                                  {"name": "C", "type": "concept"}],
                        extracted_at="2026-01-01T00:00:00Z")
    g.rebuild_edges()
    es = g.edge_stats()
    assert es["edges"] == 2
    assert es["weight_max"] == 2.0
    assert es["weight_min"] == 1.0
    assert es["edges_gt1"] == 1


def test_edge_stats_empty_graph(tmp_path: Path) -> None:
    es = _store(tmp_path).edge_stats()
    assert es["edges"] == 0 and es["weight_mean"] == 0.0


def test_total_indexed_memories_counts_distinct(tmp_path: Path) -> None:
    g = _store(tmp_path)
    for mid in ("m1", "m2", "m3"):
        g.record_extraction(memory_id=mid, memory_date="2026-01-01",
                            entities=[{"name": "Common", "type": "concept"}],
                            extracted_at="2026-01-01T00:00:00Z")
    # a memory with two entities still counts once
    g.record_extraction(memory_id="m3", memory_date="2026-01-01",
                        entities=[{"name": "Common", "type": "concept"},
                                  {"name": "Rare", "type": "concept"}],
                        extracted_at="2026-01-01T00:00:00Z")
    assert g.total_indexed_memories() == 3
    assert _store(tmp_path / "empty").total_indexed_memories() == 0


def test_entity_doc_freqs_batch_and_unknown(tmp_path: Path) -> None:
    g = _store(tmp_path)
    # "common" in 3 memories, "rare" in 1
    for mid in ("m1", "m2", "m3"):
        g.record_extraction(memory_id=mid, memory_date="2026-01-01",
                            entities=[{"name": "Common", "type": "concept"}],
                            extracted_at="2026-01-01T00:00:00Z")
    g.record_extraction(memory_id="m1", memory_date="2026-01-01",
                        entities=[{"name": "Common", "type": "concept"},
                                  {"name": "Rare", "type": "concept"}],
                        extracted_at="2026-01-01T00:00:00Z")
    df = g.entity_doc_freqs(["Common", "rare", "does-not-exist"])
    assert df["common"] == 3.0
    assert df["rare"] == 1.0
    assert "does-not-exist" not in df  # unknown names omitted
    assert g.entity_doc_freqs([]) == {}
