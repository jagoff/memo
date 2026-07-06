"""GraphStore.list_entities + merge_entity_pair — substrate for entity-canon."""

from __future__ import annotations

from memo.graph import GraphStore


def _seed(g: GraphStore) -> None:
    g.record_extraction(
        memory_id="m1",
        memory_date="2026-07-01",
        entities=[{"name": "memo recall daemon", "type": "technology"}],
        extracted_at="2026-07-01T00:00:00Z",
    )
    g.record_extraction(
        memory_id="m2",
        memory_date="2026-07-02",
        entities=[{"name": "memo recall daemons", "type": "technology"}],
        extracted_at="2026-07-02T00:00:00Z",
    )


def test_list_entities_returns_id_name_type_mentions(tmp_cfg):
    g = GraphStore(tmp_cfg.graph_db)
    _seed(g)
    rows = g.list_entities()
    assert {r["name"] for r in rows} == {"memo recall daemon", "memo recall daemons"}
    assert all({"id", "name", "type", "mention_count"} <= set(r) for r in rows)


def test_merge_entity_pair_repoints_links_and_recounts(tmp_cfg):
    g = GraphStore(tmp_cfg.graph_db)
    _seed(g)
    rows = {r["name"]: r for r in g.list_entities()}
    keep, drop = rows["memo recall daemon"], rows["memo recall daemons"]

    g.merge_entity_pair(keep["id"], drop["id"], drop["name"])

    assert set(g.entity_memories("memo recall daemon")) == {"m1", "m2"}
    assert g.entity_memories("memo recall daemons") == []
    merged = {r["name"]: r for r in g.list_entities()}
    assert set(merged) == {"memo recall daemon"}
    assert merged["memo recall daemon"]["mention_count"] == 2
