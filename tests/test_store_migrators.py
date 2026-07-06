"""Mem0/Zep dump → memo import-record mappers (adoption funnel for mlx-memo)."""

from __future__ import annotations

from memo.store_migrators import mem0_to_import_records, zep_to_import_records


def test_mem0_mapper_handles_results_envelope_and_bare_list():
    item = {
        "id": "m-1",
        "memory": "User prefers dark mode in all editors",
        "categories": ["preferences"],
        "created_at": "2025-11-02T10:00:00Z",
    }
    for payload in ([item], {"results": [item]}):
        recs = mem0_to_import_records(payload)
        assert len(recs) == 1
        rec = recs[0]
        assert rec["content"] == "User prefers dark mode in all editors"
        assert rec["title"] == "User prefers dark mode in all editors"
        assert rec["type"] == "fact"
        assert "imported:mem0" in rec["tags"] and "preferences" in rec["tags"]
        assert rec["created"] == "2025-11-02T10:00:00Z"


def test_mem0_mapper_skips_empty_and_non_dict():
    assert mem0_to_import_records([{"memory": ""}, "garbage", None]) == []


def test_zep_mapper_skips_invalidated_facts():
    payload = {
        "facts": [
            {"fact": "usuario vive en Buenos Aires", "created_at": "2025-01-01T00:00:00Z"},
            {"fact": "usuario vive en Madrid", "created_at": "2024-01-01T00:00:00Z",
             "invalid_at": "2025-01-01T00:00:00Z"},
        ]
    }
    recs = zep_to_import_records(payload)
    assert len(recs) == 1
    assert recs[0]["content"] == "usuario vive en Buenos Aires"
    assert recs[0]["tags"] == ["imported:zep"]
    assert recs[0]["type"] == "fact"
