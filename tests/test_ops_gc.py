"""Tests for pure GC logic (ported from synapse ops on deprecation)."""

from memo.ops_gc import find_exact_duplicates, find_vault_orphans


def _rec(id_, body="x", updated="2026-01-02", source="", abs_path=None):
    extra = {"source": source}
    if abs_path is not None:
        extra["abs_path"] = abs_path
    return {"id": id_, "body": body, "updated": updated, "created": "2026-01-01", "extra": extra}


def test_orphans_only_vault_ingest_missing_path():
    recs = [
        _rec("a", source="vault-ingest:notes", abs_path="/nope/gone.md"),
        _rec("b", source="vault-ingest:notes", abs_path="/exists.md"),
        _rec("c", source="chat", abs_path="/nope/gone.md"),
        _rec("d", source="vault-ingest:notes"),
    ]
    got = find_vault_orphans(recs, path_exists=lambda p: p == "/exists.md")
    assert [r["id"] for r in got] == ["a"]


def test_orphans_empty_records():
    assert find_vault_orphans([]) == []


def test_exact_duplicates_keep_newest():
    recs = [
        _rec("old", body="same", updated="2026-01-01"),
        _rec("new", body="same", updated="2026-02-01"),
        _rec("uniq", body="other"),
        _rec("empty1", body="  "),
        _rec("empty2", body="  "),
    ]
    stale = find_exact_duplicates(recs)
    assert [r["id"] for r in stale] == ["old"]


def test_exact_duplicates_falls_back_to_created():
    recs = [
        _rec("a", body="same", updated=""),
        _rec("b", body="same", updated=""),
    ]
    recs[0]["created"] = "2026-03-01"
    recs[1]["created"] = "2026-01-01"
    stale = find_exact_duplicates(recs)
    assert [r["id"] for r in stale] == ["b"]
