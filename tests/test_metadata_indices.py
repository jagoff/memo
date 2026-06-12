"""FTS5/B-tree metadata indices on `meta` for fast field/exact lookups.

`meta(path)` (exact + prefix LIKE) and `meta(type, updated)` (type-filtered
recency) were unindexed — every such query was a full table scan. The
migration is idempotent and runs on existing DBs, not just fresh ones.
"""

from __future__ import annotations


def _index_names(conn, table: str) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=?",
        (table,),
    ).fetchall()
    return {r["name"] for r in rows}


def test_meta_has_secondary_indices(mock_memory):
    names = _index_names(mock_memory.store._conn, "meta")
    assert "idx_meta_path" in names
    assert "idx_meta_type_updated" in names


def test_secondary_index_migration_is_idempotent_and_self_heals(mock_memory):
    conn = mock_memory.store._conn
    conn.execute("DROP INDEX IF EXISTS idx_meta_path")
    assert "idx_meta_path" not in _index_names(conn, "meta")
    # Re-running the migration recreates the missing index without error.
    mock_memory.store._ensure_secondary_indices()
    mock_memory.store._ensure_secondary_indices()  # idempotent second pass
    assert "idx_meta_path" in _index_names(conn, "meta")
