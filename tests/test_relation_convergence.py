from __future__ import annotations

import sqlite3

import pytest

from memo.contradict import ContradictionStore
from memo.errors import RelationConflictError
from memo.store import VecStore


@pytest.fixture
def relation_store(tmp_path):
    store = VecStore(tmp_path / "relations.db", dims=4, embedder_model="stub")
    yield store
    store.close()


def test_relation_candidate_is_idempotent_for_unordered_pair(relation_store):
    first = relation_store.create_relation_candidate(source_id="new", target_id="old")
    second = relation_store.create_relation_candidate(source_id="old", target_id="new")

    assert first["id"] == second["id"]
    assert first["source_id"] == "new"
    assert second["source_id"] == "new"
    assert relation_store.relation_stats() == {"pending": 1}


def test_relation_judgment_is_idempotent_and_conflict_safe(relation_store):
    candidate = relation_store.create_relation_candidate(source_id="new", target_id="old")
    judged = relation_store.commit_relation_judgment(
        relation_id=candidate["id"],
        relation="compatible",
        reason="same scope",
        confidence=0.8,
        actor="agent",
    )
    repeated = relation_store.commit_relation_judgment(
        relation_id=candidate["id"],
        relation="compatible",
        reason="ignored on idempotent replay",
        confidence=0.1,
    )

    assert judged["judgment_status"] == "judged"
    assert repeated["reason"] == "same scope"
    with pytest.raises(RelationConflictError):
        relation_store.commit_relation_judgment(
            relation_id=candidate["id"],
            relation="conflicts_with",
            reason=None,
            confidence=1.0,
        )
    assert relation_store.get_relation(candidate["id"])["relation"] == "compatible"


def test_relation_endpoints_are_orphaned_not_deleted(relation_store):
    candidate = relation_store.create_relation_candidate(source_id="a", target_id="b")
    assert relation_store.orphan_relations_for("a") == 1
    row = relation_store.get_relation(candidate["id"])
    assert row is not None
    assert row["judgment_status"] == "orphaned"


def test_v5_relation_rows_migrate_non_destructively(tmp_path):
    path = tmp_path / "legacy.db"
    store = VecStore(path, dims=4, embedder_model="stub")
    with store._tx() as cx:
        cx.execute("DROP INDEX IF EXISTS idx_rel_pair_unique")
        cx.execute(
            "INSERT INTO memory_relations "
            "(id, source_id, target_id, relation, judgment_status, created_at) "
            "VALUES ('legacy', 'a', 'b', 'related', 'judged', '2026-01-01')"
        )
        cx.execute("UPDATE memory_relations SET pair_key=NULL WHERE id='legacy'")
        cx.execute("PRAGMA user_version=5")
    store.close()

    migrated = VecStore(path, dims=4, embedder_model="stub")
    row = migrated.get_relation("legacy")
    assert row is not None and row["pair_key"]
    assert migrated.get_user_version() == 8
    migrated.close()


def test_relation_pair_unique_index_rejects_duplicate_low_level_insert(relation_store):
    first = relation_store.create_relation_candidate(source_id="a", target_id="b")
    with pytest.raises(sqlite3.IntegrityError):
        relation_store._conn.execute(
            "INSERT INTO memory_relations "
            "(id, pair_key, source_id, target_id, judgment_status) "
            "VALUES ('other', ?, 'b', 'a', 'pending')",
            (first["pair_key"],),
        )


def test_v5_database_without_relation_table_migrates(tmp_path):
    path = tmp_path / "missing-relations.db"
    store = VecStore(path, dims=4, embedder_model="stub")
    with store._tx() as cx:
        cx.execute("DROP TABLE memory_relations")
        cx.execute("PRAGMA user_version=5")
    store.close()

    migrated = VecStore(path, dims=4, embedder_model="stub")
    candidate = migrated.create_relation_candidate(source_id="a", target_id="b")

    assert candidate["judgment_status"] == "pending"
    assert migrated.get_user_version() == 8
    migrated.close()


def test_v7_integer_relation_table_rebuilds_with_text_identity(tmp_path):
    path = tmp_path / "integer-relations.db"
    store = VecStore(path, dims=4, embedder_model="stub")
    with store._tx() as cx:
        for index in (
            "idx_rel_source",
            "idx_rel_target",
            "idx_rel_status",
            "idx_rel_pair_unique",
            "idx_rel_migration_unique",
        ):
            cx.execute(f"DROP INDEX IF EXISTS {index}")
        cx.execute("DROP TABLE memory_relations")
        cx.execute(
            "CREATE TABLE memory_relations ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, sync_id TEXT UNIQUE, "
            "source_id INTEGER NOT NULL, target_id INTEGER NOT NULL, relation TEXT, "
            "judgment_status TEXT DEFAULT 'pending', reason TEXT, confidence REAL, "
            "session_id TEXT, created_at TEXT NOT NULL, updated_at TEXT)"
        )
        cx.execute(
            "INSERT INTO memory_relations "
            "(sync_id, source_id, target_id, relation, judgment_status, created_at) "
            "VALUES ('legacy-sync', 'a', 'b', 'related', 'judged', '2026-01-01')"
        )
        cx.execute("PRAGMA user_version=7")
    store.close()

    migrated = VecStore(path, dims=4, embedder_model="stub")
    column_types = {
        row["name"]: row["type"]
        for row in migrated._conn.execute("PRAGMA table_info(memory_relations)").fetchall()
    }
    legacy = migrated.get_relation("legacy-sync")
    candidate = migrated.create_relation_candidate(source_id="new", target_id="old")

    assert column_types["id"] == "TEXT"
    assert column_types["source_id"] == "TEXT"
    assert column_types["target_id"] == "TEXT"
    assert legacy is not None and legacy["pair_key"]
    assert candidate["id"].startswith("rel-")
    assert migrated.get_user_version() == 8
    migrated.close()


def test_legacy_contradictions_import_once_and_project_from_canonical(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_RELATION_CANDIDATES_ENABLED", "0")
    first = mock_memory.save(content="old backend choice", title="old choice")
    second = mock_memory.save(content="new backend choice", title="new choice")
    legacy = ContradictionStore(mock_memory.cfg.contradictions_db)
    pair_id = legacy.upsert_open(first.id, second.id, "contradiction", 0.91, "backend changed")
    legacy.resolve(pair_id, "dismissed", note="not actually conflicting")
    legacy.close()

    rows = mock_memory.contradict_store.list_all(status="dismissed")
    second_read = mock_memory.import_legacy_contradictions()

    assert [row.pair_id for row in rows] == [pair_id]
    assert mock_memory.store.relation_stats() == {"judged": 1}
    assert second_read == {"seen": 1, "imported": 0, "existing": 1}
