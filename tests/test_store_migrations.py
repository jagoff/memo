from __future__ import annotations

from pathlib import Path

from memo.identity import normalized_content_hash, normalized_title
from memo.store import VecStore


def _emb() -> list[float]:
    return [1.0, 0.0, 0.0, 0.0]


def _put(
    store: VecStore,
    id_: str,
    *,
    path: str,
    title: str = "Plan",
    body: str = "Body",
    topic_key: str | None = None,
    namespace: str | None = None,
) -> None:
    store.upsert(
        id_=id_,
        path=path,
        title=title,
        type_="note",
        tags=["project:memo"] if namespace == "project:memo" else [],
        created="2026-01-01T00:00:00+00:00",
        updated="2026-01-01T00:00:00+00:00",
        body_hash=f"hash-{id_}",
        embedding=_emb(),
        body_text=body,
        topic_key=topic_key,
        namespace=namespace,
    )


def test_fresh_schema_has_v5_identity_and_independent_capability(tmp_path: Path) -> None:
    store = VecStore(tmp_path / "vec.db", dims=4)
    try:
        cols = {row["name"] for row in store._conn.execute("PRAGMA table_info(meta)")}
        assert {"namespace", "normalized_title", "normalized_content_hash"} <= cols
        assert store.get_user_version() == 5
        capability = store._conn.execute(
            "SELECT value FROM schema_meta WHERE key='identity_topic_unique'"
        ).fetchone()
        assert capability["value"] == "enabled"
    finally:
        store.close()


def test_v4_migration_backfills_identity_without_touching_markdown(tmp_path: Path) -> None:
    db_path = tmp_path / "vec.db"
    markdown = tmp_path / "memory.md"
    markdown.write_bytes(b"canonical bytes stay unchanged\n")
    store = VecStore(db_path, dims=4)
    _put(
        store,
        "a" * 32,
        path="memo/2026/plan.md",
        title="  PLAN  ",
        body="Body\r\n",
        topic_key=" Release  Plan ",
        namespace="project:memo",
    )
    store._conn.execute("DROP INDEX IF EXISTS idx_meta_active_topic_unique")
    store._conn.execute("DROP INDEX IF EXISTS idx_meta_topic_identity")
    store._conn.execute("DROP INDEX IF EXISTS idx_meta_exact_identity")
    for column in ("normalized_content_hash", "normalized_title", "namespace"):
        store._conn.execute(f"ALTER TABLE meta DROP COLUMN {column}")
    store._conn.execute("DELETE FROM schema_meta WHERE key='identity_topic_unique'")
    store._conn.execute("PRAGMA user_version=4")
    store._conn.commit()
    store.close()

    migrated = VecStore(db_path, dims=4)
    try:
        keys = migrated.get_identity_keys("a" * 32)
        assert keys == {
            "namespace": "project:memo",
            "topic_key": "release plan",
            "normalized_title": normalized_title("PLAN"),
            "normalized_content_hash": normalized_content_hash("Body"),
        }
        assert migrated.get_user_version() == 5
        assert markdown.read_bytes() == b"canonical bytes stay unchanged\n"
    finally:
        migrated.close()


def test_topic_constraint_blocks_legacy_conflict_then_reenables(tmp_path: Path) -> None:
    store = VecStore(tmp_path / "vec.db", dims=4)
    try:
        store._conn.execute("DROP INDEX idx_meta_active_topic_unique")
        store._conn.execute(
            "UPDATE schema_meta SET value='blocked' WHERE key='identity_topic_unique'"
        )
        store._conn.commit()
        _put(
            store,
            "a" * 32,
            path="memo/a.md",
            topic_key="same",
            namespace="project:memo",
        )
        _put(
            store,
            "b" * 32,
            path="memo/b.md",
            topic_key="same",
            namespace="project:memo",
        )
        assert store.reconcile_identity_constraint(force=True) == "blocked"
        assert store.identity_diagnostics()["topic_collision_groups"] == 1

        store.delete("b" * 32)
        assert store.reconcile_identity_constraint(force=True) == "enabled"
        assert len(store.find_active_by_topic_identity("project:memo", "same")) == 1
    finally:
        store.close()


def test_exact_lookup_and_corroboration_share_canonical_signal(tmp_path: Path) -> None:
    store = VecStore(tmp_path / "vec.db", dims=4)
    try:
        record_id = "a" * 32
        _put(
            store,
            record_id,
            path="memo/a.md",
            title="Plan",
            body="Body",
            namespace="project:memo",
        )
        matches = store.find_active_by_exact_identity(
            "project:memo",
            "note",
            normalized_title("Plan"),
            normalized_content_hash("Body"),
        )
        assert [row["id"] for row in matches] == [record_id]
        before = store._conn.execute(
            "SELECT updated_at FROM memory_health WHERE id=?", (record_id,)
        ).fetchone()["updated_at"]
        result = store.corroborate_identity(record_id, seen_at="2026-02-01T00:00:00+00:00")
        after = store._conn.execute(
            "SELECT updated_at FROM memory_health WHERE id=?", (record_id,)
        ).fetchone()["updated_at"]
        assert result["support_count"] == 1
        assert result["duplicate_count"] == 1
        assert before == after
    finally:
        store.close()
