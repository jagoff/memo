from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from memo.errors import StorageError
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


class _ExecuteProxy:
    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        statement: str,
        mode: str,
    ) -> None:
        self._connection = connection
        self._statement = statement
        self._mode = mode
        self.triggered = False

    def execute(self, sql: str, parameters: Any = ()) -> sqlite3.Cursor:
        if not self.triggered and self._statement in sql:
            self.triggered = True
            if self._mode == "raise":
                raise sqlite3.OperationalError("injected ALTER failure")
            return self._connection.execute("SELECT 1")
        return self._connection.execute(sql, parameters)


def _inject_migration_execute(
    store: VecStore,
    monkeypatch: pytest.MonkeyPatch,
    *,
    statement: str,
    mode: str = "raise",
) -> _ExecuteProxy:
    original_tx = store._tx
    proxy: _ExecuteProxy | None = None

    @contextmanager
    def patched_tx() -> Iterator[Any]:
        nonlocal proxy
        with original_tx() as cx:
            if proxy is None:
                proxy = _ExecuteProxy(cx, statement=statement, mode=mode)
            yield proxy

    monkeypatch.setattr(store, "_tx", patched_tx)
    # Initialize the proxy immediately so callers can assert it triggered.
    with patched_tx():
        pass
    assert proxy is not None
    return proxy


def _downgrade_meta_schema(
    store: VecStore,
    *,
    user_version: int,
    columns: tuple[str, ...],
) -> None:
    with store._tx() as cx:
        cx.execute("DROP INDEX IF EXISTS idx_meta_active_topic_unique")
        cx.execute("DROP INDEX IF EXISTS idx_meta_topic_identity")
        cx.execute("DROP INDEX IF EXISTS idx_meta_exact_identity")
        for column in columns:
            cx.execute(f"ALTER TABLE meta DROP COLUMN {column}")
        cx.execute(f"PRAGMA user_version={user_version}")


def test_fresh_schema_has_v7_identity_relations_and_reviews(tmp_path: Path) -> None:
    store = VecStore(tmp_path / "vec.db", dims=4)
    try:
        cols = {row["name"] for row in store._conn.execute("PRAGMA table_info(meta)")}
        assert {"namespace", "normalized_title", "normalized_content_hash"} <= cols
        assert store.get_user_version() == 8
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
        assert migrated.get_user_version() == 8
        assert markdown.read_bytes() == b"canonical bytes stay unchanged\n"
    finally:
        migrated.close()


@pytest.mark.parametrize(
    ("user_version", "columns", "failing_statement"),
    [
        (
            2,
            (
                "topic_key",
                "normalized_hash",
                "session_id",
                "revision_count",
                "duplicate_count",
                "last_seen_at",
                "deleted_at",
                "review_after",
            ),
            "ADD COLUMN normalized_hash",
        ),
        (
            3,
            ("verification_state", "verified_at"),
            "ADD COLUMN verified_at",
        ),
    ],
)
def test_schema_migration_rolls_back_failed_alter_and_recovers_on_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    user_version: int,
    columns: tuple[str, ...],
    failing_statement: str,
) -> None:
    store = VecStore(tmp_path / "vec.db", dims=4)
    try:
        _downgrade_meta_schema(store, user_version=user_version, columns=columns)
        proxy = _inject_migration_execute(
            store,
            monkeypatch,
            statement=failing_statement,
        )

        with pytest.raises(sqlite3.OperationalError, match="injected ALTER failure"):
            store._run_migrations()

        assert proxy.triggered is True
        after_failure = {str(row["name"]) for row in store._conn.execute("PRAGMA table_info(meta)")}
        assert set(columns).isdisjoint(after_failure)
        assert store.get_user_version() == user_version

        store._run_migrations()

        after_retry = {str(row["name"]) for row in store._conn.execute("PRAGMA table_info(meta)")}
        assert set(columns) <= after_retry
        assert store.get_user_version() == 8
    finally:
        store.close()


def test_schema_migration_validates_columns_before_stamping_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = VecStore(tmp_path / "vec.db", dims=4)
    try:
        columns = ("verification_state", "verified_at")
        _downgrade_meta_schema(store, user_version=3, columns=columns)
        proxy = _inject_migration_execute(
            store,
            monkeypatch,
            statement="ADD COLUMN verified_at",
            mode="ignore",
        )

        with pytest.raises(StorageError, match=r"migration to v4.*verified_at"):
            store._run_migrations()

        assert proxy.triggered is True
        after_failure = {str(row["name"]) for row in store._conn.execute("PRAGMA table_info(meta)")}
        assert set(columns).isdisjoint(after_failure)
        assert store.get_user_version() == 3
    finally:
        store.close()


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
