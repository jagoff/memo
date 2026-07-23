from __future__ import annotations

import contextlib
import json
import logging

from ._base import _StoreBase

_log = logging.getLogger(__name__)

_CURRENT_USER_VERSION = 7


class _MigrationsMixin(_StoreBase):
    # -- schema-version helpers --------------------------------------------
    #
    # We use SQLite's built-in `PRAGMA user_version` (an INTEGER stored in
    # the DB header — zero schema cost) to track the on-disk layout of
    # store-managed paths. Versions:
    #   0 — pre-`memo init` install. Paths in `meta.path` MAY carry a
    #       legacy `<vault_subdir>/...` prefix relative to `vault_path`.
    #       Reads use the `Memory._resolve_existing` legacy fallback.
    #   1 — post-`memo migrate-vault`. Paths in `meta.path` are relative
    #       to `cfg.memory_dir` directly. Set after a successful reindex.
    #   2 — signal-table rows (access, memory_health) seeded on every meta
    #       upsert so prune/eviction queries can use direct column access
    #       instead of COALESCE over LEFT JOIN.  Migration backfills any
    #       meta rows that predate this change.
    #   3 — session pattern columns (topic_key, normalized_hash, etc.)
    #       added for tracking memory deduplication and session context.
    #   4 — verification state tracking columns (verification_state, verified_at)
    #       added for explicit verification state (UNVERIFIED/VERIFIED/STALE)
    #       and temporal decay weighting in rerank.
    #   5 — rebuildable canonical identity columns (namespace,
    #       normalized_title, normalized_content_hash).
    #   6 — canonical relation provenance and pair idempotency metadata.
    #   7 — review evidence signal table.

    def get_user_version(self) -> int:
        """Return the on-disk schema version (0 by default)."""
        cur = self._conn.execute("PRAGMA user_version")
        row = cur.fetchone()
        return int(row[0]) if row else 0

    def set_user_version(self, version: int) -> None:
        """Monotonically bump the on-disk schema version. Run inside a write tx."""
        if not isinstance(version, int) or version < 0:
            raise ValueError(f"user_version must be a non-negative int, got {version!r}")
        if version <= self.get_user_version():
            return
        self._conn.execute(f"PRAGMA user_version = {version}")

    def _run_migrations(self) -> None:
        """Run pending schema migrations in order.

        Called from _init_schema_locked after DDL creation. Fresh databases
        (version 0, empty meta table) are stamped at version 2 immediately.
        Existing databases are migrated version-by-version.
        """
        current = self.get_user_version()
        if current >= _CURRENT_USER_VERSION:
            return
        if current == 0:
            row = self._conn.execute("SELECT COUNT(*) FROM meta").fetchone()
            if row and int(row[0]) == 0:
                # Fresh DB — the schema DDL already created the current shape,
                # so stamp it current without running backfill migrations.
                self.set_user_version(_CURRENT_USER_VERSION)
                return

        # v1 → v2: backfill access + memory_health rows for meta rows
        # that predate the auto-seed on upsert.
        if current < 2:
            with self._tx() as cx:
                cx.execute(
                    "INSERT OR IGNORE INTO access (id, access_count, last_accessed) "
                    "SELECT id, 0, updated FROM meta"
                )
                cx.execute(
                    "INSERT OR IGNORE INTO memory_health (id, confidence, roi_score, updated_at) "
                    "SELECT id, 1.0, 1.0, datetime('now') FROM meta"
                )
                self.set_user_version(2)
                current = 2
            _log.info("backfilled signal rows for v2: access + memory_health")

        # v2 → v3: add session pattern columns (topic_key, normalized_hash, etc.)
        if current < 3:
            with self._tx() as cx:
                cols = {row["name"] for row in cx.execute("PRAGMA table_info(meta)").fetchall()}
                new_cols = {
                    "topic_key": "ALTER TABLE meta ADD COLUMN topic_key TEXT",
                    "normalized_hash": "ALTER TABLE meta ADD COLUMN normalized_hash TEXT",
                    "session_id": "ALTER TABLE meta ADD COLUMN session_id TEXT",
                    "revision_count": "ALTER TABLE meta ADD COLUMN revision_count INTEGER DEFAULT 1",
                    "duplicate_count": "ALTER TABLE meta ADD COLUMN duplicate_count INTEGER DEFAULT 0",
                    "last_seen_at": "ALTER TABLE meta ADD COLUMN last_seen_at TEXT",
                    "deleted_at": "ALTER TABLE meta ADD COLUMN deleted_at TEXT",
                    "review_after": "ALTER TABLE meta ADD COLUMN review_after TEXT",
                }
                for col, ddl in new_cols.items():
                    if col not in cols:
                        with contextlib.suppress(Exception):
                            cx.execute(ddl)
                self.set_user_version(3)
            _log.info("migrated to v3: session pattern columns")

        # v3 → v4: add verification state tracking columns
        if current < 4:
            with self._tx() as cx:
                cols = {row["name"] for row in cx.execute("PRAGMA table_info(meta)").fetchall()}
                new_cols = {
                    "verification_state": "ALTER TABLE meta ADD COLUMN verification_state TEXT DEFAULT 'unverified'",
                    "verified_at": "ALTER TABLE meta ADD COLUMN verified_at INTEGER",
                }
                for col, ddl in new_cols.items():
                    if col not in cols:
                        with contextlib.suppress(Exception):
                            cx.execute(ddl)
                self.set_user_version(4)
            _log.info("migrated to v4: verification state tracking columns")
            current = 4

        # v4 → v5: additive, derived identity metadata. Markdown remains
        # untouched; indexed FTS text is used only to seed rebuildable fields.
        if current < 5:
            from memo.identity import (
                canonical_topic_key,
                namespace_for_index,
                normalized_content_hash,
                normalized_title,
            )
            from memo.redact import sanitize_persisted_text

            with self._tx() as cx:
                cols = {row["name"] for row in cx.execute("PRAGMA table_info(meta)").fetchall()}
                new_cols = {
                    "namespace": "ALTER TABLE meta ADD COLUMN namespace TEXT",
                    "normalized_title": "ALTER TABLE meta ADD COLUMN normalized_title TEXT",
                    "normalized_content_hash": (
                        "ALTER TABLE meta ADD COLUMN normalized_content_hash TEXT"
                    ),
                }
                for col, ddl in new_cols.items():
                    if col not in cols:
                        cx.execute(ddl)

                rows = cx.execute(
                    "SELECT m.id, m.path, m.title, m.tags, m.topic_key, "
                    "COALESCE(f.body, '') AS body "
                    "FROM meta m LEFT JOIN fts f ON f.id = m.id"
                ).fetchall()
                for row in rows:
                    try:
                        raw_tags = json.loads(row["tags"] or "[]")
                    except (TypeError, ValueError):
                        raw_tags = []
                    tags = [str(tag) for tag in raw_tags] if isinstance(raw_tags, list) else []
                    safe_title = sanitize_persisted_text(str(row["title"] or "")).text
                    safe_body = sanitize_persisted_text(str(row["body"] or "")).text
                    cx.execute(
                        "UPDATE meta SET namespace = ?, topic_key = ?, "
                        "normalized_title = ?, normalized_content_hash = ? WHERE id = ?",
                        (
                            namespace_for_index(tags, path=str(row["path"] or "")),
                            canonical_topic_key(row["topic_key"]),
                            normalized_title(safe_title),
                            normalized_content_hash(safe_body),
                            row["id"],
                        ),
                    )
                cx.execute(
                    "CREATE INDEX IF NOT EXISTS idx_meta_topic_identity "
                    "ON meta(namespace, topic_key)"
                )
                cx.execute(
                    "CREATE INDEX IF NOT EXISTS idx_meta_exact_identity "
                    "ON meta(namespace, type, normalized_title, normalized_content_hash)"
                )
                self.set_user_version(5)
            _log.info("migrated to v5: canonical memory identity metadata")
            current = 5

        if current < 6:
            from memo.errors import ValidationError
            from memo.store.relation_queries import relation_pair_key

            with self._tx() as cx:
                # Some v5 databases predate the experimental session-relation
                # table entirely. Migrations run before schema.py's inline
                # guards, so establish the legacy baseline here first.
                cx.execute(
                    "CREATE TABLE IF NOT EXISTS memory_relations ("
                    "id TEXT PRIMARY KEY, sync_id TEXT, source_id TEXT NOT NULL, "
                    "target_id TEXT NOT NULL, relation TEXT, "
                    "judgment_status TEXT DEFAULT 'pending', reason TEXT, "
                    "confidence REAL, session_id TEXT, created_at TEXT, updated_at TEXT)"
                )
                cols = {
                    row["name"]
                    for row in cx.execute("PRAGMA table_info(memory_relations)").fetchall()
                }
                additions = {
                    "pair_key": "ALTER TABLE memory_relations ADD COLUMN pair_key TEXT",
                    "actor": "ALTER TABLE memory_relations ADD COLUMN actor TEXT",
                    "actor_kind": "ALTER TABLE memory_relations ADD COLUMN actor_kind TEXT",
                    "model": "ALTER TABLE memory_relations ADD COLUMN model TEXT",
                    "provenance_json": (
                        "ALTER TABLE memory_relations ADD COLUMN provenance_json TEXT"
                    ),
                    "migration_key": ("ALTER TABLE memory_relations ADD COLUMN migration_key TEXT"),
                    "migrated_from": ("ALTER TABLE memory_relations ADD COLUMN migrated_from TEXT"),
                }
                for column, ddl in additions.items():
                    if column not in cols:
                        cx.execute(ddl)
                rows = cx.execute(
                    "SELECT id, source_id, target_id, created_at FROM memory_relations "
                    "ORDER BY COALESCE(created_at, ''), id"
                ).fetchall()
                claimed: set[str] = set()
                for row in rows:
                    try:
                        key = relation_pair_key(str(row["source_id"]), str(row["target_id"]))
                    except ValidationError as exc:
                        _log.warning(
                            "relation migration skipped invalid row %s: %s", row["id"], exc
                        )
                        continue
                    if key in claimed:
                        cx.execute(
                            "UPDATE memory_relations SET judgment_status='orphaned' WHERE id=?",
                            (row["id"],),
                        )
                        continue
                    claimed.add(key)
                    cx.execute(
                        "UPDATE memory_relations SET pair_key=? WHERE id=?", (key, row["id"])
                    )
                cx.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_rel_pair_unique "
                    "ON memory_relations(pair_key) WHERE pair_key IS NOT NULL"
                )
                cx.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_rel_migration_unique "
                    "ON memory_relations(migration_key) WHERE migration_key IS NOT NULL"
                )
                self.set_user_version(6)
            _log.info("migrated to v6: canonical relation metadata")
            current = 6

        if current < 7:
            with self._tx() as cx:
                cx.execute(
                    "CREATE TABLE IF NOT EXISTS memory_reviews ("
                    "id TEXT PRIMARY KEY, memory_id TEXT NOT NULL, reviewed_at TEXT NOT NULL, "
                    "evidence TEXT, actor TEXT, next_review_after TEXT)"
                )
                cx.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memory_reviews_memory "
                    "ON memory_reviews(memory_id, reviewed_at)"
                )
                self.set_user_version(7)
            _log.info("migrated to v7: review evidence metadata")
