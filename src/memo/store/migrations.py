from __future__ import annotations

import contextlib
import logging

from ._base import _StoreBase

_log = logging.getLogger(__name__)


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

    def get_user_version(self) -> int:
        """Return the on-disk schema version (0 by default)."""
        cur = self._conn.execute("PRAGMA user_version")
        row = cur.fetchone()
        return int(row[0]) if row else 0

    def set_user_version(self, version: int) -> None:
        """Bump the on-disk schema version. Run inside a write tx."""
        if not isinstance(version, int) or version < 0:
            raise ValueError(f"user_version must be a non-negative int, got {version!r}")
        self._conn.execute(f"PRAGMA user_version = {version}")

    def _run_migrations(self) -> None:
        """Run pending schema migrations in order.

        Called from _init_schema_locked after DDL creation. Fresh databases
        (version 0, empty meta table) are stamped at version 2 immediately.
        Existing databases are migrated version-by-version.
        """
        current = self.get_user_version()
        if current == 0:
            row = self._conn.execute("SELECT COUNT(*) FROM meta").fetchone()
            if row and int(row[0]) == 0:
                # Fresh DB — stamp at version 2 immediately (skips 1,
                # which was only about migratable paths).
                self.set_user_version(2)
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

        # v2 → v3: add engram pattern columns (topic_key, normalized_hash, etc.)
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
            _log.info("migrated to v3: engram pattern columns")
