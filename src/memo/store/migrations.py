from __future__ import annotations

from ._base import _StoreBase


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

    def get_user_version(self) -> int:
        """Return the on-disk schema version (0 by default)."""
        cur = self._conn.execute("PRAGMA user_version")
        row = cur.fetchone()
        return int(row[0]) if row else 0

    def set_user_version(self, version: int) -> None:
        """Bump the on-disk schema version. Run inside a write tx."""
        # `PRAGMA user_version = N` doesn't accept parameter binding; we
        # interpolate after asserting the value is a small integer to
        # rule out any injection vector.
        if not isinstance(version, int) or version < 0:
            raise ValueError(f"user_version must be a non-negative int, got {version!r}")
        with self._conn:
            self._conn.execute(f"PRAGMA user_version = {version}")
