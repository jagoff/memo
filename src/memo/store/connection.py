from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager, suppress

from ._base import _StoreBase

_log = logging.getLogger(__name__)

# vec0 accepts either a JSON array (text, must be parsed) or a packed
# float32 blob. Blobs skip JSON encode on write and JSON parse on every
# search MATCH — the hot path. vec0 stores float32 internally regardless,
# so existing JSON-written rows stay readable; no migration needed.


class _ConnectionMixin(_StoreBase):
    @property
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
        return conn

    def _connect(self) -> sqlite3.Connection:
        """Open + configure a connection for the calling thread, load vec0,
        and stash it on thread-local storage. Idempotent per thread."""
        conn = sqlite3.connect(str(self.db_path), timeout=10.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 10000")
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError as exc:
            # WAL is what makes concurrent readers + a writer safe. If the
            # filesystem can't support it (e.g. some network mounts) we fall
            # back to the rollback journal — slower concurrency, still
            # correct — but surface it so the degradation isn't silent.
            _log.warning("could not enable WAL journal mode on %s: %s", self.db_path, exc)
        self._load_vec0(conn)
        self._local.conn = conn
        return conn

    def _load_vec0(self, conn: sqlite3.Connection) -> None:
        import sqlite_vec  # type: ignore[import-not-found]

        # `enable_load_extension` must be called BEFORE `load_extension`.
        # Wrapped in try/except because some Python builds disable it
        # for security reasons — we surface a clear error in that case.
        try:
            conn.enable_load_extension(True)
        except sqlite3.NotSupportedError as exc:
            from ..errors import StorageError

            raise StorageError(
                "Python's sqlite3 was compiled without `enable_load_extension`. "
                "Reinstall Python via Homebrew (`brew install python@3.13`) which "
                "bundles a sqlite3 with extension support enabled."
            ) from exc
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        # `BEGIN IMMEDIATE` acquires the write lock up-front so a
        # concurrent reader on the same connection doesn't observe a
        # half-written record. SQLite WAL mode lets readers continue
        # against the snapshot during the write.
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def _checkpoint(self) -> None:
        """Truncate the WAL back into the main DB. Call after large batch
        writes (repo indexing) so the -wal file doesn't grow unbounded
        before the default autocheckpoint (1000 pages) fires, which keeps
        crash-recovery fast. Best-effort: a checkpoint can be blocked by a
        concurrent reader, in which case autocheckpoint catches up later."""
        with suppress(sqlite3.OperationalError):
            self._conn.execute("PRAGMA wal_checkpoint(RESTART)")

    def close(self) -> None:
        # Closes the calling thread's connection. Other threads' connections
        # are released when their threads end (or at process exit) — adequate
        # for the daemon/CLI lifecycles that use this store.
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            with suppress(Exception):
                conn.close()
            self._local.conn = None
