from __future__ import annotations

import logging
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager, suppress

from ..errors import StorageError
from ..sqlite_compat import import_sqlite_vec
from ._base import _StoreBase

_log = logging.getLogger(__name__)

# Message SQLite raises when a write cannot acquire the lock within
# busy_timeout — a concurrent writer (another agent session, or an external
# tool syncing the DB inside a vault) holds it. ``sqlite3`` reports both
# SQLITE_BUSY and SQLITE_LOCKED with these strings.
_LOCK_MARKERS = ("database is locked", "database is busy")
_LOCK_MSG = "memo write blocked: the database is locked by a concurrent writer; retry the call"


def _is_lock_error(exc: sqlite3.OperationalError) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in _LOCK_MARKERS)


# vec0 accepts either a JSON array (text, must be parsed) or a packed
# float32 blob. Blobs skip JSON encode on write and JSON parse on every
# search MATCH — the hot path. vec0 stores float32 internally regardless,
# so existing JSON-written rows stay readable; no migration needed.


class _ConnectionHolder:
    """Track one thread's sqlite connection for deterministic cleanup.

    Strongly held by the store: closed by ``close()`` at shutdown, or by the
    dead-owner sweep in ``_connect()`` once the owning thread has exited —
    never by platform-dependent thread-local finalization order."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn: sqlite3.Connection | None = conn
        self.owner: threading.Thread = threading.current_thread()

    def close(self) -> None:
        conn, self.conn = self.conn, None
        if conn is not None:
            with suppress(BaseException):
                conn.close()

    def __del__(self) -> None:  # pragma: no cover - thread-exit cleanup
        with suppress(BaseException):
            self.close()


class _ConnectionMixin(_StoreBase):
    @property
    def _conn(self) -> sqlite3.Connection:
        holder = getattr(self._local, "conn_holder", None)
        if holder is None or holder.conn is None:
            return self._connect()
        return holder.conn

    @_conn.setter
    def _conn(self, conn: sqlite3.Connection) -> None:
        self._local.conn_holder = _ConnectionHolder(conn)

    @_conn.deleter
    def _conn(self) -> None:
        holder = getattr(self._local, "conn_holder", None)
        if holder is not None:
            holder.close()
        self._local.conn_holder = None

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
        try:
            self._load_vec0(conn)
        except Exception:
            conn.close()
            raise
        holder = _ConnectionHolder(conn)
        self._local.conn_holder = holder
        holders = getattr(self, "_conn_holders", None)
        holders_lock = getattr(self, "_conn_holders_lock", None)
        if holders is not None and holders_lock is not None:
            with holders_lock:
                # Sweep holders owned by exited threads: a thread-local
                # connection can never be reused after its thread dies, so
                # without this every dead worker/request thread would leak
                # one open connection for the life of the process (the
                # always-on recall daemon spawns one thread per client).
                for stale in [h for h in holders if h.conn is None or not h.owner.is_alive()]:
                    stale.close()
                    holders.discard(stale)
                holders.add(holder)
        return conn

    def _load_vec0(self, conn: sqlite3.Connection) -> None:
        sqlite_vec = import_sqlite_vec()

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
        try:
            self._conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            # Lock not acquired within busy_timeout. Surface a typed,
            # actionable StorageError instead of a raw sqlite error so callers
            # — notably the MCP write coordinator — report the real cause and
            # retry semantics rather than the opaque "write failed safely".
            # Non-lock errors keep propagating unchanged for their own handlers.
            if _is_lock_error(exc):
                raise StorageError(_LOCK_MSG) from exc
            raise
        try:
            yield self._conn
            self._conn.commit()
        except sqlite3.OperationalError as exc:
            self._conn.rollback()
            if _is_lock_error(exc):
                raise StorageError(_LOCK_MSG) from exc
            raise
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
        # Close every live thread-local holder this store has opened. FastMCP
        # HTTP worker threads may outlive the request that created them, so
        # process shutdown cannot rely on thread-exit cleanup alone.
        holders = getattr(self, "_conn_holders", None)
        holders_lock = getattr(self, "_conn_holders_lock", None)
        if holders is not None and holders_lock is not None:
            with holders_lock:
                for tracked in list(holders):
                    tracked.close()
                holders.clear()
        holder = getattr(self._local, "conn_holder", None)
        if holder is not None:
            holder.close()
            self._local.conn_holder = None
