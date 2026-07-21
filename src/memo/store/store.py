"""sqlite-vec-backed vector store for memory records.

Single file, single writer. No daemon. Metadata + vectors live in the
same DB file under separate tables; `memo reindex --rebuild` can replay
markdown-derived state while preserving signal tables, which is simpler
than splitting across qdrant + sqlite.

## Schema

Two tables:

```
CREATE TABLE meta (
    id          TEXT PRIMARY KEY,        -- UUID4 hex
    path        TEXT UNIQUE NOT NULL,    -- vault-relative .md path
    title       TEXT NOT NULL,
    type        TEXT NOT NULL,           -- decision/fact/bug/feedback/preference/note
    tags        TEXT NOT NULL,           -- json array
    created     TEXT NOT NULL,           -- ISO8601
    updated     TEXT NOT NULL,           -- ISO8601
    body_hash   TEXT NOT NULL,           -- sha256[:16] of raw body for change detection
    extra_json  TEXT                     -- arbitrary metadata bag
);

-- sqlite-vec virtual table; one row per chunk.
CREATE VIRTUAL TABLE vec USING vec0(
    id          TEXT,                    -- references meta.id (no FK; vec0 is virtual)
    embedding   FLOAT[1024]              -- L2-normalised, dot-product = cosine
);
```

`id` deliberately denormalised across both tables (vec0 is virtual,
SQLite cannot enforce a real foreign key against it). The store
guarantees consistency by always writing to both within the same
`BEGIN IMMEDIATE` transaction.

## Why dot product (not L2 distance)?

The embedder L2-normalises every vector. Under that constraint, the
cosine similarity equals `1 - 0.5 * l2_distance²`, but more directly:
`cosine = a·b`. sqlite-vec's `vec_distance_cosine` is fine; we use it
for clarity even though `vec_distance_l2` would rank identically.
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
import threading
from pathlib import Path

from .connection import _ConnectionHolder, _ConnectionMixin
from .feedback_store import _FeedbackMixin
from .migrations import _MigrationsMixin
from .queries import _QueriesMixin
from .repo_store import _RepoStoreMixin
from .schema import _SchemaMixin
from .tantivy_index import TantivyFTSIndex, _tantivy_available
from .vec_base import VecStoreBase

_log = logging.getLogger(__name__)


class VecStore(
    _ConnectionMixin,
    _SchemaMixin,
    _MigrationsMixin,
    _QueriesMixin,
    _RepoStoreMixin,
    _FeedbackMixin,
    VecStoreBase,  # last: concrete mixins shadow the template methods
):
    """sqlite-vec store for memory metadata + embeddings.

    Lifecycle:

    1. `VecStore(db_path, dims=1024)` — opens connection, runs DDL,
       loads `vec0` extension. Idempotent (safe to instantiate
       multiple times against the same file).
    2. `store.upsert(record, embedding)` — write/replace one record's
       metadata + vector under one transaction.
    3. `store.search(embedding, limit=10)` — top-k by cosine.
    4. `store.delete(id_)` — remove from both tables.
    5. `store.close()` — release the connection. Optional; SQLite
       releases on process exit anyway.

    Thread safety: each `VecStore` keeps a single connection. SQLite
    serialises writes per connection. Concurrent readers should
    instantiate their own `VecStore` (cheap — ~1ms cold open).
    """

    def __init__(
        self,
        db_path: Path,
        dims: int = 1024,
        embedder_model: str = "",
        vec_quant: str = "int8",
    ) -> None:
        self.db_path = db_path
        self.dims = dims
        self.embedder_model = embedder_model
        # vec0 storage precision for the main `vec` table. "off" = float32,
        # "int8" = int8 (1 B/dim) via vec_quantize_int8(...,'unit'). Baked into
        # the vec0 column TYPE at DDL time, so it only takes effect on a rebuild
        # (existing on-disk precision is adopted — see schema._validate_vec_quant —
        # so this default only governs a brand-new `vec` table).
        self.vec_quant = vec_quant
        self._quant_int8 = vec_quant == "int8"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # One sqlite connection PER THREAD. A single shared connection is
        # unsafe under the FastMCP HTTP transport, which dispatches sync
        # tool calls on an anyio worker threadpool: two threads issuing
        # `BEGIN IMMEDIATE` (see `_tx`) on the same connection collide
        # ("cannot start a transaction within a transaction" / recursive
        # cursor use). Per-thread connections + WAL give real reader/writer
        # concurrency instead, with `busy_timeout` absorbing writer waits.
        # The CLI and recall daemon are single-threaded / lock-serialised,
        # so they simply reuse their one thread-local connection.
        self._local = threading.local()
        # Strong references are intentional: CPython does not guarantee the
        # order in which a terminating thread clears its ``threading.local``
        # values and finalizes sqlite connections.  Retain every holder so
        # ``close()`` is the single deterministic lifecycle boundary on all
        # supported platforms.
        self._conn_holders: set[_ConnectionHolder] = set()
        self._conn_holders_lock = threading.Lock()
        try:
            self._connect()
            self._init_schema()
        except Exception:
            self.close()
            raise
        # Tantivy FTS index — optional, lives next to the sqlite DB.
        self.tantivy_index_dir: Path = db_path.parent / "tantivy"
        self._tantivy_inst: TantivyFTSIndex | None = None
        self._tantivy_init_lock: threading.Lock = threading.Lock()
        self._tantivy_write_lock: threading.Lock = threading.Lock()
        self._tantivy_healthy: bool = True
        self._maybe_rebuild_tantivy()

    @property
    def connection(self) -> sqlite3.Connection:
        """Return the calling thread's connection for ad-hoc queries.

        The connection is thread-local (one per thread, lazily created).
        Use this instead of accessing ``_conn`` directly for read-only
        queries that VecStore doesn't expose via a public method.  Write
        operations must go through the public API (``upsert``, ``delete``,
        etc.) or ``_tx()`` for transactional safety.
        """
        return self._conn

    # -- tantivy wiring --------------------------------------------------------

    def _mark_tantivy_unhealthy(self) -> None:
        """Mark the tantivy index as stale so search falls back to FTS5."""
        self._tantivy_healthy = False

    def close(self) -> None:
        """Close sqlite connection and tantivy index."""
        tantivy = getattr(self, "_tantivy_inst", None)
        if tantivy is not None:
            with contextlib.suppress(BaseException):
                tantivy.close()
        super().close()

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        with contextlib.suppress(BaseException):
            self.close()

    def _get_tantivy(self) -> TantivyFTSIndex | None:
        """Return the live TantivyFTSIndex, or None to fall back to FTS5.

        Respects `MEMO_TANTIVY_ENABLED` (kill-switch: =0 forces FTS5-only, wins
        over everything) and `MEMO_FTS_BACKEND`: 'fts5' forces FTS5; 'tantivy'
        requires tantivy (raises if absent); 'auto' (default) uses tantivy when
        installed. Lazy-opens the index on first call; thread-safe.
        Returns None when the index is known-unhealthy from a prior write failure.
        """
        if not self._tantivy_healthy:
            return None
        from ..flags import flag_bool, flag_str

        if not flag_bool("MEMO_TANTIVY_ENABLED"):
            return None
        backend = flag_str("MEMO_FTS_BACKEND")
        if backend == "fts5":
            return None
        if not _tantivy_available():
            if backend == "tantivy":
                from ..errors import StorageError

                raise StorageError(
                    "MEMO_FTS_BACKEND=tantivy but the `tantivy` package is not installed. "
                    "Run: pip install tantivy"
                )
            return None
        with self._tantivy_init_lock:
            if self._tantivy_inst is None:
                try:
                    self._tantivy_inst = TantivyFTSIndex.open_or_create(self.tantivy_index_dir)
                except Exception as exc:
                    _log.warning("tantivy open failed, falling back to FTS5: %s", exc)
                    return None
        return self._tantivy_inst

    def _maybe_rebuild_tantivy(self) -> None:
        """Build the tantivy index from the FTS5 table on first startup."""
        from ..flags import flag_bool, flag_str

        if not flag_bool("MEMO_TANTIVY_ENABLED"):
            return
        if not _tantivy_available() or flag_str("MEMO_FTS_BACKEND") == "fts5":
            return
        if TantivyFTSIndex.exists(self.tantivy_index_dir):
            return
        try:
            self._rebuild_tantivy_from_sqlite()
        except Exception as exc:
            _log.warning("tantivy initial rebuild failed, FTS5 stays primary: %s", exc)

    def _rebuild_tantivy_from_sqlite(self) -> None:
        """Bulk-rebuild the tantivy index from the current FTS5 table."""
        rows = self._conn.execute("SELECT id, title, tags, body FROM fts")
        records: list[dict[str, str]] = [
            {"id": row["id"], "title": row["title"], "tags": row["tags"], "body": row["body"]}
            for row in rows
        ]
        # rebuild() calls delete_all_documents() first, so it MUST run exactly
        # once — flushing per-batch wipes every previously committed batch and
        # leaves only the last partial batch (<5000 docs) indexed.
        self._flush_tantivy_batch(records)

    def _flush_tantivy_batch(self, records: list[dict[str, str]]) -> None:
        t = self._get_tantivy()
        if t is None:
            return
        t.rebuild(records)


__all__ = ["VecStore"]
