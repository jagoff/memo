"""sqlite-vec-backed vector store for memory records.

Single file, single writer. No daemon. Metadata + vectors live in the
same DB file under separate tables — easier reset (`rm memvec.db`) and
simpler migration than splitting across qdrant + sqlite.

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

import logging
import threading
from pathlib import Path

from .connection import _ConnectionMixin
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

    def __init__(self, db_path: Path, dims: int = 1024) -> None:
        self.db_path = db_path
        self.dims = dims
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
        self._connect()
        self._init_schema()
        # Tantivy FTS index — optional, lives next to the sqlite DB.
        self.tantivy_index_dir: Path = db_path.parent / "tantivy"
        self._tantivy_inst: TantivyFTSIndex | None = None
        self._tantivy_init_lock: threading.Lock = threading.Lock()
        self._maybe_rebuild_tantivy()

    # -- tantivy wiring --------------------------------------------------------

    def _get_tantivy(self) -> TantivyFTSIndex | None:
        """Return the live TantivyFTSIndex, or None to fall back to FTS5.

        Respects `MEMO_FTS_BACKEND`: 'fts5' forces FTS5; 'tantivy' requires
        tantivy (raises if absent); 'auto' (default) uses tantivy when installed.
        Lazy-opens the index on first call; thread-safe.
        """
        from ..flags import flag_str

        backend = flag_str("MEMO_FTS_BACKEND")
        if backend == "fts5":
            return None
        if not _tantivy_available():
            if backend == "tantivy":
                raise RuntimeError(
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
        from ..flags import flag_str

        if not _tantivy_available() or flag_str("MEMO_FTS_BACKEND") == "fts5":
            return
        if TantivyFTSIndex.exists(self.tantivy_index_dir):
            return
        try:
            self._rebuild_tantivy_from_sqlite()
        except Exception as exc:
            _log.warning("tantivy initial rebuild failed, FTS5 stays primary: %s", exc)

    def _rebuild_tantivy_from_sqlite(self) -> None:
        """Bulk-rebuild the tantivy index from the current FTS5 table.

        Uses tantivy's delete_all_documents so a single writer commit produces
        a clean, consistent snapshot of the FTS5 ground truth without touching
        the index directory.
        """
        rows = self._conn.execute("SELECT id, title, tags, body FROM fts").fetchall()
        records = [
            {"id": r["id"], "title": r["title"], "tags": r["tags"], "body": r["body"]} for r in rows
        ]
        t = self._get_tantivy()
        if t is None:
            return
        t.rebuild(records)


__all__ = ["VecStore"]
