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

import threading
from pathlib import Path

from .connection import _ConnectionMixin
from .feedback_store import _FeedbackMixin
from .migrations import _MigrationsMixin
from .queries import _QueriesMixin
from .repo_store import _RepoStoreMixin
from .schema import _SchemaMixin
from .vec_base import VecStoreBase


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


__all__ = ["VecStore"]
