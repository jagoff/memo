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

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS meta (
    id          TEXT PRIMARY KEY,
    path        TEXT UNIQUE NOT NULL,
    title       TEXT NOT NULL,
    type        TEXT NOT NULL,
    tags        TEXT NOT NULL,
    created     TEXT NOT NULL,
    updated     TEXT NOT NULL,
    body_hash   TEXT NOT NULL,
    extra_json  TEXT
);

CREATE INDEX IF NOT EXISTS idx_meta_type    ON meta(type);
CREATE INDEX IF NOT EXISTS idx_meta_updated ON meta(updated);
"""


class VecStore:
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
        self._conn = sqlite3.connect(str(db_path), timeout=10.0, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._load_vec0()
        self._init_schema()

    # -- internals ---------------------------------------------------------

    def _load_vec0(self) -> None:
        import sqlite_vec  # type: ignore[import-not-found]

        # `enable_load_extension` must be called BEFORE `load_extension`.
        # Wrapped in try/except because some Python builds disable it
        # for security reasons — we surface a clear error in that case.
        try:
            self._conn.enable_load_extension(True)
        except sqlite3.NotSupportedError as exc:
            raise RuntimeError(
                "Python's sqlite3 was compiled without `enable_load_extension`. "
                "Reinstall Python via Homebrew (`brew install python@3.13`) which "
                "bundles a sqlite3 with extension support enabled."
            ) from exc
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.executescript(_SCHEMA_DDL)
            # `vec0` is a virtual table; we can't include it in the
            # static DDL string because the dimensionality is dynamic.
            self._conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS vec USING vec0("
                f"id TEXT PRIMARY KEY, embedding FLOAT[{self.dims}])"
            )

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

    # -- public CRUD --------------------------------------------------------

    def upsert(
        self,
        *,
        id_: str,
        path: str,
        title: str,
        type_: str,
        tags: list[str],
        created: str,
        updated: str,
        body_hash: str,
        embedding: list[float],
        extra: dict[str, Any] | None = None,
    ) -> None:
        if len(embedding) != self.dims:
            raise ValueError(
                f"Embedding dim mismatch: got {len(embedding)}, expected {self.dims}",
            )
        with self._tx() as cx:
            cx.execute(
                "INSERT INTO meta (id, path, title, type, tags, created, updated, body_hash, extra_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "path=excluded.path, title=excluded.title, type=excluded.type, "
                "tags=excluded.tags, updated=excluded.updated, body_hash=excluded.body_hash, "
                "extra_json=excluded.extra_json",
                (
                    id_, path, title, type_, json.dumps(tags), created, updated, body_hash,
                    json.dumps(extra) if extra is not None else None,
                ),
            )
            # `vec0` doesn't support `ON CONFLICT` syntax — we delete
            # then insert. Within the same transaction this is atomic.
            cx.execute("DELETE FROM vec WHERE id = ?", (id_,))
            cx.execute(
                "INSERT INTO vec (id, embedding) VALUES (?, ?)",
                (id_, json.dumps(embedding)),
            )

    def get(self, id_: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT id, path, title, type, tags, created, updated, body_hash, extra_json "
            "FROM meta WHERE id = ?",
            (id_,),
        ).fetchone()
        return _row_to_dict(row) if row else None

    def get_by_path(self, path: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT id, path, title, type, tags, created, updated, body_hash, extra_json "
            "FROM meta WHERE path = ?",
            (path,),
        ).fetchone()
        return _row_to_dict(row) if row else None

    def list_recent(self, limit: int = 20, type_: str | None = None) -> list[dict[str, Any]]:
        sql = (
            "SELECT id, path, title, type, tags, created, updated, body_hash, extra_json "
            "FROM meta "
        )
        params: tuple = ()
        if type_:
            sql += "WHERE type = ? "
            params = (type_,)
        sql += "ORDER BY updated DESC LIMIT ?"
        rows = self._conn.execute(sql, (*params, limit)).fetchall()
        return [_row_to_dict(r) for r in rows]

    def search(
        self, embedding: list[float], limit: int = 10,
        type_: str | None = None,
    ) -> list[dict[str, Any]]:
        """Top-k by cosine. Returns metadata dicts with a `score` field
        added (1 - distance, so higher = more similar)."""
        if len(embedding) != self.dims:
            raise ValueError(
                f"Query embedding dim mismatch: got {len(embedding)}, expected {self.dims}",
            )
        # Pull a wider candidate set when filtering by type so the
        # final top-k still returns `limit` results after the join
        # filters out off-type rows. 5x is generous for sane type
        # distributions; degenerates gracefully when type is uncommon.
        candidate_k = limit * 5 if type_ else limit
        sql = (
            "SELECT vec.id AS id, vec.distance AS distance, "
            "       meta.path, meta.title, meta.type, meta.tags, "
            "       meta.created, meta.updated, meta.body_hash, meta.extra_json "
            "FROM vec "
            "JOIN meta ON meta.id = vec.id "
            "WHERE embedding MATCH ? AND k = ? "
        )
        params: list[Any] = [json.dumps(embedding), candidate_k]
        if type_:
            sql += "AND meta.type = ? "
            params.append(type_)
        sql += "ORDER BY distance ASC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            d = _row_to_dict(r)
            d["score"] = 1.0 - float(r["distance"])
            out.append(d)
        return out

    def delete(self, id_: str) -> bool:
        with self._tx() as cx:
            cur = cx.execute("DELETE FROM meta WHERE id = ?", (id_,))
            existed = cur.rowcount > 0
            cx.execute("DELETE FROM vec WHERE id = ?", (id_,))
        return existed

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM meta").fetchone()[0]

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = {k: row[k] for k in row.keys() if k != "distance"}
    if "tags" in d and isinstance(d["tags"], str):
        try:
            d["tags"] = json.loads(d["tags"])
        except Exception:
            d["tags"] = []
    if "extra_json" in d and d["extra_json"]:
        try:
            d["extra"] = json.loads(d["extra_json"])
        except Exception:
            d["extra"] = {}
        d.pop("extra_json", None)
    elif "extra_json" in d:
        d.pop("extra_json", None)
        d["extra"] = {}
    return d


__all__ = ["VecStore"]
