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
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

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
            # `distance_metric=cosine` makes `vec.distance` a true cosine
            # distance (1 - dot, range [0, 2]). Without it, vec0 defaults
            # to L2 distance — the ranking is monotonic for unit vectors
            # but the absolute values are wrong: an L2 distance of 0.80
            # corresponds to a cosine of 0.68, so `score = 1 - distance`
            # ends up reporting 0.20 instead of 0.68. Verified empirically
            # 2026-05-07.
            self._conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS vec USING vec0("
                f"id TEXT PRIMARY KEY, embedding FLOAT[{self.dims}] distance_metric=cosine)"
            )
            # FTS5 over title + tags + body for the BM25 side of hybrid
            # search. `unindexed=id` keeps the row id queryable but not
            # tokenised. `tokenize='unicode61 remove_diacritics 2'`
            # handles Spanish accents (so a search for "decision" matches
            # "decisión") and lowercases. Body is stored externally — we
            # write it on upsert via the `Memory` layer (the store sees
            # the body string via `body_text` arg in `upsert_text`).
            self._conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5("
                "id UNINDEXED, title, tags, body, "
                "tokenize='unicode61 remove_diacritics 2'"
                ")"
            )

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
        body_text: str = "",
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
                    # `default=str` coerces date/datetime/Path objects that
                    # frontmatter parsers leave as native Python types — common
                    # in mem-vault-style files where YAML auto-coerced dates.
                    # Without this, json.dumps raises TypeError on the first
                    # such object inside `extra`.
                    json.dumps(extra, default=str) if extra is not None else None,
                ),
            )
            # `vec0` doesn't support `ON CONFLICT` syntax — we delete
            # then insert. Within the same transaction this is atomic.
            cx.execute("DELETE FROM vec WHERE id = ?", (id_,))
            cx.execute(
                "INSERT INTO vec (id, embedding) VALUES (?, ?)",
                (id_, json.dumps(embedding)),
            )
            # Same dance for the FTS index — no upsert support.
            cx.execute("DELETE FROM fts WHERE id = ?", (id_,))
            cx.execute(
                "INSERT INTO fts (id, title, tags, body) VALUES (?, ?, ?, ?)",
                (id_, title, " ".join(tags), body_text),
            )

    def upsert_text_only(
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
        extra: dict[str, Any] | None = None,
        body_text: str = "",
    ) -> None:
        """Write metadata + FTS row without a vector embedding.

        This keeps CRUD and BM25 search usable on fresh installs or while
        models are downloading. A later `memo reindex` fills the missing vector.
        """
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
                    json.dumps(extra, default=str) if extra is not None else None,
                ),
            )
            cx.execute("DELETE FROM fts WHERE id = ?", (id_,))
            cx.execute(
                "INSERT INTO fts (id, title, tags, body) VALUES (?, ?, ?, ?)",
                (id_, title, " ".join(tags), body_text),
            )

    def has_vector(self, id_: str) -> bool:
        row = self._conn.execute("SELECT 1 FROM vec WHERE id = ? LIMIT 1", (id_,)).fetchone()
        return row is not None

    def update_meta(
        self,
        *,
        id_: str,
        title: str,
        type_: str,
        tags: list[str],
        updated: str,
        extra: dict[str, Any] | None = None,
    ) -> bool:
        """Patch metadata fields without touching the embedding. Used by
        `Memory.update()` when only title/type/tags/extra changed and
        `body_hash` is unchanged — saves an embedder forward pass.
        Returns True if a row was updated."""
        with self._tx() as cx:
            cur = cx.execute(
                "UPDATE meta SET title = ?, type = ?, tags = ?, updated = ?, extra_json = ? "
                "WHERE id = ?",
                (
                    title, type_, json.dumps(tags), updated,
                    json.dumps(extra, default=str) if extra is not None else None, id_,
                ),
            )
            # Sync FTS title + tags (body unchanged on metadata-only updates).
            # FTS5 doesn't support partial UPDATE cleanly; we delete + reinsert
            # the row preserving the existing body text.
            existing = cx.execute(
                "SELECT body FROM fts WHERE id = ?", (id_,),
            ).fetchone()
            body_text = existing["body"] if existing else ""
            cx.execute("DELETE FROM fts WHERE id = ?", (id_,))
            cx.execute(
                "INSERT INTO fts (id, title, tags, body) VALUES (?, ?, ?, ?)",
                (id_, title, " ".join(tags), body_text),
            )
            return cur.rowcount > 0

    def get(self, id_: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT id, path, title, type, tags, created, updated, body_hash, extra_json "
            "FROM meta WHERE id = ?",
            (id_,),
        ).fetchone()
        return _row_to_dict(row) if row else None

    def find_by_prefix(self, prefix: str, limit: int = 10) -> list[str]:
        """Return ids whose hex starts with `prefix`. Used by the CLI/MCP
        to let callers reference memories by the first ~7 chars (git-style)
        instead of pasting a 32-char UUID4."""
        if not prefix:
            return []
        rows = self._conn.execute(
            "SELECT id FROM meta WHERE id LIKE ? || '%' ORDER BY id LIMIT ?",
            (prefix, limit),
        ).fetchall()
        return [r["id"] for r in rows]

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
            cx.execute("DELETE FROM fts WHERE id = ?", (id_,))
        return existed

    def search_bm25(
        self, query: str, limit: int = 10, type_: str | None = None,
    ) -> list[dict[str, Any]]:
        """BM25 keyword search over title + tags + body via FTS5.

        Returns rows shaped like `search()` (vec) but with `score` set
        to `bm25_inverted` (a normalised `1 / (1 + bm25)` so higher =
        better, mirroring the cosine score convention).
        """
        if not query or not query.strip():
            return []
        # FTS5 needs an explicit MATCH expression. Pre-2026-05-07 we wrapped
        # the whole query in `"..."` (phrase match) to dodge FTS5 syntax
        # collisions on hyphens/colons. Side-effect: multi-word queries
        # required the EXACT consecutive sequence, killing recall on
        # natural Spanish queries — "Astor terapia ocupacional" would NOT
        # match a doc titled "Informe Terapia Ocupacional — Astor Ferrari"
        # because the words don't appear consecutively in that order.
        #
        # Fix: tokenize via \w+ regex (drops punctuation, keeps Unicode
        # letters via Python's \w), wrap each token in its own phrase
        # quotes, join with whitespace (FTS5's implicit AND). Result:
        # `"Astor" "terapia" "ocupacional"` — matches any doc containing
        # all 3 words anywhere, in any order.
        import re as _re
        _tokens = [t for t in _re.findall(r"\w+", query, flags=_re.UNICODE) if t]
        if not _tokens:
            return []
        match_expr = " ".join(f'"{t}"' for t in _tokens)
        candidate_k = limit * 5 if type_ else limit
        sql = (
            "SELECT fts.id AS id, bm25(fts) AS bm25_score, "
            "       meta.path, meta.title, meta.type, meta.tags, "
            "       meta.created, meta.updated, meta.body_hash, meta.extra_json "
            "FROM fts JOIN meta ON meta.id = fts.id "
            "WHERE fts MATCH ? "
        )
        params: list[Any] = [match_expr]
        if type_:
            sql += "AND meta.type = ? "
            params.append(type_)
        sql += "ORDER BY bm25_score ASC LIMIT ?"
        params.append(candidate_k)
        try:
            rows = self._conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            # Malformed FTS expression (e.g. unbalanced quotes after
            # escape). Fall back to no results — Memory.search_hybrid
            # treats this as "no BM25 signal" and uses pure vec.
            return []
        out: list[dict[str, Any]] = []
        for r in rows:
            d = _row_to_dict(r)
            # bm25() returns a NEGATIVE score for sqlite-fts5 (lower =
            # better). Invert into [0, 1] roughly: 1/(1 + abs(score)).
            bm = float(r["bm25_score"])
            d["score"] = 1.0 / (1.0 + abs(bm)) if bm < 0 else 0.0
            out.append(d)
        return out[:limit]

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM meta").fetchone()[0]

    def close(self) -> None:
        with suppress(Exception):
            self._conn.close()


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = {k: row[k] for k in row.keys() if k != "distance"}  # noqa: SIM118
    if "tags" in d and isinstance(d["tags"], str):
        try:
            d["tags"] = json.loads(d["tags"])
        except Exception:
            d["tags"] = []
    if d.get("extra_json"):
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
