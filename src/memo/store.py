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
import logging
import re
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlite_vec import serialize_float32

_log = logging.getLogger(__name__)

# vec0 accepts either a JSON array (text, must be parsed) or a packed
# float32 blob. Blobs skip JSON encode on write and JSON parse on every
# search MATCH — the hot path. vec0 stores float32 internally regardless,
# so existing JSON-written rows stay readable; no migration needed.

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

CREATE TABLE IF NOT EXISTS repo_sources (
    id          TEXT PRIMARY KEY,
    name        TEXT UNIQUE NOT NULL,
    url         TEXT NOT NULL,
    ref         TEXT NOT NULL,
    commit_sha  TEXT NOT NULL,
    clone_path  TEXT NOT NULL,
    indexed_at  TEXT NOT NULL,
    status      TEXT NOT NULL,
    extra_json  TEXT
);

CREATE INDEX IF NOT EXISTS idx_repo_sources_url_ref ON repo_sources(url, ref);

CREATE TABLE IF NOT EXISTS repo_files (
    id          TEXT PRIMARY KEY,
    repo_id     TEXT NOT NULL,
    path        TEXT NOT NULL,
    language    TEXT NOT NULL,
    size_bytes  INTEGER NOT NULL,
    sha256      TEXT NOT NULL,
    line_count  INTEGER NOT NULL,
    indexed_at  TEXT NOT NULL,
    UNIQUE(repo_id, path)
);

CREATE INDEX IF NOT EXISTS idx_repo_files_repo_path ON repo_files(repo_id, path);
CREATE INDEX IF NOT EXISTS idx_repo_files_sha       ON repo_files(sha256);

CREATE TABLE IF NOT EXISTS repo_chunks (
    id          TEXT PRIMARY KEY,
    repo_id     TEXT NOT NULL,
    file_id     TEXT NOT NULL,
    path        TEXT NOT NULL,
    chunk_seq   INTEGER NOT NULL,
    line_start  INTEGER NOT NULL,
    line_end    INTEGER NOT NULL,
    text_hash   TEXT NOT NULL,
    body_text   TEXT NOT NULL,
    indexed_at  TEXT NOT NULL,
    UNIQUE(file_id, chunk_seq)
);

CREATE INDEX IF NOT EXISTS idx_repo_chunks_repo_path ON repo_chunks(repo_id, path);
CREATE INDEX IF NOT EXISTS idx_repo_chunks_file      ON repo_chunks(file_id);

CREATE TABLE IF NOT EXISTS repo_lines (
    id          TEXT PRIMARY KEY,
    repo_id     TEXT NOT NULL,
    file_id     TEXT NOT NULL,
    path        TEXT NOT NULL,
    line_no     INTEGER NOT NULL,
    text        TEXT NOT NULL,
    text_hash   TEXT NOT NULL,
    UNIQUE(file_id, line_no)
);

CREATE INDEX IF NOT EXISTS idx_repo_lines_repo_path ON repo_lines(repo_id, path, line_no);

CREATE TABLE IF NOT EXISTS repo_embedding_cache (
    model       TEXT NOT NULL,
    dims        INTEGER NOT NULL,
    input_hash  TEXT NOT NULL,
    embedding   TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    PRIMARY KEY(model, dims, input_hash)
);

CREATE TABLE IF NOT EXISTS source_feedback (
    id          TEXT PRIMARY KEY,
    source_id   TEXT NOT NULL,
    query_text  TEXT NOT NULL,
    rating      INTEGER NOT NULL,
    created_at  TEXT NOT NULL,
    extra_json  TEXT,
    UNIQUE(source_id, query_text, rating)
);

CREATE INDEX IF NOT EXISTS idx_source_feedback_source ON source_feedback(source_id);
CREATE INDEX IF NOT EXISTS idx_source_feedback_rating ON source_feedback(rating);

CREATE TABLE IF NOT EXISTS access (
    id            TEXT PRIMARY KEY,        -- references meta.id
    access_count  INTEGER NOT NULL DEFAULT 0,
    last_accessed TEXT                     -- ISO8601 of last read/hit; NULL until first touch
);

CREATE INDEX IF NOT EXISTS idx_access_last  ON access(last_accessed);
CREATE INDEX IF NOT EXISTS idx_access_count ON access(access_count);
"""


_REQUIRED_SCHEMA_OBJECTS = frozenset(
    {
        "meta",
        "vec",
        "fts",
        "repo_sources",
        "repo_files",
        "repo_chunks",
        "repo_lines",
        "repo_vec",
        "repo_chunk_fts",
        "repo_line_fts",
        "repo_embedding_cache",
        "source_feedback",
        "source_feedback_vec",
        "access",
    }
)

# FTS5 column weights for repo retrieval BM25.
# `bm25(table, w0, w1, w2, ...)` takes one weight per column (UNINDEXED
# columns receive a weight but ignore it). Path is boosted because a
# query term matching the filename or directory is a strong canonical
# signal (e.g. `Contacts/Grecia.md` for "Grecia") that pure body
# term-density would otherwise drown in dumps with many repeats of the
# same keyword.
_BM25_REPO_NAME_WEIGHT = 0.5
_BM25_PATH_WEIGHT = 5.0
_BM25_BODY_WEIGHT = 1.0
_BM25_UNINDEXED_WEIGHT = 0.0  # harmless filler for UNINDEXED columns

# Main `fts` column weights (id UNINDEXED, title, tags, body). Title is the
# canonical-name signal (e.g. "Grecia.md" for "Grecia"); tags carry curated
# topic labels; body is full text. Without these, bm25() collapses to a
# uniform weight and short, on-topic titles get drowned by long bodies that
# happen to mention the term in passing.
_BM25_FTS_TITLE_WEIGHT = 5.0
_BM25_FTS_TAGS_WEIGHT = 3.0
_BM25_FTS_BODY_WEIGHT = 1.0

# Spanish stopwords stripped before FTS5 tokenisation. AND-of-tokens
# semantics means every kept token must hit; "cuando es el cumple de
# Grecia?" with stopwords kept becomes a 5-token AND that almost never
# matches. Drop only high-frequency function words; keep nouns/verbs.
# Conservative list — proper names, abbreviations, and content verbs stay.
_BM25_ES_STOPWORDS: frozenset[str] = frozenset({
    "a", "al", "ante", "bajo", "con", "contra", "de", "del", "desde",
    "donde", "dónde", "durante", "e", "el", "ella", "ellas", "ellos",
    "en", "entre", "era", "eres", "es", "esa", "esas", "ese", "eso",
    "esos", "esta", "estas", "este", "esto", "estos", "fue", "fueron",
    "hacia", "hasta", "hay", "la", "las", "le", "les", "lo", "los",
    "mas", "más", "me", "mi", "mis", "mí", "ni", "no", "nos", "o", "para",
    "pero", "por", "porque", "pues", "que", "qué", "quien", "quién",
    "quienes", "quiénes", "sea", "ser", "si", "sí", "sin", "sobre",
    "son", "su", "sus", "te", "ti", "tu", "tus", "tú", "un", "una",
    "unas", "uno", "unos", "u", "y", "ya", "yo", "como", "cómo",
    "cual", "cuál", "cuales", "cuáles", "cuando", "cuándo",
})


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

    # -- internals ---------------------------------------------------------

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
            raise RuntimeError(
                "Python's sqlite3 was compiled without `enable_load_extension`. "
                "Reinstall Python via Homebrew (`brew install python@3.13`) which "
                "bundles a sqlite3 with extension support enabled."
            ) from exc
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

    def _init_schema(self) -> None:
        # Most CLI commands are reads. If the schema already exists, avoid
        # no-op DDL because it still needs a write/schema lock and can fail
        # while long repo indexing is writing batches.
        if self._schema_ready():
            self._validate_vec_dims()
            return

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
            self._conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS repo_vec USING vec0("
                f"id TEXT PRIMARY KEY, embedding FLOAT[{self.dims}] distance_metric=cosine)"
            )
            # Per-source feedback uses query embeddings to detect "similar"
            # future queries. `feedback_id` mirrors `source_feedback.id` so
            # joins are cheap; `distance_metric=cosine` lets the rank hook
            # threshold on absolute cosine similarity (1 - distance).
            self._conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS source_feedback_vec USING vec0("
                f"feedback_id TEXT PRIMARY KEY, query_emb FLOAT[{self.dims}] distance_metric=cosine)"
            )
            self._validate_vec_dims()
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
            self._conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS repo_chunk_fts USING fts5("
                "id UNINDEXED, repo_name, path, body, "
                "tokenize='unicode61 remove_diacritics 2'"
                ")"
            )
            self._conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS repo_line_fts USING fts5("
                "id UNINDEXED, repo_name, path, line_no UNINDEXED, body, "
                "tokenize='unicode61 remove_diacritics 2'"
                ")"
            )

    def _schema_ready(self) -> bool:
        rows = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        ).fetchall()
        present = {str(row["name"]) for row in rows}
        return _REQUIRED_SCHEMA_OBJECTS.issubset(present)

    def _vec_table_dims(self, table: str) -> int | None:
        if table not in {"vec", "repo_vec"}:
            raise ValueError(f"unknown vector table: {table!r}")
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        if not row or not row["sql"]:
            return None
        match = re.search(r"embedding\s+FLOAT\[(\d+)\]", str(row["sql"]))
        return int(match.group(1)) if match else None

    def _validate_vec_dims(self) -> None:
        actual_dims = self._vec_table_dims("vec")
        if actual_dims is not None and actual_dims != self.dims:
            raise RuntimeError(
                f"Embedding dimension mismatch: store has {actual_dims}D vectors "
                f"but config expects {self.dims}D.\n"
                f"Fix: rm {self.db_path} && memo reindex\n"
                f"Or check: MEMO_MODEL_PROFILE={self.dims}D or MEMO_EMBEDDER_DIMS={self.dims}"
            )
        repo_actual_dims = self._vec_table_dims("repo_vec")
        if repo_actual_dims is not None and repo_actual_dims != self.dims:
            raise RuntimeError(
                f"Repo embedding dimension mismatch: store has {repo_actual_dims}D vectors "
                f"but config expects {self.dims}D.\n"
                f"Fix: rm {self.db_path} && memo reindex\n"
                f"Or check: MEMO_MODEL_PROFILE={self.dims}D or MEMO_EMBEDDER_DIMS={self.dims}"
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

    def _checkpoint(self) -> None:
        """Truncate the WAL back into the main DB. Call after large batch
        writes (repo indexing) so the -wal file doesn't grow unbounded
        before the default autocheckpoint (1000 pages) fires, which keeps
        crash-recovery fast. Best-effort: a checkpoint can be blocked by a
        concurrent reader, in which case autocheckpoint catches up later."""
        with suppress(sqlite3.OperationalError):
            self._conn.execute("PRAGMA wal_checkpoint(RESTART)")

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
        norm = sum(x * x for x in embedding) ** 0.5
        if not (0.5 < norm < 1.5):
            raise ValueError(
                f"Embedding norm {norm:.4f} out of L2-normalised range (expected ≈ 1.0) "
                f"[id={id_[:8]}] — re-embed or check MLX model output."
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
                (id_, serialize_float32(embedding)),
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

    def get_by_path_ci(self, path: str) -> dict[str, Any] | None:
        """Like `get_by_path`, but case-insensitive on the path.

        The vault sits on a case-insensitive filesystem, so `notes/Foo.md`
        and `Notes/Foo.md` are the same file. Ingest uses this to find an
        existing row regardless of the casing a re-walk produced, so it
        reuses that row's id instead of minting a duplicate. Returns the
        oldest match if several casings somehow coexist.
        """
        row = self._conn.execute(
            "SELECT id, path, title, type, tags, created, updated, body_hash, extra_json "
            "FROM meta WHERE path = ? COLLATE NOCASE ORDER BY created LIMIT 1",
            (path,),
        ).fetchone()
        return _row_to_dict(row) if row else None

    def vault_ingest_rows(self, label: str) -> list[dict[str, Any]]:
        """Rows produced by `memo ingest` under a given vault `label`.

        Used by `ingest --prune` to find stale chunks (abs_path gone). Filters
        on `source LIKE 'vault-ingest%'` so curated memorias (source NULL) are
        never returned — they're managed by `memo reindex` / `doctor --gc`.
        Returns the minimal fields the prune needs: id, path, abs_path,
        parent_path, chunk_seq.
        """
        rows = self._conn.execute(
            "SELECT id, path, "
            "json_extract(extra_json, '$.abs_path') AS abs_path, "
            "json_extract(extra_json, '$.parent_path') AS parent_path, "
            "json_extract(extra_json, '$.chunk_seq') AS chunk_seq "
            "FROM meta "
            "WHERE json_extract(extra_json, '$.vault') = ? "
            "AND json_extract(extra_json, '$.source') LIKE 'vault-ingest%'",
            (label,),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "path": r["path"],
                "abs_path": r["abs_path"],
                "parent_path": r["parent_path"],
                "chunk_seq": r["chunk_seq"],
            }
            for r in rows
        ]

    def file_rows(self, store_path: str) -> list[dict[str, Any]]:
        """All rows belonging to one ingested file: the single-chunk row
        (path == store_path) plus any multi-chunk rows (parent_path ==
        store_path). Used by per-file reconciliation to drop stale chunks
        (e.g. a multi-chunk note edited down to fewer chunks)."""
        rows = self._conn.execute(
            "SELECT id, path FROM meta "
            "WHERE path = ? "
            "OR json_extract(extra_json, '$.parent_path') = ?",
            (store_path, store_path),
        ).fetchall()
        return [{"id": r["id"], "path": r["path"]} for r in rows]

    def list_recent(
        self, limit: int = 20, type_: str | None = None,
        exclude_types: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        sql = (
            "SELECT id, path, title, type, tags, created, updated, body_hash, extra_json "
            "FROM meta "
        )
        clauses: list[str] = []
        params: list[Any] = []
        if type_:
            clauses.append("type = ?")
            params.append(type_)
        if exclude_types:
            placeholders = ",".join("?" for _ in exclude_types)
            clauses.append(f"type NOT IN ({placeholders})")
            params.extend(sorted(exclude_types))
        if clauses:
            sql += "WHERE " + " AND ".join(clauses) + " "
        sql += "ORDER BY updated DESC LIMIT ?"
        rows = self._conn.execute(sql, (*params, limit)).fetchall()
        return [_row_to_dict(r) for r in rows]

    def search(
        self, embedding: list[float], limit: int = 10,
        type_: str | None = None, exclude_types: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Top-k by cosine. Returns metadata dicts with a `score` field
        added (1 - distance, so higher = more similar).

        `exclude_types` drops rows whose `type` is in the set (pushed into
        SQL, not post-filtered, so the candidate pool isn't wasted on rows
        the caller will throw away — e.g. the recall hook excluding the
        bulk `reference` tier)."""
        if len(embedding) != self.dims:
            raise ValueError(
                f"Query embedding dim mismatch: got {len(embedding)}, expected {self.dims}",
            )
        # Pull a wider candidate set when filtering by type (in or out) so
        # the final top-k still returns `limit` results after the join
        # filters out off-type rows. 5x is generous for sane type
        # distributions; degenerates gracefully when type is uncommon.
        candidate_k = limit * 5 if (type_ or exclude_types) else limit
        sql = (
            "SELECT vec.id AS id, vec.distance AS distance, "
            "       meta.path, meta.title, meta.type, meta.tags, "
            "       meta.created, meta.updated, meta.body_hash, meta.extra_json "
            "FROM vec "
            "JOIN meta ON meta.id = vec.id "
            "WHERE embedding MATCH ? AND k = ? "
        )
        params: list[Any] = [serialize_float32(embedding), candidate_k]
        if type_:
            sql += "AND meta.type = ? "
            params.append(type_)
        if exclude_types:
            sql += f"AND meta.type NOT IN ({','.join('?' for _ in exclude_types)}) "
            params.extend(sorted(exclude_types))
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
            cx.execute("DELETE FROM access WHERE id = ?", (id_,))
        return existed

    # -- access tracking (cache tier hit counting) -------------------------
    #
    # The history log only records save/update/delete, never reads — so it
    # can't drive LRU/LFU. The `access` table fills that gap: `touch()` bumps
    # a per-memoria hit count + last-access timestamp on every search/ask
    # hit. Cheap, write-light, and decoupled from the hot `meta`/`vec` path.

    def touch(self, ids: list[str], *, ts: str | None = None) -> None:
        """Record a read/hit for each id: ++access_count, set last_accessed.

        Batch upsert in one tx. No-op on empty input. Safe to call for ids
        with no `meta` row (the access row is harmless and cleaned on delete).
        Callers should invoke this fire-and-forget off the hot path — the
        recall hook's 5s budget must not wait on it.
        """
        if not ids:
            return
        now = ts or datetime.now(UTC).isoformat()
        with self._tx() as cx:
            cx.executemany(
                "INSERT INTO access (id, access_count, last_accessed) VALUES (?, 1, ?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "access_count = access_count + 1, last_accessed = excluded.last_accessed",
                [(i, now) for i in ids],
            )

    def get_access(self, id_: str) -> dict[str, Any]:
        """Return {access_count, last_accessed} for a memoria.

        Defaults to count 0 / last_accessed None when never touched.
        """
        row = self._conn.execute(
            "SELECT access_count, last_accessed FROM access WHERE id = ?", (id_,),
        ).fetchone()
        if not row:
            return {"access_count": 0, "last_accessed": None}
        return {"access_count": int(row["access_count"]), "last_accessed": row["last_accessed"]}

    def eviction_candidates(
        self, policy: str, limit: int, *, exclude_types: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return up to `limit` memorias coldest-first under the given policy.

        Joins `meta` LEFT against `access` so never-accessed rows participate,
        falling back to `meta.updated` as their effective last-access time
        (a row written long ago and never read is genuinely cold).

          - lru: order by effective last-access ASC (coldest first).
          - lfu: order by access_count ASC, then effective last-access ASC.
          - ttl: same ordering as lru; the age cutoff is applied by the caller.

        Returns dicts: {id, type, access_count, last_accessed, updated}.
        """
        eff = "COALESCE(a.last_accessed, m.updated)"
        # lru and ttl share coldest-first-by-recency ordering
        order = f"COALESCE(a.access_count, 0) ASC, {eff} ASC" if policy == "lfu" else f"{eff} ASC"
        sql = (
            "SELECT m.id AS id, m.type AS type, "
            "COALESCE(a.access_count, 0) AS access_count, "
            "a.last_accessed AS last_accessed, m.updated AS updated "
            "FROM meta m LEFT JOIN access a ON a.id = m.id "
        )
        params: list[Any] = []
        if exclude_types:
            placeholders = ",".join("?" for _ in exclude_types)
            sql += f"WHERE m.type NOT IN ({placeholders}) "
            params.extend(sorted(exclude_types))
        sql += f"ORDER BY {order} LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def bulk_update_type(self, ids: list[str], new_type: str) -> int:
        """Reclassify the `type` of many memorias in one transaction.

        Only the `meta.type` column changes — the vec embedding and fts
        index are untouched (bodies are unchanged), so this is cheap and
        needs no re-embed. Used by `memo retier`.
        """
        if not ids:
            return 0
        with self._tx() as cx:
            cx.executemany(
                "UPDATE meta SET type = ? WHERE id = ?",
                [(new_type, i) for i in ids],
            )
        return len(ids)

    def search_bm25(
        self, query: str, limit: int = 10, type_: str | None = None,
        exclude_types: set[str] | None = None,
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
        _raw_tokens = [t for t in _re.findall(r"\w+", query, flags=_re.UNICODE) if t]
        if not _raw_tokens:
            return []
        # Strip Spanish stopwords, but only if doing so keeps ≥ 2 tokens.
        # Single-token queries like "Grecia" or stopword-heavy short
        # questions ("que es eso?") must still produce a match expression.
        _filtered = [t for t in _raw_tokens if t.lower() not in _BM25_ES_STOPWORDS]
        _tokens = _filtered if len(_filtered) >= 2 else _raw_tokens

        def _run(tokens: list[str], joiner: str) -> list[Any]:
            expr = joiner.join(f'"{t}"' for t in tokens)
            candidate_k = limit * 5 if (type_ or exclude_types) else limit
            sql = (
                "SELECT fts.id AS id, "
                "       bm25(fts, ?, ?, ?, ?) AS bm25_score, "
                "       meta.path, meta.title, meta.type, meta.tags, "
                "       meta.created, meta.updated, meta.body_hash, meta.extra_json "
                "FROM fts JOIN meta ON meta.id = fts.id "
                "WHERE fts MATCH ? "
            )
            params: list[Any] = [
                _BM25_UNINDEXED_WEIGHT,
                _BM25_FTS_TITLE_WEIGHT,
                _BM25_FTS_TAGS_WEIGHT,
                _BM25_FTS_BODY_WEIGHT,
                expr,
            ]
            if type_:
                sql += "AND meta.type = ? "
                params.append(type_)
            if exclude_types:
                sql += f"AND meta.type NOT IN ({','.join('?' for _ in exclude_types)}) "
                params.extend(sorted(exclude_types))
            sql += "ORDER BY bm25_score ASC LIMIT ?"
            params.append(candidate_k)
            try:
                return list(self._conn.execute(sql, params).fetchall())
            except sqlite3.OperationalError:
                # Malformed FTS expression (e.g. unbalanced quotes after
                # escape). Fall back to no results — Memory.search_hybrid
                # treats this as "no BM25 signal" and uses pure vec.
                return []

        rows = _run(_tokens, " ")
        # AND-of-tokens zero-recall fallback: only when the strict AND
        # match returns nothing on a multi-token query, retry with OR.
        # Triggering on `<limit` (partial recall) caused RRF rank
        # washing — OR brings in popular single-token matches that
        # demote the AND-matched correct doc once fused with the vec
        # leg. Triggering only on zero is the safe floor: it cannot
        # make a successful AND query worse.
        if not rows and len(_tokens) >= 2:
            rows = _run(_tokens, " OR ")
        out: list[dict[str, Any]] = []
        for r in rows:
            d = _row_to_dict(r)
            # bm25() returns a NEGATIVE score for sqlite-fts5 (lower =
            # better). Invert into [0, 1] roughly: 1/(1 + abs(score)).
            bm = float(r["bm25_score"])
            d["score"] = 1.0 / (1.0 + abs(bm)) if bm < 0 else 0.0
            out.append(d)
        return out[:limit]

    # -- repo corpus -------------------------------------------------------

    def get_repo_source(self, key: str) -> dict[str, Any] | None:
        """Return a repo source by id, name, or URL."""
        row = self._conn.execute(
            "SELECT id, name, url, ref, commit_sha, clone_path, indexed_at, status, extra_json "
            "FROM repo_sources WHERE id = ? OR name = ? OR url = ? "
            "ORDER BY indexed_at DESC LIMIT 1",
            (key, key, key),
        ).fetchone()
        return _row_to_dict(row) if row else None

    def get_repo_source_by_url_ref(self, url: str, ref: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT id, name, url, ref, commit_sha, clone_path, indexed_at, status, extra_json "
            "FROM repo_sources WHERE url = ? AND ref = ? LIMIT 1",
            (url, ref),
        ).fetchone()
        return _row_to_dict(row) if row else None

    def list_repo_sources(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, name, url, ref, commit_sha, clone_path, indexed_at, status, extra_json "
            "FROM repo_sources ORDER BY indexed_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def repo_file_hashes(self, repo_id: str) -> dict[str, dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, path, sha256, line_count FROM repo_files WHERE repo_id = ?",
            (repo_id,),
        ).fetchall()
        return {r["path"]: dict(r) for r in rows}

    def repo_counts(self, repo_id: str) -> dict[str, int]:
        """Return exact + semantic index counts for one repo."""
        row = self._conn.execute(
            "SELECT "
            "(SELECT COUNT(*) FROM repo_files WHERE repo_id = ?) AS files, "
            "(SELECT COUNT(*) FROM repo_lines WHERE repo_id = ?) AS lines, "
            "(SELECT COUNT(*) FROM repo_chunks WHERE repo_id = ?) AS chunks, "
            "(SELECT COUNT(*) FROM repo_vec "
            " JOIN repo_chunks ON repo_chunks.id = repo_vec.id "
            " WHERE repo_chunks.repo_id = ?) AS embedded_chunks",
            (repo_id, repo_id, repo_id, repo_id),
        ).fetchone()
        return {
            "files": int(row["files"] or 0),
            "lines": int(row["lines"] or 0),
            "chunks": int(row["chunks"] or 0),
            "embedded_chunks": int(row["embedded_chunks"] or 0),
        }

    def update_repo_status(self, repo_id: str, status: str, *, indexed_at: str | None = None) -> None:
        with self._tx() as cx:
            if indexed_at is None:
                cx.execute("UPDATE repo_sources SET status = ? WHERE id = ?", (status, repo_id))
            else:
                cx.execute(
                    "UPDATE repo_sources SET status = ?, indexed_at = ? WHERE id = ?",
                    (status, indexed_at, repo_id),
                )

    def repo_pending_chunks(self, repo_id: str, *, force: bool = False) -> list[dict[str, Any]]:
        join = "" if force else "LEFT JOIN repo_vec ON repo_vec.id = repo_chunks.id "
        where = "repo_chunks.repo_id = ?"
        if not force:
            where += " AND repo_vec.id IS NULL"
        rows = self._conn.execute(
            "SELECT repo_chunks.id, repo_chunks.repo_id, repo_sources.name AS repo_name, "
            "       repo_chunks.file_id, repo_chunks.path, repo_chunks.chunk_seq, "
            "       repo_chunks.line_start, repo_chunks.line_end, repo_chunks.body_text "
            "FROM repo_chunks "
            "JOIN repo_sources ON repo_sources.id = repo_chunks.repo_id "
            f"{join}"
            f"WHERE {where} "
            "ORDER BY repo_chunks.path, repo_chunks.chunk_seq",
            (repo_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_repo_embedding_cache(
        self,
        *,
        model: str,
        dims: int,
        input_hashes: list[str],
    ) -> dict[str, list[float]]:
        if not input_hashes:
            return {}
        out: dict[str, list[float]] = {}
        for batch in _batches(list(dict.fromkeys(input_hashes))):
            placeholders = ",".join("?" for _ in batch)
            rows = self._conn.execute(
                "SELECT input_hash, embedding FROM repo_embedding_cache "
                f"WHERE model = ? AND dims = ? AND input_hash IN ({placeholders})",
                (model, dims, *batch),
            ).fetchall()
            for row in rows:
                try:
                    emb = json.loads(row["embedding"])
                except Exception:
                    continue
                if isinstance(emb, list) and len(emb) == dims:
                    out[row["input_hash"]] = [float(x) for x in emb]
        return out

    def upsert_repo_embedding_cache(
        self,
        *,
        model: str,
        dims: int,
        embeddings: list[tuple[str, list[float]]],
        created_at: str,
    ) -> None:
        if not embeddings:
            return
        for input_hash, emb in embeddings:
            if len(emb) != dims:
                raise ValueError(
                    f"Repo cache embedding dim mismatch: got {len(emb)}, expected {dims} "
                    f"[input={input_hash[:12]}]",
                )
        with self._tx() as cx:
            cx.executemany(
                "INSERT INTO repo_embedding_cache "
                "(model, dims, input_hash, embedding, created_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(model, dims, input_hash) DO UPDATE SET "
                "embedding=excluded.embedding, created_at=excluded.created_at",
                [
                    (model, dims, input_hash, json.dumps(emb), created_at)
                    for input_hash, emb in embeddings
                ],
            )

    def upsert_repo_embeddings(
        self,
        *,
        repo_id: str,
        embeddings: list[tuple[str, list[float]]],
        status: str | None = None,
        indexed_at: str | None = None,
    ) -> None:
        for chunk_id, emb in embeddings:
            if len(emb) != self.dims:
                raise ValueError(
                    f"Repo chunk embedding dim mismatch: got {len(emb)}, expected {self.dims} "
                    f"[chunk={chunk_id[:12]}]",
                )
            norm = sum(float(x) * float(x) for x in emb) ** 0.5
            if not (0.5 < norm < 1.5):
                raise ValueError(
                    f"Repo chunk embedding norm {norm:.4f} out of L2-normalised range "
                    f"[chunk={chunk_id[:12]}]",
                )

        with self._tx() as cx:
            cx.executemany("DELETE FROM repo_vec WHERE id = ?", [(cid,) for cid, _ in embeddings])
            cx.executemany(
                "INSERT INTO repo_vec (id, embedding) VALUES (?, ?)",
                [(cid, serialize_float32(emb)) for cid, emb in embeddings],
            )
            if status is not None:
                if indexed_at is None:
                    cx.execute("UPDATE repo_sources SET status = ? WHERE id = ?", (status, repo_id))
                else:
                    cx.execute(
                        "UPDATE repo_sources SET status = ?, indexed_at = ? WHERE id = ?",
                        (status, indexed_at, repo_id),
                    )
        self._checkpoint()

    def upsert_repo_index(
        self,
        *,
        source: dict[str, Any],
        files: list[dict[str, Any]],
        delete_file_ids: list[str] | None = None,
    ) -> None:
        """Upsert one repo source and replace changed file rows.

        `files` are fully materialised file payloads. Each file dict
        contains `lines` and `chunks`; chunk embeddings are optional so
        exact indexing can commit before the slower semantic pass.
        Unchanged files can be omitted and will remain indexed.
        """
        for file_data in files:
            for chunk in file_data.get("chunks") or []:
                emb = chunk.get("embedding")
                if emb is None:
                    continue
                if len(emb) != self.dims:
                    raise ValueError(
                        f"Repo chunk embedding dim mismatch: got {len(emb)}, expected {self.dims} "
                        f"[chunk={chunk.get('id', '')[:12]}]",
                    )
                norm = sum(float(x) * float(x) for x in emb) ** 0.5
                if not (0.5 < norm < 1.5):
                    raise ValueError(
                        f"Repo chunk embedding norm {norm:.4f} out of L2-normalised range "
                        f"[chunk={chunk.get('id', '')[:12]}]",
                    )

        delete_ids = list(dict.fromkeys([*(delete_file_ids or []), *(f["id"] for f in files)]))

        with self._tx() as cx:
            cx.execute(
                "INSERT INTO repo_sources "
                "(id, name, url, ref, commit_sha, clone_path, indexed_at, status, extra_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "name=excluded.name, url=excluded.url, ref=excluded.ref, "
                "commit_sha=excluded.commit_sha, clone_path=excluded.clone_path, "
                "indexed_at=excluded.indexed_at, status=excluded.status, "
                "extra_json=excluded.extra_json",
                (
                    source["id"], source["name"], source["url"], source["ref"],
                    source["commit_sha"], source["clone_path"], source["indexed_at"],
                    source.get("status") or "ready",
                    json.dumps(source.get("extra") or {}, default=str),
                ),
            )

            self._delete_repo_file_rows(cx, delete_ids)

            for file_data in files:
                cx.execute(
                    "INSERT INTO repo_files "
                    "(id, repo_id, path, language, size_bytes, sha256, line_count, indexed_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        file_data["id"], source["id"], file_data["path"],
                        file_data.get("language") or "", int(file_data.get("size_bytes") or 0),
                        file_data["sha256"], int(file_data.get("line_count") or 0),
                        source["indexed_at"],
                    ),
                )

                line_rows = [
                    (
                        line["id"], source["id"], file_data["id"], file_data["path"],
                        int(line["line_no"]), line.get("text") or "", line["text_hash"],
                    )
                    for line in file_data.get("lines") or []
                ]
                cx.executemany(
                    "INSERT INTO repo_lines "
                    "(id, repo_id, file_id, path, line_no, text, text_hash) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    line_rows,
                )
                cx.executemany(
                    "INSERT INTO repo_line_fts (id, repo_name, path, line_no, body) "
                    "VALUES (?, ?, ?, ?, ?)",
                    [
                        (line[0], source["name"], line[3], line[4], line[5])
                        for line in line_rows
                    ],
                )

                chunk_rows = [
                    (
                        chunk["id"], source["id"], file_data["id"], file_data["path"],
                        int(chunk["chunk_seq"]), int(chunk["line_start"]),
                        int(chunk["line_end"]), chunk["text_hash"], chunk.get("body_text") or "",
                        source["indexed_at"],
                    )
                    for chunk in file_data.get("chunks") or []
                ]
                cx.executemany(
                    "INSERT INTO repo_chunks "
                    "(id, repo_id, file_id, path, chunk_seq, line_start, line_end, "
                    "text_hash, body_text, indexed_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    chunk_rows,
                )
                cx.executemany(
                    "INSERT INTO repo_vec (id, embedding) VALUES (?, ?)",
                    [
                        (chunk["id"], serialize_float32(chunk["embedding"]))
                        for chunk in file_data.get("chunks") or []
                        if chunk.get("embedding") is not None
                    ],
                )
                cx.executemany(
                    "INSERT INTO repo_chunk_fts (id, repo_name, path, body) "
                    "VALUES (?, ?, ?, ?)",
                    [
                        (chunk[0], source["name"], chunk[3], chunk[8])
                        for chunk in chunk_rows
                    ],
                )
        self._checkpoint()

    def upsert_repo_source(self, source: dict[str, Any]) -> None:
        """Upsert only the repo source row (no file payloads).

        Use this to record the target commit + indexing status BEFORE
        streaming file batches via `upsert_repo_files`. Lets resume
        logic detect a partial indexing run by checking `status`.
        """
        with self._tx() as cx:
            cx.execute(
                "INSERT INTO repo_sources "
                "(id, name, url, ref, commit_sha, clone_path, indexed_at, status, extra_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "name=excluded.name, url=excluded.url, ref=excluded.ref, "
                "commit_sha=excluded.commit_sha, clone_path=excluded.clone_path, "
                "indexed_at=excluded.indexed_at, status=excluded.status, "
                "extra_json=excluded.extra_json",
                (
                    source["id"], source["name"], source["url"], source["ref"],
                    source["commit_sha"], source["clone_path"], source["indexed_at"],
                    source.get("status") or "ready",
                    json.dumps(source.get("extra") or {}, default=str),
                ),
            )

    def upsert_repo_files(
        self,
        *,
        repo_id: str,
        repo_name: str,
        indexed_at: str,
        files: list[dict[str, Any]],
    ) -> None:
        """Append/replace a batch of file payloads for an existing repo source.

        Each batch commits in its own transaction so a long index run
        can survive interruption: rows already written stay on disk and
        the next run will skip those files via the sha256 check.
        """
        if not files:
            return
        for file_data in files:
            for chunk in file_data.get("chunks") or []:
                emb = chunk.get("embedding")
                if emb is None:
                    continue
                if len(emb) != self.dims:
                    raise ValueError(
                        f"Repo chunk embedding dim mismatch: got {len(emb)}, expected {self.dims} "
                        f"[chunk={chunk.get('id', '')[:12]}]",
                    )
                norm = sum(float(x) * float(x) for x in emb) ** 0.5
                if not (0.5 < norm < 1.5):
                    raise ValueError(
                        f"Repo chunk embedding norm {norm:.4f} out of L2-normalised range "
                        f"[chunk={chunk.get('id', '')[:12]}]",
                    )

        file_ids = [f["id"] for f in files]
        with self._tx() as cx:
            self._delete_repo_file_rows(cx, file_ids)
            for file_data in files:
                cx.execute(
                    "INSERT INTO repo_files "
                    "(id, repo_id, path, language, size_bytes, sha256, line_count, indexed_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        file_data["id"], repo_id, file_data["path"],
                        file_data.get("language") or "", int(file_data.get("size_bytes") or 0),
                        file_data["sha256"], int(file_data.get("line_count") or 0),
                        indexed_at,
                    ),
                )

                line_rows = [
                    (
                        line["id"], repo_id, file_data["id"], file_data["path"],
                        int(line["line_no"]), line.get("text") or "", line["text_hash"],
                    )
                    for line in file_data.get("lines") or []
                ]
                cx.executemany(
                    "INSERT INTO repo_lines "
                    "(id, repo_id, file_id, path, line_no, text, text_hash) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    line_rows,
                )
                cx.executemany(
                    "INSERT INTO repo_line_fts (id, repo_name, path, line_no, body) "
                    "VALUES (?, ?, ?, ?, ?)",
                    [
                        (line[0], repo_name, line[3], line[4], line[5])
                        for line in line_rows
                    ],
                )

                chunk_rows = [
                    (
                        chunk["id"], repo_id, file_data["id"], file_data["path"],
                        int(chunk["chunk_seq"]), int(chunk["line_start"]),
                        int(chunk["line_end"]), chunk["text_hash"], chunk.get("body_text") or "",
                        indexed_at,
                    )
                    for chunk in file_data.get("chunks") or []
                ]
                cx.executemany(
                    "INSERT INTO repo_chunks "
                    "(id, repo_id, file_id, path, chunk_seq, line_start, line_end, "
                    "text_hash, body_text, indexed_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    chunk_rows,
                )
                cx.executemany(
                    "INSERT INTO repo_vec (id, embedding) VALUES (?, ?)",
                    [
                        (chunk["id"], serialize_float32(chunk["embedding"]))
                        for chunk in file_data.get("chunks") or []
                        if chunk.get("embedding") is not None
                    ],
                )
                cx.executemany(
                    "INSERT INTO repo_chunk_fts (id, repo_name, path, body) "
                    "VALUES (?, ?, ?, ?)",
                    [
                        (chunk[0], repo_name, chunk[3], chunk[8])
                        for chunk in chunk_rows
                    ],
                )
        self._checkpoint()

    def delete_repo_files(self, repo_id: str, file_ids: list[str]) -> None:
        """Drop a set of files (and their lines/chunks/embeddings) from one repo."""
        if not file_ids:
            return
        with self._tx() as cx:
            self._delete_repo_file_rows(cx, file_ids)

    def _delete_repo_file_rows(self, cx: sqlite3.Connection, file_ids: list[str]) -> None:
        if not file_ids:
            return
        chunk_ids: list[str] = []
        line_ids: list[str] = []
        for batch in _batches(file_ids):
            placeholders = ",".join("?" for _ in batch)
            chunk_ids.extend(
                r["id"]
                for r in cx.execute(
                    f"SELECT id FROM repo_chunks WHERE file_id IN ({placeholders})",
                    batch,
                ).fetchall()
            )
            line_ids.extend(
                r["id"]
                for r in cx.execute(
                    f"SELECT id FROM repo_lines WHERE file_id IN ({placeholders})",
                    batch,
                ).fetchall()
            )
        cx.executemany("DELETE FROM repo_vec WHERE id = ?", [(cid,) for cid in chunk_ids])
        cx.executemany("DELETE FROM repo_chunk_fts WHERE id = ?", [(cid,) for cid in chunk_ids])
        cx.executemany("DELETE FROM repo_line_fts WHERE id = ?", [(lid,) for lid in line_ids])
        cx.executemany("DELETE FROM repo_chunks WHERE file_id = ?", [(fid,) for fid in file_ids])
        cx.executemany("DELETE FROM repo_lines WHERE file_id = ?", [(fid,) for fid in file_ids])
        cx.executemany("DELETE FROM repo_files WHERE id = ?", [(fid,) for fid in file_ids])

    def delete_repo(self, key: str) -> bool:
        source = self.get_repo_source(key)
        if source is None:
            return False
        repo_id = source["id"]
        file_ids = [r["id"] for r in self._conn.execute(
            "SELECT id FROM repo_files WHERE repo_id = ?", (repo_id,),
        ).fetchall()]
        with self._tx() as cx:
            self._delete_repo_file_rows(cx, file_ids)
            cx.execute("DELETE FROM repo_sources WHERE id = ?", (repo_id,))
        return True

    def search_repo_vec(
        self,
        embedding: list[float],
        limit: int = 10,
        repo_id: str | None = None,
        path_glob: str | None = None,
    ) -> list[dict[str, Any]]:
        if len(embedding) != self.dims:
            raise ValueError(
                f"Repo query embedding dim mismatch: got {len(embedding)}, expected {self.dims}",
            )
        candidate_k = limit * 5 if (repo_id or path_glob) else limit
        sql = (
            "SELECT repo_chunks.id AS id, repo_vec.distance AS distance, "
            "       repo_chunks.repo_id, repo_sources.name AS repo_name, repo_sources.url, "
            "       repo_sources.ref, repo_sources.commit_sha, repo_chunks.file_id, "
            "       repo_chunks.path, repo_files.language, repo_chunks.line_start, "
            "       repo_chunks.line_end, repo_chunks.body_text "
            "FROM repo_vec "
            "JOIN repo_chunks ON repo_chunks.id = repo_vec.id "
            "JOIN repo_files ON repo_files.id = repo_chunks.file_id "
            "JOIN repo_sources ON repo_sources.id = repo_chunks.repo_id "
            "WHERE embedding MATCH ? AND k = ? "
        )
        params: list[Any] = [serialize_float32(embedding), candidate_k]
        if repo_id:
            sql += "AND repo_chunks.repo_id = ? "
            params.append(repo_id)
        if path_glob:
            sql += "AND repo_chunks.path GLOB ? "
            params.append(path_glob)
        sql += "ORDER BY distance ASC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            d = _repo_row_to_dict(r)
            d["score"] = 1.0 - float(r["distance"])
            d["match_type"] = "chunk"
            out.append(d)
        return out

    def search_repo_bm25(
        self,
        query: str,
        limit: int = 10,
        repo_id: str | None = None,
        path_glob: str | None = None,
    ) -> list[dict[str, Any]]:
        match_expr = _fts_match_expr(query)
        if not match_expr:
            return []
        candidate_k = limit * 5 if (repo_id or path_glob) else limit
        # Column weights: (id UNINDEXED, repo_name, path, body)
        sql = (
            "SELECT repo_chunks.id AS id, "
            "       bm25(repo_chunk_fts, ?, ?, ?, ?) AS bm25_score, "
            "       repo_chunks.repo_id, repo_sources.name AS repo_name, repo_sources.url, "
            "       repo_sources.ref, repo_sources.commit_sha, repo_chunks.file_id, "
            "       repo_chunks.path, repo_files.language, repo_chunks.line_start, "
            "       repo_chunks.line_end, repo_chunks.body_text "
            "FROM repo_chunk_fts "
            "JOIN repo_chunks ON repo_chunks.id = repo_chunk_fts.id "
            "JOIN repo_files ON repo_files.id = repo_chunks.file_id "
            "JOIN repo_sources ON repo_sources.id = repo_chunks.repo_id "
            "WHERE repo_chunk_fts MATCH ? "
        )
        params: list[Any] = [
            _BM25_UNINDEXED_WEIGHT,
            _BM25_REPO_NAME_WEIGHT,
            _BM25_PATH_WEIGHT,
            _BM25_BODY_WEIGHT,
            match_expr,
        ]
        if repo_id:
            sql += "AND repo_chunks.repo_id = ? "
            params.append(repo_id)
        if path_glob:
            sql += "AND repo_chunks.path GLOB ? "
            params.append(path_glob)
        sql += "ORDER BY bm25_score ASC LIMIT ?"
        params.append(candidate_k)
        try:
            rows = self._conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            return []
        return [_repo_bm25_row_to_dict(r, "chunk") for r in rows[:limit]]

    def search_repo_lines(
        self,
        query: str,
        limit: int = 10,
        repo_id: str | None = None,
        path_glob: str | None = None,
    ) -> list[dict[str, Any]]:
        match_expr = _fts_match_expr(query)
        if not match_expr:
            return []
        candidate_k = limit * 5 if (repo_id or path_glob) else limit
        # Column weights: (id UNINDEXED, repo_name, path, line_no UNINDEXED, body)
        sql = (
            "SELECT repo_lines.id AS id, "
            "       bm25(repo_line_fts, ?, ?, ?, ?, ?) AS bm25_score, "
            "       repo_lines.repo_id, repo_sources.name AS repo_name, repo_sources.url, "
            "       repo_sources.ref, repo_sources.commit_sha, repo_lines.file_id, "
            "       repo_lines.path, repo_files.language, repo_lines.line_no AS line_start, "
            "       repo_lines.line_no AS line_end, repo_lines.text AS body_text "
            "FROM repo_line_fts "
            "JOIN repo_lines ON repo_lines.id = repo_line_fts.id "
            "JOIN repo_files ON repo_files.id = repo_lines.file_id "
            "JOIN repo_sources ON repo_sources.id = repo_lines.repo_id "
            "WHERE repo_line_fts MATCH ? "
        )
        params: list[Any] = [
            _BM25_UNINDEXED_WEIGHT,
            _BM25_REPO_NAME_WEIGHT,
            _BM25_PATH_WEIGHT,
            _BM25_UNINDEXED_WEIGHT,
            _BM25_BODY_WEIGHT,
            match_expr,
        ]
        if repo_id:
            sql += "AND repo_lines.repo_id = ? "
            params.append(repo_id)
        if path_glob:
            sql += "AND repo_lines.path GLOB ? "
            params.append(path_glob)
        sql += "ORDER BY bm25_score ASC LIMIT ?"
        params.append(candidate_k)
        try:
            rows = self._conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            return []
        return [_repo_bm25_row_to_dict(r, "line") for r in rows[:limit]]

    def get_repo_file(self, repo_id: str, path: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT repo_files.id, repo_files.repo_id, repo_sources.name AS repo_name, "
            "       repo_sources.url, repo_sources.ref, repo_sources.commit_sha, "
            "       repo_files.path, repo_files.language, repo_files.size_bytes, "
            "       repo_files.sha256, repo_files.line_count, repo_files.indexed_at "
            "FROM repo_files JOIN repo_sources ON repo_sources.id = repo_files.repo_id "
            "WHERE repo_files.repo_id = ? AND repo_files.path = ?",
            (repo_id, path),
        ).fetchone()
        return dict(row) if row else None

    def get_repo_file_lines(
        self,
        repo_id: str,
        path: str,
        start: int | None = None,
        end: int | None = None,
    ) -> list[dict[str, Any]]:
        start = max(1, int(start or 1))
        params: list[Any] = [repo_id, path, start]
        sql = (
            "SELECT line_no, text FROM repo_lines "
            "WHERE repo_id = ? AND path = ? AND line_no >= ? "
        )
        if end is not None:
            sql += "AND line_no <= ? "
            params.append(max(start, int(end)))
        sql += "ORDER BY line_no ASC"
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM meta").fetchone()[0]

    # -- source-level feedback (👍 / 👎) ------------------------------------

    def record_source_feedback(
        self,
        *,
        source_id: str,
        query_text: str,
        query_emb: list[float],
        rating: int,
        feedback_id: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        """Persist a 👍/👎 vote on `source_id` for `query_text`.

        Idempotent on `(source_id, query_text, rating)` — re-recording the
        same vote returns the existing feedback id. Changing rating for
        the same (source, query) replaces the old row (cancel-and-replace
        so the unique constraint stays clean).
        """
        if rating not in (-1, 1):
            raise ValueError(f"rating must be -1 or 1, got {rating!r}")
        if len(query_emb) != self.dims:
            raise ValueError(
                f"query_emb dim mismatch: got {len(query_emb)}, expected {self.dims}"
            )
        now = datetime.now(UTC).isoformat()
        # Find any existing row for this (source, query) regardless of rating.
        existing = self._conn.execute(
            "SELECT id, rating FROM source_feedback "
            "WHERE source_id = ? AND query_text = ?",
            (source_id, query_text),
        ).fetchone()
        if existing and int(existing["rating"]) == rating:
            return str(existing["id"])
        with self._conn:
            if existing:
                # Replacing rating — drop old vec + row first.
                self._conn.execute(
                    "DELETE FROM source_feedback_vec WHERE feedback_id = ?",
                    (existing["id"],),
                )
                self._conn.execute(
                    "DELETE FROM source_feedback WHERE id = ?",
                    (existing["id"],),
                )
            fid = feedback_id or uuid.uuid4().hex
            self._conn.execute(
                "INSERT INTO source_feedback "
                "(id, source_id, query_text, rating, created_at, extra_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    fid, source_id, query_text, rating, now,
                    json.dumps(extra) if extra else None,
                ),
            )
            self._conn.execute(
                "INSERT INTO source_feedback_vec (feedback_id, query_emb) "
                "VALUES (?, ?)",
                (fid, serialize_float32(query_emb)),
            )
        return fid

    def find_feedback_for_source(
        self,
        source_id: str,
        query_emb: list[float],
        *,
        threshold: float = 0.85,
        limit: int = 16,
    ) -> list[dict[str, Any]]:
        """Return prior feedback rows on `source_id` whose query embedding
        is cosine-similar to `query_emb` at >= `threshold`.

        Returns a list of dicts with keys: id, rating, query_text,
        similarity (float in [0, 1]), created_at. Empty list if none.
        """
        if len(query_emb) != self.dims:
            return []
        rows = self._conn.execute(
            "SELECT fb.id, fb.rating, fb.query_text, fb.created_at, "
            "       fv.distance "
            "FROM source_feedback fb "
            "JOIN source_feedback_vec fv ON fb.id = fv.feedback_id "
            "WHERE fb.source_id = ? "
            "  AND fv.query_emb MATCH ? "
            "  AND k = ? "
            "ORDER BY fv.distance ASC",
            (source_id, serialize_float32(query_emb), int(limit)),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            dist = float(r["distance"])
            sim = 1.0 - dist
            if sim < threshold:
                continue
            out.append({
                "id": r["id"],
                "rating": int(r["rating"]),
                "query_text": r["query_text"],
                "created_at": r["created_at"],
                "similarity": sim,
            })
        return out

    def list_source_feedback(
        self,
        *,
        source_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if source_id:
            rows = self._conn.execute(
                "SELECT id, source_id, query_text, rating, created_at, extra_json "
                "FROM source_feedback WHERE source_id = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (source_id, int(limit)),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, source_id, query_text, rating, created_at, extra_json "
                "FROM source_feedback ORDER BY created_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [dict(r) for r in rows]

    def clear_source_feedback(self, source_id: str) -> int:
        """Drop all feedback rows for a source. Returns count deleted."""
        ids = [
            r["id"] for r in self._conn.execute(
                "SELECT id FROM source_feedback WHERE source_id = ?",
                (source_id,),
            ).fetchall()
        ]
        if not ids:
            return 0
        with self._conn:
            placeholders = ",".join("?" * len(ids))
            self._conn.execute(
                f"DELETE FROM source_feedback_vec WHERE feedback_id IN ({placeholders})",
                ids,
            )
            self._conn.execute(
                f"DELETE FROM source_feedback WHERE id IN ({placeholders})",
                ids,
            )
        return len(ids)

    def close(self) -> None:
        # Closes the calling thread's connection. Other threads' connections
        # are released when their threads end (or at process exit) — adequate
        # for the daemon/CLI lifecycles that use this store.
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            with suppress(Exception):
                conn.close()
            self._local.conn = None


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


def _fts_match_expr(query: str) -> str:
    if not query or not query.strip():
        return ""
    tokens = [t for t in re.findall(r"\w+", query, flags=re.UNICODE) if t]
    return " ".join(f'"{t}"' for t in tokens)


def _repo_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    raw = dict(zip(row.keys(), row, strict=True))
    return {k: v for k, v in raw.items() if k not in {"distance", "bm25_score"}}


def _repo_bm25_row_to_dict(row: sqlite3.Row, match_type: str) -> dict[str, Any]:
    d = _repo_row_to_dict(row)
    bm = float(row["bm25_score"])
    d["score"] = 1.0 / (1.0 + abs(bm)) if bm < 0 else 0.0
    d["match_type"] = match_type
    return d


def _batches(items: list[str], size: int = 500) -> Iterator[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


__all__ = ["VecStore"]
