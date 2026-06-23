from __future__ import annotations

import logging
import re
import sqlite3
import threading

from ._base import _StoreBase

_log = logging.getLogger(__name__)

# Serialises first-touch schema creation across threads. VecStore uses
# thread-local connections, so two FastMCP worker threads can hit a fresh DB
# at once and race the (non-transactional) vec0 CREATE statements.
_SCHEMA_INIT_LOCK = threading.Lock()

_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

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
-- Exact + prefix-LIKE path lookups (queries.py filters by path) and
-- type-filtered recency scans. Added by _ensure_secondary_indices() on
-- existing DBs too — see that method.
CREATE INDEX IF NOT EXISTS idx_meta_path         ON meta(path);
CREATE INDEX IF NOT EXISTS idx_meta_type_updated ON meta(type, updated);

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

CREATE TABLE IF NOT EXISTS memory_health (
    id          TEXT PRIMARY KEY,
    confidence  REAL NOT NULL DEFAULT 1.0,
    roi_score   REAL NOT NULL DEFAULT 1.0,
    updated_at  TEXT
);

CREATE INDEX IF NOT EXISTS idx_health_roi  ON memory_health(roi_score);
CREATE INDEX IF NOT EXISTS idx_health_conf ON memory_health(confidence);
"""


_REQUIRED_SCHEMA_OBJECTS = frozenset(
    {
        "schema_meta",
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
        "memory_health",
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
_BM25_ES_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "al",
        "ante",
        "bajo",
        "con",
        "contra",
        "de",
        "del",
        "desde",
        "donde",
        "dónde",
        "durante",
        "e",
        "el",
        "ella",
        "ellas",
        "ellos",
        "en",
        "entre",
        "era",
        "eres",
        "es",
        "esa",
        "esas",
        "ese",
        "eso",
        "esos",
        "esta",
        "estas",
        "este",
        "esto",
        "estos",
        "fue",
        "fueron",
        "hacia",
        "hasta",
        "hay",
        "la",
        "las",
        "le",
        "les",
        "lo",
        "los",
        "mas",
        "más",
        "me",
        "mi",
        "mis",
        "mí",
        "ni",
        "no",
        "nos",
        "o",
        "para",
        "pero",
        "por",
        "porque",
        "pues",
        "que",
        "qué",
        "quien",
        "quién",
        "quienes",
        "quiénes",
        "sea",
        "ser",
        "si",
        "sí",
        "sin",
        "sobre",
        "son",
        "su",
        "sus",
        "te",
        "ti",
        "tu",
        "tus",
        "tú",
        "un",
        "una",
        "unas",
        "uno",
        "unos",
        "u",
        "y",
        "ya",
        "yo",
        "como",
        "cómo",
        "cual",
        "cuál",
        "cuales",
        "cuáles",
        "cuando",
        "cuándo",
    }
)


class _SchemaMixin(_StoreBase):
    def _init_schema(self) -> None:
        # Most CLI commands are reads. If the schema already exists, avoid
        # no-op DDL because it still needs a write/schema lock and can fail
        # while long repo indexing is writing batches. The lock makes the
        # check-then-create atomic across threads (thread-local connections).
        with _SCHEMA_INIT_LOCK:
            self._init_schema_locked()

    def _init_schema_locked(self) -> None:
        if not self._schema_ready():
            with self._conn:
                self._conn.executescript(_SCHEMA_DDL)
                # `vec0` virtual tables are created out of the static DDL string
                # because their dimensionality is dynamic (see
                # `_create_vec_tables`). `IF NOT EXISTS` means an existing vec
                # table keeps its old DDL — `_validate_vec_schema` (below)
                # migrates it.
                self._create_vec_tables(self._conn)
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
        # Always run dims validation + the in-place partition/metadata migration,
        # whether the DB was just created or already existed — a binary upgraded
        # over an old DB hits the second path. Both are cheap no-ops once current.
        self._validate_vec_dims()
        self._validate_vec_schema()
        self._ensure_secondary_indices()
        # Ensure schema_meta exists (older DBs predate it) then stamp/verify
        # the embedder model + dims so a model swap is caught at open time
        # rather than producing a confusing dim-mismatch error at query time.
        self._ensure_schema_meta_table()
        self._check_embedder_version()
        self._run_migrations()

    # Secondary B-tree indices on `meta` that older DBs predate. Kept out of
    # the `_schema_ready()`-gated DDL block (which only runs on fresh DBs) so a
    # binary upgraded over an existing corpus still gets them. The existence
    # pre-check is a read (no schema lock); we only take a write lock when an
    # index is actually missing, so the common already-current case is free.
    _SECONDARY_INDICES: tuple[tuple[str, str], ...] = (
        ("idx_meta_path", "CREATE INDEX IF NOT EXISTS idx_meta_path ON meta(path)"),
        (
            "idx_meta_type_updated",
            "CREATE INDEX IF NOT EXISTS idx_meta_type_updated ON meta(type, updated)",
        ),
    )

    def _ensure_secondary_indices(self) -> None:
        present = {
            str(row["name"])
            for row in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }
        missing = [ddl for name, ddl in self._SECONDARY_INDICES if name not in present]
        if not missing:
            return
        with self._conn:
            for ddl in missing:
                self._conn.execute(ddl)

    def _schema_ready(self) -> bool:
        rows = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        ).fetchall()
        present = {str(row["name"]) for row in rows}
        return _REQUIRED_SCHEMA_OBJECTS.issubset(present)

    def _vec_table_dims(self, table: str) -> int | None:
        if table not in {"vec", "repo_vec", "source_feedback_vec"}:
            raise ValueError(f"unknown vector table: {table!r}")
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        if not row or not row["sql"]:
            return None
        # Matches `embedding FLOAT[N]` (vec/repo_vec) and `query_emb FLOAT[N]`
        # (source_feedback_vec) alike.
        match = re.search(r"FLOAT\[(\d+)\]", str(row["sql"]))
        return int(match.group(1)) if match else None

    def _create_vec_tables(self, conn: sqlite3.Connection) -> None:
        """(Re)create the three vec0 virtual tables at the current dimensionality.

        `distance_metric=cosine` makes `.distance` a true cosine distance
        (1 - dot, range [0, 2]); without it vec0 defaults to L2 — monotonic for
        unit vectors but the absolute values are wrong (an L2 of 0.80 is a
        cosine of 0.68, so `score = 1 - distance` would report 0.20 not 0.68;
        verified empirically 2026-05-07).

        - `vec.type` is a METADATA column so `search()` can push a `type = ?` /
          `type != ?` filter INTO the kNN instead of over-fetching (`k * 5`) and
          filtering off-type rows out after the join. `id TEXT PRIMARY KEY`
          stays for the cheap `meta` join.
        - `repo_vec.repo_id` and `source_feedback_vec.source_id` are PARTITION
          KEYS so repo-scoped / per-source searches run an exact pre-filtered
          kNN — the global-kNN-then-filter shape could silently drop matches
          that fell outside the global top-k.
        """
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec USING vec0("
            f"id TEXT PRIMARY KEY, embedding FLOAT[{self.dims}] distance_metric=cosine, "
            f"type TEXT)"
        )
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS repo_vec USING vec0("
            f"id TEXT PRIMARY KEY, repo_id TEXT PARTITION KEY, "
            f"embedding FLOAT[{self.dims}] distance_metric=cosine)"
        )
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS source_feedback_vec USING vec0("
            f"feedback_id TEXT PRIMARY KEY, source_id TEXT PARTITION KEY, "
            f"query_emb FLOAT[{self.dims}] distance_metric=cosine)"
        )

    # vec0 round-trips the raw float32 blob on `SELECT embedding`, so a stale
    # table can be migrated in place by copying vectors into a fresh table with
    # the new partition/metadata column — no re-embedding, no data loss.
    # (table, pk_col, vec_col, partition/metadata col, source table+id for the
    # backfilled column).
    _VEC_MIGRATIONS = (
        ("vec", "id", "embedding", "type", "meta", "id", "type"),
        ("repo_vec", "id", "embedding", "repo_id", "repo_chunks", "id", "repo_id"),
        (
            "source_feedback_vec",
            "feedback_id",
            "query_emb",
            "source_id",
            "source_feedback",
            "id",
            "source_id",
        ),
    )

    def _validate_vec_schema(self) -> None:
        """Auto-migrate any pre-upgrade vec0 table (missing PARTITION KEY /
        metadata column) to the current layout, in place, preserving vectors.

        vec tables are created `IF NOT EXISTS`, so an upgraded binary opened
        against an old DB would otherwise keep the old DDL — and the new kNN
        filters reference columns that don't exist yet. Rather than hard-fail
        and force a full re-embed, copy each row's vector into a fresh table
        and backfill the new column from its companion table (`type` from
        `meta`, `repo_id` from `repo_chunks`, `source_id` from
        `source_feedback`). Runs once; idempotent thereafter.
        """
        for (
            table,
            pk_col,
            vec_col,
            new_col,
            src_table,
            src_key,
            src_col,
        ) in self._VEC_MIGRATIONS:
            row = self._conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table,),
            ).fetchone()
            if not row or not row["sql"]:
                continue
            if re.search(rf"\b{new_col}\b", str(row["sql"])) is not None:
                continue  # already migrated
            _log.warning(
                "migrating vec table `%s` to add `%s` (partition/metadata) — "
                "copying vectors in place, no re-embed",
                table,
                new_col,
            )
            rows = self._conn.execute(
                f"SELECT v.{pk_col} AS pk, v.{vec_col} AS emb, s.{src_col} AS newval "
                f"FROM {table} v LEFT JOIN {src_table} s ON s.{src_key} = v.{pk_col}"
            ).fetchall()
            payload = [
                (r["pk"], r["newval"], r["emb"]) for r in rows if r["emb"] is not None
            ]
            with self._conn:
                self._conn.execute(f"DROP TABLE {table}")
                self._create_vec_tables(self._conn)
                if payload:
                    self._conn.executemany(
                        f"INSERT INTO {table} ({pk_col}, {new_col}, {vec_col}) "
                        f"VALUES (?, ?, ?)",
                        payload,
                    )
            _log.info("migrated `%s`: %d vectors preserved", table, len(payload))

    def _validate_vec_dims(self) -> None:
        import os

        if os.environ.get("MEMO_SKIP_MODEL_VERSION_CHECK", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return
        for table, label in (
            ("vec", "Embedding"),
            ("repo_vec", "Repo embedding"),
            ("source_feedback_vec", "Feedback embedding"),
        ):
            actual_dims = self._vec_table_dims(table)
            if actual_dims is not None and actual_dims != self.dims:
                raise RuntimeError(
                    f"{label} dimension mismatch: store has {actual_dims}D vectors "
                    f"but config expects {self.dims}D. "
                    f"This usually happens after switching between model profiles "
                    f"(e.g., from 'default' with 1024D to 'quality' with 2560D) "
                    f"without running 'memo reindex'.\n"
                    f"Fix: Run 'memo reindex --rebuild' to rebuild the index with the correct dimensions.\n"
                    f"Or check your model profile: MEMO_MODEL_PROFILE (current: {self.dims}D) "
                    f"or MEMO_EMBEDDER_DIMS (current: {self.dims})."
                )

    def _ensure_schema_meta_table(self) -> None:
        """Create schema_meta if it does not exist (existing DBs predate it).

        The DDL block in _init_schema_locked only runs on fresh DBs (when
        _schema_ready() returns False). Existing corpora skip it, so we must
        ensure schema_meta is present via an idempotent CREATE IF NOT EXISTS
        outside that block.
        """
        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'schema_meta'"
        ).fetchone()
        if row is None:
            with self._conn:
                self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS schema_meta ("
                    "key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )

    def _check_embedder_version(self) -> None:
        """Stamp or verify embedder_model + embedder_dims in schema_meta.

        On first open: INSERT OR IGNORE writes the current model/dims.
        On subsequent opens: SELECT and compare. A mismatch raises StorageError
        with a clear instruction to run 'memo reindex --rebuild'.

        Bypassed when:
        - MEMO_SKIP_MODEL_VERSION_CHECK=1  (tests with stub embedders)
        - The stored model name is 'stub'  (legacy test DBs)
        - The current model name is 'stub' (test stub embedded inline)
        """
        import os

        # Fast path: bypass for tests / CI
        if os.environ.get("MEMO_SKIP_MODEL_VERSION_CHECK", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return

        # The store layer cannot import memo.flags (circular dep risk), so
        # read the env var directly (same pattern as bm25_queries._env_float).
        current_model: str = getattr(self, "embedder_model", "") or ""
        current_dims: int = self.dims

        # Skip check when using a test stub model (by convention, model=""
        # or model contains "stub").
        if not current_model or "stub" in current_model.lower():
            return

        with self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO schema_meta (key, value) VALUES (?, ?)",
                ("embedder_model", current_model),
            )
            self._conn.execute(
                "INSERT OR IGNORE INTO schema_meta (key, value) VALUES (?, ?)",
                ("embedder_dims", str(current_dims)),
            )

        stored_model_row = self._conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'embedder_model'"
        ).fetchone()
        stored_dims_row = self._conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'embedder_dims'"
        ).fetchone()

        if stored_model_row is None or stored_dims_row is None:
            return

        stored_model = str(stored_model_row["value"])
        stored_dims_str = str(stored_dims_row["value"])

        # Skip check if stored model is a test stub
        if "stub" in stored_model.lower():
            return

        try:
            stored_dims = int(stored_dims_str)
        except ValueError:
            from ..errors import StorageError

            raise StorageError(
                f"Corrupted embedder_dims in schema_meta: {stored_dims_str!r}. "
                "Run 'memo reindex --rebuild' to fix."
            ) from None

        if stored_model != current_model or stored_dims != current_dims:
            from ..errors import StorageError

            raise StorageError(
                f"Embedder model mismatch: index was built with {stored_model} "
                f"({stored_dims}d) but current config is {current_model} "
                f"({current_dims}d). Run 'memo reindex --rebuild' to rebuild the index."
            )
