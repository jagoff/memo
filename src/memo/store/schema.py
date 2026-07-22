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
    extra_json  TEXT,
    valid_at    TEXT,
    invalid_at  TEXT
);

CREATE INDEX IF NOT EXISTS idx_meta_type    ON meta(type);
CREATE INDEX IF NOT EXISTS idx_meta_updated ON meta(updated);
CREATE INDEX IF NOT EXISTS idx_meta_created ON meta(created);
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
CREATE INDEX IF NOT EXISTS idx_access_count_last ON access(access_count, last_accessed);

CREATE TABLE IF NOT EXISTS memory_health (
    id            TEXT PRIMARY KEY,
    confidence    REAL NOT NULL DEFAULT 1.0,
    roi_score     REAL NOT NULL DEFAULT 1.0,
    updated_at    TEXT,
    support_count INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_health_roi  ON memory_health(roi_score);
CREATE INDEX IF NOT EXISTS idx_health_conf ON memory_health(confidence);

CREATE TABLE IF NOT EXISTS secret_store (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL CHECK (kind IN (
        'api_token', 'password', 'ssh_key', 'db_credential', 'certificate', 'generic'
    )),
    encrypted_blob BLOB NOT NULL,
    nonce BLOB NOT NULL,
    created_at TEXT NOT NULL,
    accessed_at TEXT,
    accessed_count INTEGER DEFAULT 0,
    detection_method TEXT CHECK (detection_method IN ('regex', 'llm', 'manual')),
    confidence REAL CHECK (confidence >= 0.0 AND confidence <= 1.0)
);

CREATE INDEX IF NOT EXISTS idx_secret_name ON secret_store(name);
CREATE INDEX IF NOT EXISTS idx_secret_kind ON secret_store(kind);
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
            # Run schema DDL OUTSIDE a transaction. `executescript` issues an
            # implicit COMMIT (which would void a wrapping BEGIN IMMEDIATE), and
            # the vec0/fts5 virtual-table CREATEs aren't transactional anyway.
            # Every statement is idempotent (CREATE ... IF NOT EXISTS) and
            # individually atomic in the connection's autocommit mode.
            self._conn.executescript(_SCHEMA_DDL)
            self._create_vec_tables(self._conn)
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
        # Guard quant mode BEFORE _validate_vec_schema, whose in-place migration
        # re-inserts stored `vec` blobs — a dtype mismatch there would corrupt.
        self._validate_vec_quant()
        self._validate_vec_schema()
        self._ensure_secondary_indices()
        self._ensure_schema_meta_table()
        self._check_embedder_version()
        # Always run migrations (not gated by _schema_ready)
        self._run_migrations()
        # Inline v2→v3 migration: add pattern columns to existing meta table
        cols = {row["name"] for row in self._conn.execute("PRAGMA table_info(meta)").fetchall()}
        new_cols = {
            "topic_key": "ALTER TABLE meta ADD COLUMN topic_key TEXT",
            "normalized_hash": "ALTER TABLE meta ADD COLUMN normalized_hash TEXT",
            "session_id": "ALTER TABLE meta ADD COLUMN session_id TEXT",
            "revision_count": "ALTER TABLE meta ADD COLUMN revision_count INTEGER DEFAULT 1",
            "duplicate_count": "ALTER TABLE meta ADD COLUMN duplicate_count INTEGER DEFAULT 0",
            "last_seen_at": "ALTER TABLE meta ADD COLUMN last_seen_at TEXT",
            "deleted_at": "ALTER TABLE meta ADD COLUMN deleted_at TEXT",
            "review_after": "ALTER TABLE meta ADD COLUMN review_after TEXT",
        }
        added = False
        for col, ddl in new_cols.items():
            if col not in cols:
                try:
                    self._conn.execute(ddl)
                    cols.add(col)
                    added = True
                except Exception as e:
                    _log.debug("schema migration col %r failed: %s", col, e)
        version_row = self._conn.execute("PRAGMA user_version").fetchone()
        user_version = int(version_row[0]) if version_row else 0
        if added and user_version < 3:
            self.set_user_version(3)
        # Cache pattern-column presence so upsert/upsert_text_only skip a
        # per-write PRAGMA table_info(meta). `cols` reflects the post-migration
        # column set (updated as the ALTERs above succeed).
        self._has_pattern_cols = "topic_key" in cols and "normalized_hash" in cols

        # Inline migration (C1): corroboration counter on memory_health.
        # CREATE IF NOT EXISTS skips existing tables, so pre-existing DBs
        # need the column added here. Idempotent, no user_version dance
        # (the fresh-DB early-return in _run_migrations would skip a v4).
        hcols = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(memory_health)").fetchall()
        }
        if "support_count" not in hcols:
            try:
                self._conn.execute(
                    "ALTER TABLE memory_health ADD COLUMN support_count INTEGER NOT NULL DEFAULT 0"
                )
            except Exception as e:
                _log.debug("schema migration memory_health.support_count failed: %s", e)

        # Inline migration (V1): verification state tracking columns for state machine.
        # CREATE IF NOT EXISTS skips existing tables, so pre-existing DBs
        # need the columns added here. Idempotent, no user_version dance
        # (the fresh-DB early-return in _run_migrations would skip v4).
        mcols = {row["name"] for row in self._conn.execute("PRAGMA table_info(meta)").fetchall()}
        if "verification_state" not in mcols:
            try:
                self._conn.execute(
                    "ALTER TABLE meta ADD COLUMN verification_state TEXT DEFAULT 'unverified'"
                )
            except Exception as e:
                _log.debug("schema migration meta.verification_state failed: %s", e)
        if "verified_at" not in mcols:
            try:
                self._conn.execute("ALTER TABLE meta ADD COLUMN verified_at INTEGER")
            except Exception as e:
                _log.debug("schema migration meta.verified_at failed: %s", e)

        # Inline migration (T1): record-level bi-temporal validity. valid_at =
        # world-validity start (defaults to created on save); invalid_at =
        # world-validity end (NULL = currently true, closed at successor's
        # valid_at on contradiction-supersede). Distinct from created/updated
        # (learned-time). CREATE IF NOT EXISTS skips the existing meta table so
        # pre-existing DBs need these added here. Idempotent.
        if "valid_at" not in mcols:
            try:
                self._conn.execute("ALTER TABLE meta ADD COLUMN valid_at TEXT")
            except Exception as e:
                _log.debug("schema migration meta.valid_at failed: %s", e)
        if "invalid_at" not in mcols:
            try:
                self._conn.execute("ALTER TABLE meta ADD COLUMN invalid_at TEXT")
            except Exception as e:
                _log.debug("schema migration meta.invalid_at failed: %s", e)
        # Partial index keeps the default-recall "currently valid" filter cheap.
        try:
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_meta_invalid_at "
                "ON meta(invalid_at) WHERE invalid_at IS NOT NULL"
            )
        except Exception as e:
            _log.debug("schema migration idx_meta_invalid_at failed: %s", e)

        # Ensure sessions table exists (session pattern)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS sessions ("
            "id TEXT PRIMARY KEY, project TEXT NOT NULL, directory TEXT, "
            "started_at TEXT NOT NULL, ended_at TEXT, summary TEXT, status TEXT DEFAULT 'active')"
        )

        # Ensure memory_relations table exists (session pattern)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS memory_relations ("
            "id TEXT PRIMARY KEY, sync_id TEXT, source_id TEXT NOT NULL, target_id TEXT NOT NULL, "
            "relation TEXT, judgment_status TEXT DEFAULT 'pending', reason TEXT, "
            "confidence REAL, session_id TEXT, created_at TEXT, updated_at TEXT)"
        )
        # Ensure indexes for relations
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rel_source ON memory_relations(source_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rel_target ON memory_relations(target_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rel_status ON memory_relations(judgment_status)"
        )

    # Secondary B-tree indices on `meta` that older DBs predate. Kept out of
    # the `_schema_ready()`-gated DDL block (which only runs on fresh DBs) so a
    # binary upgraded over an existing corpus still gets them. The existence
    # pre-check is a read (no schema lock); we only take a write lock when an
    # index is actually missing, so the common already-current case is free.
    _SECONDARY_INDICES: tuple[tuple[str, str], ...] = (
        ("idx_meta_path", "CREATE INDEX IF NOT EXISTS idx_meta_path ON meta(path)"),
        (
            "idx_meta_path_nocase",
            "CREATE INDEX IF NOT EXISTS idx_meta_path_nocase ON meta(path COLLATE NOCASE)",
        ),
        (
            "idx_meta_type_updated",
            "CREATE INDEX IF NOT EXISTS idx_meta_type_updated ON meta(type, updated)",
        ),
        (
            "idx_meta_extra_vault",
            "CREATE INDEX IF NOT EXISTS idx_meta_extra_vault "
            "ON meta(json_extract(extra_json, '$.vault'))",
        ),
        (
            "idx_meta_extra_source",
            "CREATE INDEX IF NOT EXISTS idx_meta_extra_source "
            "ON meta(json_extract(extra_json, '$.source'))",
        ),
        (
            "idx_meta_extra_parent_path",
            "CREATE INDEX IF NOT EXISTS idx_meta_extra_parent_path "
            "ON meta(json_extract(extra_json, '$.parent_path'))",
        ),
        (
            "idx_meta_extra_parent_id",
            "CREATE INDEX IF NOT EXISTS idx_meta_extra_parent_id "
            "ON meta(json_extract(extra_json, '$.parent_id'))",
        ),
        (
            "idx_access_count_last",
            "CREATE INDEX IF NOT EXISTS idx_access_count_last "
            "ON access(access_count, last_accessed)",
        ),
        (
            "idx_meta_created",
            "CREATE INDEX IF NOT EXISTS idx_meta_created ON meta(created)",
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
        with self._tx() as cx:
            for ddl in missing:
                cx.execute(ddl)

    def _schema_ready(self) -> bool:
        rows = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        ).fetchall()
        present = {str(row["name"]) for row in rows}
        return _REQUIRED_SCHEMA_OBJECTS.issubset(present)

    def _vec_table_dims(self, table: str) -> int | None:
        if table not in {
            "vec",
            "repo_vec",
            "source_feedback_vec",
            "hype_vec",
            "episode_vec",
        }:
            raise ValueError(f"unknown vector table: {table!r}")
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        if not row or not row["sql"]:
            return None
        # Matches `embedding FLOAT[N]` (vec/repo_vec) and `query_emb FLOAT[N]`
        # (source_feedback_vec) alike. Also `int8[N]` once the `vec` table is
        # quantized — otherwise an int8 DDL would return None and silently
        # disable the dims-mismatch guard.
        match = re.search(r"(?:FLOAT|int8)\[(\d+)\]", str(row["sql"]))
        return int(match.group(1)) if match else None

    def _vec_table_dtype(self, table: str) -> str:
        """Physical vec0 element dtype for `table`, derived from its live DDL.

        Returns ``"int8"`` when the column is declared ``int8[N]``, else
        ``"off"`` (float32 — the historical default). The DDL is the single
        source of truth for the quant guard: a self-written schema_meta stamp
        would self-bless a legacy float32 DB opened under an int8 flag.
        """
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        if not row or not row["sql"]:
            return "off"
        return "int8" if re.search(r"\bint8\[\d+\]", str(row["sql"])) else "off"

    def _vec_dtype_ddl(self) -> str:
        """vec0 column type token for the main `vec` table (quant-dependent)."""
        return "int8" if self._quant_int8 else "FLOAT"

    def _vec_bind_new(self) -> str:
        """Bind expression for a FRESH float32 embedding param going into `vec`.

        Under int8 the L2-normalised float32 vector is quantized in SQL via
        vec_quantize_int8(...,'unit'); the bound param stays serialize_float32
        bytes either way (embeddings never leave float32 in RAM — MLX invariant).
        """
        return "vec_quantize_int8(vec_f32(?), 'unit')" if self._quant_int8 else "?"

    def _vec_bind_stored(self) -> str:
        """Bind expression for re-inserting a blob ALREADY read from `vec`.

        Under int8 that blob is already 1 B/dim int8 bytes, so it must be typed
        with vec_int8() (never re-quantized through vec_f32, which would
        reinterpret the bytes as float32 and corrupt the vector)."""
        return "vec_int8(?)" if self._quant_int8 else "?"

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
            f"id TEXT PRIMARY KEY, embedding {self._vec_dtype_ddl()}[{self.dims}] distance_metric=cosine, "
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
            # Snapshot + DROP + re-insert share one BEGIN IMMEDIATE tx: a row
            # committed by another process (e.g. a still-running recall daemon
            # on the old binary) between an outside-tx snapshot and the DROP
            # would be silently lost. The immediate lock means the read below
            # sees the final pre-migration state.
            with self._tx() as cx:
                rows = cx.execute(
                    f"SELECT v.{pk_col} AS pk, v.{vec_col} AS emb, s.{src_col} AS newval "  # noqa: S608
                    f"FROM {table} v LEFT JOIN {src_table} s ON s.{src_key} = v.{pk_col}"
                ).fetchall()
                payload = [(r["pk"], r["newval"], r["emb"]) for r in rows if r["emb"] is not None]
                cx.execute(f"DROP TABLE {table}")
                self._create_vec_tables(cx)
                if payload:
                    # The `vec` blob is already the stored dtype (int8 bytes
                    # when quantized) — re-type it with vec_int8(), never
                    # re-quantize. repo_vec/source_feedback_vec stay float32.
                    vec_bind = self._vec_bind_stored() if table == "vec" else "?"
                    cx.executemany(
                        f"INSERT INTO {table} ({pk_col}, {new_col}, {vec_col}) "  # noqa: S608
                        f"VALUES (?, ?, {vec_bind})",
                        payload,
                    )
            _log.info("migrated `%s`: %d vectors preserved", table, len(payload))

    def _validate_vec_dims(self) -> None:
        import os

        # Raw os.environ read is intentional: the store layer cannot import
        # memo.flags (circular dep risk — store is imported by Memory/Config
        # which the flags system depends on).  The env var is registered in
        # flags_misc.py for documentation/audit but gated through config.py
        # for production use; this bypass is only for dev/test toggling.
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
                from ..errors import StorageError

                raise StorageError(
                    f"{label} dimension mismatch: store has {actual_dims}D vectors "
                    f"but config expects {self.dims}D. "
                    f"This usually happens after switching between model profiles "
                    f"(e.g., from 'default' with 1024D to 'quality' with 2560D) "
                    f"without running 'memo reindex'.\n"
                    f"Fix: Run 'memo reindex --rebuild' to rebuild the index with the correct dimensions.\n"
                    f"Or check your model profile: MEMO_MODEL_PROFILE (current: {self.dims}D) "
                    f"or MEMO_EMBEDDER_DIMS (current: {self.dims})."
                )

    def _validate_vec_quant(self) -> None:
        """Adopt the on-disk `vec` element dtype when it differs from config.

        DDL-derived (never a self-written stamp): the physical `vec` column type
        is the source of truth. MEMO_VEC_QUANTIZE's default governs FRESH
        indexes only (a brand-new `vec` table is created at the configured
        dtype by `_create_vec_tables` before this guard runs, so a fresh DB
        never mismatches here). An EXISTING index built at a different
        precision than the current config — e.g. a float32 install opened
        after the default graduated to int8 — is never broken: this adopts
        the on-disk precision instead of raising, so the configured default
        only ever changes what a *new* index is built as. Honours
        MEMO_SKIP_MODEL_VERSION_CHECK so `memo reindex --rebuild` can
        intentionally flip an existing store's precision.
        """
        import os

        # See _validate_vec_dims for why the store layer reads env directly.
        if os.environ.get("MEMO_SKIP_MODEL_VERSION_CHECK", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            # Rebuild path: honour the configured self.vec_quant so
            # replace_memory_index's quant_changed branch can flip it.
            return
        stored = self._vec_table_dtype("vec")
        if stored != self.vec_quant:
            _log.info(
                "adopting on-disk vec precision %r (configured %r); "
                "run 'memo reindex --rebuild' to change it",
                stored,
                self.vec_quant,
            )
            self.vec_quant = stored
            self._quant_int8 = stored == "int8"

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
            with self._tx() as cx:
                cx.execute(
                    "CREATE TABLE IF NOT EXISTS schema_meta ("
                    "key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )

    def _check_embedder_version(self) -> None:
        """Stamp or verify embedder_model + embedder_dims in schema_meta.

        On first open: INSERT OR IGNORE writes the current model/dims.
        On subsequent opens: SELECT and compare. A mismatch raises StorageError
        with a clear instruction to run 'memo reindex --rebuild'.

        A legacy (pre-schema_meta) index that already contains vectors is
        never stamped with the current model — that would silently bless an
        unknown vector space. It warns and stays unstamped until
        'memo reindex --rebuild' re-embeds and stamps it.

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

        stamped_row = self._conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'embedder_model'"
        ).fetchone()
        if stamped_row is None:
            # A pre-schema_meta index that already contains vectors is an
            # UNKNOWN vector space: stamping the current model would silently
            # bless it and permanently disarm the mismatch guard (the vectors
            # may have been built by a different same-width model). Warn and
            # leave it unstamped — `memo reindex --rebuild` re-embeds from the
            # markdown source of truth and stamps (replace_memory_index).
            # Absent stamps keep the pre-stamp behaviour in the self-describing
            # readers (config._index_embedder_profile derives the model from
            # dims; runtime.mcp._actual_embedder_config falls back gracefully).
            has_vectors = self._conn.execute("SELECT 1 FROM vec LIMIT 1").fetchone() is not None
            if has_vectors:
                _log.warning(
                    "index predates embedder version stamping and already has "
                    "vectors; cannot verify they were built with %s (%dd). "
                    "Run 'memo reindex --rebuild' to re-embed and stamp the index.",
                    current_model,
                    current_dims,
                )
                return

        with self._tx() as cx:
            cx.execute(
                "INSERT OR IGNORE INTO schema_meta (key, value) VALUES (?, ?)",
                ("embedder_model", current_model),
            )
            cx.execute(
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
