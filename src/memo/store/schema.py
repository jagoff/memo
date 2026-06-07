from __future__ import annotations

import re
import threading

from ._base import _StoreBase

# Serialises first-touch schema creation across threads. VecStore uses
# thread-local connections, so two FastMCP worker threads can hit a fresh DB
# at once and race the (non-transactional) vec0 CREATE statements.
_SCHEMA_INIT_LOCK = threading.Lock()

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


class _SchemaMixin(_StoreBase):
    def _init_schema(self) -> None:
        # Most CLI commands are reads. If the schema already exists, avoid
        # no-op DDL because it still needs a write/schema lock and can fail
        # while long repo indexing is writing batches. The lock makes the
        # check-then-create atomic across threads (thread-local connections).
        with _SCHEMA_INIT_LOCK:
            self._init_schema_locked()

    def _init_schema_locked(self) -> None:
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

    def _validate_vec_dims(self) -> None:
        for table, label in (
            ("vec", "Embedding"),
            ("repo_vec", "Repo embedding"),
            ("source_feedback_vec", "Feedback embedding"),
        ):
            actual_dims = self._vec_table_dims(table)
            if actual_dims is not None and actual_dims != self.dims:
                raise RuntimeError(
                    f"{label} dimension mismatch: store has {actual_dims}D vectors "
                    f"but config expects {self.dims}D.\n"
                    f"Fix: rm {self.db_path} && memo reindex\n"
                    f"Or check: MEMO_MODEL_PROFILE={self.dims}D or MEMO_EMBEDDER_DIMS={self.dims}"
                )
