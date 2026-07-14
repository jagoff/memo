from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, datetime
from typing import Any

from ..sqlite_compat import import_sqlite_vec
from .bm25_queries import _BM25QueriesMixin
from .rows import _row_to_dict
from .signal_queries import _SignalQueriesMixin

serialize_float32 = import_sqlite_vec().serialize_float32

_log = logging.getLogger(__name__)

META_SELECT_COLUMNS = (
    "id, path, title, type, tags, created, updated, body_hash, extra_json, "
    "verification_state, verified_at"
)


def _parse_filter_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


class _QueriesMixin(_BM25QueriesMixin, _SignalQueriesMixin):
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
        topic_key: str | None = None,
        normalized_hash: str | None = None,
    ) -> None:
        if len(embedding) != self.dims:
            raise ValueError(
                f"Embedding dimension mismatch: got {len(embedding)}, store expects {self.dims}. "
                f"The embedder produced a vector of the wrong length — usually a swapped "
                f"model/MEMO_EMBEDDER_DIMS (e.g. 1024 vs 2560). A wrong-dims vector can still be "
                f"unit-norm, so the norm check alone won't catch this. "
                f"Run 'memo reindex --rebuild' after restoring the correct model/dims.",
            )
        norm = sum(x * x for x in embedding) ** 0.5
        if norm != norm or norm == float("inf") or norm == float("-inf"):
            raise ValueError(
                f"Embedding contains NaN or Inf (norm={norm}). "
                f"The embedder produced corrupt output — re-embed or check MLX model."
            )
        if not (0.5 < norm < 1.5):
            raise ValueError(
                f"Embedding norm {norm:.4f} out of L2-normalised range (expected ≈ 1.0) "
                f"[id={id_[:8]}] — re-embed or check MLX model output."
            )
        if not (0.95 < norm < 1.05):
            _log.warning(
                "Embedding norm %.4f outside expected [0.95, 1.05] for id=%s"
                " — possible model mismatch or quantization issue; "
                "search quality may be degraded.",
                norm,
                id_[:8],
            )
        with self._tx() as cx:
            # Build query dynamically based on available columns. Pattern-column
            # presence is cached at schema init (`_has_pattern_cols`) so writes
            # skip a per-upsert PRAGMA table_info(meta).
            if self._has_pattern_cols:
                cx.execute(
                    "INSERT INTO meta (id, path, title, type, tags, created, updated, body_hash, extra_json, topic_key, normalized_hash) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(id) DO UPDATE SET "
                    "path=excluded.path, title=excluded.title, type=excluded.type, "
                    "tags=excluded.tags, updated=excluded.updated, body_hash=excluded.body_hash, "
                    "deleted_at=NULL, extra_json=excluded.extra_json, topic_key=excluded.topic_key, normalized_hash=excluded.normalized_hash",
                    (
                        id_,
                        path,
                        title,
                        type_,
                        json.dumps(tags),
                        created,
                        updated,
                        body_hash,
                        json.dumps(extra, default=str) if extra is not None else None,
                        topic_key,
                        normalized_hash,
                    ),
                )
            else:
                cx.execute(
                    "INSERT INTO meta (id, path, title, type, tags, created, updated, body_hash, extra_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(id) DO UPDATE SET "
                    "path=excluded.path, title=excluded.title, type=excluded.type, "
                    "tags=excluded.tags, updated=excluded.updated, body_hash=excluded.body_hash, "
                    "deleted_at=NULL, extra_json=excluded.extra_json",
                    (
                        id_,
                        path,
                        title,
                        type_,
                        json.dumps(tags),
                        created,
                        updated,
                        body_hash,
                        json.dumps(extra, default=str) if extra is not None else None,
                    ),
                )
            # `vec0` doesn't support `ON CONFLICT` syntax — we delete
            # then insert. Within the same transaction this is atomic.
            cx.execute("DELETE FROM vec WHERE id = ?", (id_,))
            cx.execute(
                "INSERT INTO vec (id, embedding, type) VALUES (?, ?, ?)",
                (id_, serialize_float32(embedding), type_),
            )
            # Same dance for the FTS index — no upsert support.
            cx.execute("DELETE FROM fts WHERE id = ?", (id_,))
            cx.execute(
                "INSERT INTO fts (id, title, tags, body) VALUES (?, ?, ?, ?)",
                (id_, title, " ".join(tags), body_text),
            )
            # Seed signal rows so prune/eviction queries don't need COALESCE.
            cx.execute(
                "INSERT OR IGNORE INTO access (id, access_count, last_accessed) VALUES (?, 0, ?)",
                (id_, updated),
            )
            cx.execute(
                "INSERT OR IGNORE INTO memory_health (id, confidence, roi_score, updated_at) "
                "VALUES (?, 1.0, 1.0, datetime('now'))",
                (id_,),
            )
        # Dual-write to tantivy (outside the sqlite tx — separate index).
        with self._tantivy_write_lock:
            tantivy = self._get_tantivy()
            if tantivy is not None:
                try:
                    tantivy.delete_document(id_)
                    tantivy.add_document(id_, title, " ".join(tags), body_text)
                    tantivy.commit()
                except Exception as exc:
                    _log.warning("tantivy upsert failed (FTS5 still current): %s", exc)
                    self._mark_tantivy_unhealthy()

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
        topic_key: str | None = None,
        normalized_hash: str | None = None,
    ) -> None:
        """Write metadata + FTS row without a vector embedding.

        This keeps CRUD and BM25 search usable on fresh installs or while
        models are downloading. A later `memo reindex` fills the missing vector.
        """
        with self._tx() as cx:
            if self._has_pattern_cols:
                cx.execute(
                    "INSERT INTO meta (id, path, title, type, tags, created, updated, body_hash, extra_json, topic_key, normalized_hash) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(id) DO UPDATE SET "
                    "path=excluded.path, title=excluded.title, type=excluded.type, "
                    "tags=excluded.tags, updated=excluded.updated, body_hash=excluded.body_hash, "
                    "deleted_at=NULL, extra_json=excluded.extra_json, topic_key=excluded.topic_key, normalized_hash=excluded.normalized_hash",
                    (
                        id_,
                        path,
                        title,
                        type_,
                        json.dumps(tags),
                        created,
                        updated,
                        body_hash,
                        json.dumps(extra, default=str) if extra is not None else None,
                        topic_key,
                        normalized_hash,
                    ),
                )
            else:
                cx.execute(
                    "INSERT INTO meta (id, path, title, type, tags, created, updated, body_hash, extra_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(id) DO UPDATE SET "
                    "path=excluded.path, title=excluded.title, type=excluded.type, "
                    "tags=excluded.tags, updated=excluded.updated, body_hash=excluded.body_hash, "
                    "deleted_at=NULL, extra_json=excluded.extra_json",
                    (
                        id_,
                        path,
                        title,
                        type_,
                        json.dumps(tags),
                        created,
                        updated,
                        body_hash,
                        json.dumps(extra, default=str) if extra is not None else None,
                    ),
                )
            cx.execute("DELETE FROM fts WHERE id = ?", (id_,))
            cx.execute(
                "INSERT INTO fts (id, title, tags, body) VALUES (?, ?, ?, ?)",
                (id_, title, " ".join(tags), body_text),
            )
            # Drop any stale vector — this method writes text-only, so a
            # pre-existing vec row for the same id would make semantic search
            # see the old body. Callers that later embed will re-insert via
            # upsert() with the correct vector.
            cx.execute("DELETE FROM vec WHERE id = ?", (id_,))
            # Seed signal rows so prune/eviction queries don't need COALESCE.
            cx.execute(
                "INSERT OR IGNORE INTO access (id, access_count, last_accessed) VALUES (?, 0, ?)",
                (id_, updated),
            )
            cx.execute(
                "INSERT OR IGNORE INTO memory_health (id, confidence, roi_score, updated_at) "
                "VALUES (?, 1.0, 1.0, datetime('now'))",
                (id_,),
            )
        # Dual-write to tantivy.
        with self._tantivy_write_lock:
            tantivy = self._get_tantivy()
            if tantivy is not None:
                try:
                    tantivy.delete_document(id_)
                    tantivy.add_document(id_, title, " ".join(tags), body_text)
                    tantivy.commit()
                except Exception as exc:
                    _log.warning("tantivy upsert_text_only failed: %s", exc)
                    self._mark_tantivy_unhealthy()

    def has_vector(self, id_: str) -> bool:
        row = self._conn.execute("SELECT 1 FROM vec WHERE id = ? LIMIT 1", (id_,)).fetchone()
        return row is not None

    def get_embedding_blob(self, id_: str) -> bytes | None:
        """Return the raw embedding blob for ``id_``, or None if missing."""
        row = self._conn.execute("SELECT embedding FROM vec WHERE id = ?", (id_,)).fetchone()
        return row["embedding"] if row else None

    def get_fts_body(self, id_: str) -> str:
        """Return the FTS body text for ``id_``, or empty string."""
        row = self._conn.execute("SELECT body FROM fts WHERE id = ?", (id_,)).fetchone()
        return str(row["body"]) if row else ""

    def get_dedup_keys(self, id_: str) -> tuple[str | None, str | None]:
        """Return ``(topic_key, normalized_hash)`` for ``id_`` — the dedup keys.

        These live ONLY in the sqlite ``meta`` index (not in the ``.md``
        frontmatter), so ``get()`` omits them. A delete-rollback that restores
        a row from ``get()`` would drop them, letting a later same-topic upsert
        create a duplicate instead of updating. Delete pre-fetches them here so
        the rollback restores faithfully. Older stores without the pattern
        columns return ``(None, None)``.
        """
        if not self._has_pattern_cols:
            return (None, None)
        row = self._conn.execute(
            "SELECT topic_key, normalized_hash FROM meta WHERE id = ?", (id_,)
        ).fetchone()
        if not row:
            return (None, None)
        return (row["topic_key"], row["normalized_hash"])

    def get_fts_bodies(self, ids: list[str]) -> dict[str, str]:
        """Batch-fetch FTS body text for many ids in one query.

        Used to feed the rerank candidate pool without one disk-read +
        frontmatter parse per hit. Returns {id: body} for ids present in fts.
        """
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"SELECT id, body FROM fts WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        return {r["id"]: str(r["body"]) for r in rows}

    def find_by_topic_key(self, topic_key: str) -> dict[str, str] | None:
        """Return the active row keyed by ``topic_key`` for save-path upserts."""
        if not topic_key:
            return None
        try:
            row = self._conn.execute(
                "SELECT id, path, created FROM meta "
                "WHERE topic_key = ? AND (deleted_at IS NULL OR deleted_at = '') "
                "LIMIT 1",
                (topic_key,),
            ).fetchone()
        except sqlite3.OperationalError as exc:
            # Older stores may not have pattern columns yet.
            if "no such column" not in str(exc):
                raise
            return None
        return dict(row) if row else None

    def get_fts_body_by_path(self, path: str) -> str:
        """Return indexed FTS body text for a memory path, or an empty string."""
        try:
            row = self._conn.execute(
                "SELECT fts.body AS body FROM fts JOIN meta ON meta.id = fts.id "
                "WHERE meta.path = ? AND (meta.deleted_at IS NULL OR meta.deleted_at = '') "
                "LIMIT 1",
                (path,),
            ).fetchone()
        except sqlite3.OperationalError as exc:
            if "no such column" not in str(exc):
                raise
            row = self._conn.execute(
                "SELECT fts.body AS body FROM fts JOIN meta ON meta.id = fts.id "
                "WHERE meta.path = ? LIMIT 1",
                (path,),
            ).fetchone()
        return str(row["body"]) if row and row["body"] else ""

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
                    title,
                    type_,
                    json.dumps(tags),
                    updated,
                    json.dumps(extra, default=str) if extra is not None else None,
                    id_,
                ),
            )
            # Sync FTS title + tags only when the meta row was actually updated.
            # Unconditional delete+insert would create a ghost FTS row for a
            # non-existent id_ (rowcount 0 means no meta row matched).
            body_text = ""
            if cur.rowcount > 0:
                existing = cx.execute(
                    "SELECT body FROM fts WHERE id = ?",
                    (id_,),
                ).fetchone()
                body_text = existing["body"] if existing else ""
                cx.execute("DELETE FROM fts WHERE id = ?", (id_,))
                cx.execute(
                    "INSERT INTO fts (id, title, tags, body) VALUES (?, ?, ?, ?)",
                    (id_, title, " ".join(tags), body_text),
                )
            # Sync vec.type — vec0 has no UPDATE, so delete + reinsert
            existing_emb = cx.execute("SELECT embedding FROM vec WHERE id = ?", (id_,)).fetchone()
            if existing_emb is not None:
                cx.execute("DELETE FROM vec WHERE id = ?", (id_,))
                cx.execute(
                    "INSERT INTO vec (id, embedding, type) VALUES (?, ?, ?)",
                    (id_, existing_emb["embedding"], type_),
                )
            was_updated = cur.rowcount > 0
        # Dual-write to tantivy (preserve existing body, just update title/tags).
        if was_updated:
            with self._tantivy_write_lock:
                tantivy = self._get_tantivy()
                if tantivy is not None:
                    try:
                        tantivy.delete_document(id_)
                        tantivy.add_document(id_, title, " ".join(tags), body_text)
                        tantivy.commit()
                    except Exception as exc:
                        _log.warning("tantivy update_meta failed: %s", exc)
                        self._mark_tantivy_unhealthy()
        return was_updated

    def get(self, id_: str) -> dict[str, Any] | None:
        # Check for soft delete column
        try:
            row = self._conn.execute(
                f"SELECT {META_SELECT_COLUMNS} FROM meta WHERE id = ? AND deleted_at IS NULL",
                (id_,),
            ).fetchone()
        except sqlite3.OperationalError as e:
            if "no such column" not in str(e):
                raise
            # Fallback for old DBs without deleted_at column
            row = self._conn.execute(
                f"SELECT {META_SELECT_COLUMNS} FROM meta WHERE id = ?",
                (id_,),
            ).fetchone()
        return _row_to_dict(row) if row else None

    def get_batch(self, ids: list[str]) -> list[dict[str, Any]]:
        """Fetch multiple memories by ID in a single query."""
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        # Check for soft delete column
        try:
            rows = self._conn.execute(
                f"SELECT {META_SELECT_COLUMNS} "
                f"FROM meta WHERE id IN ({placeholders}) AND deleted_at IS NULL",
                ids,
            ).fetchall()
        except sqlite3.OperationalError as e:
            if "no such column" not in str(e):
                raise
            # Fallback for old DBs
            rows = self._conn.execute(
                f"SELECT {META_SELECT_COLUMNS} FROM meta WHERE id IN ({placeholders})",
                ids,
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def find_by_prefix(self, prefix: str, limit: int = 10) -> list[str]:
        """Return ids whose hex starts with `prefix`. Used by the CLI/MCP
        to let callers reference memories by the first ~7 chars (git-style)
        instead of pasting a 32-char UUID4."""
        if not prefix:
            return []
        try:
            rows = self._conn.execute(
                "SELECT id FROM meta WHERE id LIKE ? || '%' AND deleted_at IS NULL "
                "ORDER BY id LIMIT ?",
                (prefix, limit),
            ).fetchall()
        except sqlite3.OperationalError as e:
            if "no such column" not in str(e):
                raise
            # Fallback for old DBs without deleted_at column
            rows = self._conn.execute(
                "SELECT id FROM meta WHERE id LIKE ? || '%' ORDER BY id LIMIT ?",
                (prefix, limit),
            ).fetchall()
        return [r["id"] for r in rows]

    def get_by_path(self, path: str, *, include_deleted: bool = False) -> dict[str, Any] | None:
        """Row for `path`, or None. `include_deleted=True` also matches
        soft-deleted rows — needed by callers reclaiming a path, because a
        tombstone still occupies the UNIQUE(path) index and blocks INSERTs.
        """
        deleted_filter = "" if include_deleted else " AND (deleted_at IS NULL OR deleted_at = '')"
        try:
            row = self._conn.execute(
                f"SELECT {META_SELECT_COLUMNS} FROM meta WHERE path = ?{deleted_filter}",
                (path,),
            ).fetchone()
        except sqlite3.OperationalError as exc:
            if "no such column" not in str(exc):
                raise
            # Fallback for old DBs without deleted_at column
            row = self._conn.execute(
                f"SELECT {META_SELECT_COLUMNS} FROM meta WHERE path = ?",
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
        try:
            row = self._conn.execute(
                f"SELECT {META_SELECT_COLUMNS} "
                "FROM meta WHERE path = ? COLLATE NOCASE AND (deleted_at IS NULL OR deleted_at = '') ORDER BY created LIMIT 1",
                (path,),
            ).fetchone()
        except sqlite3.OperationalError as exc:
            if "no such column" not in str(exc):
                raise
            # Fallback for old DBs without deleted_at column
            row = self._conn.execute(
                f"SELECT {META_SELECT_COLUMNS} "
                "FROM meta WHERE path = ? COLLATE NOCASE ORDER BY created LIMIT 1",
                (path,),
            ).fetchone()
        return _row_to_dict(row) if row else None

    def vault_ingest_rows(self, label: str) -> list[dict[str, Any]]:
        """Rows produced by `memo ingest` under a given vault `label`.

        Used by `ingest --prune` to find stale chunks (abs_path gone). Filters
        on `source LIKE 'vault-ingest%'` so curated memories (source NULL) are
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

    def chunks_by_parent(self, parent_path: str, limit: int = 3) -> list[dict[str, Any]]:
        """Newest chunks of one ingested note, by title (descending).

        Dated transcript chunk titles end in `— YYYY-MM-DD`, which sorts
        lexicographically by date, so `ORDER BY title DESC` surfaces the most
        recent day first. Used by recency asks to guarantee the latest chunk of
        a retrieved transcript enters the candidate pool even when it scores too
        low semantically to be retrieved on its own.
        """
        rows = self._conn.execute(
            f"SELECT {META_SELECT_COLUMNS} "
            "FROM meta "
            "WHERE json_extract(extra_json, '$.parent_path') = ? "
            "ORDER BY title DESC LIMIT ?",
            (parent_path, limit),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def chunks_by_parent_id(self, parent_id: str) -> list[dict[str, Any]]:
        """All chunks whose extra_json contains the given parent_id.
        Used by reindex to prune stale chunks without scanning every row."""
        rows = self._conn.execute(
            f"SELECT {META_SELECT_COLUMNS} "
            "FROM meta "
            "WHERE json_extract(extra_json, '$.parent_id') = ?",
            (parent_id,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def chunks_adjacent(
        self, parent_path: str, seq: int, *, before: int = 2, after: int = 2
    ) -> list[dict[str, Any]]:
        """Seq-window of sibling chunks of one ingested note: chunk_seq in
        [seq-before, seq+after], anchor row included, ordered by chunk_seq ASC.
        Cheap SQL over extra_json — no embedder, no MLX."""
        rows = self._conn.execute(
            f"SELECT {META_SELECT_COLUMNS} "
            "FROM meta "
            "WHERE json_extract(extra_json, '$.parent_path') = ? "
            "AND CAST(json_extract(extra_json, '$.chunk_seq') AS INTEGER) BETWEEN ? AND ? "
            "ORDER BY CAST(json_extract(extra_json, '$.chunk_seq') AS INTEGER) ASC",
            (parent_path, seq - before, seq + after),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def records_around_created(
        self,
        created: str,
        *,
        before: int = 2,
        after: int = 2,
        exclude_types: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Chronological neighbourhood by `created`: up to `before` strictly
        earlier rows + `after` strictly later rows (ISO strings compare
        chronologically). Returned oldest -> newest, anchor excluded."""
        ex_sql = ""
        ex_params: list[Any] = []
        if exclude_types:
            _ph = ",".join("?" for _ in exclude_types)
            ex_sql = f" AND type NOT IN ({_ph})"
            ex_params = sorted(exclude_types)
        cols = META_SELECT_COLUMNS
        older = self._conn.execute(
            f"SELECT {cols} FROM meta "
            f"WHERE coalesce(julianday(created), -1e300) < julianday(?)"
            f"{ex_sql} ORDER BY julianday(created) DESC LIMIT ?",
            (created, *ex_params, before),
        ).fetchall()
        newer = self._conn.execute(
            f"SELECT {cols} FROM meta "
            f"WHERE coalesce(julianday(created), -1e300) > julianday(?)"
            f"{ex_sql} ORDER BY julianday(created) ASC LIMIT ?",
            (created, *ex_params, after),
        ).fetchall()
        return [_row_to_dict(r) for r in [*reversed(older), *newer]]

    def list_by_tag(self, tag: str, limit: int = 500) -> list[dict[str, Any]]:
        """Rows whose JSON tags array contains `tag` exactly (tags are stored as
        json.dumps(list), so the quoted token is an exact-tag match)."""
        rows = self._conn.execute(
            f"SELECT {META_SELECT_COLUMNS} "
            "FROM meta WHERE tags LIKE ? ORDER BY julianday(updated) DESC LIMIT ?",
            (f'%"{tag}"%', limit),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def list_recent(
        self,
        limit: int = 20,
        type_: str | None = None,
        exclude_types: set[str] | None = None,
        updated_since: str | None = None,
    ) -> list[dict[str, Any]]:
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(meta)").fetchall()}
        has_deleted = "deleted_at" in cols
        sql = f"SELECT {META_SELECT_COLUMNS} FROM meta "
        clauses: list[str] = []
        params: list[Any] = []
        if has_deleted:
            clauses.append("deleted_at IS NULL")
        if type_:
            clauses.append("type = ?")
            params.append(type_)
        if exclude_types:
            placeholders = ",".join("?" for _ in exclude_types)
            clauses.append(f"type NOT IN ({placeholders})")
            params.extend(sorted(exclude_types))
        if updated_since:
            # Compare UTC instants so mixed offsets don't drop rows. Wrapping
            # updated in julianday() forces a full table scan since idx_meta_updated
            # is defined on bare updated, not julianday(updated). For large
            # corpora, consider adding: CREATE INDEX idx_meta_updated_jd ON meta(julianday(updated))
            clauses.append("coalesce(julianday(updated), -1e300) >= julianday(?)")
            params.append(updated_since)
        if clauses:
            sql += "WHERE " + " AND ".join(clauses) + " "
        sql += "ORDER BY julianday(updated) DESC LIMIT ?"
        rows = self._conn.execute(sql, (*params, limit)).fetchall()
        return [_row_to_dict(r) for r in rows]

    def search(
        self,
        embedding: list[float],
        limit: int = 10,
        type_: str | None = None,
        exclude_types: set[str] | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        exclude_tags: set[str] | None = None,
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
        # `type` is a vec0 METADATA column, so type filters are pushed INTO
        # the kNN — the candidate pool is already filtered before the JOINed
        # meta row is read. sqlite-vec supports chained `AND type != ?`
        # predicates, so multi-value exclude_types becomes N individual push
        # predicates rather than a post-join `NOT IN` that required 5x
        # over-fetch to fill the top-k after off-type rows were discarded.
        push_clauses: list[str] = []
        push_params: list[Any] = []
        if type_:
            push_clauses.append("vec.type = ?")
            push_params.append(type_)
        if exclude_types:
            for ex_type in sorted(exclude_types):
                push_clauses.append("vec.type != ?")
                push_params.append(ex_type)
        # Date windows compare instants, not ISO text: stored/imported rows may
        # carry non-UTC offsets. Fetch a wider kNN pool and filter in Python.
        from_dt = _parse_filter_ts(date_from)
        to_dt = _parse_filter_ts(date_to)
        has_date_filter = from_dt is not None or to_dt is not None
        tag_clauses: list[str] = []
        tag_params: list[Any] = []
        for tag in sorted(exclude_tags or ()):
            # tags column stores json.dumps(list[str]); the quoted token is an
            # exact-tag match (mirrors reference-tier exclusion, but tags can't
            # be pushed into vec0 so this filters on the joined meta row).
            tag_clauses.append("meta.tags NOT LIKE ?")
            tag_params.append(f'%"{tag}"%')
        k_fetch = limit * 4 if (has_date_filter or tag_clauses) else limit

        # Check for deleted_at column and build filter
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(meta)").fetchall()}
        has_deleted = "deleted_at" in cols

        deleted_filter = "meta.deleted_at IS NULL" if has_deleted else "1=1"

        sql = (
            "SELECT vec.id AS id, vec.distance AS distance, "
            "       meta.path, meta.title, meta.type, meta.tags, "
            "       meta.created, meta.updated, meta.body_hash, meta.extra_json, "
            "       meta.verification_state, meta.verified_at "
            f"FROM vec JOIN meta ON vec.id = meta.id AND {deleted_filter} "
            "WHERE vec.embedding MATCH ? AND vec.k = ? "
        )
        params: list[Any] = [serialize_float32(embedding), k_fetch]
        for clause in push_clauses:
            sql += f"AND {clause} "
        params.extend(push_params)
        for clause in tag_clauses:
            sql += f"AND {clause} "
        params.extend(tag_params)
        sql += "ORDER BY distance ASC LIMIT ?"
        params.append(k_fetch if has_date_filter else limit)
        rows = self._conn.execute(sql, params).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            d = _row_to_dict(r)
            if has_date_filter:
                updated_dt = _parse_filter_ts(str(d.get("updated") or ""))
                if updated_dt is None:
                    continue
                if from_dt is not None and updated_dt < from_dt:
                    continue
                if to_dt is not None and updated_dt > to_dt:
                    continue
            d["score"] = max(0.0, 1.0 - float(r["distance"]))
            out.append(d)
            if len(out) >= limit:
                break
        return out

    def clear_memory_index(self) -> int:
        """Truncate the markdown-DERIVABLE memory tables (`meta`, `vec`, `fts`)
        so they can be fully replayed from the `.md` source of truth.

        Deliberately does NOT touch the user-signal tables — `access`,
        `memory_health`, `source_feedback`/`source_feedback_vec` — because those
        are PRIMARY data not present in markdown. They key on the stable
        memory `id`, so they re-join after the replay; rows whose memory no
        longer exists on disk become harmless orphans (cleanable by `gc`).
        Repo corpus tables are untouched (separate ingest surface).

        Returns the number of `meta` rows cleared. Use only for a full
        `reindex(rebuild=True)` — never per-row (that path is `delete()`).
        """
        with self._tx() as cx:
            n = cx.execute("SELECT COUNT(*) FROM meta").fetchone()[0]
            cx.execute("DELETE FROM meta")
            cx.execute("DELETE FROM vec")
            cx.execute("DELETE FROM fts")
        with self._tantivy_write_lock:
            tantivy = self._get_tantivy()
            if tantivy is not None:
                try:
                    # rebuild([]) clears the index while holding tantivy._lock
                    # for the whole delete+commit (direct _writer access didn't).
                    tantivy.rebuild([])
                except Exception as exc:
                    _log.warning("tantivy clear failed during rebuild: %s", exc)
                    self._mark_tantivy_unhealthy()
            elif self.tantivy_index_dir.is_dir():
                # Tantivy unavailable but stale index directory exists — nuke it
                # so _maybe_rebuild_tantivy will do a full rebuild when tantivy
                # becomes available again, rather than serving ghost documents.
                import shutil

                try:
                    shutil.rmtree(self.tantivy_index_dir)
                except Exception as exc:
                    _log.warning("failed to remove stale tantivy index: %s", exc)
        return int(n)

    def delete(self, id_: str) -> bool:
        # Check if soft delete column exists (session pattern)
        from ..flags import flag_bool

        _use_soft = flag_bool("MEMO_SOFT_DELETE")
        has_soft_delete = False
        if _use_soft:
            try:
                col_check = self._conn.execute("PRAGMA table_info(meta)").fetchall()
                columns = {row["name"] for row in col_check}
                has_soft_delete = "deleted_at" in columns
            except Exception:
                _log.debug("soft-delete column check failed")

        if has_soft_delete and _use_soft:
            # Soft delete pattern: mark deleted_at + remove from vec/fts indexes
            import datetime

            now = datetime.datetime.now(datetime.UTC).isoformat()
            with self._tx() as cx:
                cur = cx.execute(
                    "UPDATE meta SET deleted_at = ? WHERE id = ?",
                    (now, id_),
                )
                existed = cur.rowcount > 0
                cx.execute("DELETE FROM vec WHERE id = ?", (id_,))
                cx.execute("DELETE FROM fts WHERE id = ?", (id_,))
            if existed:
                with self._tantivy_write_lock:
                    tantivy = self._get_tantivy()
                    if tantivy is not None:
                        try:
                            tantivy.delete_document(id_)
                            tantivy.commit()
                        except Exception as exc:
                            _log.warning("tantivy soft-delete failed: %s", exc)
                            self._mark_tantivy_unhealthy()
            return existed

        # Hard delete fallback (for old DBs without deleted_at)
        with self._tx() as cx:
            cur = cx.execute("DELETE FROM meta WHERE id = ?", (id_,))
            existed = cur.rowcount > 0
            cx.execute("DELETE FROM vec WHERE id = ?", (id_,))
            cx.execute("DELETE FROM fts WHERE id = ?", (id_,))
            cx.execute("DELETE FROM access WHERE id = ?", (id_,))
            cx.execute("DELETE FROM memory_health WHERE id = ?", (id_,))
            cx.execute("DELETE FROM source_feedback_vec WHERE source_id = ?", (id_,))
            cx.execute("DELETE FROM source_feedback WHERE source_id = ?", (id_,))
        if existed:
            with self._tantivy_write_lock:
                tantivy = self._get_tantivy()
                if tantivy is not None:
                    try:
                        tantivy.delete_document(id_)
                        tantivy.commit()
                    except Exception as exc:
                        _log.warning("tantivy delete failed: %s", exc)
                        self._mark_tantivy_unhealthy()
        return existed

    def hard_delete(self, id_: str) -> bool:
        """Permanently delete a memory bypassing soft-delete (vacuum path)."""
        with self._tx() as cx:
            cur = cx.execute("DELETE FROM meta WHERE id = ?", (id_,))
            existed = cur.rowcount > 0
            cx.execute("DELETE FROM vec WHERE id = ?", (id_,))
            cx.execute("DELETE FROM fts WHERE id = ?", (id_,))
            cx.execute("DELETE FROM access WHERE id = ?", (id_,))
            cx.execute("DELETE FROM memory_health WHERE id = ?", (id_,))
            cx.execute("DELETE FROM source_feedback_vec WHERE source_id = ?", (id_,))
            cx.execute("DELETE FROM source_feedback WHERE source_id = ?", (id_,))
        if existed:
            with self._tantivy_write_lock:
                tantivy = self._get_tantivy()
                if tantivy is not None:
                    try:
                        tantivy.delete_document(id_)
                        tantivy.commit()
                    except Exception as exc:
                        _log.warning("tantivy hard_delete failed: %s", exc)
                        self._mark_tantivy_unhealthy()
        return existed

    def list_soft_deleted(self, before: str | None = None) -> list[str]:
        """List IDs of soft-deleted records, optionally filtered by age."""
        query = "SELECT id FROM meta WHERE deleted_at IS NOT NULL"
        params: list[Any] = []
        if before is not None:
            query += " AND coalesce(julianday(deleted_at), -1e300) < julianday(?)"
            params.append(before)
        rows = self._conn.execute(query, params).fetchall()
        return [r["id"] for r in rows]

    def bulk_update_type(self, ids: list[str], new_type: str) -> int:
        """Reclassify the `type` of many memories in one transaction.

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
            # Batch delete + insert for vec.type: read embeddings, then
            # issue two executemany calls instead of 2×N individual stmts.
            placeholders = ",".join("?" for _ in ids)
            vec_rows = {
                r["id"]: r["embedding"]
                for r in cx.execute(
                    f"SELECT id, embedding FROM vec WHERE id IN ({placeholders})", ids
                ).fetchall()
            }
            to_update = [(id_,) for id_ in ids if id_ in vec_rows]
            if to_update:
                cx.executemany("DELETE FROM vec WHERE id = ?", to_update)
                cx.executemany(
                    "INSERT INTO vec (id, embedding, type) VALUES (?, ?, ?)",
                    [(id_, vec_rows[id_], new_type) for (id_,) in to_update],
                )
        return len(ids)

    def secret_store_insert(
        self,
        *,
        id: str,
        name: str,
        kind: str,
        encrypted_blob: bytes,
        nonce: bytes,
        created_at: str,
        detection_method: str | None = None,
        confidence: float | None = None,
    ) -> None:
        """Insert a secret into secret_store."""
        with self._tx() as cursor:
            cursor.execute(
                """
                INSERT INTO secret_store
                (id, name, kind, encrypted_blob, nonce, created_at, detection_method, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (id, name, kind, encrypted_blob, nonce, created_at, detection_method, confidence),
            )

    def secret_store_get(self, name: str) -> dict[str, Any] | None:
        """Fetch a secret by name."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT id, kind, encrypted_blob, nonce, accessed_count FROM secret_store WHERE name = ?",
            (name,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "name": name,
            "kind": row[1],
            "encrypted_blob": row[2],
            "nonce": row[3],
            "accessed_count": row[4],
        }

    def secret_store_list(self, kind: str | None = None) -> list[dict[str, Any]]:
        """List secrets (metadata only)."""
        cursor = self._conn.cursor()
        if kind:
            cursor.execute(
                "SELECT id, name, kind, accessed_count FROM secret_store WHERE kind = ? ORDER BY name",
                (kind,),
            )
        else:
            cursor.execute("SELECT id, name, kind, accessed_count FROM secret_store ORDER BY name")
        rows = cursor.fetchall()
        return [
            {
                "id": row[0],
                "name": row[1],
                "kind": row[2],
                "accessed_count": row[3],
            }
            for row in rows
        ]

    def secret_store_delete(self, name: str) -> bool:
        """Delete a secret by name. Returns True if deleted, False if not found."""
        with self._tx() as cx:
            cursor = cx.execute("DELETE FROM secret_store WHERE name = ?", (name,))
            return cursor.rowcount > 0

    def secret_store_increment_access(self, name: str) -> None:
        """Increment access count and update accessed_at timestamp."""
        now = datetime.now(UTC).isoformat(timespec="seconds")
        with self._tx() as cx:
            cx.execute(
                """
                UPDATE secret_store
                SET accessed_count = accessed_count + 1, accessed_at = ?
                WHERE name = ?
                """,
                (now, name),
            )
