from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import UTC, datetime
from typing import Any

from sqlite_vec import serialize_float32

from ._base import _StoreBase
from .rows import _row_to_dict
from .schema import (
    _BM25_ES_STOPWORDS,
    _BM25_FTS_BODY_WEIGHT,
    _BM25_FTS_TAGS_WEIGHT,
    _BM25_FTS_TITLE_WEIGHT,
    _BM25_UNINDEXED_WEIGHT,
)

_log = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    """Parse a float env var, falling back to `default` when unset/blank/bad.

    The store layer is a foundation module and cannot import memo.flags, so
    these tuning knobs (registered there for `memo config validate`) are read
    directly from the environment here.
    """
    raw = os.environ.get(name)
    if not raw or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class _QueriesMixin(_StoreBase):
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
                    id_,
                    path,
                    title,
                    type_,
                    json.dumps(tags),
                    created,
                    updated,
                    body_hash,
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
                "INSERT INTO vec (id, embedding, type) VALUES (?, ?, ?)",
                (id_, serialize_float32(embedding), type_),
            )
            # Same dance for the FTS index — no upsert support.
            cx.execute("DELETE FROM fts WHERE id = ?", (id_,))
            cx.execute(
                "INSERT INTO fts (id, title, tags, body) VALUES (?, ?, ?, ?)",
                (id_, title, " ".join(tags), body_text),
            )
        # Dual-write to tantivy (outside the sqlite tx — separate index).
        tantivy = self._get_tantivy()
        if tantivy is not None:
            try:
                tantivy.delete_document(id_)
                tantivy.add_document(id_, title, " ".join(tags), body_text)
                tantivy.commit()
            except Exception as exc:
                _log.warning("tantivy upsert failed (FTS5 still current): %s", exc)

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
        # Dual-write to tantivy.
        tantivy = self._get_tantivy()
        if tantivy is not None:
            try:
                tantivy.delete_document(id_)
                tantivy.add_document(id_, title, " ".join(tags), body_text)
                tantivy.commit()
            except Exception as exc:
                _log.warning("tantivy upsert_text_only failed: %s", exc)

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
                    title,
                    type_,
                    json.dumps(tags),
                    updated,
                    json.dumps(extra, default=str) if extra is not None else None,
                    id_,
                ),
            )
            # Sync FTS title + tags (body unchanged on metadata-only updates).
            # FTS5 doesn't support partial UPDATE cleanly; we delete + reinsert
            # the row preserving the existing body text.
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
            was_updated = cur.rowcount > 0
        # Dual-write to tantivy (preserve existing body, just update title/tags).
        if was_updated:
            tantivy = self._get_tantivy()
            if tantivy is not None:
                try:
                    tantivy.delete_document(id_)
                    tantivy.add_document(id_, title, " ".join(tags), body_text)
                    tantivy.commit()
                except Exception as exc:
                    _log.warning("tantivy update_meta failed: %s", exc)
        return was_updated

    def get(self, id_: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT id, path, title, type, tags, created, updated, body_hash, extra_json "
            "FROM meta WHERE id = ?",
            (id_,),
        ).fetchone()
        return _row_to_dict(row) if row else None

    def get_batch(self, ids: list[str]) -> list[dict[str, Any]]:
        """Fetch multiple memorias by ID in a single query."""
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            "SELECT id, path, title, type, tags, created, updated, body_hash, extra_json "
            f"FROM meta WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

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

    def chunks_by_parent(self, parent_path: str, limit: int = 3) -> list[dict[str, Any]]:
        """Newest chunks of one ingested note, by title (descending).

        Dated transcript chunk titles end in `— YYYY-MM-DD`, which sorts
        lexicographically by date, so `ORDER BY title DESC` surfaces the most
        recent day first. Used by recency asks to guarantee the latest chunk of
        a retrieved transcript enters the candidate pool even when it scores too
        low semantically to be retrieved on its own.
        """
        rows = self._conn.execute(
            "SELECT id, path, title, type, tags, created, updated, body_hash, extra_json "
            "FROM meta "
            "WHERE json_extract(extra_json, '$.parent_path') = ? "
            "ORDER BY title DESC LIMIT ?",
            (parent_path, limit),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def list_recent(
        self,
        limit: int = 20,
        type_: str | None = None,
        exclude_types: set[str] | None = None,
        updated_since: str | None = None,
    ) -> list[dict[str, Any]]:
        sql = (
            "SELECT id, path, title, type, tags, created, updated, body_hash, extra_json FROM meta "
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
        if updated_since:
            # ISO-8601 timestamps sort lexicographically, so a string >=
            # compares chronologically. Indexed by idx_meta_updated /
            # idx_meta_type_updated so incremental scans stay cheap.
            clauses.append("updated >= ?")
            params.append(updated_since)
        if clauses:
            sql += "WHERE " + " AND ".join(clauses) + " "
        sql += "ORDER BY updated DESC LIMIT ?"
        rows = self._conn.execute(sql, (*params, limit)).fetchall()
        return [_row_to_dict(r) for r in rows]

    def search(
        self,
        embedding: list[float],
        limit: int = 10,
        type_: str | None = None,
        exclude_types: set[str] | None = None,
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
        # `type` is a vec0 metadata column, so an include filter (`type = ?`)
        # and a single-value exclude (`type != ?` — the recall hook dropping
        # the bulk `reference` tier) are pushed INTO the kNN: the candidate
        # pool is already on-type, so no over-fetch is needed. Only a
        # multi-value `exclude_types` (rare) can't be a single vec0 predicate,
        # so it stays a post-join `NOT IN` and keeps the 5x over-fetch so the
        # final top-k still fills after off-type rows drop out.
        push_clauses: list[str] = []
        push_params: list[Any] = []
        join_clauses: list[str] = []
        join_params: list[Any] = []
        if type_:
            push_clauses.append("vec.type = ?")
            push_params.append(type_)
        if exclude_types:
            ex = sorted(exclude_types)
            if len(ex) == 1:
                push_clauses.append("vec.type != ?")
                push_params.append(ex[0])
            else:
                join_clauses.append(f"meta.type NOT IN ({','.join('?' for _ in ex)})")
                join_params.extend(ex)
        candidate_k = limit * 5 if join_clauses else limit
        sql = (
            "SELECT vec.id AS id, vec.distance AS distance, "
            "       meta.path, meta.title, meta.type, meta.tags, "
            "       meta.created, meta.updated, meta.body_hash, meta.extra_json "
            "FROM vec "
            "JOIN meta ON meta.id = vec.id "
            "WHERE embedding MATCH ? AND k = ? "
        )
        params: list[Any] = [serialize_float32(embedding), candidate_k]
        for clause in push_clauses:
            sql += f"AND {clause} "
        params.extend(push_params)
        for clause in join_clauses:
            sql += f"AND {clause} "
        params.extend(join_params)
        sql += "ORDER BY distance ASC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            d = _row_to_dict(r)
            d["score"] = 1.0 - float(r["distance"])
            out.append(d)
        return out

    def clear_memoria_index(self) -> int:
        """Truncate the markdown-DERIVABLE memoria tables (`meta`, `vec`, `fts`)
        so they can be fully replayed from the `.md` source of truth.

        Deliberately does NOT touch the user-signal tables — `access`,
        `memory_health`, `source_feedback`/`source_feedback_vec` — because those
        are PRIMARY data not present in markdown. They key on the stable
        memoria `id`, so they re-join after the replay; rows whose memoria no
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
        tantivy = self._get_tantivy()
        if tantivy is not None:
            try:
                tantivy._writer.delete_all_documents()
                tantivy.commit()
            except Exception as exc:
                _log.warning("tantivy clear failed during rebuild: %s", exc)
        return int(n)

    def delete(self, id_: str) -> bool:
        with self._tx() as cx:
            cur = cx.execute("DELETE FROM meta WHERE id = ?", (id_,))
            existed = cur.rowcount > 0
            cx.execute("DELETE FROM vec WHERE id = ?", (id_,))
            cx.execute("DELETE FROM fts WHERE id = ?", (id_,))
            cx.execute("DELETE FROM access WHERE id = ?", (id_,))
        if existed:
            tantivy = self._get_tantivy()
            if tantivy is not None:
                try:
                    tantivy.delete_document(id_)
                    tantivy.commit()
                except Exception as exc:
                    _log.warning("tantivy delete failed: %s", exc)
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
            "SELECT access_count, last_accessed FROM access WHERE id = ?",
            (id_,),
        ).fetchone()
        if not row:
            return {"access_count": 0, "last_accessed": None}
        return {"access_count": int(row["access_count"]), "last_accessed": row["last_accessed"]}

    # -- memory health (confidence + roi_score) ----------------------------

    def get_health_batch(self, ids: list[str]) -> dict[str, dict[str, float]]:
        """Return {id: {confidence, roi_score}} for the given IDs.

        IDs not in the table are absent from the result (callers treat missing
        as defaults: confidence=1.0, roi_score=1.0).
        """
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"SELECT id, confidence, roi_score FROM memory_health WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        return {
            r["id"]: {"confidence": float(r["confidence"]), "roi_score": float(r["roi_score"])}
            for r in rows
        }

    def boost_roi_batch(
        self,
        ids: list[str],
        delta: float = 0.05,
        cap: float = 1.5,
    ) -> None:
        """Increment roi_score for each id, capped at `cap`. Upserts new rows."""
        if not ids:
            return
        with self._tx() as cx:
            cx.executemany(
                "INSERT INTO memory_health(id, confidence, roi_score, updated_at) "
                "VALUES(?, 1.0, min(?, 1.0 + ?), datetime('now')) "
                "ON CONFLICT(id) DO UPDATE SET "
                "roi_score = min(?, roi_score + ?), "
                "updated_at = datetime('now')",
                [(i, cap, delta, cap, delta) for i in ids],
            )

    def penalize_confidence_batch(
        self,
        ids: list[str],
        delta: float = 0.15,
        floor: float = 0.1,
    ) -> None:
        """Decrement confidence for each id (e.g. open contradiction). Floor at `floor`."""
        if not ids:
            return
        with self._tx() as cx:
            cx.executemany(
                "INSERT INTO memory_health(id, confidence, roi_score, updated_at) "
                "VALUES(?, max(?, 1.0 - ?), 1.0, datetime('now')) "
                "ON CONFLICT(id) DO UPDATE SET "
                "confidence = max(excluded.confidence, confidence - ?), "
                "updated_at = datetime('now')",
                [(i, floor, delta, delta) for i in ids],
            )

    def set_confidence_batch(
        self,
        pairs: list[tuple[str, float]],
        floor: float = 0.1,
    ) -> None:
        """Set an absolute confidence for each (id, confidence) pair, floored at
        ``floor``. Unlike :meth:`penalize_confidence_batch` (relative decrement),
        this writes the value directly — used to stamp OCR'd-image records with
        their measured quality so low-confidence screenshots rank below clean
        notes (search score x confidence). roi_score left neutral (1.0)."""
        if not pairs:
            return
        with self._tx() as cx:
            cx.executemany(
                "INSERT INTO memory_health(id, confidence, roi_score, updated_at) "
                "VALUES(?, max(?, ?), 1.0, datetime('now')) "
                "ON CONFLICT(id) DO UPDATE SET "
                "confidence = min(confidence, ?), "
                "updated_at = datetime('now')",
                [(i, floor, c, c) for i, c in pairs],
            )

    def decay_roi(
        self,
        factor: float = 0.98,
        older_than_days: int = 30,
    ) -> int:
        """Multiply roi_score by `factor` for memorias not accessed in `older_than_days`.

        Returns the count of rows updated. Used by Dream mode nightly pipeline.
        """
        with self._tx() as cx:
            cur = cx.execute(
                "UPDATE memory_health SET roi_score = max(0.1, roi_score * ?), "
                "updated_at = datetime('now') "
                "WHERE updated_at < datetime('now', ? || ' days') "
                "OR updated_at IS NULL",
                (factor, f"-{older_than_days}"),
            )
            return cur.rowcount

    def eviction_candidates(
        self,
        policy: str,
        limit: int,
        *,
        exclude_types: set[str] | None = None,
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
        if policy not in {"lru", "lfu", "ttl"}:
            raise ValueError(f"unknown eviction policy: {policy!r}")
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
        self,
        query: str,
        limit: int = 10,
        type_: str | None = None,
        exclude_types: set[str] | None = None,
        field_boost: str | None = None,
    ) -> list[dict[str, Any]]:
        """BM25 keyword search. Dispatches to tantivy when available, FTS5 otherwise.

        Returns rows shaped like `search()` (vec) — metadata dict with `score`
        in [0,1] where higher = more relevant.

        `field_boost="exact"` forces the FTS5 path (so the elevated tag/title
        weights apply deterministically regardless of backend) and runs a
        strict AND with no OR fallback.
        """
        if not query or not query.strip():
            return []
        if field_boost == "exact":
            return self._search_bm25_fts5(
                query, limit, type_, exclude_types, field_boost="exact"
            )
        t = self._get_tantivy()
        if t is not None:
            return self._search_bm25_tantivy(query, limit, type_, exclude_types, t)
        return self._search_bm25_fts5(query, limit, type_, exclude_types)

    def _search_bm25_tantivy(
        self,
        query: str,
        limit: int,
        type_: str | None,
        exclude_types: set[str] | None,
        t: Any,
    ) -> list[dict[str, Any]]:
        # Fetch more candidates when filtering by type so we can honour `limit`
        # after post-filtering against the meta table.
        candidate_k = limit * 5 if (type_ or exclude_types) else limit
        hits = t.search_bm25(query, candidate_k)
        if not hits:
            return []
        # Resolve metadata from sqlite in one batched query.
        id_score = {h["id"]: h["score"] for h in hits}
        placeholders = ",".join("?" for _ in id_score)
        sql = (
            "SELECT id, path, title, type, tags, created, updated, body_hash, extra_json "
            f"FROM meta WHERE id IN ({placeholders})"
        )
        params: list[Any] = list(id_score.keys())
        if type_:
            sql += " AND type = ?"
            params.append(type_)
        if exclude_types:
            sql += f" AND type NOT IN ({','.join('?' for _ in exclude_types)})"
            params.extend(sorted(exclude_types))
        rows = self._conn.execute(sql, params).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            d = _row_to_dict(r)
            d["score"] = id_score.get(d["id"], 0.0)
            out.append(d)
        out.sort(key=lambda x: x["score"], reverse=True)
        return out[:limit]

    def _search_bm25_fts5(
        self,
        query: str,
        limit: int,
        type_: str | None,
        exclude_types: set[str] | None,
        field_boost: str | None = None,
    ) -> list[dict[str, Any]]:
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

        # `exact` mode applies a preconfigured field boost favouring curated
        # metadata (title/tags) over body prose. Weights are tunable via
        # MEMO_EXACT_TITLE_WEIGHT / MEMO_EXACT_TAGS_WEIGHT (registered in
        # flags.py for `memo config validate`; read here via os.environ because
        # the store layer is a foundation module and cannot import memo.flags).
        if field_boost == "exact":
            title_w = _env_float("MEMO_EXACT_TITLE_WEIGHT", 10.0)
            tags_w = _env_float("MEMO_EXACT_TAGS_WEIGHT", 8.0)
            body_w = _BM25_FTS_BODY_WEIGHT
        else:
            title_w = _BM25_FTS_TITLE_WEIGHT
            tags_w = _BM25_FTS_TAGS_WEIGHT
            body_w = _BM25_FTS_BODY_WEIGHT

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
                title_w,
                tags_w,
                body_w,
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
            except sqlite3.OperationalError as _bm25_err:
                # Malformed FTS expression (e.g. unbalanced quotes after
                # escape). Fall back to no results — Memory.search_hybrid
                # treats this as "no BM25 signal" and uses pure vec.
                _log.warning("BM25 search failed (falling back to vec-only): %s", _bm25_err)
                return []

        rows = _run(_tokens, " ")
        # AND-of-tokens zero-recall fallback: only when the strict AND
        # match returns nothing on a multi-token query, retry with OR.
        # Triggering on `<limit` (partial recall) caused RRF rank
        # washing — OR brings in popular single-token matches that
        # demote the AND-matched correct doc once fused with the vec
        # leg. Triggering only on zero is the safe floor: it cannot
        # make a successful AND query worse.
        # exact mode is strict: never loosen AND into OR. A missing term means
        # no match, by design.
        if not rows and len(_tokens) >= 2 and field_boost != "exact":
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

    def search_fuzzy(
        self,
        query: str,
        limit: int = 10,
        type_: str | None = None,
        exclude_types: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Fuzzy (typo-tolerant) BM25 search via tantivy.

        Falls back to `_search_bm25_fts5` when tantivy is not available
        (no true fuzzy without tantivy, but better than nothing).
        Returns rows shaped like `search()`.
        """
        if not query or not query.strip():
            return []
        t = self._get_tantivy()
        if t is None:
            return self._search_bm25_fts5(query, limit, type_, exclude_types)
        candidate_k = limit * 5 if (type_ or exclude_types) else limit
        hits = t.search_fuzzy(query, candidate_k)
        if not hits:
            return []
        id_score = {h["id"]: h["score"] for h in hits}
        placeholders = ",".join("?" for _ in id_score)
        sql = (
            "SELECT id, path, title, type, tags, created, updated, body_hash, extra_json "
            f"FROM meta WHERE id IN ({placeholders})"
        )
        params: list[Any] = list(id_score.keys())
        if type_:
            sql += " AND type = ?"
            params.append(type_)
        if exclude_types:
            sql += f" AND type NOT IN ({','.join('?' for _ in exclude_types)})"
            params.extend(sorted(exclude_types))
        rows = self._conn.execute(sql, params).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            d = _row_to_dict(r)
            d["score"] = id_score.get(d["id"], 0.0)
            out.append(d)
        out.sort(key=lambda x: x["score"], reverse=True)
        return out[:limit]

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM meta").fetchone()[0]
