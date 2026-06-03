from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, datetime
from typing import Any

_log = logging.getLogger(__name__)

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

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM meta").fetchone()[0]
