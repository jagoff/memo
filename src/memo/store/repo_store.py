from __future__ import annotations

import json
import logging
import sqlite3
import struct
from typing import Any

from ..sqlite_compat import import_sqlite_vec
from ._base import _StoreBase
from .rows import (
    _batches,
    _fts_match_expr,
    _repo_bm25_row_to_dict,
    _repo_row_to_dict,
    _row_to_dict,
)
from .schema import (
    _BM25_BODY_WEIGHT,
    _BM25_PATH_WEIGHT,
    _BM25_REPO_NAME_WEIGHT,
    _BM25_UNINDEXED_WEIGHT,
)

serialize_float32 = import_sqlite_vec().serialize_float32

_log = logging.getLogger(__name__)

_TEST_SCOPE_SQL = (
    "({column} GLOB 'test/*' OR {column} GLOB 'tests/*' "
    "OR {column} GLOB '*/test/*' OR {column} GLOB '*/tests/*' "
    "OR {column} GLOB '__tests__/*' OR {column} GLOB '*/__tests__/*' "
    "OR {column} GLOB 'spec/*' OR {column} GLOB 'specs/*' "
    "OR {column} GLOB '*/spec/*' OR {column} GLOB '*/specs/*' "
    "OR {column} GLOB 'fixtures/*' OR {column} GLOB '*/fixtures/*' "
    "OR {column} GLOB 'testdata/*' OR {column} GLOB '*/testdata/*' "
    "OR {column} GLOB 'test_*.*' OR {column} GLOB '*/test_*.*' "
    "OR {column} GLOB '*_test.*' OR {column} GLOB '*/*_test.*' "
    "OR {column} GLOB '*.test.*' OR {column} GLOB '*/*.test.*' "
    "OR {column} GLOB '*_spec.*' OR {column} GLOB '*/*_spec.*' "
    "OR {column} GLOB '*.spec.*' OR {column} GLOB '*/*.spec.*')"
)
_VENDOR_SCOPE_SQL = (
    "({column} GLOB 'vendor/*' OR {column} GLOB 'vendored/*' "
    "OR {column} GLOB '*/vendor/*' OR {column} GLOB '*/vendored/*' "
    "OR {column} GLOB 'third_party/*' OR {column} GLOB 'third-party/*' "
    "OR {column} GLOB '*/third_party/*' OR {column} GLOB '*/third-party/*' "
    "OR {column} GLOB 'node_modules/*' OR {column} GLOB '*/node_modules/*' "
    "OR {column} GLOB 'dist/*' OR {column} GLOB '*/dist/*' "
    "OR {column} GLOB 'build/*' OR {column} GLOB '*/build/*' "
    "OR {column} GLOB 'generated/*' OR {column} GLOB '*/generated/*' "
    "OR {column} GLOB 'coverage/*' OR {column} GLOB '*/coverage/*' "
    "OR {column} GLOB 'target/*' OR {column} GLOB '*/target/*')"
)


def _repo_scope_sql(column: str, scope: str) -> str:
    normalized = str(scope or "all").strip().lower()
    if column not in {"repo_chunks.path", "repo_lines.path"}:
        raise ValueError(f"unsupported repo path column: {column}")
    lowered = f"lower({column})"
    tests = _TEST_SCOPE_SQL.format(column=lowered)
    vendor = _VENDOR_SCOPE_SQL.format(column=lowered)
    if normalized == "all":
        return "1 = 1"
    if normalized == "tests":
        return tests
    if normalized == "vendor":
        return vendor
    if normalized == "production":
        return f"NOT {tests} AND NOT {vendor}"
    raise ValueError(f"invalid repo scope {scope!r}; expected all, production, tests, or vendor")


def _attach_repo_generation(row: dict[str, Any]) -> None:
    raw = row.pop("extra_json", None)
    try:
        extra = json.loads(raw or "{}")
    except (TypeError, ValueError):
        extra = {}
    evidence = extra.get("code_evidence") if isinstance(extra, dict) else {}
    if not isinstance(evidence, dict):
        evidence = {}
    row["index_generation"] = str(evidence.get("index_generation") or "")


def _provider_target_line(entry: dict[str, Any]) -> int | None:
    for evidence in entry.get("evidence") or []:
        if not isinstance(evidence, dict):
            continue
        try:
            return int(evidence["line_start"])
        except (KeyError, TypeError, ValueError):
            continue
    return None


class _RepoStoreMixin(_StoreBase):
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

    def replace_repo_coverage(
        self,
        *,
        repo_id: str,
        generation: str,
        rows: list[dict[str, Any]],
        recorded_at: str,
    ) -> None:
        """Atomically replace recorded coverage gaps for one index generation."""
        with self._tx() as cx:
            cx.execute(
                "DELETE FROM repo_coverage WHERE repo_id = ? AND generation = ?",
                (repo_id, generation),
            )
            cx.executemany(
                "INSERT INTO repo_coverage "
                "(repo_id, generation, path, reason, detail, line_start, line_end, recorded_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        repo_id,
                        generation,
                        str(row["path"]),
                        str(row["reason"]),
                        str(row.get("detail") or ""),
                        row.get("line_start"),
                        row.get("line_end"),
                        recorded_at,
                    )
                    for row in rows
                ],
            )

    def list_repo_coverage(self, repo_id: str, generation: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT path, reason, detail, line_start, line_end, recorded_at "
            "FROM repo_coverage WHERE repo_id = ? AND generation = ? "
            "ORDER BY path, reason",
            (repo_id, generation),
        ).fetchall()
        return [dict(row) for row in rows]

    def update_repo_status(
        self, repo_id: str, status: str, *, indexed_at: str | None = None
    ) -> None:
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
            "SELECT repo_chunks.id, repo_chunks.repo_id, repo_sources.name AS repo_name, "  # noqa: S608
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
                "SELECT input_hash, embedding FROM repo_embedding_cache "  # noqa: S608
                f"WHERE model = ? AND dims = ? AND input_hash IN ({placeholders})",
                (model, dims, *batch),
            ).fetchall()
            for row in rows:
                try:
                    raw = row["embedding"]
                    if isinstance(raw, (bytes, bytearray, memoryview)):
                        if len(raw) != dims * 4:
                            continue
                        emb = list(struct.unpack(f"<{dims}f", bytes(raw)))
                    else:
                        emb = json.loads(raw)
                except (ValueError, TypeError):
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
                    f"[input={input_hash[:12]}].\n"
                    f"Fix: memo reindex --rebuild\n"
                    f"Or check: MEMO_MODEL_PROFILE={dims}D"
                )
        with self._tx() as cx:
            cx.executemany(
                "INSERT INTO repo_embedding_cache "
                "(model, dims, input_hash, embedding, created_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(model, dims, input_hash) DO UPDATE SET "
                "embedding=excluded.embedding, created_at=excluded.created_at",
                [
                    (model, dims, input_hash, serialize_float32(emb), created_at)
                    for input_hash, emb in embeddings
                ],
            )

    def prune_repo_embedding_cache(
        self,
        *,
        keep_models: set[tuple[str, int]],
        older_than: str | None = None,
        dry_run: bool = False,
    ) -> int:
        """Remove rebuildable cache rows for inactive model identities.

        The cache is derived data, unlike Markdown or user feedback. Keeping
        one identity per active model/dimension pair prevents model revisions
        from silently accumulating hundreds of megabytes of JSON embeddings.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if older_than:
            clauses.append("created_at < ?")
            params.append(older_than)
        sql = "SELECT model, dims, input_hash FROM repo_embedding_cache"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        rows = self._conn.execute(sql, params).fetchall()
        stale = [
            (str(row["model"]), int(row["dims"]), str(row["input_hash"]))
            for row in rows
            if (str(row["model"]), int(row["dims"])) not in keep_models
        ]
        if dry_run or not stale:
            return len(stale)
        with self._tx() as cx:
            cx.executemany(
                "DELETE FROM repo_embedding_cache WHERE model = ? AND dims = ? AND input_hash = ?",
                stale,
            )
        return len(stale)

    def compact_repo_embedding_cache(self, *, dry_run: bool = False) -> int:
        """Pack legacy JSON cache rows as float32 blobs in place.

        SQLite's dynamic typing lets this migrate without a schema rewrite;
        readers remain backward-compatible with both representations.
        """
        rows = self._conn.execute(
            "SELECT model, dims, input_hash, embedding FROM repo_embedding_cache"
        ).fetchall()
        updates: list[tuple[bytes, str, int, str]] = []
        for row in rows:
            raw = row["embedding"]
            if isinstance(raw, (bytes, bytearray, memoryview)):
                continue
            try:
                emb = json.loads(raw)
                dims = int(row["dims"])
                if not isinstance(emb, list) or len(emb) != dims:
                    continue
                updates.append(
                    (serialize_float32([float(value) for value in emb]), str(row["model"]), dims, str(row["input_hash"]))
                )
            except (ValueError, TypeError, json.JSONDecodeError):
                continue
        if dry_run or not updates:
            return len(updates)
        with self._tx() as cx:
            cx.executemany(
                "UPDATE repo_embedding_cache SET embedding = ? "
                "WHERE model = ? AND dims = ? AND input_hash = ?",
                updates,
            )
        return len(updates)

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
                    f"[chunk={chunk_id[:12]}].\n"
                    f"Fix: rm {self.db_path} && memo reindex\n"
                    f"Or check: MEMO_MODEL_PROFILE={self.dims}D"
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
                "INSERT INTO repo_vec (id, repo_id, embedding) VALUES (?, ?, ?)",
                [(cid, repo_id, serialize_float32(emb)) for cid, emb in embeddings],
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
                    source["id"],
                    source["name"],
                    source["url"],
                    source["ref"],
                    source["commit_sha"],
                    source["clone_path"],
                    source["indexed_at"],
                    source.get("status") or "ready",
                    json.dumps(source.get("extra") or {}, default=str),
                ),
            )

    def patch_repo_source_extra(self, repo_id: str, patch: dict[str, Any]) -> None:
        """Merge provider metadata without clobbering index/evidence fields."""
        with self._tx() as cx:
            row = cx.execute(
                "SELECT extra_json FROM repo_sources WHERE id = ?",
                (repo_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"repo not found: {repo_id}")
            try:
                current = json.loads(row["extra_json"] or "{}")
            except (TypeError, ValueError):
                current = {}
            if not isinstance(current, dict):
                current = {}
            current.update(patch)
            cx.execute(
                "UPDATE repo_sources SET extra_json = ? WHERE id = ?",
                (json.dumps(current, default=str), repo_id),
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
                        f"[chunk={chunk.get('id', '')[:12]}].\n"
                        f"Fix: rm {self.db_path} && memo reindex\n"
                        f"Or check: MEMO_MODEL_PROFILE={self.dims}D"
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
                        file_data["id"],
                        repo_id,
                        file_data["path"],
                        file_data.get("language") or "",
                        int(file_data.get("size_bytes") or 0),
                        file_data["sha256"],
                        int(file_data.get("line_count") or 0),
                        indexed_at,
                    ),
                )

                line_rows = [
                    (
                        line["id"],
                        repo_id,
                        file_data["id"],
                        file_data["path"],
                        int(line["line_no"]),
                        line.get("text") or "",
                        line["text_hash"],
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
                    [(line[0], repo_name, line[3], line[4], line[5]) for line in line_rows],
                )

                chunk_rows = [
                    (
                        chunk["id"],
                        repo_id,
                        file_data["id"],
                        file_data["path"],
                        int(chunk["chunk_seq"]),
                        int(chunk["line_start"]),
                        int(chunk["line_end"]),
                        chunk["text_hash"],
                        chunk.get("body_text") or "",
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
                    "INSERT INTO repo_vec (id, repo_id, embedding) VALUES (?, ?, ?)",
                    [
                        (chunk["id"], repo_id, serialize_float32(chunk["embedding"]))
                        for chunk in file_data.get("chunks") or []
                        if chunk.get("embedding") is not None
                    ],
                )
                cx.executemany(
                    "INSERT INTO repo_chunk_fts (id, repo_name, path, body) VALUES (?, ?, ?, ?)",
                    [(chunk[0], repo_name, chunk[3], chunk[8]) for chunk in chunk_rows],
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
        for batch in _batches(file_ids):
            placeholders = ",".join("?" for _ in batch)
            chunk_ids = [
                r["id"]
                for r in cx.execute(
                    f"SELECT id FROM repo_chunks WHERE file_id IN ({placeholders})",  # noqa: S608
                    batch,
                ).fetchall()
            ]
            line_ids = [
                r["id"]
                for r in cx.execute(
                    f"SELECT id FROM repo_lines WHERE file_id IN ({placeholders})",  # noqa: S608
                    batch,
                ).fetchall()
            ]
            cx.executemany("DELETE FROM repo_vec WHERE id = ?", [(cid,) for cid in chunk_ids])
            cx.executemany("DELETE FROM repo_chunk_fts WHERE id = ?", [(cid,) for cid in chunk_ids])
            cx.executemany("DELETE FROM repo_line_fts WHERE id = ?", [(lid,) for lid in line_ids])
            cx.executemany("DELETE FROM repo_chunks WHERE file_id = ?", [(fid,) for fid in batch])
            cx.executemany("DELETE FROM repo_lines WHERE file_id = ?", [(fid,) for fid in batch])
            cx.executemany("DELETE FROM repo_files WHERE id = ?", [(fid,) for fid in batch])

    def delete_repo(self, key: str) -> bool:
        source = self.get_repo_source(key)
        if source is None:
            return False
        repo_id = source["id"]
        with self._tx() as cx:
            file_ids = [
                r["id"]
                for r in cx.execute(
                    "SELECT id FROM repo_files WHERE repo_id = ?",
                    (repo_id,),
                ).fetchall()
            ]
            self._delete_repo_file_rows(cx, file_ids)
            cx.execute("DELETE FROM repo_coverage WHERE repo_id = ?", (repo_id,))
            cx.execute("DELETE FROM repo_sources WHERE id = ?", (repo_id,))
        return True

    def search_repo_vec(
        self,
        embedding: list[float],
        limit: int = 10,
        repo_id: str | None = None,
        path_glob: str | None = None,
        scope: str = "all",
    ) -> list[dict[str, Any]]:
        if len(embedding) != self.dims:
            raise ValueError(
                f"Repo query embedding dim mismatch: got {len(embedding)}, expected {self.dims}.\n"
                f"Fix: rm {self.db_path} && memo reindex\n"
                f"Or check: MEMO_MODEL_PROFILE={self.dims}D"
            )
        # `repo_id` is a vec0 PARTITION KEY → `repo_vec.repo_id = ?` pre-filters
        # the kNN to that repo (exact, no over-fetch). Only `path_glob` (a GLOB,
        # not vec0-filterable) still needs the `k * 5` over-fetch + post-filter.
        candidate_k = limit * 20 if (path_glob or scope != "all" or repo_id is None) else limit
        sql = (
            "SELECT repo_chunks.id AS id, repo_vec.distance AS distance, "
            "       repo_chunks.repo_id, repo_sources.name AS repo_name, repo_sources.url, "
            "       repo_sources.ref, repo_sources.commit_sha, repo_chunks.file_id, "
            "       repo_chunks.path, repo_files.language, repo_chunks.line_start, "
            "       repo_chunks.line_end, repo_chunks.body_text, repo_sources.extra_json "
            "FROM repo_vec "
            "JOIN repo_chunks ON repo_chunks.id = repo_vec.id "
            "JOIN repo_files ON repo_files.id = repo_chunks.file_id "
            "JOIN repo_sources ON repo_sources.id = repo_chunks.repo_id "
            "WHERE embedding MATCH ? AND k = ? "
            "AND repo_sources.status != 'indexing' "
        )
        params: list[Any] = [serialize_float32(embedding), candidate_k]
        if repo_id:
            sql += "AND repo_vec.repo_id = ? "
            params.append(repo_id)
        if path_glob:
            sql += "AND repo_chunks.path GLOB ? "
            params.append(path_glob)
        sql += f"AND {_repo_scope_sql('repo_chunks.path', scope)} "
        sql += "ORDER BY distance ASC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            d = _repo_row_to_dict(r)
            _attach_repo_generation(d)
            d["score"] = max(0.0, 1.0 - float(r["distance"]))
            d["match_type"] = "vec"
            out.append(d)
        return out

    def search_repo_bm25(
        self,
        query: str,
        limit: int = 10,
        repo_id: str | None = None,
        path_glob: str | None = None,
        scope: str = "all",
    ) -> list[dict[str, Any]]:
        match_expr = _fts_match_expr(query)
        if not match_expr:
            return []
        candidate_k = limit * 20 if (repo_id or path_glob or scope != "all") else limit
        # Column weights: (id UNINDEXED, repo_name, path, body)
        sql = (
            "SELECT repo_chunks.id AS id, "
            "       bm25(repo_chunk_fts, ?, ?, ?, ?) AS bm25_score, "
            "       repo_chunks.repo_id, repo_sources.name AS repo_name, repo_sources.url, "
            "       repo_sources.ref, repo_sources.commit_sha, repo_chunks.file_id, "
            "       repo_chunks.path, repo_files.language, repo_chunks.line_start, "
            "       repo_chunks.line_end, repo_chunks.body_text, repo_sources.extra_json "
            "FROM repo_chunk_fts "
            "JOIN repo_chunks ON repo_chunks.id = repo_chunk_fts.id "
            "JOIN repo_files ON repo_files.id = repo_chunks.file_id "
            "JOIN repo_sources ON repo_sources.id = repo_chunks.repo_id "
            "WHERE repo_chunk_fts MATCH ? AND repo_sources.status != 'indexing' "
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
        sql += f"AND {_repo_scope_sql('repo_chunks.path', scope)} "
        sql += "ORDER BY bm25_score ASC LIMIT ?"
        params.append(candidate_k)
        try:
            rows = self._conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError as exc:
            _log.warning("repo bm25 (chunk) query failed: %s", exc)
            return []
        out = [_repo_bm25_row_to_dict(r, "bm25") for r in rows[:limit]]
        for row in out:
            _attach_repo_generation(row)
        return out

    def search_repo_lines(
        self,
        query: str,
        limit: int = 10,
        repo_id: str | None = None,
        path_glob: str | None = None,
        scope: str = "all",
    ) -> list[dict[str, Any]]:
        match_expr = _fts_match_expr(query)
        if not match_expr:
            return []
        candidate_k = limit * 20 if (repo_id or path_glob or scope != "all") else limit
        # Column weights: (id UNINDEXED, repo_name, path, line_no UNINDEXED, body)
        sql = (
            "SELECT repo_lines.id AS id, "
            "       bm25(repo_line_fts, ?, ?, ?, ?, ?) AS bm25_score, "
            "       repo_lines.repo_id, repo_sources.name AS repo_name, repo_sources.url, "
            "       repo_sources.ref, repo_sources.commit_sha, repo_lines.file_id, "
            "       repo_lines.path, repo_files.language, repo_lines.line_no AS line_start, "
            "       repo_lines.line_no AS line_end, repo_lines.text AS body_text, "
            "       repo_sources.extra_json "
            "FROM repo_line_fts "
            "JOIN repo_lines ON repo_lines.id = repo_line_fts.id "
            "JOIN repo_files ON repo_files.id = repo_lines.file_id "
            "JOIN repo_sources ON repo_sources.id = repo_lines.repo_id "
            "WHERE repo_line_fts MATCH ? AND repo_sources.status != 'indexing' "
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
        sql += f"AND {_repo_scope_sql('repo_lines.path', scope)} "
        sql += "ORDER BY bm25_score ASC LIMIT ?"
        params.append(candidate_k)
        try:
            rows = self._conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError as exc:
            _log.warning("repo bm25 (line) query failed: %s", exc)
            return []
        out = [_repo_bm25_row_to_dict(r, "line") for r in rows[:limit]]
        for row in out:
            _attach_repo_generation(row)
        return out

    def repo_chunks_for_paths(
        self,
        repo_id: str,
        path_entries: list[dict[str, Any]],
        *,
        limit: int = 40,
    ) -> list[dict[str, Any]]:
        """Resolve provider path evidence to the most relevant active chunk."""
        if not path_entries:
            return []
        by_path = {
            str(entry.get("path") or ""): entry
            for entry in path_entries
            if str(entry.get("path") or "")
        }
        rows: list[sqlite3.Row] = []
        for batch in _batches(list(by_path)):
            placeholders = ",".join("?" for _ in batch)
            query = (
                "SELECT repo_chunks.id, repo_chunks.repo_id, "  # noqa: S608
                "       repo_sources.name AS repo_name, repo_sources.url, "
                "       repo_sources.ref, repo_sources.commit_sha, "
                "       repo_sources.extra_json, repo_chunks.file_id, "
                "       repo_chunks.path, repo_files.language, "
                "       repo_chunks.line_start, repo_chunks.line_end, "
                "       repo_chunks.body_text "
                "FROM repo_chunks "
                "JOIN repo_files ON repo_files.id = repo_chunks.file_id "
                "JOIN repo_sources ON repo_sources.id = repo_chunks.repo_id "
                "WHERE repo_chunks.repo_id = ? "
                "AND repo_sources.status != 'indexing' "
                f"AND repo_chunks.path IN ({placeholders}) "
                "ORDER BY repo_chunks.path, repo_chunks.chunk_seq"
            )
            rows.extend(
                self._conn.execute(
                    query,
                    (repo_id, *batch),
                ).fetchall()
            )

        candidates: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            candidates.setdefault(str(row["path"]), []).append(row)
        out: list[dict[str, Any]] = []
        for path, entry in by_path.items():
            path_rows = candidates.get(path) or []
            if not path_rows:
                continue
            target_line = _provider_target_line(entry)
            selected = next(
                (
                    row
                    for row in path_rows
                    if target_line is not None
                    and int(row["line_start"]) <= target_line <= int(row["line_end"])
                ),
                path_rows[0],
            )
            item = _repo_row_to_dict(selected)
            _attach_repo_generation(item)
            item["score"] = float(entry.get("score") or 0.0)
            item["match_type"] = str(entry.get("match_type") or "provider")
            item["provider_evidence"] = list(entry.get("evidence") or [])
            item["provider_metadata"] = {
                key: value
                for key, value in entry.items()
                if key not in {"path", "score", "evidence", "match_type"}
            }
            out.append(item)
        return sorted(
            out,
            key=lambda item: (-float(item["score"]), str(item["path"])),
        )[:limit]

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
            "SELECT line_no, text FROM repo_lines WHERE repo_id = ? AND path = ? AND line_no >= ? "
        )
        if end is not None:
            sql += "AND line_no <= ? "
            params.append(max(start, int(end)))
        sql += "ORDER BY line_no ASC"
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
