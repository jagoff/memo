from __future__ import annotations

import json
import sqlite3
from typing import Any

from sqlite_vec import serialize_float32

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
