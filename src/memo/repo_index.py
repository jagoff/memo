"""Git repository ingestion and retrieval for memo.

This subsystem is intentionally separate from curated memories. Repo
content is source code / document corpus material: it can be large, it
changes by commit, and callers need line-accurate retrieval rather than
one note-shaped Markdown file per record.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path
from typing import Any

from memo.config import Config
from memo.embedder import MLXEmbedder, assert_valid_embedding
from memo.ingest_helpers import enrich_with_ocr
from memo.ocr import ocr_enabled_via_env

# Re-export constants so external callers that import them from this module
# still work without changes.
from memo.repo_index_helpers import (  # noqa: F401
    DEFAULT_CHUNK_OVERLAP_LINES,
    DEFAULT_CHUNK_TARGET_CHARS,
    DEFAULT_EMBED_BATCH,
    DEFAULT_EXCLUDE_DIRS,
    DEFAULT_EXCLUDE_GLOBS,
    DEFAULT_FLUSH_BATCH,
    DEFAULT_MAX_FILE_BYTES,
    MIN_CHUNK_CHARS,
    MIN_EMBED_BATCH,
    MIN_FLUSH_BATCH,
    STATUS_INDEXING,
    ProgressCallback,
    RepoEmbedInput,
    _chunk_lines,
    _derive_repo_name,
    _embed_cache_model,
    _emit,
    _git,
    _git_timeout,
    _is_excluded,
    _is_noise_chunk,
    _language_for_path,
    _looks_binary,
    _now_iso,
    _repo_embed_batch_size,
    _repo_embed_input,
    _repo_flush_batch_size,
    _safe_repo_name,
    _semantic_status,
    _stable_id,
    _tracked_files,
)
from memo.repo_index_search import (
    RepoSearchHit,
    _boost_and_resort,
    _extract_query_terms,
    _hits_from_rows,
    _path_name_boost,  # noqa: F401 — re-exported for backward compat
    _rrf_fuse_repo,
)
from memo.store import VecStore
from memo.util import sha256_short as _short_hash

_logger = logging.getLogger(__name__)


class RepoCorpus:
    """Index and search Git repositories inside memo's sqlite-vec store."""

    def __init__(
        self,
        cfg: Config,
        *,
        store: VecStore | None = None,
        embedder: MLXEmbedder | None = None,
    ) -> None:
        self.cfg = cfg
        cfg.ensure_dirs()
        self.store = store or VecStore(
            cfg.db_path, dims=cfg.embedder_dims, embedder_model=cfg.embedder_model
        )
        self.embedder = embedder or MLXEmbedder(
            model_path=cfg.embedder_model,
            expected_dims=cfg.embedder_dims,
        )

    def index(
        self,
        url: str,
        *,
        name: str | None = None,
        ref: str | None = None,
        force: bool = False,
        with_embeddings: bool = True,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        max_file_bytes: int | None = None,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        if not url or not url.strip():
            raise ValueError("repo url must be non-empty")

        ref_name = (ref or "HEAD").strip() or "HEAD"
        repo_id = _stable_id("repo", url.strip(), ref_name)
        repo_name = _safe_repo_name(name or _derive_repo_name(url))
        if not repo_name:
            repo_name = f"repo-{repo_id[:8]}"

        existing_by_name = self.store.get_repo_source(repo_name)
        if existing_by_name and existing_by_name["id"] != repo_id:
            if name:
                raise ValueError(f"repo name {repo_name!r} already exists")
            repo_name = f"{repo_name}-{repo_id[:8]}"

        clone_path = self.cfg.state_dir / "repos" / f"{repo_name}-{repo_id[:8]}"
        clone_path.parent.mkdir(parents=True, exist_ok=True)

        _emit(progress, "clone_start", url=url.strip(), path=str(clone_path), ref=ref_name)
        self._sync_clone(url.strip(), clone_path, ref_name)
        commit_sha = _git(["git", "-C", str(clone_path), "rev-parse", "HEAD"]).strip()
        _emit(progress, "clone_done", commit_sha=commit_sha)

        existing_source = self.store.get_repo_source_by_url_ref(url.strip(), ref_name)
        resuming_partial = (
            existing_source is not None
            and existing_source.get("commit_sha") == commit_sha
            and existing_source.get("status") == STATUS_INDEXING
        )
        if (
            existing_source
            and existing_source.get("commit_sha") == commit_sha
            and existing_source.get("status") != STATUS_INDEXING
            and not force
        ):
            counts = self.store.repo_counts(existing_source["id"])
            semantic_status = _semantic_status(existing_source.get("status"), counts)
            embed_counts: dict[str, Any] = {}
            if with_embeddings and semantic_status != "semantic_ready":
                embed_counts = self.embed(existing_source["id"], force=False, progress=progress)
                counts = self.store.repo_counts(existing_source["id"])
                semantic_status = _semantic_status(embed_counts["semantic_status"], counts)
                return {
                    "repo_id": repo_id,
                    "name": existing_source["name"],
                    "url": url.strip(),
                    "ref": ref_name,
                    "commit_sha": commit_sha,
                    "clone_path": str(clone_path),
                    "checked_files": 0,
                    "indexed_files": 0,
                    "unchanged_files": len(self.store.repo_file_hashes(repo_id)),
                    "deleted_files": 0,
                    "indexed_chunks": 0,
                    "indexed_lines": 0,
                    "embedded_chunks": counts["embedded_chunks"],
                    "model_chunks": int(embed_counts.get("model_chunks") or 0),
                    "cached_chunks": int(embed_counts.get("cached_chunks") or 0),
                    "pending_chunks": counts["chunks"] - counts["embedded_chunks"],
                    "semantic_status": semantic_status,
                    "skipped_binary": 0,
                    "skipped_excluded": 0,
                    "skipped_too_large": 0,
                    "errors": 0,
                    "skipped_repo_unchanged": True,
                    "resumed_partial": False,
                }
            return {
                "repo_id": repo_id,
                "name": existing_source["name"],
                "url": url.strip(),
                "ref": ref_name,
                "commit_sha": commit_sha,
                "clone_path": str(clone_path),
                "checked_files": 0,
                "indexed_files": 0,
                "unchanged_files": len(self.store.repo_file_hashes(repo_id)),
                "deleted_files": 0,
                "indexed_chunks": 0,
                "indexed_lines": 0,
                "embedded_chunks": counts["embedded_chunks"],
                "model_chunks": 0,
                "cached_chunks": 0,
                "pending_chunks": counts["chunks"] - counts["embedded_chunks"],
                "semantic_status": semantic_status,
                "skipped_binary": 0,
                "skipped_excluded": 0,
                "skipped_too_large": 0,
                "errors": 0,
                "skipped_repo_unchanged": True,
                "resumed_partial": False,
            }

        max_bytes = max_file_bytes
        if max_bytes is None:
            from memo.flags import flag_int

            max_bytes = flag_int("MEMO_REPO_MAX_FILE_BYTES") or DEFAULT_MAX_FILE_BYTES
        include_globs = list(include or [])
        exclude_globs = [*DEFAULT_EXCLUDE_GLOBS, *(exclude or [])]

        existing_files = self.store.repo_file_hashes(repo_id)
        # Track all paths that were part of the scan input, regardless of
        # processing success. Used for deletion detection below — without this
        # a transient read/OCR/stat error on a previously-indexed file would
        # make it disappear from tracked_paths and get falsely deleted.
        tracked_paths: set[str] = set()
        # (dev, inode) of every file already indexed this run. On a
        # case-insensitive filesystem (APFS default) git can track the same
        # physical file under two casings (e.g. `notes/x.md` + `Notes/x.md`),
        # which otherwise index twice as distinct file_ids and duplicate every
        # chunk. Inode collision catches that without skipping genuinely
        # distinct files on a case-sensitive FS.
        seen_inodes: set[tuple[int, int]] = set()
        skipped_dup_path = 0
        pending_files: list[dict[str, Any]] = []
        checked = unchanged = skipped_binary = skipped_excluded = skipped_too_large = errors = 0
        indexed_files_total = indexed_chunks = indexed_lines = 0
        flushed_files = 0

        indexed_at = _now_iso()
        flush_batch = _repo_flush_batch_size()

        # Persist target commit + indexing status BEFORE the scan so a
        # partial run is recoverable: file-batch commits below remain on
        # disk and the next call resumes via the sha256 short-circuit.
        self.store.upsert_repo_source(
            {
                "id": repo_id,
                "name": repo_name,
                "url": url.strip(),
                "ref": ref_name,
                "commit_sha": commit_sha,
                "clone_path": str(clone_path),
                "indexed_at": indexed_at,
                "status": STATUS_INDEXING,
                "extra": {
                    "include": include_globs,
                    "exclude": list(exclude or []),
                    "max_file_bytes": max_bytes,
                },
            }
        )

        def _flush() -> None:
            nonlocal pending_files, flushed_files
            if not pending_files:
                return
            self.store.upsert_repo_files(
                repo_id=repo_id,
                repo_name=repo_name,
                indexed_at=indexed_at,
                files=pending_files,
            )
            flushed_files += len(pending_files)
            _emit(progress, "batch_flush", files=len(pending_files), total_flushed=flushed_files)
            pending_files = []

        tracked_files = _tracked_files(clone_path)
        _emit(
            progress,
            "scan_start",
            total=len(tracked_files),
            resuming_partial=resuming_partial,
        )
        _emit(progress, "write_start", flush_batch=flush_batch)

        for rel_posix in tracked_files:
            rel = Path(rel_posix)
            path = clone_path / rel
            if _is_excluded(rel, rel_posix, include_globs, exclude_globs):
                skipped_excluded += 1
                _emit(progress, "file_skipped", path=rel_posix, reason="excluded")
                continue
            checked += 1
            tracked_paths.add(rel_posix)
            try:
                st = path.stat()
                size = st.st_size
                inode_key = (st.st_dev, st.st_ino)
                if inode_key in seen_inodes:
                    # Same physical file already indexed under another casing.
                    skipped_dup_path += 1
                    _emit(progress, "file_skipped", path=rel_posix, reason="duplicate_path")
                    continue
                if size > max_bytes:
                    skipped_too_large += 1
                    _emit(progress, "file_skipped", path=rel_posix, reason="too_large")
                    continue
                raw = path.read_bytes()
                if _looks_binary(raw):
                    skipped_binary += 1
                    _emit(progress, "file_skipped", path=rel_posix, reason="binary")
                    continue
                text = raw.decode("utf-8", errors="replace")
                sha = hashlib.sha256(raw).hexdigest()
                seen_inodes.add(inode_key)

                # OCR enrichment for .md files with embedded images
                # (`![[image.png]]`). Appends extracted text to the body so
                # the embedder sees screenshot content. Hash is composed
                # with each image's bytes hash so changing the image
                # invalidates the cache.
                if ocr_enabled_via_env() and rel_posix.lower().endswith(".md"):
                    enriched, _resolved, img_hashes = enrich_with_ocr(
                        text,
                        path,
                        clone_path,
                        self.cfg.state_dir,
                    )
                    if img_hashes:
                        text = enriched
                        h = hashlib.sha256(raw)
                        for piece in img_hashes:
                            h.update(piece)
                        sha = h.hexdigest()

                file_id = _stable_id("repo-file", repo_id, rel_posix)
                existing = existing_files.get(rel_posix)
                if existing and existing["sha256"] == sha and not force:
                    unchanged += 1
                    _emit(progress, "file_skipped", path=rel_posix, reason="unchanged")
                    continue

                file_payload = self._build_file_payload(
                    path=rel_posix,
                    file_id=file_id,
                    text=text,
                    raw_size=size,
                    sha=sha,
                )
                pending_files.append(file_payload)
                indexed_files_total += 1
                indexed_chunks += len(file_payload["chunks"])
                indexed_lines += len(file_payload["lines"])
                _emit(
                    progress,
                    "file_indexed",
                    path=rel_posix,
                    chunks=len(file_payload["chunks"]),
                    lines=len(file_payload["lines"]),
                )
                if len(pending_files) >= flush_batch:
                    _flush()
            except Exception:
                _logger.debug("file error for %s", rel_posix, exc_info=True)
                errors += 1
                _emit(progress, "file_error", path=rel_posix)

        _flush()

        deleted_file_ids = [
            meta["id"] for path, meta in existing_files.items() if path not in tracked_paths
        ]
        if deleted_file_ids:
            self.store.delete_repo_files(repo_id, deleted_file_ids)

        semantic_status = "semantic_pending"
        self.store.update_repo_status(repo_id, semantic_status, indexed_at=_now_iso())
        _emit(
            progress,
            "write_done",
            files=indexed_files_total,
            chunks=indexed_chunks,
            lines=indexed_lines,
            flushed_files=flushed_files,
        )

        index_embed_counts: dict[str, Any] = {}
        if with_embeddings:
            index_embed_counts = self.embed(repo_id, force=False, progress=progress)
            semantic_status = index_embed_counts["semantic_status"]
        counts_after = self.store.repo_counts(repo_id)
        semantic_status = _semantic_status(semantic_status, counts_after)

        return {
            "repo_id": repo_id,
            "name": repo_name,
            "url": url.strip(),
            "ref": ref_name,
            "commit_sha": commit_sha,
            "clone_path": str(clone_path),
            "checked_files": checked,
            "indexed_files": indexed_files_total,
            "unchanged_files": unchanged,
            "deleted_files": len(deleted_file_ids),
            "indexed_chunks": indexed_chunks,
            "indexed_lines": indexed_lines,
            "embedded_chunks": counts_after["embedded_chunks"],
            "model_chunks": int(index_embed_counts.get("model_chunks") or 0),
            "cached_chunks": int(index_embed_counts.get("cached_chunks") or 0),
            "pending_chunks": counts_after["chunks"] - counts_after["embedded_chunks"],
            "semantic_status": semantic_status,
            "skipped_binary": skipped_binary,
            "skipped_excluded": skipped_excluded,
            "skipped_too_large": skipped_too_large,
            "skipped_dup_path": skipped_dup_path,
            "errors": errors,
            "skipped_repo_unchanged": False,
            "resumed_partial": resuming_partial,
        }

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        repo: str | None = None,
        path: str | None = None,
        mode: str = "hybrid",
    ) -> list[RepoSearchHit]:
        if not query or not query.strip():
            return []
        repo_id = self._resolve_repo_id(repo) if repo else None
        if repo and repo_id is None:
            return []

        # Over-fetch so the title/filename boost has a chance to promote
        # notes whose body BM25 / line match score is dwarfed by noisy
        # transcripts. Without this, a procedural note can sit at rank 20
        # behind a YouTube transcript and never reach the boost stage.
        oversample = max(limit * 4, 40)

        if mode == "line":
            return _boost_and_resort(
                _hits_from_rows(
                    self.store.search_repo_lines(
                        query, limit=oversample, repo_id=repo_id, path_glob=path
                    )
                ),
                query=query,
                limit=limit,
            )
        if mode == "bm25":
            return _boost_and_resort(
                _hits_from_rows(
                    self.store.search_repo_bm25(
                        query, limit=oversample, repo_id=repo_id, path_glob=path
                    )
                ),
                query=query,
                limit=limit,
            )

        query_terms = _extract_query_terms(query)

        if repo_id is not None and self.store.repo_counts(repo_id)["embedded_chunks"] == 0:
            if mode == "vec":
                return []
            rows = _rrf_fuse_repo(
                [
                    self.store.search_repo_bm25(
                        query, limit=max(limit * 2, 30), repo_id=repo_id, path_glob=path
                    ),
                    self.store.search_repo_lines(
                        query, limit=max(limit * 2, 30), repo_id=repo_id, path_glob=path
                    ),
                ],
                limit=limit,
                query_terms=query_terms,
            )
            return _boost_and_resort(_hits_from_rows(rows), query=query, limit=limit)

        emb = self.embedder.embed_query(query)
        assert_valid_embedding(emb, self.cfg.embedder_dims, context="repo search query")

        if mode == "vec":
            return _boost_and_resort(
                _hits_from_rows(
                    self.store.search_repo_vec(emb, limit=limit, repo_id=repo_id, path_glob=path)
                ),
                query=query,
                limit=limit,
            )

        input_k = max(limit * 2, 30)
        vec_hits = self.store.search_repo_vec(emb, limit=input_k, repo_id=repo_id, path_glob=path)
        bm_hits = self.store.search_repo_bm25(query, limit=input_k, repo_id=repo_id, path_glob=path)
        line_hits = self.store.search_repo_lines(
            query, limit=input_k, repo_id=repo_id, path_glob=path
        )
        return _boost_and_resort(
            _hits_from_rows(
                _rrf_fuse_repo(
                    [vec_hits, bm_hits, line_hits],
                    limit=max(limit * 2, 30),
                    query_terms=query_terms,
                )
            ),
            query=query,
            limit=limit,
        )

    def embed(
        self,
        repo: str,
        *,
        force: bool = False,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        repo_id = self._resolve_repo_id(repo)
        if repo_id is None:
            raise ValueError(f"repo not found: {repo}")
        source = self.store.get_repo_source(repo_id)
        if source is None:
            raise ValueError(f"repo not found: {repo}")

        _emit(progress, "semantic_prepare", repo=source["name"])
        pending_rows = self.store.repo_pending_chunks(repo_id, force=force)
        _emit(progress, "semantic_start", repo=source["name"], chunks=len(pending_rows))
        if not pending_rows:
            counts = self.store.repo_counts(repo_id)
            status = _semantic_status(source.get("status"), counts)
            self.store.update_repo_status(repo_id, status, indexed_at=_now_iso())
            _emit(progress, "semantic_done", repo=source["name"], embedded=0, total=0)
            return {
                "repo_id": repo_id,
                "name": source["name"],
                "embedded_chunks": 0,
                "model_chunks": 0,
                "cached_chunks": 0,
                "total_chunks": counts["chunks"],
                "pending_chunks": counts["chunks"] - counts["embedded_chunks"],
                "semantic_status": status,
            }

        pending = sorted(
            (_repo_embed_input(chunk) for chunk in pending_rows),
            key=lambda item: len(item.text),
        )
        batch_size = _repo_embed_batch_size()
        cache_model = _embed_cache_model(self.embedder, self.cfg)
        embedded_total = 0
        model_total = 0
        cached_total = 0
        cursor = 0
        while cursor < len(pending):
            batch = pending[cursor : cursor + batch_size]
            cache_hits = self.store.get_repo_embedding_cache(
                model=cache_model,
                dims=self.cfg.embedder_dims,
                input_hashes=[item.input_hash for item in batch],
            )
            cached_embeddings = [
                (item.chunk["id"], cache_hits[item.input_hash])
                for item in batch
                if item.input_hash in cache_hits
            ]
            missing = [item for item in batch if item.input_hash not in cache_hits]
            new_cache_rows: list[tuple[str, list[float]]] = []
            embeddings: list[tuple[str, list[float]]] = []
            if missing:
                try:
                    vectors = self.embedder.embed([item.text for item in missing])
                except RuntimeError:
                    if batch_size <= MIN_EMBED_BATCH:
                        raise
                    batch_size = max(MIN_EMBED_BATCH, batch_size // 2)
                    _emit(
                        progress,
                        "semantic_batch_resize",
                        repo=source["name"],
                        batch_size=batch_size,
                    )
                    continue
                for item, vector in zip(missing, vectors, strict=True):
                    chunk = item.chunk
                    assert_valid_embedding(
                        vector,
                        self.cfg.embedder_dims,
                        context=f"repo chunk {chunk['path']}:{chunk['line_start']}-{chunk['line_end']}",
                    )
                    embeddings.append((chunk["id"], vector))
                    new_cache_rows.append((item.input_hash, vector))

            if new_cache_rows:
                self.store.upsert_repo_embedding_cache(
                    model=cache_model,
                    dims=self.cfg.embedder_dims,
                    embeddings=new_cache_rows,
                    created_at=_now_iso(),
                )
            all_embeddings = [*cached_embeddings, *embeddings]
            if all_embeddings:
                self.store.upsert_repo_embeddings(repo_id=repo_id, embeddings=all_embeddings)
            cached_total += len(cached_embeddings)
            model_total += len(embeddings)
            embedded_total += len(all_embeddings)
            cursor += len(batch)
            _emit(
                progress,
                "semantic_batch",
                repo=source["name"],
                completed=cursor,
                total=len(pending),
                cached=cached_total,
                model=model_total,
                batch_size=batch_size,
            )

        counts = self.store.repo_counts(repo_id)
        status = _semantic_status("semantic_ready", counts)
        self.store.update_repo_status(repo_id, status, indexed_at=_now_iso())
        _emit(
            progress,
            "semantic_done",
            repo=source["name"],
            embedded=embedded_total,
            cached=cached_total,
            model=model_total,
            total=len(pending),
        )
        return {
            "repo_id": repo_id,
            "name": source["name"],
            "embedded_chunks": embedded_total,
            "model_chunks": model_total,
            "cached_chunks": cached_total,
            "total_chunks": counts["chunks"],
            "pending_chunks": counts["chunks"] - counts["embedded_chunks"],
            "semantic_status": status,
        }

    def status(self, repo: str) -> dict[str, Any] | None:
        repo_id = self._resolve_repo_id(repo)
        if repo_id is None:
            return None
        source = self.store.get_repo_source(repo_id)
        if source is None:
            return None
        counts = self.store.repo_counts(repo_id)
        semantic_status = _semantic_status(source.get("status"), counts)
        return {
            **source,
            **counts,
            "pending_chunks": counts["chunks"] - counts["embedded_chunks"],
            "semantic_status": semantic_status,
        }

    def get_file(
        self,
        repo: str,
        path: str,
        *,
        start: int | None = None,
        end: int | None = None,
    ) -> dict[str, Any] | None:
        repo_id = self._resolve_repo_id(repo)
        if repo_id is None:
            return None
        meta = self.store.get_repo_file(repo_id, path)
        if meta is None:
            return None
        lines = self.store.get_repo_file_lines(repo_id, path, start=start, end=end)
        text = "\n".join(line["text"] for line in lines)
        return {
            **meta,
            "start": lines[0]["line_no"] if lines else (start or 1),
            "end": lines[-1]["line_no"] if lines else (end or start or 1),
            "text": text,
            "lines": lines,
        }

    def list(self, *, limit: int = 100) -> list[dict[str, Any]]:
        return self.store.list_repo_sources(limit=limit)

    def delete(self, repo: str, *, remove_clone: bool = True) -> bool:
        source = self.store.get_repo_source(repo)
        if source is None:
            return False
        ok = self.store.delete_repo(source["id"])
        if ok and remove_clone:
            clone_path = source.get("clone_path")
            if clone_path:
                shutil.rmtree(clone_path, ignore_errors=True)
        return ok

    def _sync_clone(self, url: str, clone_path: Path, ref: str) -> None:
        # A previous run may have been killed mid-clone, leaving a `.git`
        # directory without a usable HEAD. Detect that and wipe before
        # `git fetch` runs against a corrupt repo.
        if clone_path.exists() and (clone_path / ".git").is_dir():
            head = _git(
                ["git", "-C", str(clone_path), "rev-parse", "--verify", "HEAD"],
                check=False,
            ).strip()
            if not head:
                shutil.rmtree(clone_path, ignore_errors=True)

        if not clone_path.exists():
            _git(["git", "clone", url, str(clone_path)])
        elif not (clone_path / ".git").is_dir():
            shutil.rmtree(clone_path)
            _git(["git", "clone", url, str(clone_path)])
        else:
            _git(["git", "-C", str(clone_path), "fetch", "--all", "--tags", "--prune"])

        if ref != "HEAD":
            _git(["git", "-C", str(clone_path), "checkout", "--detach", ref])
            return

        branch = _git(
            ["git", "-C", str(clone_path), "branch", "--show-current"],
            check=False,
        ).strip()
        if branch:
            _git(["git", "-C", str(clone_path), "pull", "--ff-only"], check=False)

    def _build_file_payload(
        self,
        *,
        path: str,
        file_id: str,
        text: str,
        raw_size: int,
        sha: str,
    ) -> dict[str, Any]:
        lines = text.splitlines()
        line_rows = [
            {
                "id": _stable_id("repo-line", file_id, str(i)),
                "line_no": i,
                "text": line,
                "text_hash": _short_hash(line),
            }
            for i, line in enumerate(lines, start=1)
        ]

        chunk_payloads: list[dict[str, Any]] = []
        # The noise filter is markdown-centric (headings, wikilinks, frontmatter
        # fragments). Code files can be legitimately short — a 2-line function is
        # real signal, not noise — so only filter markdown.
        is_markdown = path.lower().endswith((".md", ".markdown"))
        for seq, line_start, line_end, body in _chunk_lines(lines):
            if is_markdown and _is_noise_chunk(body):
                continue  # near-empty md chunk, no heading/link — ingest noise
            chunk_id = _stable_id("repo-chunk", file_id, str(seq), _short_hash(body))
            chunk_payloads.append(
                {
                    "id": chunk_id,
                    "chunk_seq": seq,
                    "line_start": line_start,
                    "line_end": line_end,
                    "text_hash": _short_hash(body),
                    "body_text": body,
                }
            )

        return {
            "id": file_id,
            "path": path,
            "language": _language_for_path(path),
            "size_bytes": raw_size,
            "sha256": sha,
            "line_count": len(lines),
            "lines": line_rows,
            "chunks": chunk_payloads,
        }

    def _resolve_repo_id(self, key: str | None) -> str | None:
        if not key:
            return None
        source = self.store.get_repo_source(key)
        return source["id"] if source else None
