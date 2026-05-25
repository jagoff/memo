"""Git repository ingestion and retrieval for memo.

This subsystem is intentionally separate from curated memorias. Repo
content is source code / document corpus material: it can be large, it
changes by commit, and callers need line-accurate retrieval rather than
one note-shaped Markdown file per record.
"""

from __future__ import annotations

import contextlib
import fnmatch
import hashlib
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from memo.config import Config
from memo.embedder import MLXEmbedder, assert_valid_embedding
from memo.store import VecStore

DEFAULT_EXCLUDE_DIRS = frozenset({
    ".git", ".hg", ".svn",
    ".claude", ".codex", ".devin",
    ".venv", "venv", "env",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "node_modules", ".next", ".nuxt", ".turbo",
    "dist", "build", "target", "coverage",
    ".idea", ".vscode",
})

DEFAULT_EXCLUDE_GLOBS = (
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp", "*.ico",
    "*.pdf", "*.zip", "*.gz", "*.bz2", "*.xz", "*.7z", "*.tar",
    "*.sqlite", "*.sqlite3", "*.db", "*.dylib", "*.so", "*.a",
    "*.pyc", "*.class", "*.o", "*.wasm",
)

DEFAULT_MAX_FILE_BYTES = 2_000_000
DEFAULT_CHUNK_TARGET_CHARS = 3500
DEFAULT_CHUNK_OVERLAP_LINES = 8
DEFAULT_EMBED_BATCH = 64
MIN_EMBED_BATCH = 1
DEFAULT_FLUSH_BATCH = 25
MIN_FLUSH_BATCH = 1
STATUS_INDEXING = "indexing"

ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class RepoSearchHit:
    id: str
    repo_id: str
    repo_name: str
    url: str
    ref: str
    commit_sha: str
    file_id: str
    path: str
    language: str
    line_start: int
    line_end: int
    text: str
    score: float | None
    match_type: str

    @property
    def locator(self) -> str:
        commit = self.commit_sha[:8] if self.commit_sha else "unknown"
        return f"repo:{self.repo_name}:{self.path}:{self.line_start}-{self.line_end}@{commit}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "repo_id": self.repo_id,
            "repo_name": self.repo_name,
            "url": self.url,
            "ref": self.ref,
            "commit_sha": self.commit_sha,
            "file_id": self.file_id,
            "path": self.path,
            "language": self.language,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "text": self.text,
            "score": self.score,
            "match_type": self.match_type,
            "locator": self.locator,
        }


@dataclass(frozen=True)
class RepoEmbedInput:
    chunk: dict[str, Any]
    text: str
    input_hash: str


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
        self.store = store or VecStore(cfg.db_path, dims=cfg.embedder_dims)
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
            max_bytes = int(os.environ.get("MEMO_REPO_MAX_FILE_BYTES", DEFAULT_MAX_FILE_BYTES))
        include_globs = list(include or [])
        exclude_globs = [*DEFAULT_EXCLUDE_GLOBS, *(exclude or [])]

        existing_files = self.store.repo_file_hashes(repo_id)
        seen_paths: set[str] = set()
        pending_files: list[dict[str, Any]] = []
        checked = unchanged = skipped_binary = skipped_excluded = skipped_too_large = errors = 0
        indexed_files_total = indexed_chunks = indexed_lines = 0
        flushed_files = 0

        indexed_at = _now_iso()
        flush_batch = _repo_flush_batch_size()

        # Persist target commit + indexing status BEFORE the scan so a
        # partial run is recoverable: file-batch commits below remain on
        # disk and the next call resumes via the sha256 short-circuit.
        self.store.upsert_repo_source({
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
        })

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
            try:
                size = path.stat().st_size
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
                seen_paths.add(rel_posix)

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
                errors += 1
                _emit(progress, "file_error", path=rel_posix)

        _flush()

        deleted_file_ids = [
            meta["id"]
            for path, meta in existing_files.items()
            if path not in seen_paths
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

        if mode == "line":
            return _hits_from_rows(
                self.store.search_repo_lines(query, limit=limit, repo_id=repo_id, path_glob=path)
            )
        if mode == "bm25":
            return _hits_from_rows(
                self.store.search_repo_bm25(query, limit=limit, repo_id=repo_id, path_glob=path)
            )

        query_terms = _extract_query_terms(query)

        if repo_id is not None and self.store.repo_counts(repo_id)["embedded_chunks"] == 0:
            if mode == "vec":
                return []
            rows = _rrf_fuse_repo(
                [
                    self.store.search_repo_bm25(query, limit=max(limit * 2, 30), repo_id=repo_id, path_glob=path),
                    self.store.search_repo_lines(query, limit=max(limit * 2, 30), repo_id=repo_id, path_glob=path),
                ],
                limit=limit,
                query_terms=query_terms,
            )
            return _hits_from_rows(rows)

        emb = self.embedder.embed_query(query)
        assert_valid_embedding(emb, self.cfg.embedder_dims, context="repo search query")

        if mode == "vec":
            return _hits_from_rows(
                self.store.search_repo_vec(emb, limit=limit, repo_id=repo_id, path_glob=path)
            )

        input_k = max(limit * 2, 30)
        vec_hits = self.store.search_repo_vec(emb, limit=input_k, repo_id=repo_id, path_glob=path)
        bm_hits = self.store.search_repo_bm25(query, limit=input_k, repo_id=repo_id, path_glob=path)
        line_hits = self.store.search_repo_lines(query, limit=input_k, repo_id=repo_id, path_glob=path)
        return _hits_from_rows(_rrf_fuse_repo(
            [vec_hits, bm_hits, line_hits],
            limit=limit,
            query_terms=query_terms,
        ))

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
        for seq, line_start, line_end, body in _chunk_lines(lines):
            chunk_id = _stable_id("repo-chunk", file_id, str(seq), _short_hash(body))
            chunk_payloads.append({
                "id": chunk_id,
                "chunk_seq": seq,
                "line_start": line_start,
                "line_end": line_end,
                "text_hash": _short_hash(body),
                "body_text": body,
            })

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


def _git(args: list[str], *, check: bool = True) -> str:
    try:
        proc = subprocess.run(args, check=False, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("`git` not found on PATH") from exc
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"git command failed ({proc.returncode}): {' '.join(args)}\n{detail}")
    return proc.stdout


def _emit(progress: ProgressCallback | None, event: str, **data: Any) -> None:
    if progress is not None:
        progress(event, data)


def _repo_embed_input(chunk: dict[str, Any]) -> RepoEmbedInput:
    text = (
        f"repo: {chunk['repo_name']}\npath: {chunk['path']}\n"
        f"lines: {chunk['line_start']}-{chunk['line_end']}\n\n{chunk['body_text']}"
    )
    return RepoEmbedInput(
        chunk=chunk,
        text=text,
        input_hash=hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest(),
    )


def _repo_embed_batch_size() -> int:
    raw = os.environ.get("MEMO_REPO_EMBED_BATCH")
    if raw:
        with contextlib.suppress(ValueError):
            return max(MIN_EMBED_BATCH, int(raw))
    return DEFAULT_EMBED_BATCH


def _repo_flush_batch_size() -> int:
    """Number of files to accumulate before flushing to the store.

    Lower values trade write overhead for finer-grained resume
    granularity if the run is interrupted.
    """
    raw = os.environ.get("MEMO_REPO_FLUSH_BATCH")
    if raw:
        with contextlib.suppress(ValueError):
            return max(MIN_FLUSH_BATCH, int(raw))
    return DEFAULT_FLUSH_BATCH


def _embed_cache_model(embedder: MLXEmbedder, cfg: Config) -> str:
    model = getattr(embedder, "model_path", None)
    return str(model or cfg.embedder_model)


def _tracked_files(clone_path: Path) -> list[str]:
    """Return Git-tracked files without walking generated/untracked trees."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(clone_path), "ls-files", "-z"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("`git` not found on PATH") from exc
    if proc.returncode == 0:
        paths = [p for p in proc.stdout.split("\0") if p]
        return sorted(paths)
    return sorted(
        p.relative_to(clone_path).as_posix()
        for p in clone_path.rglob("*")
        if p.is_file()
    )


def _semantic_status(current: str | None, counts: dict[str, int]) -> str:
    if counts["chunks"] == counts["embedded_chunks"]:
        return "semantic_ready"
    if current == "semantic_indexing":
        return "semantic_indexing"
    return "semantic_pending"


def _derive_repo_name(url: str) -> str:
    s = url.rstrip("/").removesuffix(".git")
    return re.split(r"[/:\s]+", s)[-1] or "repo"


def _safe_repo_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9._-]+", "-", name)
    return name.strip("-._")[:80]


def _stable_id(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8", errors="replace"))
        h.update(b"\0")
    return h.hexdigest()[:32]


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _now_iso() -> str:
    return datetime.now(tz=UTC).astimezone().isoformat(timespec="milliseconds")


def _looks_binary(raw: bytes) -> bool:
    sample = raw[:8192]
    return b"\0" in sample


def _is_excluded(rel: Path, rel_posix: str, include: list[str], exclude: list[str]) -> bool:
    if any(part in DEFAULT_EXCLUDE_DIRS for part in rel.parts):
        return True
    if include and not any(fnmatch.fnmatch(rel_posix, pat) for pat in include):
        return True
    return any(fnmatch.fnmatch(rel_posix, pat) for pat in exclude)


def _chunk_lines(
    lines: list[str],
    *,
    target_chars: int = DEFAULT_CHUNK_TARGET_CHARS,
    overlap_lines: int = DEFAULT_CHUNK_OVERLAP_LINES,
) -> list[tuple[int, int, int, str]]:
    if not lines:
        return []

    chunks: list[tuple[int, int, int, str]] = []
    seq = 0
    start = 0
    while start < len(lines):
        # Minified/generated files often contain one very long line. Keep
        # the exact full line in repo_lines, but never hand that whole
        # line to the embedder as one sequence.
        if len(lines[start]) > target_chars:
            for part in _split_long_line(lines[start], target_chars):
                chunks.append((seq, start + 1, start + 1, part))
                seq += 1
            start += 1
            continue

        end = start
        chars = 0
        while end < len(lines) and chars < target_chars:
            if len(lines[end]) > target_chars:
                break
            chars += len(lines[end]) + 1
            end += 1
        if end == start:
            # Defensive: the long-line branch above should handle this,
            # but guarantee forward progress if target_chars is tiny.
            end += 1
        body = "\n".join(lines[start:end])
        chunks.append((seq, start + 1, end, body))
        seq += 1
        if end >= len(lines):
            break
        start = max(start + 1, end - overlap_lines)
    return chunks


def _split_long_line(line: str, target_chars: int) -> list[str]:
    target = max(1, target_chars)
    return [line[i : i + target] for i in range(0, len(line), target)]


def _language_for_path(path: str) -> str:
    suffix = Path(path).suffix.lower().lstrip(".")
    return suffix or "text"


def _hits_from_rows(rows: list[dict[str, Any]]) -> list[RepoSearchHit]:
    return [
        RepoSearchHit(
            id=r["id"],
            repo_id=r["repo_id"],
            repo_name=r["repo_name"],
            url=r["url"],
            ref=r["ref"],
            commit_sha=r["commit_sha"],
            file_id=r["file_id"],
            path=r["path"],
            language=r.get("language") or "",
            line_start=int(r["line_start"]),
            line_end=int(r["line_end"]),
            text=r.get("body_text") or "",
            score=r.get("score"),
            match_type=r.get("match_type") or "chunk",
        )
        for r in rows
    ]


# Paths that should NEVER get a filename boost — these are ingest dumps
# (e.g. raw Claude/agent transcripts) that mention canonical names many
# times but are not themselves the canonical source. Boosting them
# defeats the purpose of preferring `99-Contacts/Grecia.md` over a dump
# whose filename happens to also include "Grecia".
_INGEST_PATH_MARKERS = (
    "99-obsidian/99-AI/external-ingest/",
    "external-ingest/",
)

# Lightweight stopword + token-min-length guard. We strip puncuation,
# lowercase, and drop tokens shorter than 3 chars so noise like "es",
# "de", or "?" doesn't trigger spurious path-name boosts.
_QUERY_TERM_MIN_LEN = 3
_QUERY_TERM_STOPWORDS = frozenset({
    "the", "and", "for", "with", "que", "los", "las", "una", "del",
    "como", "qué", "cuál", "quién", "donde", "cuando", "este", "esta",
})


def _extract_query_terms(query: str) -> list[str]:
    """Tokenize a query into significant terms for path-name boosting.

    Lowercased, punctuation-stripped, stopwords + short tokens removed.
    Used to compare query intent against path basenames in the post-RRF
    boost — not used for FTS5 matching (FTS5 has its own tokenizer).
    """
    if not query:
        return []
    raw = re.split(r"[^\w]+", query.lower(), flags=re.UNICODE)
    return [
        tok for tok in raw
        if len(tok) >= _QUERY_TERM_MIN_LEN and tok not in _QUERY_TERM_STOPWORDS
    ]


def _path_name_boost(path: str, terms: list[str]) -> float:
    """Compute a boost in [0.0, 1.0] for filename-match relevance.

    1.0: basename (sans extension) matches a query term exactly.
    0.5: basename contains a query term as substring.
    0.0: no match, OR path lies under an ingest-dump prefix (noisy).
    """
    if not path or not terms:
        return 0.0
    if any(marker in path for marker in _INGEST_PATH_MARKERS):
        return 0.0
    basename = Path(path).stem.lower()  # strips dir + extension
    if not basename:
        return 0.0
    for term in terms:
        if basename == term:
            return 1.0
    for term in terms:
        if term in basename:
            return 0.5
    return 0.0


def _rrf_fuse_repo(
    hit_lists: list[list[dict[str, Any]]],
    *,
    limit: int,
    k: int = 60,
    query_terms: list[str] | None = None,
) -> list[dict[str, Any]]:
    fused: dict[str, float] = {}
    canon: dict[str, dict[str, Any]] = {}
    for hits in hit_lists:
        for rank, hit in enumerate(hits):
            rid = hit["id"]
            fused[rid] = fused.get(rid, 0.0) + 1.0 / (k + rank + 1)
            canon.setdefault(rid, hit)
    if query_terms:
        for rid, score in list(fused.items()):
            boost = _path_name_boost(canon[rid].get("path") or "", query_terms)
            if boost:
                fused[rid] = score * (1.0 + boost)
    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    out: list[dict[str, Any]] = []
    for rid, score in ranked:
        d = dict(canon[rid])
        d["score"] = score
        out.append(d)
    return out
