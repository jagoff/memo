"""Repo-corpus operations for `Memory` — index / embed / status / search.

`_RepoOpsMixin` holds the repository-indexing surface (clone, embed, search,
get-file, list, delete) and the `_repo_corpus` accessor, moved verbatim from
the former `memory.py` god-file.
"""

from __future__ import annotations

from typing import Any

from memo.memory._base import _MemoryBase


class _RepoOpsMixin(_MemoryBase):
    # -- repo corpus -------------------------------------------------------

    def _repo_corpus(self):
        from memo.repo_index import RepoCorpus

        return RepoCorpus(self.cfg, store=self.store, embedder=self.embedder)

    def repo_index(
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
        progress=None,
    ) -> dict[str, Any]:
        index_kwargs: dict[str, Any] = dict(
            name=name,
            ref=ref,
            force=force,
            with_embeddings=with_embeddings,
            include=include,
            exclude=exclude,
            max_file_bytes=max_file_bytes,
        )
        # Optional off-request path: when MEMO_INGEST_VIA_DAEMON=1 and the
        # ingest daemon is reachable, hand the batch index to its serialized
        # writer (returns a job_id; poll via `memo ingest-daemon status`).
        # `progress` callbacks can't cross the socket, so daemon routing is
        # only used when no progress sink is requested. Any miss (flag off,
        # daemon down, progress requested) runs in-process exactly as before.
        from memo.flags import flag_bool

        if progress is None and flag_bool("MEMO_INGEST_VIA_DAEMON"):
            from memo import ingest_client

            job_id = ingest_client.enqueue("repo", {"url": url, **index_kwargs})
            if job_id is not None:
                return {"queued": True, "job_id": job_id, "via": "ingest-daemon"}
            # daemon unreachable → fall through to in-process (graceful)
        return self._repo_corpus().index(url, progress=progress, **index_kwargs)

    def repo_embed(self, repo: str, *, force: bool = False, progress=None) -> dict[str, Any]:
        return self._repo_corpus().embed(repo, force=force, progress=progress)

    def repo_status(self, repo: str) -> dict[str, Any] | None:
        return self._repo_corpus().status(repo)

    def repo_search(
        self,
        query: str,
        *,
        limit: int = 10,
        repo: str | None = None,
        path: str | None = None,
        mode: str = "hybrid",
    ):
        return self._repo_corpus().search(
            query, limit=limit, repo=repo, path=path, mode=mode,
        )

    def repo_get_file(
        self,
        repo: str,
        path: str,
        *,
        start: int | None = None,
        end: int | None = None,
    ) -> dict[str, Any] | None:
        return self._repo_corpus().get_file(repo, path, start=start, end=end)

    def repo_list(self, *, limit: int = 100) -> list[dict[str, Any]]:
        return self._repo_corpus().list(limit=limit)

    def repo_delete(self, repo: str, *, remove_clone: bool = True) -> bool:
        return self._repo_corpus().delete(repo, remove_clone=remove_clone)

