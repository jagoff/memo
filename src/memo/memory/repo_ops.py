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
        return self._repo_corpus().index(
            url,
            name=name,
            ref=ref,
            force=force,
            with_embeddings=with_embeddings,
            include=include,
            exclude=exclude,
            max_file_bytes=max_file_bytes,
            progress=progress,
        )

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

