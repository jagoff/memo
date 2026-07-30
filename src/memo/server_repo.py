"""MCP tools — repo domain (split from server.py).

Registered by `build_server()` via `register(server, memory)`. Tool names,
signatures, defaults, docstrings and bodies are identical to the originals;
only the enclosing function and indentation changed.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from memo.memory import Memory
from memo.server_annotations import DESTRUCTIVE, READ_ONLY, WRITE_IDEMPOTENT, annotated_tool


def register(server: FastMCP, memory: Memory) -> None:
    @annotated_tool(server, **WRITE_IDEMPOTENT)
    def memo_repo_index(
        url: str,
        name: str | None = None,
        ref: str | None = None,
        force: bool = False,
        refresh: bool = False,
        with_embeddings: bool = True,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        max_file_bytes: int | None = None,
    ) -> dict[str, Any]:
        """Clone/fetch a Git repo and index all included text files.

        Auth is delegated to local `git`: public repos work directly;
        private repos work when SSH agent / credential helpers / tokens
        already let `git clone <url>` succeed on this machine.
        """
        kwargs: dict[str, Any] = {
            "name": name,
            "ref": ref,
            "force": force,
            "with_embeddings": with_embeddings,
            "include": include,
            "exclude": exclude,
            "max_file_bytes": max_file_bytes,
        }
        if refresh:
            kwargs["refresh"] = True
        return memory.repo_index(url, **kwargs)

    @annotated_tool(server, **WRITE_IDEMPOTENT)
    def memo_repo_embed(repo: str, force: bool = False) -> dict[str, Any]:
        """Embed pending repo chunks. Runs automatically during repo index by default."""
        return memory.repo_embed(repo, force=force)

    @annotated_tool(server, **READ_ONLY)
    def memo_repo_status(
        repo: str,
        paths: list[str] | None = None,
        scopes: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Return index counts plus coverage/freshness evidence for one repo.

        `paths` requests byte-level freshness for exact files. `scopes`
        restricts recorded coverage gaps to directory prefixes.
        """
        if paths is None and scopes is None:
            return memory.repo_status(repo)
        return memory.repo_status(repo, paths=paths, scopes=scopes)

    @annotated_tool(server, **READ_ONLY)
    def memo_repo_search(
        query: str,
        limit: int = 10,
        repo: str | None = None,
        path: str | None = None,
        mode: str = "hybrid",
        scope: str = "all",
        include_evidence: bool = True,
    ) -> list[dict[str, Any]]:
        """Search indexed repositories.

        Modes: `unified` adds CodeGraph and Git co-change providers to the
        semantic/lexical fusion; `hybrid` fuses embeddings and lexical
        channels; `lexical` is BM25 + exact lines; `vec`, `bm25`, and
        `line` select one channel. Scope is `all`, `production`, `tests`,
        or `vendor`.
        """
        search_kwargs: dict[str, Any] = {
            "limit": limit,
            "repo": repo,
            "path": path,
            "mode": mode,
        }
        if scope != "all":
            search_kwargs["scope"] = scope
        hits = memory.repo_search(query, **search_kwargs)
        out: list[dict[str, Any]] = []
        evidence_cache: dict[tuple[str, str], dict[str, Any]] = {}
        for hit in hits:
            row = hit.to_dict()
            if include_evidence:
                repo_key = str(repo or row.get("repo_id") or row.get("repo_name") or "")
                hit_path = str(row.get("path") or "")
                cache_key = (repo_key, hit_path)
                if cache_key not in evidence_cache:
                    evidence_cache[cache_key] = memory.repo_evidence(
                        repo_key,
                        paths=[hit_path] if hit_path else None,
                    )
                row["code_evidence"] = evidence_cache[cache_key]
            out.append(row)
        return out

    @annotated_tool(server, **READ_ONLY)
    def memo_repo_get_file(
        repo: str,
        path: str,
        start: int | None = None,
        end: int | None = None,
    ) -> dict[str, Any] | None:
        """Return indexed text for one repo file or line range."""
        return memory.repo_get_file(repo, path, start=start, end=end)

    @annotated_tool(server, **READ_ONLY)
    def memo_repo_list(limit: int = 100) -> list[dict[str, Any]]:
        """List indexed repositories."""
        return memory.repo_list(limit=limit)

    @annotated_tool(server, **DESTRUCTIVE)
    def memo_repo_delete(repo: str, remove_clone: bool = True) -> dict[str, bool]:
        """Delete one indexed repo and optionally remove memo's managed clone."""
        return {"deleted": memory.repo_delete(repo, remove_clone=remove_clone)}
