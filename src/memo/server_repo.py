"""MCP tools — repo domain (split from server.py).

Registered by `build_server()` via `register(server, memory)`. Tool names,
signatures, defaults, docstrings and bodies are identical to the originals;
only the enclosing function and indentation changed.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from memo.memory import Memory

def register(server: FastMCP, memory: Memory) -> None:
    @server.tool()
    def memory_repo_index(
        url: str,
        name: str | None = None,
        ref: str | None = None,
        force: bool = False,
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
        return memory.repo_index(
            url,
            name=name,
            ref=ref,
            force=force,
            with_embeddings=with_embeddings,
            include=include,
            exclude=exclude,
            max_file_bytes=max_file_bytes,
        )

    @server.tool()
    def memory_repo_embed(repo: str, force: bool = False) -> dict[str, Any]:
        """Embed pending repo chunks. Runs automatically during repo index by default."""
        return memory.repo_embed(repo, force=force)

    @server.tool()
    def memory_repo_status(repo: str) -> dict[str, Any] | None:
        """Return exact and semantic index counts for one repo."""
        return memory.repo_status(repo)

    @server.tool()
    def memory_repo_search(
        query: str,
        limit: int = 10,
        repo: str | None = None,
        path: str | None = None,
        mode: str = "hybrid",
    ) -> list[dict[str, Any]]:
        """Search indexed repositories.

        Modes: `hybrid` fuses chunk embeddings, chunk BM25, and line
        BM25; `vec` is semantic chunk search; `bm25` is keyword chunk
        search; `line` searches the exact per-line index.
        """
        return [
            hit.to_dict()
            for hit in memory.repo_search(
                query, limit=limit, repo=repo, path=path, mode=mode,
            )
        ]

    @server.tool()
    def memory_repo_get_file(
        repo: str,
        path: str,
        start: int | None = None,
        end: int | None = None,
    ) -> dict[str, Any] | None:
        """Return indexed text for one repo file or line range."""
        return memory.repo_get_file(repo, path, start=start, end=end)

    @server.tool()
    def memory_repo_list(limit: int = 100) -> list[dict[str, Any]]:
        """List indexed repositories."""
        return memory.repo_list(limit=limit)

    @server.tool()
    def memory_repo_delete(repo: str, remove_clone: bool = True) -> dict[str, bool]:
        """Delete one indexed repo and optionally remove memo's managed clone."""
        return {"deleted": memory.repo_delete(repo, remove_clone=remove_clone)}
