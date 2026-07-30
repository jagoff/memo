"""MCP tools — repo domain (split from server.py).

Registered by `build_server()` via `register(server, memory)`. Tool names,
signatures, defaults, docstrings and bodies are identical to the originals;
only the enclosing function and indentation changed.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastmcp import Context, FastMCP
from pydantic import Field

from memo.memory import Memory
from memo.server_annotations import DESTRUCTIVE, READ_ONLY, WRITE_IDEMPOTENT, annotated_tool


def register(server: FastMCP, memory: Memory) -> None:
    @annotated_tool(server, **WRITE_IDEMPOTENT)
    def memo_repo_index(
        url: Annotated[
            str,
            Field(
                description="Git clone URL (https, ssh, scp-like, or local path) passed to "
                "local `git clone`. Remote-helper schemes (`ext::`, `fd::`) and leading-dash "
                "URLs are rejected."
            ),
        ],
        name: Annotated[
            str | None,
            Field(
                description="Repo name override; None derives it from the URL. Sanitized to "
                "lowercase [a-z0-9._-] (max 80 chars); an explicit name already used by a "
                "different repo is an error."
            ),
        ] = None,
        ref: Annotated[
            str | None,
            Field(
                description="Branch, tag, or commit to check out (detached); None or empty "
                "indexes the remote's default branch (HEAD). Leading-dash refs are rejected."
            ),
        ] = None,
        force: Annotated[
            bool,
            Field(
                description="Re-index files even when the commit and per-file sha256 are "
                "unchanged; default skips an already-indexed identical state."
            ),
        ] = False,
        refresh: Annotated[
            bool,
            Field(
                description="Refresh an existing checkout incrementally even when HEAD is "
                "unchanged; only files whose sha256 changed are rewritten."
            ),
        ] = False,
        with_embeddings: Annotated[
            bool,
            Field(
                description="Embed new chunks right after indexing; False leaves the semantic "
                "index pending (run memo_repo_embed later)."
            ),
        ] = True,
        include: Annotated[
            list[str] | None,
            Field(
                description="fnmatch globs over repo-relative paths; when set, only matching "
                "files are indexed. None indexes every tracked text file."
            ),
        ] = None,
        exclude: Annotated[
            list[str] | None,
            Field(
                description="Extra fnmatch globs to skip, added on top of built-in excludes "
                "(binary/media/archive extensions and vendor/build/VCS dirs like node_modules, "
                ".git, dist)."
            ),
        ] = None,
        max_file_bytes: Annotated[
            int | None,
            Field(
                description="Skip files larger than this many bytes; None uses the "
                "MEMO_REPO_MAX_FILE_BYTES env flag or the 2,000,000-byte default."
            ),
        ] = None,
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
    def memo_repo_embed(
        repo: Annotated[
            str,
            Field(description="Repo id, name, or URL of an indexed repo; unknown repo errors."),
        ],
        force: Annotated[
            bool,
            Field(
                description="Re-embed every chunk instead of only chunks missing a stored "
                "vector; cached embeddings are still reused."
            ),
        ] = False,
    ) -> dict[str, Any]:
        """Embed pending repo chunks. Runs automatically during repo index by default.

        Idempotent: without `force` only chunks lacking a stored vector are
        embedded, so re-running on a fully embedded repo writes nothing.
        """
        return memory.repo_embed(repo, force=force)

    @annotated_tool(server, **READ_ONLY)
    def memo_repo_status(
        repo: Annotated[
            str,
            Field(description="Repo id, name, or URL; unknown repo returns null."),
        ],
        paths: Annotated[
            list[str] | None,
            Field(
                description="Exact repo-relative files whose current bytes should be checked "
                "against the recorded index generation; None skips byte-level freshness."
            ),
        ] = None,
        scopes: Annotated[
            list[str] | None,
            Field(
                description="Repo-relative directory prefixes used to filter recorded coverage "
                "gaps; None reports repository-wide coverage."
            ),
        ] = None,
    ) -> dict[str, Any] | None:
        """Return index counts plus coverage/freshness evidence for one repo."""
        if paths is None and scopes is None:
            return memory.repo_status(repo)
        return memory.repo_status(repo, paths=paths, scopes=scopes)

    @annotated_tool(server, **READ_ONLY)
    def memo_repo_search(
        query: Annotated[
            str,
            Field(description="Search text; empty or whitespace-only returns no hits."),
        ],
        limit: Annotated[
            int,
            Field(description="Maximum hits returned after fusion and boosting."),
        ] = 10,
        repo: Annotated[
            str | None,
            Field(
                description="Restrict to one repo by id, name, or URL; an unknown repo returns "
                "no hits. None searches every indexed repo."
            ),
        ] = None,
        path: Annotated[
            str | None,
            Field(
                description="SQLite GLOB pattern over repo-relative file paths "
                "(e.g. 'src/*.py'); None searches all files."
            ),
        ] = None,
        mode: Annotated[
            str,
            Field(
                description="One of 'hybrid', 'lexical', 'unified', 'vec', 'bm25', or 'line'. "
                "'unified' adds CodeGraph and Git co-change channels; without embeddings it "
                "continues with available lexical and structural providers."
            ),
        ] = "hybrid",
        scope: Annotated[
            str,
            Field(
                description="Path class filter: 'all', 'production', 'tests', or 'vendor'."
            ),
        ] = "all",
        include_evidence: Annotated[
            bool,
            Field(
                description="Attach a coverage/freshness evidence envelope for each hit path."
            ),
        ] = True,
    ) -> list[dict[str, Any]]:
        """Search indexed repositories.

        `unified` adds CodeGraph and Git co-change providers to the
        semantic/lexical fusion. Scope can isolate production, tests, or
        vendored code without treating missing structural data as negative
        evidence.
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
        repo: Annotated[
            str,
            Field(description="Repo id, name, or URL; unknown repo returns null."),
        ],
        path: Annotated[
            str,
            Field(
                description="Repo-relative file path exactly as indexed; unknown path returns null."
            ),
        ],
        start: Annotated[
            int | None,
            Field(
                description="First line number to return (1-based, clamped to >= 1); "
                "None starts at line 1."
            ),
        ] = None,
        end: Annotated[
            int | None,
            Field(
                description="Last line number to return (inclusive, clamped to >= start); "
                "None reads to end of file."
            ),
        ] = None,
    ) -> dict[str, Any] | None:
        """Return indexed text for one repo file or line range."""
        return memory.repo_get_file(repo, path, start=start, end=end)

    @annotated_tool(server, **READ_ONLY)
    def memo_repo_list(
        limit: Annotated[
            int,
            Field(description="Maximum repos returned, most recently indexed first."),
        ] = 100,
    ) -> list[dict[str, Any]]:
        """List indexed repositories."""
        return memory.repo_list(limit=limit)

    @annotated_tool(server, **DESTRUCTIVE)
    async def memo_repo_delete(
        repo: Annotated[
            str,
            Field(description="Repo id, name, or URL; unknown repo returns deleted=false."),
        ],
        remove_clone: Annotated[
            bool,
            Field(
                description="Also delete memo's managed on-disk clone directory; False keeps "
                "the clone and removes only the index rows."
            ),
        ] = True,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Delete one indexed repo and optionally remove memo's managed clone.

        Removal is permanent (elicitation-capable clients are asked to
        confirm); an unknown repo returns deleted=false instead of raising.
        """
        from memo.server_elicit import abort_result, confirm_destructive, sanitize_fragment

        try:
            source = memory.store.get_repo_source(repo)
        except Exception:
            source = None
        if source is not None:
            name = sanitize_fragment(source.get("name") or repo)
            clone_path = source.get("clone_path") if remove_clone else None
            blast = (
                f" and its on-disk clone at {sanitize_fragment(clone_path, limit=200)}"
                if clone_path
                else " from the index"
            )
            gate = await confirm_destructive(
                ctx,
                action="delete",
                detail=f"Delete indexed repo '{name}'{blast}? Removal is permanent.",
            )
            if not gate.proceed:
                return abort_result(
                    gate,
                    memory,
                    tool="memo_repo_delete",
                    action="delete",
                    target=f"repo '{name}'",
                )
        # Bind execution to the row the user confirmed: `repo` is id-or-name-
        # or-URL and a re-resolve after the human-latency confirmation window
        # could match a different row (re-indexed same-name repo) and rmtree a
        # clone the user never saw.
        resolved = str(source["id"]) if source is not None and source.get("id") else repo
        return {"deleted": memory.repo_delete(resolved, remove_clone=remove_clone)}
