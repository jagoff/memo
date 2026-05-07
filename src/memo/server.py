"""MCP server — `memo-mcp` entry point.

Exposes the `Memory` API as MCP tools so any MCP-aware client (Claude
Code, Devin, Claude Desktop, etc) can save / search / list / get /
update / delete entries from natural language.

Tools are deliberately small and verb-shaped so the LLM picks them up
without prompt engineering. Each tool returns plain JSON-serialisable
dicts (no SimpleNamespaces, no dataclasses) — the MCP transport
serialises everything anyway, but a dict surfaces field names in the
tool result message which helps the LLM decide what to do next.

Run via the installed entry point:

    memo-mcp

Or programmatically:

    from memo.server import build_server
    server = build_server()
    server.run()
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from memo.config import Config
from memo.memory import AmbiguousIdError, Memory


def build_server(memory: Memory | None = None) -> FastMCP:
    """Build the MCP server. Accepts an explicit `Memory` for tests.

    The default constructs from `Config.from_env()` — production runs
    pick up `MEMO_*` env vars set by the calling shell or by Claude
    Code's `claude mcp add` invocation.
    """
    if memory is None:
        memory = Memory(Config.from_env())

    server = FastMCP("memo")

    @server.tool()
    def memory_save(
        content: str,
        title: str | None = None,
        type: str = "note",
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Persist a memory to the vault + index it.

        Args:
            content: Markdown body. Required, non-empty. The first
                non-empty line is used as title if `title` is omitted.
            title: Optional short title. Defaults to first line of
                content. Truncated to 80 chars.
            type: One of `decision`, `fact`, `bug`, `feedback`,
                `preference`, `note`, `manual`. Default `note`.
            tags: Optional list. Lower-cased + de-duplicated.

        Returns the persisted record (id, path, title, ...).
        """
        rec = memory.save(content=content, title=title, type_=type, tags=tags)
        return rec.to_dict()

    @server.tool()
    def memory_search(
        query: str,
        limit: int = 10,
        type: str | None = None,
        body_chars: int = 280,
    ) -> list[dict[str, Any]]:
        """Top-k semantic search over the memory index.

        Returns records ordered by descending similarity. Each result
        has a `score` field (0..1, higher = more similar).

        Args:
            query: Free-text query. Required, non-empty.
            limit: Max results. Defaults to 10.
            type: Optional filter by record type (e.g. only `decision`).
            body_chars: Truncate the `body` field to this many chars. The
                default keeps results compact for the LLM context — call
                `memory_get(id)` for the full body. Pass a very large
                number to disable truncation.
        """
        out: list[dict[str, Any]] = []
        for r in memory.search(query, limit=limit, type_=type):
            d = r.to_dict()
            body = d.get("body") or ""
            if body_chars >= 0 and len(body) > body_chars:
                d["body"] = body[:body_chars].rstrip() + "…"
                d["body_truncated"] = True
            out.append(d)
        return out

    @server.tool()
    def memory_list(
        limit: int = 20, type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Recent memories ordered by `updated` desc. No vector lookup
        — useful when you want to inspect "what did I save lately"
        without a search query."""
        return [r.to_dict() for r in memory.list(limit=limit, type_=type)]

    @server.tool()
    def memory_get(id: str) -> dict[str, Any] | None:
        """Fetch one memory record by id (full UUID hex or unique prefix
        ≥4 chars). Returns None when not found. On ambiguous prefix
        returns `{"error": "ambiguous", "matches": [...]}` instead of
        raising — keeps the MCP transport happy and lets the LLM read
        the candidates."""
        try:
            rec = memory.get(id)
        except AmbiguousIdError as exc:
            return {"error": "ambiguous", "prefix": exc.prefix, "matches": exc.matches}
        return rec.to_dict() if rec else None

    @server.tool()
    def memory_update(
        id: str,
        title: str | None = None,
        type: str | None = None,
        tags: list[str] | None = None,
        content: str | None = None,
    ) -> dict[str, Any] | None:
        """Patch fields on an existing memory. Re-embeds only when the
        body actually changed (saves a forward pass for retag/rename).

        Returns the updated record, or None if `id` is unknown. Tags
        replace the existing list (use `memory_get` first if you want to
        merge). On ambiguous prefix returns `{"error": "ambiguous",
        "matches": [...]}` so the LLM can surface candidates.
        """
        try:
            rec = memory.update(
                id, title=title, type_=type, tags=tags, content=content,
            )
        except AmbiguousIdError as exc:
            return {"error": "ambiguous", "prefix": exc.prefix, "matches": exc.matches}
        return rec.to_dict() if rec else None

    @server.tool()
    def memory_reindex() -> dict[str, int]:
        """Re-scan the memory dir, re-embed entries whose on-disk body
        diverged from the indexed `body_hash`. Picks up edits the user
        made to memory `.md` files in Obsidian.

        Returns `{"checked", "reindexed", "added", "skipped"}`.
        """
        return memory.reindex()

    @server.tool()
    def memory_delete(id: str) -> dict[str, Any]:
        """Delete one memory by id (full or unique prefix). Removes both
        the vec entry and the backing `.md` file. Returns
        `{"deleted": true|false}` on success, or
        `{"error": "ambiguous", "matches": [...]}` if the prefix matches
        multiple records."""
        try:
            return {"deleted": memory.delete(id)}
        except AmbiguousIdError as exc:
            return {"error": "ambiguous", "prefix": exc.prefix, "matches": exc.matches}

    @server.tool()
    def memory_history(
        limit: int = 20, op: str | None = None, id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Recent save/update/delete events from the audit log.

        Args:
            limit: Max events. Defaults to 20.
            op: Optional filter: `save`, `update`, or `delete`.
            id: Optional filter to events for one record (full id or
                unique prefix ≥4 chars).
        """
        record_id = id
        if record_id and len(record_id) < 32:
            try:
                resolved = memory.resolve_id(record_id)
            except AmbiguousIdError as exc:
                return [{"error": "ambiguous", "prefix": exc.prefix, "matches": exc.matches}]
            if resolved is None:
                return []
            record_id = resolved
        return memory.history.list_recent(limit=limit, op=op, record_id=record_id)

    @server.tool()
    def memory_stats() -> dict[str, Any]:
        """Summary stats — total records, recent counts. No body load."""
        return {
            "total": memory.store.count(),
            "vault_path": str(memory.cfg.vault_path),
            "memory_dir": str(memory.cfg.memory_dir),
            "db_path": str(memory.cfg.db_path),
            "embedder_model": memory.cfg.embedder_model,
        }

    return server


def main() -> None:
    """Entry point for `memo-mcp` console script."""
    server = build_server()
    server.run()


if __name__ == "__main__":
    main()
