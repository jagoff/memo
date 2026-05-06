"""MCP server — `mem-lmx-mcp` entry point.

Exposes the `Memory` API as MCP tools so any MCP-aware client (Claude
Code, Devin, Claude Desktop, etc) can save / search / list / get /
update / delete entries from natural language.

Tools are deliberately small and verb-shaped so the LLM picks them up
without prompt engineering. Each tool returns plain JSON-serialisable
dicts (no SimpleNamespaces, no dataclasses) — the MCP transport
serialises everything anyway, but a dict surfaces field names in the
tool result message which helps the LLM decide what to do next.

Run via the installed entry point:

    mem-lmx-mcp

Or programmatically:

    from mem_lmx.server import build_server
    server = build_server()
    server.run()
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from mem_lmx.config import Config
from mem_lmx.memory import Memory


def build_server(memory: Memory | None = None) -> FastMCP:
    """Build the MCP server. Accepts an explicit `Memory` for tests.

    The default constructs from `Config.from_env()` — production runs
    pick up `MEM_LMX_*` env vars set by the calling shell or by Claude
    Code's `claude mcp add` invocation.
    """
    if memory is None:
        memory = Memory(Config.from_env())

    server = FastMCP("mem-lmx")

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
        query: str, limit: int = 10, type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Top-k semantic search over the memory index.

        Returns records ordered by descending similarity. Each result
        has a `score` field (0..1, higher = more similar).

        Args:
            query: Free-text query. Required, non-empty.
            limit: Max results. Defaults to 10.
            type: Optional filter by record type (e.g. only `decision`).
        """
        return [r.to_dict() for r in memory.search(query, limit=limit, type_=type)]

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
        """Fetch one memory record by id. Returns None when not found."""
        rec = memory.get(id)
        return rec.to_dict() if rec else None

    @server.tool()
    def memory_delete(id: str) -> dict[str, bool]:
        """Delete one memory by id. Removes both the vec entry and the
        backing `.md` file. Returns `{"deleted": true|false}`."""
        return {"deleted": memory.delete(id)}

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
    """Entry point for `mem-lmx-mcp` console script."""
    server = build_server()
    server.run()


if __name__ == "__main__":
    main()
