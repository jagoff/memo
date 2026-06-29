"""MCP tools — entity-extraction domain (split from server.py).

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
    def memo_extract_entities(
        ids: list[str] | None = None,
        all_: bool = False,
        force: bool = False,
    ) -> dict[str, int]:
        """Extract named entities (person/project/technology/file/org/concept)
        from memory bodies via Qwen2.5-3B and write them to the graph DB.

        Args:
            ids: Specific memory ids to process (full UUID hex). Mutually
                exclusive with `all_`.
            all_: Process every memory in the store.
            force: Re-extract even if entity links already exist
                (default skips already-indexed memories).

        Returns counts: `{processed, entities_extracted, links_written, skipped, errors}`.
        Cost: ~0.5-1s per memory. Use `all_=True` once after a fresh
        install, then incrementally on new memories.
        """
        return memory.extract_entities(
            ids=ids,
            all_=all_,
            skip_already_indexed=not force,
        )

    @server.tool()
    def memo_entities(
        limit: int = 30,
        type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Top entities in the knowledge graph, ranked by mention count.

        Args:
            limit: Max entities. Default 30.
            type: Optional filter (`person`/`project`/`technology`/
                `file`/`org`/`concept`).
        """
        return memory.graph.top_entities(limit=limit, type_=type)

    @server.tool()
    def memo_entity(name: str, type: str | None = None) -> list[str]:
        """Memory IDs that mention `name` (and optionally a specific
        entity type). Returns a list of full UUIDs."""
        return memory.graph.entity_memories(name, type_=type)
