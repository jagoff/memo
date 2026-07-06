"""MCP tools — wikilinks domain (split from server.py).

Registered by `build_server()` via `register(server, memory)`. Tool names,
signatures, defaults, docstrings and bodies are identical to the originals;
only the enclosing function and indentation changed.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from fastmcp import FastMCP

from memo.memory import Memory
from memo.server_annotations import READ_ONLY, annotated_tool


def register(server: FastMCP, memory: Memory) -> None:
    @annotated_tool(server, **READ_ONLY)
    def memo_links_backlinks(
        memory_id: str,
    ) -> list[dict[str, Any]]:
        """Show all memories that reference this one.

        Returns backlinks (incoming links) to the specified memory.
        Useful for understanding how a memory is connected to others.

        Args:
            memory_id: The memory ID to find backlinks for.
        """
        backlinks = memory.crossref.get_backlinks(memory_id)
        return [dataclasses.asdict(b) for b in backlinks]

    @annotated_tool(server, **READ_ONLY)
    def memo_links_outlinks(
        memory_id: str,
    ) -> list[dict[str, Any]]:
        """Show all memories that this one references.

        Returns outlinks (outgoing links) from the specified memory.
        Useful for understanding what a memory connects to.

        Args:
            memory_id: The memory ID to find outlinks for.
        """
        outlinks = memory.crossref.get_outlinks(memory_id)
        return [dataclasses.asdict(o) for o in outlinks]

    @annotated_tool(server, **READ_ONLY)
    def memo_links_suggest(
        content: str,
        title: str | None = None,
        tags: list[str] | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Suggest links to existing memories based on content.

        Returns suggested memories to link to, based on semantic similarity
        and other heuristics. Useful when saving a new memory to discover
        related existing content.

        Args:
            content: The memory content being saved.
            title: Optional title of the memory.
            tags: Optional tags of the memory.
            limit: Maximum suggestions to return.
        """
        suggestions = memory.link_suggester.suggest_links(
            content=content,
            title=title or "",
            tags=tags or [],
            limit=limit,
        )
        return [dataclasses.asdict(s) for s in suggestions]

    @annotated_tool(server, **READ_ONLY)
    def memo_links_format(
        memory_id: str,
        title: str | None = None,
    ) -> str:
        """Format a memory ID as a wikilink.

        Returns a wikilink string like [[memory-id]] or [[memory-id|Title]].
        Use this to insert links into memory content.

        Args:
            memory_id: The memory ID to format as a wikilink.
            title: Optional display title for the link.
        """
        return memory.link_suggester.format_wikilink(memory_id, title)
