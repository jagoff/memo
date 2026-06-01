"""MCP tools — wikilinks domain (split from server.py).

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
    def memory_links_backlinks(
        memoria_id: str,
    ) -> list[dict[str, Any]]:
        """Show all memorias that reference this one.

        Returns backlinks (incoming links) to the specified memoria.
        Useful for understanding how a memoria is connected to others.

        Args:
            memoria_id: The memoria ID to find backlinks for.
        """
        backlinks = memory.crossref.get_backlinks(memoria_id)
        return [b.__dict__ for b in backlinks]

    @server.tool()
    def memory_links_outlinks(
        memoria_id: str,
    ) -> list[dict[str, Any]]:
        """Show all memorias that this one references.

        Returns outlinks (outgoing links) from the specified memoria.
        Useful for understanding what a memoria connects to.

        Args:
            memoria_id: The memoria ID to find outlinks for.
        """
        outlinks = memory.crossref.get_outlinks(memoria_id)
        return [o.__dict__ for o in outlinks]

    @server.tool()
    def memory_links_suggest(
        content: str,
        title: str | None = None,
        tags: list[str] | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Suggest links to existing memorias based on content.

        Returns suggested memorias to link to, based on semantic similarity
        and other heuristics. Useful when saving a new memoria to discover
        related existing content.

        Args:
            content: The memoria content being saved.
            title: Optional title of the memoria.
            tags: Optional tags of the memoria.
            limit: Maximum suggestions to return.
        """
        suggestions = memory.link_suggester.suggest_links(
            content=content,
            title=title or "",
            tags=tags or [],
            limit=limit,
        )
        return [s.__dict__ for s in suggestions]

    @server.tool()
    def memory_links_format(
        memoria_id: str,
        title: str | None = None,
    ) -> str:
        """Format a memoria ID as a wikilink.

        Returns a wikilink string like [[memoria-id]] or [[memoria-id|Title]].
        Use this to insert links into memoria content.

        Args:
            memoria_id: The memoria ID to format as a wikilink.
            title: Optional display title for the link.
        """
        return memory.link_suggester.format_wikilink(memoria_id, title)
