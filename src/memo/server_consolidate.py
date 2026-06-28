"""MCP tools — consolidation domain (split from server.py).

Registered by `build_server()` via `register(server, memory)`. Tool names,
signatures, defaults, docstrings and bodies are identical to the originals;
only the enclosing function and indentation changed.
"""

from __future__ import annotations

import logging
from typing import Any

from fastmcp import FastMCP

from memo.memory import Memory

_log = logging.getLogger(__name__)


def register(server: FastMCP, memory: Memory) -> None:
    @server.tool()
    def memo_consolidate_list_archived() -> list[dict[str, Any]]:
        """List all archived memories.

        Returns a list of archived memory entries with metadata including
        the replacement memory ID and archival timestamp.
        """

        import frontmatter

        archival_dir = memory.cfg.memory_dir / "archived"
        if not archival_dir.is_dir():
            return []

        archived = []
        for f in archival_dir.glob("*.md"):
            try:
                post = frontmatter.loads(f.read_text(encoding="utf-8"))
                archived.append(
                    {
                        "id": f.stem,
                        "title": post.get("title", ""),
                        "archived_for": post.get("archived_for", ""),
                        "archived_at": post.get("archived_at", ""),
                    }
                )
            except Exception as exc:
                _log.warning("skipping %s: %s", f, exc)
                continue
        return archived
