"""MCP tools — consolidation domain (split from server.py).

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
    def memory_consolidate_propose(
        threshold: float = 0.85,
        max_clusters: int = 20,
        type: str | None = None,
    ) -> dict[str, Any]:
        """Detect clusters and propose merge strategies (read-only).

        Returns a dict with detected clusters and merge proposals. Does not
        modify the corpus. Use `memory_consolidate_apply` to execute merges.

        Args:
            threshold: Cosine similarity threshold (default 0.85).
            max_clusters: Maximum clusters to process (default 20).
            type: Optional filter by memoria type.
        """
        return memory.consolidator.consolidate_all(
            threshold=threshold,
            max_clusters=max_clusters,
            type_=type,
            auto_apply=False,
            dry_run=True,
        )

    @server.tool()
    def memory_consolidate_apply(
        threshold: float = 0.85,
        max_clusters: int = 20,
        type: str | None = None,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Apply merge proposals to consolidate the corpus.

        Executes the consolidation pipeline: detect clusters, propose merges,
        and apply them. Archives old memorias to an `archived/` subdirectory.

        SAFE DEFAULT: `dry_run=True` — this previews the merges without
        mutating the corpus. Pass `dry_run=False` to actually merge and
        archive (data-loss operation).

        Args:
            threshold: Cosine similarity threshold (default 0.85).
            max_clusters: Maximum clusters to process (default 20).
            type: Optional filter by memoria type.
            dry_run: If True (default), show what would happen without
                applying changes. Set False to apply.
        """
        return memory.consolidator.consolidate_all(
            threshold=threshold,
            max_clusters=max_clusters,
            type_=type,
            auto_apply=True,
            dry_run=dry_run,
        )

    @server.tool()
    def memory_consolidate_list_archived() -> list[dict[str, Any]]:
        """List all archived memorias.

        Returns a list of archived memoria entries with metadata including
        the replacement memoria ID and archival timestamp.
        """

        import frontmatter

        archival_dir = memory.cfg.memory_dir / "archived"
        if not archival_dir.is_dir():
            return []

        archived = []
        for f in archival_dir.glob("*.md"):
            post = frontmatter.loads(f.read_text(encoding="utf-8"))
            archived.append(
                {
                    "id": f.stem,
                    "title": post.get("title", ""),
                    "archived_for": post.get("archived_for", ""),
                    "archived_at": post.get("archived_at", ""),
                }
            )
        return archived
