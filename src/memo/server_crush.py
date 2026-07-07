"""MCP tools for JSON crushing — retrieval of offloaded low-relevance rows.

Wave 1 token economy: memo_crush_retrieve recovers the original JSON
from crush cache using a marker hash.
"""

from __future__ import annotations

from typing import Any

from memo.memory import Memory
from memo.server_annotations import READ_ONLY, annotated_tool


def register(server: Any, memory: Memory) -> None:
    """Register crush-related MCP tools."""

    @annotated_tool(server, **READ_ONLY)
    def memo_crush_retrieve(hash_marker: str) -> dict[str, Any]:
        """Retrieve original JSON from crush cache.

        When memo crushes a large JSON array during ingest, it offloads
        low-relevance rows to cache and embeds only the top-K rows.
        This tool recovers the original using the crush marker hash.

        Args:
            hash_marker: Marker string from crushed output, e.g.,
                        "<<memo-crush:abc123def456>>"

        Returns:
            {"original": <full_json_string>, "hash": <hash_val>} on success,
            or {"error": <message>} on failure (missing or expired cache entry).
        """
        from memo.store.crush_cache import CrushCache

        # Parse marker format: <<memo-crush:HASH>> -> extract HASH
        if not hash_marker.startswith("<<memo-crush:") or not hash_marker.endswith(">>"):
            return {"error": f"Invalid marker format: {hash_marker}"}

        hash_val = hash_marker[13:-2]  # Strip <<memo-crush: and >>
        cache = CrushCache(memory.cfg.state_dir)

        original = cache.retrieve(hash_val)
        if original is None:
            return {"error": f"Cache entry not found or expired: {hash_val}"}

        return {"original": original, "hash": hash_val}
