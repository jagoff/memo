"""MCP tools for JSON crushing — retrieval of offloaded low-relevance rows.

Wave 1 token economy: memo_crush_retrieve recovers the original JSON
from crush cache using a marker hash.

Registered unconditionally in `server.py` (not gated behind the advanced
surface) because it is also the ONE recovery path the context-compression
proxy's markers point at (`memo.proxy.ccr.marker()`) — a cut can happen
regardless of which MCP profile the caller has active, so the tool that
undoes it must be reachable from all of them too. See `memo.proxy.ccr`'s
own docstring for why it deliberately reuses this cache instead of a
second one.
"""

from __future__ import annotations

from typing import Any

from memo.memory import Memory
from memo.server_annotations import READ_ONLY, annotated_tool


def register(server: Any, memory: Memory) -> None:
    """Register crush-related MCP tools."""

    @annotated_tool(server, **READ_ONLY)
    def memo_crush_retrieve(hash_marker: str) -> dict[str, Any]:
        """Retrieve original content from crush cache.

        When memo crushes a large JSON array during ingest, it offloads
        low-relevance rows to cache and embeds only the top-K rows. The
        context-compression proxy reuses the same cache to make a cut
        reversible. This tool recovers the original from either.

        Args:
            hash_marker: Either the ingest-time wrapped form,
                        "<<memo-crush:abc123def456>>", or the proxy's bare
                        hex key, "abc123def456" — both name the same cache.

        Returns:
            {"original": <full_original_string>, "hash": <hash_val>} on
            success, or {"error": <message>} on failure (missing or expired
            cache entry).
        """
        from memo.store.crush_cache import CrushCache

        # Parse marker format: <<memo-crush:HASH>> -> extract HASH; anything
        # else is taken as a bare hex key (CrushCache validates the shape).
        if hash_marker.startswith("<<memo-crush:") and hash_marker.endswith(">>"):
            hash_val = hash_marker[13:-2]  # Strip <<memo-crush: and >>
        else:
            hash_val = hash_marker

        cache = CrushCache(memory.cfg.state_dir)

        original = cache.retrieve(hash_val)
        if original is None:
            return {"error": f"Cache entry not found or expired: {hash_val}"}

        return {"original": original, "hash": hash_val}
