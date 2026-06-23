"""MCP tools — cache domain (split from server.py).

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
    def memo_cache_stats() -> dict[str, Any]:
        """Cache-tier status: mode, backend, entry count, capacity, overflow.

        When MEMO_CACHE_MODE=off (the default) `enabled` is False and memo is
        behaving as a durable store with no eviction.
        """
        return memory.cache.stats()

    @server.tool()
    def memo_cache_evict() -> dict[str, Any]:
        """Force a capacity-bound eviction pass now (coldest-first, per
        MEMO_CACHE_EVICTION). Dirty entries are flushed to the backing store
        before removal. Returns the evicted memory ids.

        No-op unless MEMO_CACHE_MODE != off and MEMO_CACHE_MAX_ENTRIES > 0.
        """
        evicted = memory.cache.evict_if_needed()
        return {"evicted": evicted, "count": len(evicted)}

    @server.tool()
    def memo_cache_flush() -> dict[str, Any]:
        """Push all dirty (write-back, un-persisted) memories to the backing
        store and clear their dirty flags. Returns {flushed, failed,
        dirty_remaining}. No-op when cache mode is off or no backend exists.
        """
        return memory.cache.flush_all()
