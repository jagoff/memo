"""MCP tools — cache domain (split from server.py).

Registered by `build_server()` via `register(server, memory)`. Tool names,
signatures, defaults, docstrings and bodies are identical to the originals;
only the enclosing function and indentation changed.
"""

from __future__ import annotations

from typing import Any

from fastmcp import Context, FastMCP

from memo.memory import Memory
from memo.server_annotations import DESTRUCTIVE, READ_ONLY, annotated_tool


def register(server: FastMCP, memory: Memory) -> None:
    @annotated_tool(server, **READ_ONLY)
    def memo_cache_stats() -> dict[str, Any]:
        """Cache-tier status: mode, backend, entry count, capacity, overflow.

        When MEMO_CACHE_MODE=off (the default) `enabled` is False and memo is
        behaving as a durable store with no eviction.
        """
        return memory.cache.stats()

    @annotated_tool(server, **DESTRUCTIVE)
    async def memo_cache_evict(ctx: Context | None = None) -> dict[str, Any]:
        """Force a capacity-bound eviction pass now (coldest-first, per
        MEMO_CACHE_EVICTION). Dirty entries are flushed to the backing store
        before removal. Evicted memories are deleted permanently, so
        elicitation-capable clients are asked to confirm when the pass
        would actually evict. Returns the evicted memory ids.

        No-op unless MEMO_CACHE_MODE != off and MEMO_CACHE_MAX_ENTRIES > 0.
        """
        from memo.server_elicit import abort_result, confirm_destructive

        stats = memory.cache.stats()
        over = int(stats.get("over_capacity") or 0)
        if stats.get("enabled") and over > 0:
            gate = await confirm_destructive(
                ctx,
                action="evict",
                detail=(
                    f"Evict ~{over} coldest memories from the cache tier "
                    f"({stats.get('eviction')} policy)? Dirty entries are "
                    "flushed to the backing store first; evicted memories are "
                    "then deleted."
                ),
            )
            if not gate.proceed:
                return abort_result(
                    gate,
                    memory,
                    tool="memo_cache_evict",
                    action="evict",
                    target=f"~{over} cache-tier memories",
                )
        evicted = memory.cache.evict_if_needed()
        return {"evicted": evicted, "count": len(evicted)}

    @annotated_tool(server, **DESTRUCTIVE)
    def memo_cache_flush() -> dict[str, Any]:
        """Push all dirty (write-back, un-persisted) memories to the backing
        store and clear their dirty flags. Returns {flushed, failed,
        dirty_remaining}. No-op when cache mode is off or no backend exists.
        """
        return memory.cache.flush_all()
