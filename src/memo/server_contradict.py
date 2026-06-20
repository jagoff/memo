"""MCP tools — contradiction domain (split from server.py).

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
    def memo_contradict_scan(
        top_k: int = 5,
        sim_floor: float = 0.55,
        confidence_threshold: float = 0.7,
        min_days_apart: int = 1,
        max_memorias: int = 2000,
        max_pairs: int = 500,
        since: str | None = None,
        type: str | None = None,
    ) -> dict[str, Any]:
        """Scan the corpus for contradiction / evolution pairs and persist them.

        Unlike `memo_temporal_contradictions`, which requires an entity name
        and returns ephemeral results, this walks all memorias, uses vec
        neighborhoods to surface candidate pairs, and persists detected
        contradictions to a sidecar DB. The same pair is never re-classified
        once the user resolves it.

        Args:
            top_k: Neighbors to consider per memoria.
            sim_floor: Cosine floor for candidate pairs.
            confidence_threshold: Min LLM confidence to persist.
            min_days_apart: Skip pairs whose `updated` are within N days.
            max_memorias: Hard cap on memorias visited.
            max_pairs: Hard cap on pairs sent to the LLM.
            since: ISO date; only memorias updated on/after are scanned.
            type: Optional memoria type filter.
        """
        result = memory.contradict_scanner.scan_corpus(
            top_k=top_k,
            sim_floor=sim_floor,
            confidence_threshold=confidence_threshold,
            min_days_apart=min_days_apart,
            max_memorias=max_memorias,
            max_pairs=max_pairs,
            since=since,
            type_=type,
        )
        return {
            "scanned_memorias": result.scanned_memorias,
            "pairs_examined": result.pairs_examined,
            "pairs_inserted": result.pairs_inserted,
            "pairs_refreshed": result.pairs_refreshed,
            "pairs_skipped_resolved": result.pairs_skipped_resolved,
            "contradictions_found": result.contradictions_found,
            "evolutions_found": result.evolutions_found,
        }

    @server.tool()
    def memo_contradict_list(
        status: str = "open",
        limit: int = 20,
        min_confidence: float = 0.0,
        relationship: str | None = None,
    ) -> list[dict[str, Any]]:
        """List contradiction pairs from the sidecar DB.

        Args:
            status: One of open|fused|kept_newer|kept_older|evolved|dismissed.
            limit: Max rows.
            min_confidence: Filter to pairs at/above this LLM confidence.
            relationship: Optional 'contradiction' or 'evolution' filter.
        """
        if status == "open":
            pairs = memory.contradict_store.list_open(
                limit=limit,
                min_confidence=min_confidence,
                relationship=relationship,
            )
        else:
            pairs = memory.contradict_store.list_all(status=status, limit=limit)
            if relationship:
                pairs = [p for p in pairs if p.relationship == relationship]
        return [p.__dict__ for p in pairs]

    @server.tool()
    def memo_contradict_resolve(
        pair_id: int,
        status: str,
        note: str | None = None,
    ) -> dict[str, Any]:
        """Mark a contradiction pair as resolved.

        Valid statuses (excluding `open`):
          - `fused`       — both merged into a new memoria
          - `kept_newer`  — newer side won, older deleted/archived
          - `kept_older`  — older side won
          - `evolved`     — legitimate evolution, both kept
          - `dismissed`   — false positive

        This tool only updates the sidecar; it does NOT itself delete the
        memorias or run a merge. Use `memo_consolidate_apply` or
        `memo_delete` first if the resolution implies destructive ops.

        Args:
            pair_id: The integer pair_id returned by `memo_contradict_list`.
            status: One of the valid statuses listed above.
            note: Optional free-form resolution note.
        """
        ok = memory.contradict_store.resolve(pair_id, status, note=note)
        return {"updated": ok, "pair_id": pair_id, "status": status}

    @server.tool()
    def memo_contradict_stats() -> dict[str, int]:
        """Return counts of contradiction pairs grouped by status."""
        return memory.contradict_store.stats()
