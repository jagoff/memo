"""MCP tools — time-machine (as-of) domain (split from server.py).

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
    def memo_search_as_of(
        query: str,
        as_of: str,
        limit: int = 10,
        type: str | None = None,
        mode: str = "hybrid",
    ) -> dict[str, Any]:
        """Search the corpus as it existed on a past date (time-machine).

        Reconstructs a snapshot by replaying `history.db` events in
        reverse from "now", then runs the live search and post-filters
        to the snapshot's record set. Use this when you want to know
        what the user knew (or had decided) at a specific past point.

        Args:
            query: Free-text query.
            as_of: `YYYY-MM-DD` (interpreted as end-of-day UTC) or a
                full ISO-8601 timestamp.
            limit: Max results.
            type: Optional record-type filter.
            mode: `hybrid` (default), `vec`, or `bm25`.

        Returns: `{"as_of", "snapshot_size", "results": [...]}` where
        each result has the same shape as `memo_search` items.
        """
        from memo.time_machine import reconstruct

        snap = reconstruct(memory, as_of=as_of)
        hits = snap.search(query, limit=limit, mode=mode)
        if type:
            hits = [h for h in hits if h.type == type]
        return {
            "as_of": snap.as_of.isoformat(),
            "snapshot_size": len(snap),
            "results": [h.to_dict() for h in hits],
        }

    @server.tool()
    def memo_ask_as_of(question: str, as_of: str, k: int = 5) -> dict[str, Any]:
        """RAG question against a past snapshot.

        Same contract as `memo_ask` but the corpus view is rewound to
        `as_of`. The LLM is told explicitly that the view is historical
        so it doesn't smuggle in facts that only became known later.

        Args:
            question: Free-text question.
            as_of: `YYYY-MM-DD` or full ISO 8601.
            k: Top-k memories to feed the model as context.
        """
        from memo.time_machine import reconstruct

        snap = reconstruct(memory, as_of=as_of)
        return snap.ask(question, k=k)

    @server.tool()
    def memo_diff(from_ts: str, to_ts: str | None = None) -> dict[str, Any]:
        """Diff the corpus between two snapshots — added, removed, updated.

        Args:
            from_ts: Start date — YYYY-MM-DD or full ISO 8601.
            to_ts: End date (defaults to "now" when omitted).
        """
        from datetime import UTC
        from datetime import datetime as _dt

        from memo.time_machine import diff as _diff

        to_resolved = to_ts or _dt.now(UTC).isoformat()
        d = _diff(memory, from_ts=from_ts, to_ts=to_resolved)
        return {
            "from_ts": d.from_ts.isoformat(),
            "to_ts": d.to_ts.isoformat(),
            "summary": d.summary(),
            "added": [{"id": r.id, "title": r.title, "type": r.type} for r in d.added],
            "removed": [{"id": r.id, "title": r.title, "type": r.type} for r in d.removed],
            "updated": d.updated,
        }
