"""MCP surface for explicit lexical transcript search."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import Field

from memo.memory import Memory
from memo.server_annotations import READ_ONLY, annotated_tool
from memo.server_common import log_consult, now_ms
from memo.store.turn_store import MAX_VERBATIM_RESULTS


def register(server: Any, memory: Memory) -> None:
    @annotated_tool(server, **READ_ONLY)
    def memo_verbatim_search(
        query: Annotated[
            str,
            Field(
                description="FTS5 query text; tokenized on word characters and AND-joined "
                "(every token must match, order-independent), falling back to OR when the "
                "AND match returns zero rows. Empty/no-token queries return no hits."
            ),
        ],
        session_id: Annotated[
            str | None,
            Field(
                description="Restrict hits to one indexed session's turns; "
                "None searches all sessions."
            ),
        ] = None,
        since: Annotated[
            str | None,
            Field(
                description="ISO8601 timestamp lower bound; only turns with ts >= since "
                "(compared lexicographically as ISO strings). None applies no time filter."
            ),
        ] = None,
        limit: Annotated[
            int,
            Field(description="Maximum hits to return (clamped to 1-100)."),
        ] = 10,
        source: Annotated[
            str,
            Field(
                description="Calling layer for consult attribution "
                "(e.g. 'claude-code', 'codex'); empty falls back to client info."
            ),
        ] = "",
    ) -> dict[str, Any]:
        """Search explicitly indexed transcript turns with FTS5.

        Read-only and excluded from automatic recall. Results are bounded to
        100; `session_id` and ISO8601 `since` narrow the private local index.
        """
        from memo.store.turn_store import TurnStore

        t0 = now_ms()
        bounded_limit = max(1, min(int(limit), MAX_VERBATIM_RESULTS))
        hits: list[dict[str, Any]] = []
        if memory.cfg.verbatim_db.is_file():
            store = TurnStore(memory.cfg.verbatim_db)
            try:
                hits = store.search(
                    query,
                    limit=bounded_limit,
                    session_id=session_id,
                    since=since,
                )
            finally:
                store.close()
        log_consult(
            memory,
            tool="verbatim_search",
            query=query,
            hits=hits,
            t0_ms=t0,
            source=source,
        )
        return {"hits": hits}


__all__ = ["register"]
