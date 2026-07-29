"""MCP tools — feedback domain (split from server.py).

Registered by `build_server()` via `register(server, memory)`. Tool names,
signatures, defaults, docstrings and bodies are identical to the originals;
only the enclosing function and indentation changed.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastmcp import Context, FastMCP
from pydantic import Field

from memo.memory import Memory
from memo.server_annotations import DESTRUCTIVE, READ_ONLY, WRITE, annotated_tool


def register(server: FastMCP, memory: Memory) -> None:
    @annotated_tool(server, **WRITE)
    def memo_feedback_record(
        source_id: str,
        query: str,
        rating: str,
    ) -> dict[str, Any]:
        """Record a feedback signal on a memory for a given query text.

        Supported signal values for `rating`:
          "thumbs_up" / "up"  — explicit positive vote; boosts score ~0.15 per vote.
          "click"              — implicit positive (user used/viewed this result); ~0.08 boost.
          "thumbs_down" / "down" — explicit rejection; hard-excludes from future similar queries.
          "ignore"             — implicit negative (user skipped); soft 0.7× score penalty.

        Idempotent on (source_id, query, rating). Recording a different signal
        for the same (source, query) pair replaces the prior vote.

        Args:
            source_id: meta.id (full or unique prefix >= 4 chars).
            query: query text the feedback applies to.
            rating: "thumbs_up", "click", "thumbs_down", "ignore", "up", or "down".
        """
        return memory.feedback_record(source_id, query_text=query, rating=rating)

    @annotated_tool(server, **WRITE)
    def memo_feedback_flag(
        source_id: str,
        kind: str,
        superseded_by: str | None = None,
    ) -> dict[str, Any]:
        """Flag a surfaced memory whose CONTENT is no longer true.

        Distinct from `memo_feedback_record` (which teaches the retriever per
        query): this routes to the memory lifecycle, not ranking.

          kind="outdated" — archive the memory (stale, not contradicted).
          kind="wrong"    — archive it; pass `superseded_by` (id or prefix of
                            the replacement memory) to record the supersede.

        Archive is reversible (the `memo maintain`/`dream` primitive), never a
        hard delete — an over-eager flag is recoverable. Use this when you
        find a recalled memory that is stale or contradicted rather than
        silently working around it.

        Args:
            source_id: meta.id (full or unique prefix >= 4 chars).
            kind: "outdated" or "wrong".
            superseded_by: for kind="wrong", the replacement memory id/prefix.
        """
        return memory.feedback_flag(source_id, kind=kind, superseded_by=superseded_by)

    @annotated_tool(server, **READ_ONLY)
    def memo_feedback_list(
        source_id: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """List recorded source-level feedback, newest first.

        Args:
            source_id: optional filter (full id or unique prefix).
            limit: max rows.
        """
        rows = memory.feedback_list(source_id=source_id, limit=limit)
        return {"rows": rows, "count": len(rows)}

    @annotated_tool(server, **DESTRUCTIVE)
    async def memo_feedback_clear(
        source_id: Annotated[
            str,
            Field(
                description=(
                    "meta.id of the memory whose feedback rows to delete — full 32-char id "
                    "or a unique prefix (must resolve to exactly one memory)."
                ),
            ),
        ],
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Drop all feedback rows for `source_id`. Returns count deleted.

        Deletes every ranking-feedback row recorded for the memory (the memory
        itself is untouched). Irreversible — these user signals are
        deliberately preserved by reindex — so elicitation-capable clients
        are asked to confirm. Idempotent: a repeat call deletes 0 rows.
        """
        from memo.server_elicit import abort_result, confirm_destructive, sanitize_fragment

        rows: int | None
        try:
            rows = len(memory.feedback_list(source_id=source_id, limit=100_000))
        except Exception:
            rows = None
        if rows is None or rows > 0:
            safe_id = sanitize_fragment(source_id)
            scope = f"all {rows} feedback rows" if rows is not None else "all feedback rows"
            gate = await confirm_destructive(
                ctx,
                action="clear",
                detail=(
                    f"Delete {scope} for memory {safe_id}? These user ranking "
                    "signals are deliberately preserved by reindex and are not "
                    "recoverable once cleared."
                ),
            )
            if not gate.proceed:
                return abort_result(
                    gate,
                    memory,
                    tool="memo_feedback_clear",
                    action="clear",
                    target=f"feedback rows for {safe_id}",
                )
        n = memory.feedback_clear(source_id)
        return {"source_id": source_id, "deleted": n}
