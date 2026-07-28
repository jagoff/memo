"""MCP tools for context offloading — `memo_offload` (memo_get is the drill-down)."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import Field

from memo.memory import Memory
from memo.server_annotations import WRITE_IDEMPOTENT, annotated_tool


def register(server: Any, memory: Memory) -> None:
    @annotated_tool(server, **WRITE_IDEMPOTENT)
    def memo_offload(
        content: Annotated[
            str,
            Field(
                description="Raw payload to store verbatim. Rejected when empty/"
                "whitespace-only or longer than the configured max_content_chars "
                "(MEMO_MAX_CONTENT_CHARS); identical content deduplicates by "
                "sha256 to the existing memory id."
            ),
        ],
        title: Annotated[
            str | None,
            Field(
                description="Optional label used as the stored memory's title and "
                "markdown heading; None auto-generates 'offload:<kind> <sha256[:12]>'."
            ),
        ] = None,
    ) -> dict[str, Any]:
        """Offload a bulky payload (tool output, log, dump) out of the context
        window: memo stores it content-addressed as a reference-tier memory
        and returns `{id, sha256, kind, synopsis, deduplicated, drill_down}`.

        Use memo_offload for bulk working-context dumps; use memo_save for
        curated durable facts meant to be recalled. Idempotent per payload:
        re-offloading identical content returns the existing id with
        `deduplicated: true` instead of writing a new memory.

        The synopsis is deterministic (no LLM): JSON keys, CSV headers, code
        symbols, or compressed text. Reference tier is excluded from
        auto-recall, so offloaded blobs never appear in the recall hook.
        Fetch the full payload later with `memo_get(id)`.
        """
        from memo.offload import offload

        return offload(memory, content, title=title)
