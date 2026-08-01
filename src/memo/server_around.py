"""MCP surface: memo_around — timeline context around one memory."""

from __future__ import annotations

import logging
from typing import Annotated, Any

from pydantic import Field

from memo.memory import Memory
from memo.server_annotations import READ_ONLY, annotated_tool

_log = logging.getLogger(__name__)


def register(server: Any, memory: Memory) -> None:
    @annotated_tool(server, **READ_ONLY)
    def memo_around(
        id: Annotated[
            str,
            Field(
                description="Anchor memory id: full 32-char id or a unique prefix "
                "(git-style, e.g. 7 chars); derived reference-chunk ids must be "
                "given exactly. Unresolvable ids return anchor=null."
            ),
        ],
        before: Annotated[
            int,
            Field(
                description="Neighbours to include before the anchor — earlier "
                "sibling chunks (chunk_seq mode) or earlier-created memories "
                "(created mode). Clamped to 0-10."
            ),
        ] = 2,
        after: Annotated[
            int,
            Field(
                description="Neighbours to include after the anchor — later "
                "sibling chunks (chunk_seq mode) or later-created memories "
                "(created mode). Clamped to 0-10."
            ),
        ] = 2,
    ) -> dict[str, Any]:
        """Timeline context around one memory: seq-adjacent sibling chunks for
        reference-tier chunks (vault docs), created-time neighbours
        for durable memories. Turns a search hit into narrative context.
        Read-only; window clamped to ±10."""
        before = max(0, min(int(before), 10))
        after = max(0, min(int(after), 10))
        return memory.around(id, before=before, after=after)
