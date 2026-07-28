"""MCP tools for explicit review, verification, and truth validity."""

from __future__ import annotations

from typing import Annotated, Any

from fastmcp import FastMCP
from pydantic import Field

from memo.memory import Memory
from memo.server_annotations import DESTRUCTIVE, READ_ONLY, WRITE_IDEMPOTENT, annotated_tool


def register(server: FastMCP, memory: Memory) -> None:
    @annotated_tool(server, **READ_ONLY)
    def memo_review_due(
        project: Annotated[
            str | None,
            Field(
                description=(
                    "Project name to filter by (matched against its slugified"
                    " project namespace); omit to list all projects."
                ),
            ),
        ] = None,
        limit: Annotated[
            int,
            Field(
                description=(
                    "Maximum records to return (clamped to 1-200). Records with an"
                    " open conflict sort first, then by earliest review date."
                ),
            ),
        ] = 50,
    ) -> dict[str, Any]:
        """List records whose explicit review date passed or have an open conflict."""
        rows = memory.list_due_reviews(project=project, limit=max(1, min(limit, 200)))
        return {"due": rows, "count": len(rows)}

    @annotated_tool(server, **WRITE_IDEMPOTENT)
    def memo_mark_reviewed(
        id: Annotated[
            str,
            Field(description="ID of the memory to mark as reviewed; errors if not found."),
        ],
        evidence: Annotated[
            str | None,
            Field(
                description=(
                    "Free-text evidence supporting the review; sanitized before"
                    " being persisted with the review record."
                ),
            ),
        ] = None,
        actor: Annotated[
            str | None,
            Field(description="Identifier of who performed the review, stored with the evidence."),
        ] = None,
    ) -> dict[str, Any]:
        """Record review evidence, verify the memory, and schedule its next review.

        Replaying the latest still-current review with identical evidence and
        actor is a no-op, so retries are idempotent.
        """
        return memory.mark_reviewed(id, evidence=evidence, actor=actor).to_dict()

    @annotated_tool(server, **DESTRUCTIVE)
    def memo_invalidate(
        id: Annotated[
            str,
            Field(description="ID of the memory whose validity to close; errors if not found."),
        ],
        reason: Annotated[
            str,
            Field(
                description=(
                    "Why the memory is no longer valid; sanitized and logged in the update history."
                ),
            ),
        ],
        at: Annotated[
            str | None,
            Field(
                description=(
                    "ISO-8601 timestamp when the memory became invalid (naive"
                    " values are treated as UTC); defaults to now."
                ),
            ),
        ] = None,
    ) -> dict[str, Any]:
        """Close one memory's truth-validity interval without deleting it.

        Sets invalid_at in the record's frontmatter and index; the record
        itself is retained and stays readable.
        """
        return memory.invalidate(id, reason=reason, at=at).to_dict()

    @annotated_tool(server, **DESTRUCTIVE)
    def memo_supersede(
        old_id: Annotated[
            str,
            Field(
                description=(
                    "ID of the memory being superseded; its validity interval is"
                    " closed. Must differ from new_id; errors if not found."
                ),
            ),
        ],
        new_id: Annotated[
            str,
            Field(
                description=(
                    "ID of the successor memory; the old memory is closed at this"
                    " record's validity start (or creation time). Errors if not found."
                ),
            ),
        ],
        reason: Annotated[
            str,
            Field(
                description=(
                    "Why the successor replaces the old memory; logged as"
                    " 'superseded by <new_id>: <reason>' in the update history."
                ),
            ),
        ],
    ) -> dict[str, Any]:
        """Close the old memory at the successor's validity start.

        Only the old record is modified; the close time never precedes the old
        record's own validity start.
        """
        return memory.supersede(old_id, new_id, reason=reason).to_dict()
