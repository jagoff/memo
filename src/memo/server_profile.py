"""MCP profile surface built on Memo's existing durable state."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import Field

from memo.memory import Memory
from memo.memory_profile import build_memory_profile
from memo.server_annotations import READ_ONLY, annotated_tool
from memo.server_common import log_consult, now_ms


def register(server: Any, memory: Memory) -> None:
    @annotated_tool(server, **READ_ONLY)
    def memo_profile(
        scope: Annotated[
            str,
            Field(
                description=(
                    "Descriptive label echoed into the payload: one of 'current', 'user', "
                    "'project', 'agent' (anything else returns a memo.error.v1 invalid_scope "
                    "envelope). Does not filter or authorize — content selection stays "
                    "governed by Memo's profile files and record provenance."
                ),
            ),
        ] = "current",
        limit: Annotated[
            int,
            Field(
                description=(
                    "Maximum recent memory records in the 'active' section "
                    "(clamped to 0-50; forgotten records are excluded)."
                ),
            ),
        ] = 8,
        budget_chars: Annotated[
            int,
            Field(
                description=(
                    "Character budget for the 'stable' profile text (clamped to 256-12000). "
                    "Longer text is truncated with a trailing ellipsis and an entry in "
                    "'omissions'."
                ),
            ),
        ] = 4000,
        cwd: Annotated[
            str | None,
            Field(
                description=(
                    "Working directory used to resolve the current project (git toplevel) so "
                    "the project-specific profile document is included in the stable section. "
                    "Defaults to the server process's working directory."
                ),
            ),
        ] = None,
        source: Annotated[
            str,
            Field(
                description=(
                    "Caller identity recorded in the recall consult log. Empty string falls "
                    "back to the MEMO_SOURCE env var, then the MCP client's declared "
                    "clientInfo name."
                ),
            ),
        ] = "",
    ) -> dict[str, Any]:
        """Return bounded stable and active memory with evidence metadata."""
        t0 = now_ms()
        try:
            payload = build_memory_profile(
                memory,
                scope=scope,
                limit=limit,
                budget_chars=budget_chars,
                cwd=cwd,
            )
        except ValueError as exc:
            return {
                "schema": "memo.error.v1",
                "error": {"code": "invalid_scope", "message": str(exc)},
                "scope": scope,
            }
        hits = [
            item
            for section in ("stable", "active")
            for item in payload.get(section, [])
            if isinstance(item, dict)
        ]
        log_consult(memory, tool="profile", query=scope, hits=hits, t0_ms=t0, source=source)
        return payload
