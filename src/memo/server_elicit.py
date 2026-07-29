"""MCP elicitation gate for irreversible tools.

Design: docs/SPECS/2026-07-28-mcp-elicitation-destructive-ops-design.md.

Six tools are gated on **irreversibility** (not on the advisory DESTRUCTIVE
annotation): the gate list is private to the server code and client-facing
annotations are unchanged. The gate is strictly **fail-open** — a client
without the elicitation capability (HTTP json_response daemons, headless SDK
clients, old CLIs) proceeds exactly as before; an unguarded ``ctx.elicit``
against such a client raises ``McpError`` and would brick the tool, so the
capability check runs FIRST.

Decline vs cancel (MCP three-action elicitation model): **decline** is an
explicit refusal and — flag-gated by ``MEMO_ELICIT_DECLINE_SIGNAL`` — is
persisted as a durable ``type=feedback`` memory so the refusal itself feeds
memo's feedback loop. **Cancel** is "no decision" and is a pure no-op.
The signal write is itself fail-open: a failed save never blocks the abort.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from fastmcp import Context

    from memo.memory import Memory

_log = logging.getLogger(__name__)

# Tools gated on irreversibility. Private list — see the design doc for why
# reversible DESTRUCTIVE-annotated tools (memo_forget, memo_update, ...) are
# deliberately exempt.
GATED_TOOLS: frozenset[str] = frozenset(
    {
        "memo_delete",
        "memo_synthesize_delete",
        "memo_backup_restore",
        "memo_feedback_clear",
        "memo_repo_delete",
        "memo_cache_evict",
    }
)

Outcome = Literal["accepted", "declined", "cancelled", "unsupported", "disabled", "error"]


@dataclass(frozen=True)
class ElicitOutcome:
    """Result of one confirmation attempt: what happened + whether to proceed."""

    outcome: Outcome
    proceed: bool


async def confirm_destructive(ctx: Context | None, *, action: str, detail: str) -> ElicitOutcome:
    """Ask the client to confirm an irreversible operation. Fail-open.

    ``detail`` is the elicitation message and must state the blast radius.
    ``action`` is the affirmative choice word; the client picks between it
    and "cancel". Proceeds without confirmation when the flag is off, the
    client lacks the elicitation capability, or the round-trip errors.
    """
    from memo.flags import flag_bool

    if not flag_bool("MEMO_ELICIT_CONFIRM"):
        return ElicitOutcome("disabled", proceed=True)
    if ctx is None:  # defensive: FastMCP injects ctx inside a request context
        return ElicitOutcome("unsupported", proceed=True)

    import mcp.types

    # Capability check FIRST: ctx.elicit against a non-elicitation client
    # raises McpError -> ToolError, which would brick the tool for every
    # non-elicitation client (HTTP json_response, headless SDKs, daemons).
    supported = ctx.session.check_client_capability(
        mcp.types.ClientCapabilities(elicitation=mcp.types.ElicitationCapability())
    )
    if not supported:
        return ElicitOutcome("unsupported", proceed=True)

    from fastmcp.server.elicitation import (
        AcceptedElicitation,
        CancelledElicitation,
        DeclinedElicitation,
    )
    from mcp.shared.exceptions import McpError

    try:
        # fastmcp 3.4.4 has a `response_type: list[str]` overload
        # (fastmcp/server/context.py:1050-1058) but stray docstrings between
        # the @overload defs break mypy's overload chain, so it only sees the
        # `None` overload. Runtime behavior is probe-verified.
        result = await ctx.elicit(detail, response_type=[action, "cancel"])  # type: ignore[arg-type]
    except McpError:
        # Belt and braces: clients that advertise the capability but
        # misimplement it must not brick the tool.
        return ElicitOutcome("error", proceed=True)
    if isinstance(result, AcceptedElicitation) and result.data == action:
        return ElicitOutcome("accepted", proceed=True)
    if isinstance(result, DeclinedElicitation):
        return ElicitOutcome("declined", proceed=False)
    if isinstance(result, CancelledElicitation):
        return ElicitOutcome("cancelled", proceed=False)
    # Accepted the form but chose "cancel" (or anything unexpected): treat as
    # a cancel — no decision, no signal.
    return ElicitOutcome("cancelled", proceed=False)


def record_decline_signal(memory: Memory, *, tool: str, action: str, target: str) -> None:
    """Persist an explicit refusal as a ``type=feedback`` memory. Fail-open.

    Flag-gated by ``MEMO_ELICIT_DECLINE_SIGNAL``. Saved with
    ``defer_embed=True`` so the abort response never waits on (or fails
    with) an MLX embed — reindex replays the pending embedding later.
    """
    from memo.flags import flag_bool

    if not flag_bool("MEMO_ELICIT_DECLINE_SIGNAL"):
        return
    from datetime import UTC, datetime

    try:
        memory.save(
            content=f"User refused {action} of {target} at the {tool} confirmation prompt.",
            title=f"Declined {action}: {target}"[:120],
            type_="feedback",
            tags=["elicit-decline"],
            extra={
                "elicit_tool": tool,
                "elicit_action": action,
                "elicit_target": target,
                "elicit_declined_at": datetime.now(UTC).isoformat(),
            },
            enforce_write_policy=False,
            defer_embed=True,
        )
    except Exception:
        _log.debug("decline-signal save failed (fail-open)", exc_info=True)


def abort_result(
    gate: ElicitOutcome,
    memory: Memory,
    *,
    tool: str,
    action: str,
    target: str,
) -> dict[str, Any]:
    """Envelope for a user-aborted gated tool; writes the decline signal.

    A user's "no" is a valid outcome, not a protocol error — gated tools
    return this normal result instead of raising.
    """
    if gate.outcome == "declined":
        record_decline_signal(memory, tool=tool, action=action, target=target)
    return {"ok": False, "aborted": gate.outcome}
