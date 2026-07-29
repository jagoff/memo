"""MCP elicitation gate for irreversible tools.

Design: docs/SPECS/2026-07-28-mcp-elicitation-destructive-ops-design.md.

Six tools are gated on **irreversibility** (not on the advisory DESTRUCTIVE
annotation): the gate list is private to the server code and client-facing
annotations are unchanged. The gate is **fail-open before asking**: a client
without the elicitation capability (headless SDK clients, old CLIs) proceeds
exactly as before — an unguarded ``ctx.elicit`` against such a client raises
``McpError`` and would brick the tool, so the capability check runs FIRST.
The HTTP daemon in ``json_response`` mode cannot deliver a mid-call
server->client request at all (the SDK discards it and the call deadlocks),
so ``server.py`` marks that transport via
:func:`mark_transport_elicit_unsupported` and the gate skips eliciting.

Once the question HAS been sent, the gate **fails closed**: a round-trip that
errors mid-flight (client handler crashed, connection died while a human was
deciding) aborts instead of running the irreversible op unconfirmed. The one
exception is an up-front ``Method not found`` rejection — the client
advertised the capability but does not actually serve it — which is treated
like a missing capability (fail-open).

Decline vs cancel (MCP three-action elicitation model): **decline** is an
explicit refusal and — flag-gated by ``MEMO_ELICIT_DECLINE_SIGNAL`` — is
persisted as a durable ``type=feedback`` memory so the refusal itself feeds
memo's feedback loop. **Cancel** is "no decision" and is a pure no-op.
The signal write is itself fail-open: a failed save never blocks the abort.
"""

from __future__ import annotations

import logging
import re
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

# Set by server.py when the serving transport cannot deliver a mid-call
# server->client request (HTTP daemon in json_response mode: the SDK's
# streamable_http JSON path discards elicitation/create, the call deadlocks,
# and the eventual disconnect surfaces as an McpError). Checked BEFORE the
# capability check because check_client_capability is client-declared and
# transport-blind.
_TRANSPORT_ELICIT_UNSUPPORTED = False

# Control chars (incl. newline/CR/ESC — kills ANSI sequences) that would let
# a hostile memory title rewrite or hide the blast-radius warning shown to
# the human. C1 range included: some terminals honor 8-bit control codes.
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def mark_transport_elicit_unsupported() -> None:
    """Disable elicitation process-wide for transports that cannot carry it."""
    global _TRANSPORT_ELICIT_UNSUPPORTED
    _TRANSPORT_ELICIT_UNSUPPORTED = True


def sanitize_fragment(value: object, limit: int = 80) -> str:
    """Neutralize untrusted text interpolated into a confirmation prompt.

    Strips control characters (newlines, ANSI escapes), collapses whitespace,
    and caps length — memory titles / repo names originate from untrusted
    content (auto-capture, LLM derivation, ingested repos) and the prompt
    exists precisely to protect the user from a confused or hostile agent.
    """
    text = _CONTROL_CHARS.sub(" ", str(value))
    text = " ".join(text.split())
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text


@dataclass(frozen=True)
class ElicitOutcome:
    """Result of one confirmation attempt: what happened + whether to proceed."""

    outcome: Outcome
    proceed: bool


async def confirm_destructive(ctx: Context | None, *, action: str, detail: str) -> ElicitOutcome:
    """Ask the client to confirm an irreversible operation.

    ``detail`` is the elicitation message and must state the blast radius.
    ``action`` is the affirmative choice word; the client picks between it
    and "cancel". Fail-open BEFORE asking (flag off, transport can't carry
    elicitation, client lacks the capability, up-front Method-not-found);
    fail-closed AFTER asking (mid-flight error or an uninterpretable answer
    aborts — an irreversible op never runs on a question with no answer).
    """
    from memo.flags import flag_bool

    if not flag_bool("MEMO_ELICIT_CONFIRM"):
        return ElicitOutcome("disabled", proceed=True)
    if _TRANSPORT_ELICIT_UNSUPPORTED:
        # json_response HTTP: elicitation/create never reaches the client and
        # the call deadlocks — skip BEFORE eliciting, exactly like a client
        # without the capability.
        return ElicitOutcome("unsupported", proceed=True)
    if ctx is None:  # defensive: FastMCP injects ctx inside a request context
        return ElicitOutcome("unsupported", proceed=True)

    import mcp.types

    # Capability check FIRST: ctx.elicit against a non-elicitation client
    # raises McpError -> ToolError, which would brick the tool for every
    # non-elicitation client (headless SDKs, daemons, old CLIs).
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
    from pydantic import ValidationError

    # Untrusted fragments (titles, repo names) are sanitized at their build
    # sites; re-sanitize the whole message here as the single choke point.
    detail = sanitize_fragment(detail, limit=500)
    try:
        # fastmcp 3.4.4 has a `response_type: list[str]` overload
        # (fastmcp/server/context.py:1050-1058) but stray docstrings between
        # the @overload defs break mypy's overload chain, so it only sees the
        # `None` overload. Runtime behavior is probe-verified.
        result = await ctx.elicit(detail, response_type=[action, "cancel"])  # type: ignore[arg-type]
    except McpError as exc:
        if exc.error.code == mcp.types.METHOD_NOT_FOUND:
            # The client advertised the capability but rejected the request
            # up-front — equivalent to not having it. Fail-open.
            return ElicitOutcome("unsupported", proceed=True)
        # The question was sent but the round-trip died (client handler
        # crashed -> INTERNAL_ERROR, connection closed while the human was
        # deciding, ...). The user never confirmed: abort, don't destroy.
        return ElicitOutcome("error", proceed=False)
    except ValidationError:
        # The client accepted with content that fails the response schema
        # (enum mismatch, empty auto-accept). The user interacted but intent
        # is unknown — treat as cancel: no decision, no signal, no destroy.
        return ElicitOutcome("cancelled", proceed=False)
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
    ``target`` is sanitized/capped (it embeds an untrusted title) and the
    save carries a stable ``topic_key`` so repeated declines of the same
    (tool, target) revise/corroborate ONE memory instead of minting an
    unbounded stream; the write goes through the normal write policy.
    """
    from memo.flags import flag_bool

    if not flag_bool("MEMO_ELICIT_DECLINE_SIGNAL"):
        return
    from datetime import UTC, datetime

    safe_target = sanitize_fragment(target, limit=120)
    try:
        memory.save(
            content=f"User refused {action} of {safe_target} at the {tool} confirmation prompt.",
            title=f"Declined {action}: {safe_target}"[:120],
            type_="feedback",
            tags=["elicit-decline"],
            extra={
                "elicit_tool": tool,
                "elicit_action": action,
                "elicit_target": safe_target,
                "elicit_declined_at": datetime.now(UTC).isoformat(),
            },
            topic_key=f"elicit-decline:{tool}:{safe_target}",
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
