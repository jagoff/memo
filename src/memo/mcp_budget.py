"""Response-size budget for the MCP surface.

A `limit` parameter bounds an internal loop; it does not bound what the caller
receives. Measured 2026-08-06: `memo_graph verb=impact` returned 27.5k tokens
WITH `limit=3`, over the client's own tool-result cap. memo sells token savings,
so a read tool that costs 27k tokens inverts the product.

Three mechanisms, deliberately not overlapping:

* `bounded_list` -- opt-in, COUNT-based. A tool that knows which of its fields
  is elastic trims it and reports the real total. Generalises the fix already
  applied in `EntityNeighbors.to_bounded_dict()`.
* `fit_to_budget` -- opt-in, SIZE-based. For fields whose unit is a
  variable-length string, where no fixed count bounds the payload.
* the middleware (`make_response_budget_middleware`) -- automatic. It knows
  nothing about payload shape, so it never truncates; over cap it substitutes a
  structured error. Loud failure, bounded cost.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

from memo.flags import REGISTRY, flag_int

# Derived from the registry, not a second hard-coded literal: flags.py builds
# REGISTRY from flags_misc.SPECS (and never imports mcp_budget back), so this
# import direction carries no cycle. `.default` types as `Any` on FlagSpec;
# narrowed explicitly rather than trusting a loose inference.
DEFAULT_CAP_TOKENS: int = int(REGISTRY["MEMO_MCP_RESPONSE_BUDGET_TOKENS"].default)

# Per-tool overrides. Every entry carries the reason it is not the default.
CAPS: dict[str, int] = {
    # A raw embedding vector has exactly ONE plausible size, set by the
    # embedder profile the operator configured -- 2560 floats here, measured
    # 2026-08-08 at 21,924 tokens on the wire (42,566 JSON chars + 45,130 str
    # chars; a float32 component reads back as -0.0012969970703125). Nothing
    # the caller passes and nothing in the store can grow it, so it is not the
    # failure mode this budget exists for (a payload that scales with the
    # corpus), and under the default 10k cap the tool could NEVER return its
    # one and only result. It is exempted rather than trimmed because a
    # truncated vector is not a vector: the tool's whole contract is "the
    # exact vector memo would use for retrieval".
    #
    # Deliberately NOT extended to `memo_embed_batch`: its payload is
    # len(texts) x dim, i.e. caller-scaled, so "pass fewer texts" is a real
    # remedy there and an exemption would be a blank cheque.
    "memo_embed_query": 0,
}

# Headroom `fit_to_budget` leaves between its own estimate and the cap the
# middleware enforces. Two known, small, one-directional gaps make the real
# wire payload LARGER than `wire_tokens` reports, so a fit computed against
# the bare cap can land a few tokens over and be refused anyway:
#   1. FastMCP wraps a non-object tool return as `{"result": <payload>}` in
#      structured content -- ~5 tokens a tool sizing a bare list cannot see.
#   2. FastMCP's serializer is not `json.dumps`; their whitespace agrees in
#      practice but is not contractually identical.
# 64 tokens is 0.6% of the default cap and comfortably over both; it is
# pinned by `test_wire_tokens_stays_within_the_slack_of_result_text`.
_WIRE_SLACK_TOKENS = 64


def est_tokens(text: str) -> int:
    """The house estimate: 4 chars per token, rounded up."""
    return (len(text) + 3) // 4


def cap_for(tool: str) -> int:
    """Cap for one tool: explicit override > flag > built-in default. 0 = unlimited.

    `flag_int` is typed `int | None` because it is a generic accessor over any
    flag, including string/bool ones coerced elsewhere -- but
    MEMO_MCP_RESPONSE_BUDGET_TOKENS is registered with a concrete int default
    (DEFAULT_CAP_TOKENS), so the flag resolution chain (env > markdown config >
    tuned overlay > built-in default) always bottoms out on that default, never
    None. The assert documents the invariant instead of re-implementing the
    fallback that `flag()` already guarantees.
    """
    override = CAPS.get(tool)
    if override is not None:
        return override
    value = flag_int("MEMO_MCP_RESPONSE_BUDGET_TOKENS")
    assert value is not None, "MEMO_MCP_RESPONSE_BUDGET_TOKENS is registered with a default"
    return value


def bounded_list[T](
    items: Sequence[T],
    *,
    cap: int,
    key: Callable[[T], Any] | None = None,
) -> tuple[list[T], dict[str, Any]]:
    """Trim an elastic field to `cap` and report what was really there.

    `key`, when given, is a DISTANCE -- lower is better -- not a score: items
    are sorted ascending by `key` and the first `cap` are kept (nearest /
    lowest first). A caller holding a relevance SCORE (higher is better) must
    invert it (e.g. `key=lambda x: -x.score`) before passing it here.

    Returns the kept items plus the metadata a tool should splat into its
    result: shown / total / truncated.
    """
    ordered = sorted(items, key=key) if key is not None else list(items)
    kept = ordered[:cap]
    return kept, {"shown": len(kept), "total": len(items), "truncated": len(kept) < len(items)}


def wire_tokens(payload: Any) -> int:
    """Estimated tokens a tool's return costs on the wire.

    Counts the payload TWICE, mirroring `result_text`: fastmcp serialises the
    same object into both the content block and the structured content, and
    the middleware sizes their sum. A tool sizing its own result before
    returning it must therefore do the same -- counting once would let it hand
    back 2x its cap and still be refused.

    `default=str` because this is a measurement, never the wire payload
    itself: an exotic value must yield a number, not raise inside a budget
    check.
    """
    try:
        content = json.dumps(payload, separators=(",", ":"), default=str)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        content = str(payload)
    return est_tokens(content) + est_tokens(str(payload))


def fit_to_budget[T](
    items: Sequence[T],
    *,
    cap: int,
    render: Callable[[Sequence[T]], Any],
) -> tuple[list[T], Any]:
    """Longest prefix of `items` whose rendered payload fits `cap` tokens.

    `bounded_list` bounds a COUNT, which is the right bound when every item
    costs about the same. It is the wrong bound when the item is a
    variable-length string: `memo_graph_export`'s 500-edge cap measured 11,365
    tokens as DOT and 27,413 as JSON off one graph, and a community's 50
    entity names cost whatever those names happen to be. No single count is
    correct for both, so this bounds the size directly.

    `render` turns a prefix into the payload the tool would return, and is
    called O(log n) times (binary search over the prefix length, which is
    sound because payload size is monotone in prefix length). Give it a pool
    already trimmed by the tool's own count cap, so no render sees the whole
    graph.

    Returns the kept items AND their rendered payload -- callers that need the
    dict take the second, callers that rebuild from the items take the first.
    `cap <= 0` (unlimited) renders everything untouched.
    """
    ordered = list(items)
    full = render(ordered)
    budget = cap - _WIRE_SLACK_TOKENS
    if cap <= 0 or wire_tokens(full) <= budget:
        return ordered, full

    # Nothing beyond `ordered[:high]` can fit, since the full render did not.
    low, high = 0, len(ordered) - 1
    best_count = 0
    best_payload = render(ordered[:0])
    while low <= high:
        mid = (low + high) // 2
        candidate = render(ordered[:mid])
        if wire_tokens(candidate) <= budget:
            best_count, best_payload = mid, candidate
            low = mid + 1
        else:
            high = mid - 1
    return ordered[:best_count], best_payload


def budget_exceeded_payload(tool: str, tokens: int, cap: int, hint: str = "") -> dict[str, Any]:
    """The substitute response. Small, structured, and actionable."""
    return {
        "error": "response_budget_exceeded",
        "tool": tool,
        "tokens": tokens,
        "cap": cap,
        "hint": hint or "narrow the request (pass a smaller limit=) or use a summarising verb",
    }


def result_text(result: Any) -> str:
    """Textual projection of a tool result, for measurement only.

    Content blocks AND structured content, concatenated -- because
    `ToolResult.to_mcp_result()` (fastmcp 3.4.5) puts BOTH on the wire:
    `CallToolResult(content=..., structuredContent=...)` when `meta`/
    `is_error` is set, and the `(content, structured_content)` tuple
    otherwise. For a dict-returning tool that is the same JSON twice --
    measured on the 10k conformance corpus, `memo_list` is 4,586 content
    tokens PLUS 4,868 structured tokens. Measuring only `content`, as this
    did originally, reported roughly half of what the caller actually pays
    for, so the effective ceiling was ~2x the configured cap.

    This runs on every tool call, so it must never raise -- a budget layer
    that can throw converts a working tool into a broken one. Every branch
    is guarded, including the final fallback.
    """
    try:
        parts: list[str] = []
        blocks = getattr(result, "content", None)
        if isinstance(blocks, list):
            parts.extend(str(getattr(b, "text", "") or "") for b in blocks)
        structured = getattr(result, "structured_content", None)
        if structured is not None:
            parts.append(str(structured))
        joined = "".join(parts)
        return joined if joined else str(result)
    except Exception:
        try:
            return str(result)
        except Exception:
            return ""


def make_response_budget_middleware() -> Any:
    """Substitute an over-cap tool result with a structured error.

    Returns None when FastMCP middleware is unavailable, matching
    `_make_trace_middleware` (`server.py`) so `build_server` can skip wiring
    it without failing.

    `on_call_tool` must return a `fastmcp.tools.base.ToolResult` -- FastMCP's
    own `Middleware.on_call_tool` is typed to return one, and every built-in
    middleware (e.g. `ResponseLimitingMiddleware`) does the same. A bare dict
    is not a legal return here.

    Ordering (see `server.py`): FastMCP's `_run_middleware` walks
    `self.middleware` via `reversed()`, so the FIRST middleware registered
    via `add_middleware` ends up OUTERMOST -- the last to see/transform the
    result on the way back out, i.e. what the caller actually receives. The
    LAST middleware registered is INNERMOST and sees the tool's raw result
    first. This middleware is meant to size what the caller receives, so it
    must be registered before every other middleware, not after.

    "Size" is an estimate, not a byte count: `est_tokens` is the 4-chars-per-
    token house rule over `result_text`, which sums the two fields
    `to_mcp_result()` serialises (content + structured content). It tracks
    the wire payload closely enough to gate on; it is not the JSON-RPC
    frame's exact length.

    Also the sole committer of `server_common`'s emission-ledger stage (see
    `_LEDGER_STAGE` there): `apply_ledger` (called from inside a tool body,
    which FastMCP runs in a worker thread) can only STAGE what it would
    write, because a plain `ContextVar.set()` performed in that thread never
    becomes visible back here (a copied-context/thread boundary, not a
    fastmcp quirk). This middleware is the one place that both (a) sits in
    the SAME context the tool's stage list was bound in, so it can read what
    got staged, and (b) knows whether the caller actually received the
    payload those writes describe -- being outermost, it is registered
    before every other middleware, so its stage covers every tool
    unconditionally, not just the ones that happen to use the ledger. A tool
    that never calls `apply_ledger` simply stages nothing, at negligible
    cost (one `ContextVar.set([])` + `.reset()` per call).
    """
    try:
        from fastmcp.server.middleware import Middleware
        from fastmcp.tools.base import ToolResult
    except ImportError:  # pragma: no cover - supported FastMCP provides it
        return None

    class _ResponseBudgetMiddleware(Middleware):
        async def on_call_tool(self, context: Any, call_next: Any) -> Any:
            from memo.server_common import (
                commit_ledger_stage,
                discard_ledger_stage,
                open_ledger_stage,
            )

            token = open_ledger_stage()
            try:
                result = await call_next(context)
            except Exception:
                # The tool body raised: whatever it staged describes bodies
                # the caller never received (they got an error, not the
                # payload). Discard, then let the original exception
                # propagate exactly as it would have before this ran.
                discard_ledger_stage(token)
                raise
            name = str(getattr(context.message, "name", "") or "")
            cap = cap_for(name)
            if cap <= 0:
                commit_ledger_stage(token)
                return result
            tokens = est_tokens(result_text(result))
            if tokens <= cap:
                commit_ledger_stage(token)
                return result
            # Over cap: the substituted payload below, not `result`, is what
            # the caller actually receives. Whatever apply_ledger staged for
            # `result`'s bodies describes content that never reached them --
            # discard rather than commit.
            discard_ledger_stage(token)
            payload = budget_exceeded_payload(name, tokens, cap)
            # meta={} (not is_error=True): every other refusal on this MCP
            # surface (server_graph_tool.py, server_core_records.py,
            # server_multimodal.py, server_sync.py, ...) returns a plain
            # {"error": ...} dict as a NORMAL successful call -- is_error
            # would make FastMCP's Client.call_tool() (raise_on_error=True by
            # default) raise ToolError instead, indistinguishable from a
            # crash, breaking that convention for this one payload. `meta`
            # is what actually buys the output_schema bypass:
            # ToolResult.to_mcp_result()'s gate is `self.meta is not None or
            # self.is_error`, and FastMCP's own ResponseLimitingMiddleware
            # sets `meta={}` (not is_error) for this exact reason.
            return ToolResult(structured_content=payload, meta={})

    return _ResponseBudgetMiddleware()
