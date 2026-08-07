"""Response-size budget for the MCP surface.

A `limit` parameter bounds an internal loop; it does not bound what the caller
receives. Measured 2026-08-06: `memo_graph verb=impact` returned 27.5k tokens
WITH `limit=3`, over the client's own tool-result cap. memo sells token savings,
so a read tool that costs 27k tokens inverts the product.

Two mechanisms, deliberately not overlapping:

* `bounded_list` -- opt-in. A tool that knows which of its fields is elastic
  trims it and reports the real total. Generalises the fix already applied in
  `EntityNeighbors.to_bounded_dict()`.
* the middleware (`make_response_budget_middleware`) -- automatic. It knows
  nothing about payload shape, so it never truncates; over cap it substitutes a
  structured error. Loud failure, bounded cost.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from memo.flags import REGISTRY, flag_int

# Derived from the registry, not a second hard-coded literal: flags.py builds
# REGISTRY from flags_misc.SPECS (and never imports mcp_budget back), so this
# import direction carries no cycle. `.default` types as `Any` on FlagSpec;
# narrowed explicitly rather than trusting a loose inference.
DEFAULT_CAP_TOKENS: int = int(REGISTRY["MEMO_MCP_RESPONSE_BUDGET_TOKENS"].default)

# Per-tool overrides. Every entry carries the reason it is not the default.
CAPS: dict[str, int] = {}


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

    Handles the FastMCP shapes (content blocks, structured content) and
    falls back to `str`. This runs on every tool call, so it must never
    raise -- a budget layer that can throw converts a working tool into a
    broken one. Every branch is guarded, including the final fallback.
    """
    try:
        blocks = getattr(result, "content", None)
        if isinstance(blocks, list):
            parts = [str(getattr(b, "text", "") or "") for b in blocks]
            if any(parts):
                return "".join(parts)
        structured = getattr(result, "structured_content", None)
        if structured is not None:
            return str(structured)
        return str(result)
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
    first. This middleware measures the byte count the caller gets, so it
    must be registered before every other middleware, not after.
    """
    try:
        from fastmcp.server.middleware import Middleware
        from fastmcp.tools.base import ToolResult
    except ImportError:  # pragma: no cover - supported FastMCP provides it
        return None

    class _ResponseBudgetMiddleware(Middleware):
        async def on_call_tool(self, context: Any, call_next: Any) -> Any:
            result = await call_next(context)
            name = str(getattr(context.message, "name", "") or "")
            cap = cap_for(name)
            if cap <= 0:
                return result
            tokens = est_tokens(result_text(result))
            if tokens <= cap:
                return result
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
