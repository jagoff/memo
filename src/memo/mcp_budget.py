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

from memo.flags import flag_int

# Keep in sync with the MEMO_MCP_RESPONSE_BUDGET_TOKENS default in
# flags_misc.py -- two literals rather than one import to avoid a
# flags_misc -> mcp_budget -> flags import cycle (flags.py aggregates
# flags_misc.SPECS).
DEFAULT_CAP_TOKENS = 10000

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
