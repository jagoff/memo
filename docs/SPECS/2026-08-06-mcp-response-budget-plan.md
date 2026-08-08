Status: partially shipped — this plan's own deliverable (`src/memo/mcp_budget.py`) is absent from master; the underlying problem (unbounded MCP payloads) was instead closed via five individual per-surface fixes in #209, not the systemic middleware this plan builds. A complete implementation of this exact plan exists on local branch `feat/conformance-budget-deadline-admission`, not yet opened as a PR.

# MCP Response Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make it impossible for an MCP tool to return an unbounded payload without the caller being told, and impossible for a new tool to merge without a cap that holds against a 10,000-memory corpus.

**Architecture:** A new `memo.mcp_budget` module owns three things: the house token estimator, a `bounded_list` helper tools opt into, and a FastMCP middleware that measures every tool result and replaces an over-cap one with a structured error rather than truncating a shape it does not understand. A conformance test enumerates every registered tool and asserts its result fits.

**Tech Stack:** FastMCP middleware (`fastmcp.server.middleware.Middleware`, `on_call_tool`), the `MEMO_*` flag registry, pytest.

Spec: `docs/SPECS/2026-08-06-mcp-response-budget-design.md`.
Depends on: `docs/SPECS/2026-08-06-corpus-conformance-plan.md` Task 2 (the `big_corpus` fixture).

## Global Constraints

- No tool is removed, renamed, or given a new required parameter. memo ships on PyPI, MCP registries and the Claude store; the surface is a public contract.
- Flags go in the registry (`flags_misc.py`), never `os.environ.get` inline. `memo config validate` must stay clean.
- Token estimate is `(chars + 3) // 4` — the estimator already used in `dashboard_logs.py:227`, `cli_recall_hook.py:435`, `recall_logic.py:1465`, `web_build.py:636`. Those four call sites are NOT refactored here (out of scope).
- FastMCP is pinned `>=0.5,<4`. Every middleware import degrades to `None` on `ImportError`, matching `_make_trace_middleware` (`server.py:84`) and `make_write_coordinator_middleware` (`server_write_coordinator.py:159`).
- Shared working tree: stage explicit paths only.

---

### Task 1: The budget module

**Files:**
- Create: `src/memo/mcp_budget.py`
- Create: `tests/test_mcp_budget.py`
- Modify: `src/memo/flags_misc.py` (append to `SPECS`)

**Interfaces:**
- Produces:
  - `est_tokens(text: str) -> int`
  - `DEFAULT_CAP_TOKENS: int` (= 10000)
  - `CAPS: dict[str, int]` — per-tool overrides, tool name → cap
  - `cap_for(tool: str) -> int` — override, else the flag, else `DEFAULT_CAP_TOKENS`; `0` means unlimited
  - `bounded_list(items: list[T], *, cap: int, key: Callable[[T], Any] | None = None) -> tuple[list[T], dict[str, Any]]`
  - `budget_exceeded_payload(tool: str, tokens: int, cap: int, hint: str = "") -> dict[str, Any]`

- [ ] **Step 1: Write the failing tests**

`tests/test_mcp_budget.py`:

```python
"""The budget module: estimator, opt-in trimming, and the error payload.

The middleware itself is covered in test 2; these are the pure pieces."""

from __future__ import annotations

import pytest

from memo import mcp_budget


def test_est_tokens_matches_the_house_estimator() -> None:
    assert mcp_budget.est_tokens("") == 0
    assert mcp_budget.est_tokens("abcd") == 1
    assert mcp_budget.est_tokens("a" * 4000) == 1000


def test_bounded_list_passes_a_short_list_through_untouched() -> None:
    shown, meta = mcp_budget.bounded_list([1, 2, 3], cap=10)
    assert shown == [1, 2, 3]
    assert meta == {"shown": 3, "total": 3, "truncated": False}


def test_bounded_list_trims_and_reports_the_real_total() -> None:
    shown, meta = mcp_budget.bounded_list(list(range(100)), cap=5)
    assert shown == [0, 1, 2, 3, 4]
    assert meta == {"shown": 5, "total": 100, "truncated": True}


def test_bounded_list_keeps_the_best_by_key() -> None:
    items = [{"d": 9}, {"d": 1}, {"d": 5}]
    shown, meta = mcp_budget.bounded_list(items, cap=2, key=lambda x: x["d"])
    assert [x["d"] for x in shown] == [1, 5]
    assert meta["total"] == 3


def test_cap_for_prefers_the_per_tool_override(monkeypatch) -> None:
    monkeypatch.setitem(mcp_budget.CAPS, "memo_export_json", 500_000)
    assert mcp_budget.cap_for("memo_export_json") == 500_000
    assert mcp_budget.cap_for("memo_search") == mcp_budget.DEFAULT_CAP_TOKENS


def test_cap_for_honours_the_flag(monkeypatch) -> None:
    monkeypatch.setenv("MEMO_MCP_RESPONSE_BUDGET_TOKENS", "42")
    assert mcp_budget.cap_for("memo_search") == 42


def test_zero_cap_means_unlimited(monkeypatch) -> None:
    monkeypatch.setenv("MEMO_MCP_RESPONSE_BUDGET_TOKENS", "0")
    assert mcp_budget.cap_for("memo_search") == 0


def test_budget_exceeded_payload_names_the_tool_and_both_numbers() -> None:
    payload = mcp_budget.budget_exceeded_payload("memo_graph", 27500, 10000, hint="pass limit=")
    assert payload["error"] == "response_budget_exceeded"
    assert payload["tool"] == "memo_graph"
    assert payload["tokens"] == 27500
    assert payload["cap"] == 10000
    assert "limit=" in payload["hint"]
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --no-sync pytest tests/test_mcp_budget.py -q`
Expected: `ModuleNotFoundError: No module named 'memo.mcp_budget'`.

- [ ] **Step 3: Register the flag**

Append to the `SPECS` tuple in `src/memo/flags_misc.py`, before the closing `)`:

```text
    _spec(
        "MEMO_MCP_RESPONSE_BUDGET_TOKENS",
        "int",
        10000,
        "mcp",
        "Cap on the estimated token size of any single MCP tool result. Over "
        "cap the response is REPLACED by a structured "
        "`response_budget_exceeded` error naming the tool, the size and the "
        "cap -- never truncated, because truncating an arbitrary payload "
        "silently corrupts its contract. Per-tool overrides live in "
        "`mcp_budget.CAPS`. 0 = unlimited (escape hatch for a client with no "
        "cap of its own). Motivated by `memo_graph verb=impact` returning "
        "27.5k tokens with limit=3.",
        min_val=0,
    ),
```

- [ ] **Step 4: Write the module**

`src/memo/mcp_budget.py`:

```python
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
from typing import Any, TypeVar

from memo.flags import flag_int

T = TypeVar("T")

DEFAULT_CAP_TOKENS = 10000

# Per-tool overrides. Every entry carries the reason it is not the default.
CAPS: dict[str, int] = {}


def est_tokens(text: str) -> int:
    """The house estimate: 4 chars per token, rounded up."""
    return (len(text) + 3) // 4


def cap_for(tool: str) -> int:
    """Cap for one tool: explicit override > flag > built-in. 0 = unlimited."""
    override = CAPS.get(tool)
    if override is not None:
        return override
    flagged = flag_int("MEMO_MCP_RESPONSE_BUDGET_TOKENS")
    return DEFAULT_CAP_TOKENS if flagged is None else flagged


def bounded_list(
    items: Sequence[T],
    *,
    cap: int,
    key: Callable[[T], Any] | None = None,
) -> tuple[list[T], dict[str, Any]]:
    """Trim an elastic field to `cap` and report what was really there.

    `key` orders by relevance first (nearest, highest-scoring) so the trim keeps
    the best rather than the first. Returns the kept items plus the metadata a
    tool should splat into its result: shown / total / truncated.
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
```

- [ ] **Step 5: Run the tests**

Run: `uv run --no-sync pytest tests/test_mcp_budget.py -q`
Expected: 8 passed.

- [ ] **Step 6: Validate the flag registry**

Run: `uv run --no-sync memo config validate`
Expected: clean — no unknown-var or duplicate-registration complaint.

- [ ] **Step 7: Commit**

```bash
git add src/memo/mcp_budget.py tests/test_mcp_budget.py src/memo/flags_misc.py
git commit -m "feat(mcp): add the response-budget primitives"
```

---

### Task 2: The enforcement middleware

**Files:**
- Modify: `src/memo/mcp_budget.py` (add the middleware factory)
- Modify: `src/memo/server.py:281-284` (wire it after the write-coordinator middleware)
- Modify: `tests/test_mcp_budget.py` (append)

**Interfaces:**
- Consumes: `est_tokens`, `cap_for`, `budget_exceeded_payload` from Task 1.
- Produces: `make_response_budget_middleware() -> Any` — a FastMCP `Middleware` instance, or `None` when FastMCP middleware is unavailable.
- Produces: `result_text(result: Any) -> str` — best-effort textual projection of a tool result, used for measurement.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_mcp_budget.py`:

```python
class _Block:
    def __init__(self, text: str) -> None:
        self.text = text


class _Result:
    def __init__(self, text: str) -> None:
        self.content = [_Block(text)]


def test_result_text_reads_content_blocks() -> None:
    assert mcp_budget.result_text(_Result("hello")) == "hello"


def test_result_text_falls_back_to_str() -> None:
    assert mcp_budget.result_text(1234) == "1234"


@pytest.mark.asyncio
async def test_middleware_passes_a_small_result_through(monkeypatch) -> None:
    monkeypatch.setenv("MEMO_MCP_RESPONSE_BUDGET_TOKENS", "1000")
    mw = mcp_budget.make_response_budget_middleware()
    assert mw is not None

    small = _Result("ok")

    class _Ctx:
        message = type("M", (), {"name": "memo_search"})()

    async def _call_next(_ctx):
        return small

    assert await mw.on_call_tool(_Ctx(), _call_next) is small


@pytest.mark.asyncio
async def test_middleware_replaces_an_over_cap_result(monkeypatch) -> None:
    monkeypatch.setenv("MEMO_MCP_RESPONSE_BUDGET_TOKENS", "10")
    mw = mcp_budget.make_response_budget_middleware()

    class _Ctx:
        message = type("M", (), {"name": "memo_graph"})()

    async def _call_next(_ctx):
        return _Result("x" * 4000)

    out = await mw.on_call_tool(_Ctx(), _call_next)
    assert out["error"] == "response_budget_exceeded"
    assert out["tool"] == "memo_graph"
    assert out["tokens"] == 1000
    assert out["cap"] == 10


@pytest.mark.asyncio
async def test_zero_cap_disables_enforcement(monkeypatch) -> None:
    monkeypatch.setenv("MEMO_MCP_RESPONSE_BUDGET_TOKENS", "0")
    mw = mcp_budget.make_response_budget_middleware()
    huge = _Result("x" * 40000)

    class _Ctx:
        message = type("M", (), {"name": "memo_graph"})()

    async def _call_next(_ctx):
        return huge

    assert await mw.on_call_tool(_Ctx(), _call_next) is huge
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --no-sync pytest tests/test_mcp_budget.py -q -k middleware`
Expected: `AttributeError: module 'memo.mcp_budget' has no attribute 'make_response_budget_middleware'`.

If `pytest.mark.asyncio` is unavailable, check how the existing async MCP tests are marked (`grep -rn "asyncio" tests/test_server.py | head`) and match that convention.

- [ ] **Step 3: Implement**

Append to `src/memo/mcp_budget.py`:

```python
def result_text(result: Any) -> str:
    """Textual projection of a tool result, for measurement only.

    Handles the FastMCP shapes (content blocks, structured content) and falls
    back to `str`. Measurement must never raise -- a budget layer that can throw
    is worse than no budget layer.
    """
    blocks = getattr(result, "content", None)
    if isinstance(blocks, list):
        parts = [str(getattr(b, "text", "") or "") for b in blocks]
        if any(parts):
            return "".join(parts)
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        return str(structured)
    return str(result)


def make_response_budget_middleware() -> Any:
    """Substitute an over-cap tool result with a structured error.

    Returns None when FastMCP middleware is unavailable, matching
    `_make_trace_middleware` so `build_server` can skip wiring it.
    """
    try:
        from fastmcp.server.middleware import Middleware
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
            return budget_exceeded_payload(name, tokens, cap)

    return _ResponseBudgetMiddleware()
```

- [ ] **Step 4: Run the tests**

Run: `uv run --no-sync pytest tests/test_mcp_budget.py -q`
Expected: all pass.

- [ ] **Step 5: Wire it into the server**

In `src/memo/server.py`, immediately after the write-coordinator middleware block (which ends at line 284 with `server.add_middleware(_write_mw)`), add:

```text
    # Response budget LAST so it measures what the caller actually receives,
    # after every other middleware has had its turn.
    from memo.mcp_budget import make_response_budget_middleware

    _budget_mw = make_response_budget_middleware()
    if _budget_mw is not None:
        server.add_middleware(_budget_mw)
```

- [ ] **Step 6: Verify the server still builds and its tests pass**

Run: `uv run --no-sync pytest tests/test_server.py -q`
Expected: green.

- [ ] **Step 7: Commit**

```bash
git add src/memo/mcp_budget.py src/memo/server.py tests/test_mcp_budget.py
git commit -m "feat(mcp): enforce a response budget on every tool result"
```

---

### Task 3: The gate that enumerates every tool

**Files:**
- Create: `tests/conformance/test_mcp_response_budget.py`
- Modify: `src/memo/mcp_budget.py` (`CAPS` entries for legitimately-large tools)
- Modify: the offending `server_*.py` modules (adopt `bounded_list`)

**Interfaces:**
- Consumes: `big_corpus` from the conformance plan Task 2; `cap_for`, `est_tokens`, `result_text` from Tasks 1-2.

This is the task that makes the whole thing hold. Everything before it is machinery.

- [ ] **Step 1: Write the failing test**

```python
"""Every registered tool, invoked against 10k memories, fits its cap.

This is the gate the five payload defects of 2026-08-06 would have hit. It
enumerates rather than lists, so a newly added tool is covered the day it is
added and cannot be forgotten."""

from __future__ import annotations

import pytest

from memo import mcp_budget
from memo.memory.facade import Memory
from memo.server import build_server_for_memory

from .conftest import DIMS, seeded_id

pytestmark = pytest.mark.conformance

# Arguments for tools that cannot be called bare. A tool that is neither
# callable with defaults nor listed here FAILS -- it is never skipped, so the
# enumeration cannot silently shrink.
ARGS: dict[str, dict[str, object]] = {
    "memo_get": {"id": seeded_id(0)},
    "memo_search": {"query": "topic00"},
    "memo_graph": {"verb": "neighbors", "entity": "topic00"},
    "memo_related": {"id": seeded_id(0)},
}

# Tools with a side effect or an unbounded external dependency. Each entry is a
# deliberate exclusion with a stated reason, not a convenience.
SKIP: dict[str, str] = {
    "memo_delete": "destructive",
    "memo_reindex": "rebuilds the whole index; covered by the rebuild test",
}


@pytest.mark.asyncio
async def test_every_tool_result_fits_its_cap(big_corpus) -> None:
    memory = Memory(big_corpus)
    try:
        server = build_server_for_memory(memory)
        tools = await server.get_tools()

        over: list[str] = []
        uncallable: list[str] = []
        for name in sorted(tools):
            if name in SKIP:
                continue
            try:
                result = await server._call_tool(name, ARGS.get(name, {}))
            except TypeError:
                uncallable.append(name)
                continue
            tokens = mcp_budget.est_tokens(mcp_budget.result_text(result))
            cap = mcp_budget.cap_for(name)
            if cap and tokens > cap:
                over.append(f"{name}: {tokens} > {cap}")

        assert not uncallable, f"tools needing an ARGS entry: {uncallable}"
        assert not over, "tools over budget:\n" + "\n".join(over)
    finally:
        memory.close()
```

- [ ] **Step 2: Run it**

Run: `MEMO_CONFORMANCE_CORPUS_N=10000 uv run --no-sync pytest tests/conformance/test_mcp_response_budget.py -q`
Expected: FAIL, with a concrete list.

The exact accessors are the risk here. Confirm three things against the installed FastMCP before assuming the test is wrong:
- `server.get_tools()` — `server.get_tool(name)` is already used at `server_write_coordinator.py:168`, so the plural form is the natural companion; verify with `python -c "import fastmcp, inspect; print([m for m in dir(fastmcp.FastMCP) if 'tool' in m])"`.
- the invocation entry point (`_call_tool` vs `call_tool`) — same command.
- `build_server_for_memory` — the real name of the function around `server.py:249` ("Register the complete surface around an already constructed Memory"). Use whatever it is actually called.

- [ ] **Step 3: Fix the over-budget tools**

For each name in the failure list, in order of overage:

- If the tool has an elastic field, trim it with `bounded_list` and splat the metadata, exactly as `EntityNeighbors.to_bounded_dict()` does.
- If the tool is legitimately large (export, backup), add a `CAPS` entry with the reason in a comment, or change it to write a file and return the path.

Do not raise the default cap to make a red tool green.

- [ ] **Step 4: Re-run until green**

Run: `MEMO_CONFORMANCE_CORPUS_N=10000 uv run --no-sync pytest tests/conformance/test_mcp_response_budget.py -q`
Expected: PASS.

- [ ] **Step 5: Prove the gate is real**

Temporarily remove the `bounded_list` call from one tool you just fixed, re-run, confirm RED, then restore it. A gate that has never been seen red is not known to work.

- [ ] **Step 6: Full check**

```bash
uv run --no-sync pytest -m "not slow and not conformance" -q
uv run --no-sync mypy src/memo/mcp_budget.py src/memo/server.py
uv run --no-sync ruff check src/memo/mcp_budget.py && uv run --no-sync ruff format src/memo/mcp_budget.py
```

- [ ] **Step 7: Commit**

```bash
git add tests/conformance/test_mcp_response_budget.py src/memo/mcp_budget.py
git commit -m "test(conformance): every MCP tool result fits its budget at 10k memories"
```

---

## Self-review notes

- Spec coverage: enforcement middleware (Task 2), `bounded_list` compliance helper (Task 1, adopted in Task 3), caps and the flag (Task 1), per-tool override table (Tasks 1 and 3), the enumerating gate (Task 3), the export-family exemption path (Task 3 Step 3).
- Non-goal respected: no tool is removed or renamed; `surface.py` is untouched.
- Unverified identifiers, each flagged in-step with a command to confirm: `server.get_tools()`, `server._call_tool`, `build_server_for_memory`, and the async test marker convention.
