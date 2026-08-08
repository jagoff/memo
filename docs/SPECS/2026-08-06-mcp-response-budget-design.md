Status: partially shipped — the five measured payload defects (`memo_graph_export`, `memo_graph verb=impact/neighbors`, `memo_lint`, `memo_operational_state`) were fixed ad hoc, per-surface, in #209, but the systemic mechanism this design proposes (`src/memo/mcp_budget.py`, `_ResponseBudgetMiddleware`, `MEMO_MCP_RESPONSE_BUDGET_TOKENS`, `bounded_list`) does not exist on master — grep confirms zero matches. A complete implementation exists on local branch `feat/conformance-budget-deadline-admission` (`src/memo/mcp_budget.py`), not yet opened as a PR.

# MCP response budget — design

**Date:** 2026-08-06
**Status:** proposed
**Scope:** `src/memo/mcp_budget.py` (new), `src/memo/server.py`, per-tool opt-in call sites, `tests/conformance/`

## Problem

A `limit` parameter bounds an internal loop. It does not bound the payload the
caller receives. Measured during the 2026-08-06 sweep against the live corpus
(11,383 memories at the time; 10,496 today after consolidation):

| Surface | Measured | Note |
|---|---|---|
| `memo_graph verb=impact` | 109.9k chars ≈ 27.5k tok | **with `limit=3`**; exceeds the client's tool-result cap, verb unusable on any real change |
| `memo_graph verb=neighbors` | 35.9k chars ≈ 9k tok | default `limit=8` on the `memo` hub |
| `memo_graph_export` | 3.6MB DOT / 6.0MB JSON | no bounding parameter at all |
| `memo_lint` | 725k chars | no parameters on the MCP surface |
| `memo_operational_state` | ~10k tok | every open conflict, unbounded, +9/day |

Twelve defects of this family were found by exercising surfaces against the live
corpus. **Zero** were caught by 6,955 passing tests, mypy over 501 files, ruff,
`memo definitive check`, or `memo journey-check`.

Each was fixed by hand, at the call site. There are ~150 tool functions across
`server_*.py` (162 registered tools on the `full` profile). The next one added
repeats the bug, because nothing in the system makes an unbounded response fail.

memo's value proposition is token savings. A read tool that returns 27.5k tokens
inverts it.

## Goals

1. No MCP tool can return a payload above its declared cap without the caller
   being told, explicitly, in-band.
2. A newly added tool cannot merge without a cap that holds against a
   10k-memory corpus.
3. Zero breaking changes to the tool surface: no tool is removed, no tool is
   renamed, no required parameter is added. (Decided 2026-08-06 — memo ships on
   PyPI, MCP registries and the Claude store.)

## Non-goals

- Reducing the number of registered tools. `src/memo/surface.py` already
  provides profiles (`agent` 41 / `core`,`slim` 58 / `full` 162) and that
  mechanism is adequate.
- Changing any tool's semantics or field names.
- Streaming or pagination cursors. Out of scope; a bounded response plus an
  honest `total` is enough for the measured failures.

## Design

Two mechanisms that do not overlap. One is a safety net that assumes nothing
about payload shape; the other is a convention tools opt into.

### 1. Enforcement — a FastMCP result middleware (hard, automatic)

`_ResponseBudgetMiddleware.on_call_tool` wraps `call_next`, serialises the
result, and measures it. This mirrors the two middlewares already wired in
`build_server` (`_make_trace_middleware`, `make_write_coordinator_middleware`),
including their `ImportError` → `None` degradation when FastMCP middleware is
unavailable.

Over cap, the middleware **does not truncate**. Truncating a dict whose shape it
does not understand silently corrupts a contract. It replaces the result with a
structured error:

```json
{
  "error": "response_budget_exceeded",
  "tool": "memo_graph",
  "tokens": 27500,
  "cap": 10000,
  "hint": "verb=impact on 13 changed files; pass a smaller limit= or use verb=architecture"
}
```

Loud failure, bounded cost. The caller learns what happened and what to do; the
client's context survives.

Token estimate uses `(chars + 3) // 4`, the estimator already used in
`dashboard_logs.py:227`, `cli_recall_hook.py:435`, `recall_logic.py:1465` and
`web_build.py:636`. `mcp_budget.py` owns the single definition; those four call
sites are left alone (out of scope, pre-existing).

### 2. Compliance — `bounded_list` (soft, opt-in)

A tool that knows which of its fields is elastic trims it before returning:

```python
shown, meta = bounded_list(neighbors, cap=cfg.cap, key=lambda n: n.distance)
return {"neighbors": shown, **meta}
# meta == {"shown": 5, "total": 378, "truncated": True}
```

This generalises the fix already applied in `EntityNeighbors.to_bounded_dict()`
(5 exemplar ids + `neighbor_memory_counts`, 11-19× reduction). It is the pattern
that worked; the module makes it reusable instead of re-derived per tool.

Tools that use it never reach the net. The net exists for the ones that do not.

### 3. Caps

- Default: `MEMO_MCP_RESPONSE_BUDGET_TOKENS`, built-in default **10000**.
  Registered in `flags_misc.py` per the flags-registry convention.
- Per-tool override declared at registration, in a table in `mcp_budget.py`
  keyed by tool name. Export/backup-family tools (`memo_export_*`,
  `memo_backup_*`, `memo_graph_export`) either declare a high cap or return a
  written `path` instead of a body — decided per tool during implementation,
  recorded in the table with a one-line reason.
- Setting the flag to `0` disables enforcement. Escape hatch for a user whose
  client has no cap.

## Testing

Unit tests for `mcp_budget.py`: estimator, `bounded_list` metadata, middleware
over/under cap, middleware absent-FastMCP degradation.

**The gate that matters** — `tests/conformance/test_mcp_response_budget.py`:
enumerate every tool registered on a built server, invoke each with default
arguments against the seeded 10k-memory corpus fixture, assert
`est_tokens(result) <= cap(tool)`.

This is the test that would have caught all five payload defects at once. It
depends on the corpus fixture from the deadline-and-conformance spec, which is
why that fixture is built first.

Tools requiring arguments (ids, paths) get them from the fixture's known seed
data via a small per-tool argument table in the test module; a tool absent from
that table and not invokable with defaults fails the test rather than being
skipped, so the enumeration cannot silently shrink.

## Rollout

Enforcement on by default. It cannot regress a working client: the only
behaviour change is that a response which would have blown the client's cap
becomes a small error instead. If the conformance gate is green, no compliant
tool ever hits it.

## Success criteria

- Every registered tool ≤ its cap against the 10k fixture.
- The five payload defects from the sweep are covered by the enumerating gate
  rather than by five individual fixes.
- Adding a tool with an unbounded elastic field fails CI.

## Risks

- **Tool enumeration depends on the FastMCP API** (`get_tools` / tool manager).
  Confirm the exact accessor at implementation; if unavailable, fall back to
  memo's own registry of `register()` modules, which is enumerable from
  `server.py` regardless.
- **A legitimately large export becomes an error.** Mitigated by the per-tool
  table; each exemption carries a written reason.
- **`(chars + 3) // 4` is an estimate, not a tokenizer.** It is the house
  convention and is conservative enough for a cap whose purpose is preventing a
  27k-token response, not billing.
