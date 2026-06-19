# Memo Token Economy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make memo reduce agent context substantially by default while preserving full administrative capabilities as an explicit opt-in.

**Architecture:** Introduce a five-tool `agent` MCP surface as the default, keep `core` and `full` compatibility profiles, and remove repeated server prose. Claude hooks receive compact startup and recall packets. Exact injected-context costs are logged and subtracted from estimated savings, while destructive Dream compression becomes opt-in.

**Tech Stack:** Python 3.13+, Click, FastMCP, pytest, JSONL telemetry, Markdown agent skills.

## Global Constraints

- Markdown remains the source of truth; no migration or destructive rewrite.
- Existing `core` and explicit `full` MCP profiles remain supported.
- The default agent surface exposes only briefing, search, ask, get, and save.
- Compact startup context is capped at 480 characters.
- Ambient recall is capped at 160 estimated tokens and one memoria.
- Token KPIs report gross context cost and net estimated savings separately.

---

### Task 1: Default five-tool agent MCP surface

**Files:**
- Modify: `src/memo/surface.py`
- Modify: `src/memo/server.py`
- Modify: `src/memo/flags_misc.py`
- Modify: `src/memo/runtime/mcp.py`
- Test: `tests/test_surface_profiles.py`
- Test: `tests/test_runtime_isolation.py`

**Interfaces:**
- Produces: `mcp_profile() -> str`, `mcp_allowed_tools() -> frozenset[str] | None`.
- Default agent allowlist: `memory_unified_briefing`, `memory_search`, `memory_ask`, `memory_get`, `memory_save`.

- [x] Write failing tests asserting the default five-tool list, explicit `core` compatibility, explicit `full` advanced tools, and installer propagation of `MEMO_MCP_PROFILE=agent`.
- [x] Run focused tests and confirm failures are caused by the current full default surface.
- [x] Implement profile selection, post-registration tool filtering, terse server instructions, and installer environment propagation.
- [x] Run the focused tests until green.

### Task 2: Compact startup briefing

**Files:**
- Modify: `src/memo/briefing.py`
- Modify: `src/memo/cli_briefing.py`
- Modify: `src/memo/server_core_search.py`
- Modify: `hooks/hooks.json`
- Test: `tests/test_briefing_unified.py`

**Interfaces:**
- Produces: `compact_text(text: str, max_chars: int = 480) -> str`.
- CLI: `memo briefing --compact` avoids Synapse, open-loop, and memory-of-day expansion.
- MCP unified briefing returns at most 480 characters.

- [x] Write failing tests for the 480-character cap and compact CLI output.
- [x] Verify the tests fail against the rich-only implementation.
- [x] Implement compact formatting and wire `SessionStart` to `memo briefing --compact`.
- [x] Run briefing tests until green.

### Task 3: Bound ambient recall and record exact context cost

**Files:**
- Modify: `src/memo/dashboard_logs.py`
- Modify: `src/memo/recall_logic.py`
- Modify: `src/memo/cli_recall_hook.py`
- Modify: `src/memo/flags_recall.py`
- Modify: `hooks/hooks.json`
- Test: `tests/test_dashboard.py`
- Test: `tests/test_recall_hooks.py`
- Test: `tests/test_recall_server.py`

**Interfaces:**
- Produces: `append_context_cost_log(... kind, chars, client, session_id, turn)` and `read_context_cost_log()`.
- Recall directive appears once per session; body size adapts to score.
- Hook defaults: `TOKEN_BUDGET=160`, `TOP_K=1`, `FEEDBACK_HINT=0`.

- [x] Write failing tests for context-cost JSONL, directive-once, strict total budget, and hook command defaults.
- [x] Verify failures.
- [x] Implement the smallest shared behavior in daemon and subprocess renderers and log exact `additionalContext` characters.
- [x] Run recall/dashboard-log tests until green.

### Task 4: Report gross and net token estimates

**Files:**
- Modify: `web/build.py`
- Modify: `tests/test_dashboard_build.py`

**Interfaces:**
- `token_detail` adds `context_tokens`, `net`, and a per-kind `context_costs` mapping.
- `gerencial` adds `tokens_net` and `tokens_net_human`; existing gross fields remain backward compatible.

- [x] Write a failing dashboard test with one grounded recall and explicit briefing/recall costs.
- [x] Verify the missing net fields fail.
- [x] Subtract logged context tokens from estimated gross savings without clamping.
- [x] Run dashboard tests until green.

### Task 5: Make compression safe and slim the agent skill

**Files:**
- Modify: `src/memo/flags_misc.py`
- Modify: `skills/memo/SKILL.md`
- Modify: `README.md`
- Test: `tests/test_flags.py`

**Interfaces:**
- `MEMO_DREAM_COMPRESS_THRESHOLD` defaults to `0`; explicit values retain existing behavior.
- Root memo skill matches the compact Codex plugin router instead of embedding implementation history.

- [x] Write a failing test that the compression default is disabled.
- [x] Verify failure.
- [x] Change the default, document opt-in behavior, and replace the verbose skill with the compact router.
- [x] Run flag/profile documentation checks.

### Task 6: Full verification and publication

**Files:**
- Review: all changed files

- [x] Run `uv run --no-sync ruff check src/ tests/`.
- [x] Run `uv run --no-sync mypy src/memo`.
- [x] Run `uv run --no-sync pytest -m "not slow" -n auto --timeout=120 -q`.
- [x] Measure default/full/core MCP tool counts and schema characters.
- [ ] Commit with a concise Conventional Commit message.
- [ ] Push the verified branch commit directly to `origin/master` as explicitly requested.
