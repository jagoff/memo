# Memo Token Economy Options Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce token usage for memo-powered agents by making every high-frequency surface preview-first, dedup-aware, and profile-aware without breaking existing CLI or MCP compatibility.

**Architecture:** The savings come from four layers. First, stop injecting recall when the result set is already in context. Second, make search/list/resource outputs compact by default and reserve full bodies for explicit fetches. Third, shrink RAG prompt payloads and reuse session-local context where possible. Fourth, align defaults, metrics, and tests so the low-token behavior stays honest across profiles and does not drift back toward verbose outputs.

**Tech Stack:** Python 3.13, Click, FastMCP, pytest, rich, memo flags/config, memo dashboard logs.

## Global Constraints

- Do not read `MEMO_*` values with raw `os.environ.get(...)`; use `memo.flags`.
- Keep `mlx` / `mlx_lm` imports deferred inside functions.
- Tests must remain hermetic; use `tmp_cfg` or an isolated `Config` and never touch the real vault.
- Hook commands must preserve `MEMO_NONINTERACTIVE=1`.
- Markdown files are source of truth; sqlite is rebuildable.
- Preserve compatibility for existing CLI and MCP entry points; add opt-in flags instead of breaking default command names.

---

## File Structure

- `src/memo/cli_recall_hook.py` (modify) — skip rendering when dedup leaves no relevant hits.
- `src/memo/recall_logic.py` (modify) — add a compact recall format and keep budget trimming strict.
- `src/memo/cli_search.py` (modify) — add compact/full controls for JSON output on search, ask, and recall-style commands.
- `src/memo/server_resources.py` (modify) — cap `memo://memory/{id}` previews and keep full fetch explicit.
- `src/memo/server_core_search.py` (modify) — keep MCP search/ask outputs compact by default and pass session hints where available.
- `src/memo/memory/ask_ops.py` (modify) — lower default snippet payloads, reuse cached retrieval context, and cap verbatim short-circuit output for MCP-facing calls.
- `src/memo/flags_recall.py` (modify) — align recall defaults with the installed agent profile.
- `src/memo/surface.py` (modify) — fix token-cost estimates so doctor output matches reality.
- `tests/test_recall_hooks.py` (modify) — assert no empty recall envelope is injected after session dedup.
- `tests/test_server.py` or `tests/test_surface_profiles.py` (modify) — assert compact resource/search payloads and profile-visible behavior.
- `tests/test_memory_ask.py` (modify) — assert snippet sizing and cached retrieval behavior.

---

### Task 1: Stop useless recall injections

**Files:**
- Modify: `src/memo/cli_recall_hook.py`
- Modify: `src/memo/recall_logic.py`
- Test: `tests/test_recall_hooks.py`

**Interfaces:**
- Consumes: recalled-id session state from `memo.session`.
- Produces: recall hook returns `{}` when no unrecalled hits remain; rendered recall stays within budget and remains stable for first-turn injections.

- [ ] **Step 1: Write the failing test**

Add a regression test in `tests/test_recall_hooks.py` that seeds recalled IDs for a session, feeds the same top hit again, and asserts the hook bails instead of emitting the recall header/footer with no memories.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --no-sync pytest tests/test_recall_hooks.py::test_recall_hook_bails_when_all_hits_were_already_recalled -v`
Expected: fail because the hook still renders an empty recall envelope.

- [ ] **Step 3: Implement the minimal fix**

In `src/memo/cli_recall_hook.py`, after filtering `relevant = [h for h in relevant if h.id not in _prev_recalled]`, add a bail path when `not relevant`.

In `src/memo/recall_logic.py`, keep the current budget logic, but make sure the renderer never gets called with an empty hit list for this case.

- [ ] **Step 4: Run the test again**

Run: `uv run --no-sync pytest tests/test_recall_hooks.py::test_recall_hook_bails_when_all_hits_were_already_recalled -v`
Expected: pass.

- [ ] **Step 5: Commit**

Use a focused commit such as `fix: skip empty recall injections`.

---

### Task 2: Make search and resource outputs preview-first

**Files:**
- Modify: `src/memo/cli_search.py`
- Modify: `src/memo/server_resources.py`
- Modify: `src/memo/server_core_search.py`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `MemoryRecord.to_dict()` and current `memo search` / MCP search outputs.
- Produces: compact JSON by default, explicit full-body opt-in, and resource previews that do not dump full note bodies unless the user asks for them.

- [ ] **Step 1: Write the failing tests**

Add tests that assert:
1. `memo search --json` truncates `body` by default and marks it as truncated.
2. `memo://memory/{id}` returns a preview body or explicit truncation marker instead of the entire note body.
3. MCP `memo_search` preserves the compact `body_chars` behavior for programmatic callers.

- [ ] **Step 2: Run the tests to verify they fail**

Run:
`uv run --no-sync pytest tests/test_server.py::test_memo_search_resource_and_mcp_outputs_are_preview_first -v`

Expected: fail until the preview/full split is implemented.

- [ ] **Step 3: Implement the minimal fix**

In `src/memo/cli_search.py`, add a `--body-chars` option for JSON output and default it to a compact preview size. Keep `--json` as the same command, but make full bodies opt-in with `--full` or `--body-chars -1`.

In `src/memo/server_resources.py`, cap `memo://memory/{id}` output to a preview length and append a clear indicator that the caller should use `memo_get` for the full record.

In `src/memo/server_core_search.py`, keep `body_chars` as a first-class cap for MCP search and avoid regressing the default.

- [ ] **Step 4: Run the tests again**

Run: `uv run --no-sync pytest tests/test_server.py::test_memo_search_resource_and_mcp_outputs_are_preview_first -v`
Expected: pass.

- [ ] **Step 5: Commit**

Use a focused commit such as `feat: compact search and resource payloads`.

---

### Task 3: Reduce ask/RAG payloads

**Files:**
- Modify: `src/memo/memory/ask_ops.py`
- Modify: `src/memo/server_core_search.py`
- Modify: `src/memo/cli_search.py`
- Test: `tests/test_memory_ask.py`

**Interfaces:**
- Consumes: `snippet_chars`, `include_repos`, and optional `session_id`.
- Produces: smaller retrieved snippets, better reuse of cached context for repeated questions, and less duplicated source text in JSON/MCP envelopes.

- [ ] **Step 1: Write the failing tests**

Add tests that assert:
1. The CLI `memo ask` path uses a smaller default snippet size than the current 2000-character document snippet.
2. Passing a `session_id` enables reuse of the cached retrieval context on repeated asks.
3. The verbatim short-circuit path does not return oversized bodies when the caller is an agent-facing surface.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --no-sync pytest tests/test_memory_ask.py::test_ask_defaults_to_compact_snippets_and_reuses_session_cache -v`
Expected: fail until the default and cache behavior are changed.

- [ ] **Step 3: Implement the minimal fix**

In `src/memo/memory/ask_ops.py`, lower the default `snippet_chars` for agent-facing asks, keep the full body only for explicit `get`, and thread `session_id` into the retrieval cache path.

In `src/memo/server_core_search.py`, pass a session hint when the MCP layer has one available and keep `memo_ask` source snippets compact.

In `src/memo/cli_search.py`, expose the compact snippet sizing as an option so humans can opt into full context when needed.

- [ ] **Step 4: Run the tests again**

Run: `uv run --no-sync pytest tests/test_memory_ask.py::test_ask_defaults_to_compact_snippets_and_reuses_session_cache -v`
Expected: pass.

- [ ] **Step 5: Commit**

Use a focused commit such as `feat: shrink ask context payloads`.

---

### Task 4: Align defaults, format, and doctor metrics

**Files:**
- Modify: `src/memo/flags_recall.py`
- Modify: `src/memo/recall_logic.py`
- Modify: `src/memo/surface.py`
- Test: `tests/test_hook_contract.py`
- Test: `tests/test_surface_profiles.py`

**Interfaces:**
- Consumes: current agent hook defaults and profile reporting.
- Produces: a low-token agent profile whose defaults match installed behavior and whose doctor output reflects the real tool surface.

- [ ] **Step 1: Write the failing tests**

Add tests that assert:
1. The agent recall token budget default matches the installed hook behavior.
2. The compact recall renderer can emit an ultra-compact mode without breaking the first-turn directive behavior.
3. `mcp_profile_token_cost()` reports the correct tool-count/cost pair for `agent`, `core`, and `full`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --no-sync pytest tests/test_hook_contract.py tests/test_surface_profiles.py -v`
Expected: fail on the mismatched defaults and stale surface estimate.

- [ ] **Step 3: Implement the minimal fix**

In `src/memo/flags_recall.py`, align the default budget with the production hook target or make the active profile resolve to the smaller budget automatically.

In `src/memo/recall_logic.py`, add a compact format mode that removes avoidable markdown overhead while preserving citations and the closing footer.

In `src/memo/surface.py`, update the profile-cost table so `core` no longer reports the same tool cost as `agent`.

- [ ] **Step 4: Run the tests again**

Run: `uv run --no-sync pytest tests/test_hook_contract.py tests/test_surface_profiles.py -v`
Expected: pass.

- [ ] **Step 5: Commit**

Use a focused commit such as `chore: align token defaults and cost reporting`.

---

## Recommended Execution Order

1. Task 1 first: it is the lowest-risk token win and removes dead recall injections.
2. Task 2 next: it cuts the largest accidental payloads from CLI and MCP outputs.
3. Task 3 after that: it reduces repeated ask/RAG cost without changing user-facing semantics.
4. Task 4 last: it hardens the defaults and keeps the savings from drifting back.

## Coverage Check

- Recall path: Task 1 and Task 4.
- Search/resource outputs: Task 2.
- Ask/RAG path: Task 3.
- Profile/default honesty: Task 4.
- Regression guardrails: all tasks add or update tests on the exact serialized surfaces.
