# M1 — MCP Surface Honesty + memo-MCP Dedup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Task 3 is an environment/ops change (edits the live `~/.claude.json`) — the controller executes it directly, NOT a subagent.**

**Goal:** Make the `agent` MCP profile self-honest (definition matches the 9 tools it actually exposes), fix `memo doctor`'s false "~118 tools / ~35k tokens" report for the agent profile, and collapse the two parallel memo MCP servers in the live environment down to one.

**Architecture:** The agent profile already works at runtime (`server.py` registers core+idle_capture modules, then `mcp_tools_to_remove()` strips `CORE − AGENT`, leaving 9 tools). The defects are cosmetic-but-misleading: `AGENT_MCP_TOOLS` lists only 5 of the 9, and `cli_doctor.py` re-reads the profile flag with the wrong default (`"default"` instead of the real `agent`) and has no branch for `agent`, so it reports the full-surface cost. Fixes are surgical: extend the frozenset, add a pure cost helper in `surface.py`, and point doctor at `mcp_profile()`. The environment has two memo MCP servers (a manual `~/.claude.json` entry + the `plugin:memo` plugin); dedup removes the manual one.

**Tech Stack:** Python 3.13, Click, Rich, FastMCP, pytest, mypy, ruff.

## Global Constraints

- Run tests with `uv run --no-sync pytest tests/...`; type-check with `uv run --no-sync mypy src/memo/`; lint with `uv run --no-sync ruff check src/`.
- Never read env via raw `os.environ` — use `flag_str`/`flag_bool` (or, for the resolved MCP profile, `memo.surface.mcp_profile()`).
- Files stay **< 800 lines**.
- The token-cost doctor line is an **advisory** (`[green]✓[/green]` reduced / `[yellow]![/yellow]` full); it must not flip `ok` / exit code.
- **Decision taken (2026-06-26):** the agent surface is **9 tools** (honest), NOT trimmed to 5. The 4 extra tools (`memo_idle_capture`, `memo_pop_notification`, `memo_start_session`, `memo_save_text`) are session/notification plumbing registered outside the advanced gate; they stay on the agent surface.
- Adding the 4 names to `AGENT_MCP_TOOLS` must NOT change `mcp_tools_to_remove()`: those 4 are not in `CORE_MCP_TOOLS`, so `CORE − AGENT` is unchanged and runtime behavior is identical — only the definition becomes truthful.

---

## File Structure

- `src/memo/surface.py` (modify) — extend `AGENT_MCP_TOOLS` to 9; add pure `mcp_profile_token_cost()` helper.
- `src/memo/flags_misc.py` (modify) — fix the `MEMO_MCP_PROFILE` description ("5 tools" → "9 tools").
- `src/memo/cli_doctor.py` (modify) — token-cost block uses `mcp_profile()` + the new helper.
- `tests/test_surface_profiles.py` (modify) — strengthen the agent test to exact equality; add helper + drop-exclusion tests.
- `~/.claude.json` (environment, Task 3) — remove the manual top-level `mcpServers.memo` entry. Not a repo file.

Tasks are independent; recommended order Task 1 → Task 2 → Task 3.

---

## Task 1: Make the agent surface definition honest (9 tools)

`AGENT_MCP_TOOLS` claims 5 tools but the agent profile exposes 9 at runtime. Lock the definition to runtime so the drift can't silently return.

**Files:**
- Modify: `src/memo/surface.py:45-53` (`AGENT_MCP_TOOLS`)
- Modify: `src/memo/flags_misc.py:83-88` (`MEMO_MCP_PROFILE` description)
- Modify: `tests/test_surface_profiles.py` (strengthen agent test + add unit test)

**Interfaces:**
- Consumes: `mcp_tools_to_remove()`, `build_server()` (unchanged).
- Produces: `AGENT_MCP_TOOLS` now has exactly 9 members.

- [ ] **Step 1: Write the failing tests**

In `tests/test_surface_profiles.py`, replace the body of `test_mcp_agent_profile_is_default_and_exposes_core_tools` (currently lines 80-109) so it asserts the built agent surface equals `AGENT_MCP_TOOLS` exactly:

```python
def test_mcp_agent_profile_is_default_and_exposes_core_tools(tmp_path, monkeypatch) -> None:
    from memo.surface import AGENT_MCP_TOOLS

    cfg = Config(
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        vault_path=tmp_path / "vault",
        embedder_dims=4,
    )
    monkeypatch.delenv("MEMO_MCP_PROFILE", raising=False)
    monkeypatch.delenv("MEMO_MCP_SLIM", raising=False)
    mem = Memory(cfg)
    try:
        tools = asyncio.run(build_server(memory=mem).list_tools())
        tool_names = {tool.name for tool in tools}
        # Definition must equal runtime — no silent drift either way.
        assert tool_names == set(AGENT_MCP_TOOLS)
        # Advanced tools the agent profile must NOT have
        assert "memo_graph_nodes" not in tool_names
        assert "memo_contradict_scan" not in tool_names
    finally:
        mem.close()
```

Append a pure unit test at the end of the file:

```python
def test_agent_tools_definition_is_nine_and_excludes_idle_from_removal(monkeypatch) -> None:
    monkeypatch.setenv("MEMO_MCP_PROFILE", "agent")
    monkeypatch.delenv("MEMO_MCP_SLIM", raising=False)
    from memo.surface import AGENT_MCP_TOOLS, mcp_tools_to_remove

    removed = mcp_tools_to_remove()
    assert len(AGENT_MCP_TOOLS) == 9
    for name in (
        "memo_idle_capture",
        "memo_pop_notification",
        "memo_start_session",
        "memo_save_text",
    ):
        assert name in AGENT_MCP_TOOLS
        assert name not in removed
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --no-sync pytest tests/test_surface_profiles.py::test_agent_tools_definition_is_nine_and_excludes_idle_from_removal tests/test_surface_profiles.py::test_mcp_agent_profile_is_default_and_exposes_core_tools -q`
Expected: FAIL — `len(AGENT_MCP_TOOLS) == 9` is `5`, and `tool_names == set(AGENT_MCP_TOOLS)` is `9 != 5`.

- [ ] **Step 3: Extend `AGENT_MCP_TOOLS`**

In `src/memo/surface.py`, replace the `AGENT_MCP_TOOLS` frozenset (lines 45-53) with:

```python
AGENT_MCP_TOOLS: frozenset[str] = frozenset(
    {
        "memo_ask",
        "memo_get",
        "memo_save",
        "memo_search",
        "memo_unified_briefing",
        # Session/notification plumbing registered by _srv_idle_capture outside
        # the advanced gate and never removed — so the real agent surface is 9.
        "memo_idle_capture",
        "memo_pop_notification",
        "memo_start_session",
        "memo_save_text",
    }
)
```

- [ ] **Step 4: Fix the flag description**

In `src/memo/flags_misc.py`, change the `MEMO_MCP_PROFILE` description (line 87) from:

```python
        "MCP surface profile: agent (default, 5 tools) | core/slim (stable core) | full/default (all tools).",
```

to:

```python
        "MCP surface profile: agent (default, 9 tools) | core/slim (stable core) | full/default (all tools).",
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_surface_profiles.py -q`
Expected: PASS (all profile tests green).

- [ ] **Step 6: Lint + types**

Run: `uv run --no-sync ruff check src/memo/surface.py src/memo/flags_misc.py && uv run --no-sync mypy src/memo/surface.py`
Expected: `All checks passed!` and `Success: no issues found`.

- [ ] **Step 7: Commit**

```bash
git add src/memo/surface.py src/memo/flags_misc.py tests/test_surface_profiles.py
git commit -m "fix(mcp): make AGENT_MCP_TOOLS honest (9 tools) and lock def to runtime"
```

---

## Task 2: Fix `memo doctor`'s false token-cost report for the agent profile

Doctor re-reads `MEMO_MCP_PROFILE` with default `"default"` and has no `agent` branch, so it warns "~118 tools / ~35k tokens" even though the live default profile is `agent` (~9 tools). Route doctor through the resolved profile and a pure cost helper.

**Files:**
- Modify: `src/memo/surface.py` (add `mcp_profile_token_cost`)
- Modify: `src/memo/cli_doctor.py:274-294` (token-cost block)
- Modify: `tests/test_surface_profiles.py` (add helper tests)

**Interfaces:**
- Consumes: `mcp_profile()` (already in `surface.py`).
- Produces: `mcp_profile_token_cost(profile: str | None = None) -> tuple[str, str, bool]` returning `(tool_count_label, token_label, is_reduced)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_surface_profiles.py`:

```python
def test_token_cost_recognizes_agent() -> None:
    from memo.surface import mcp_profile_token_cost

    count, cost, reduced = mcp_profile_token_cost("agent")
    assert count == "~9"
    assert cost == "~2.4k"
    assert reduced is True


def test_token_cost_full_is_not_reduced() -> None:
    from memo.surface import mcp_profile_token_cost

    for profile in ("full", "default"):
        count, cost, reduced = mcp_profile_token_cost(profile)
        assert count == "~118"
        assert cost == "~35k"
        assert reduced is False


def test_token_cost_active_default_resolves_to_agent(monkeypatch) -> None:
    monkeypatch.delenv("MEMO_MCP_PROFILE", raising=False)
    monkeypatch.delenv("MEMO_MCP_SLIM", raising=False)
    from memo.surface import mcp_profile_token_cost

    count, cost, reduced = mcp_profile_token_cost()
    assert count == "~9"
    assert reduced is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --no-sync pytest tests/test_surface_profiles.py -k token_cost -q`
Expected: FAIL — `ImportError: cannot import name 'mcp_profile_token_cost'`.

- [ ] **Step 3: Add the pure helper to `surface.py`**

In `src/memo/surface.py`, immediately after `mcp_tools_to_remove()` (after current line 118), add:

```python
# Per-profile token-cost estimates for the `memo doctor` advisory. Reduced
# profiles (agent/core/slim) are cheap; only the full/default surface warns.
_PROFILE_TOKEN_COST: dict[str, tuple[str, str]] = {
    "agent": ("~9", "~2.4k"),
    "core": ("~25", "~2.4k"),
    "slim": ("~25", "~2.4k"),
}


def mcp_profile_token_cost(profile: str | None = None) -> tuple[str, str, bool]:
    """Return ``(tool_count_label, token_label, is_reduced)`` for ``profile``
    (or the active profile when ``None``). ``is_reduced`` is False only for the
    full/default surface — the costly one doctor warns about."""
    resolved = profile if profile is not None else mcp_profile()
    count, cost = _PROFILE_TOKEN_COST.get(resolved, ("~118", "~35k"))
    return count, cost, resolved in _PROFILE_TOKEN_COST
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_surface_profiles.py -k token_cost -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Wire doctor to the resolved profile + helper**

In `src/memo/cli_doctor.py`, replace the token-efficiency block (current lines 274-294, from the `# Token efficiency summary` comment through the end of the `else:` `console.print(...)` for the full-profile warning — **stop before** the inner `try:` that imports `compute_roi`) with:

```python
    # Token efficiency summary — quick snapshot of profile cost and ROI.
    try:
        from memo.surface import mcp_profile, mcp_profile_token_cost

        _profile = mcp_profile()
        _tool_count, _tok_cost, _is_reduced = mcp_profile_token_cost(_profile)
        _profile_label = f"MEMO_MCP_PROFILE={_profile}"
        if _is_reduced:
            console.print(
                f"[green]✓[/green] token cost: {_profile_label}  {_tool_count} tools "
                f"({_tok_cost} tokens/connection)"
            )
        else:
            console.print(
                f"[yellow]![/yellow] token cost: {_profile_label}  {_tool_count} tools "
                f"({_tok_cost} tokens/connection)  "
                "[dim](set MEMO_MCP_PROFILE=agent or use `memo install-mcp --profile core` "
                "to reduce to ~9 tools / ~2.4k tokens for constrained clients)[/dim]"
            )
```

Leave everything from the inner `try: from memo.cli_roi import compute_roi` onward untouched.

- [ ] **Step 6: Lint, types, and a manual doctor smoke**

Run: `uv run --no-sync ruff check src/memo/surface.py src/memo/cli_doctor.py && uv run --no-sync mypy src/memo/surface.py src/memo/cli_doctor.py`
Expected: `All checks passed!` and `Success: no issues found`.

Run: `MEMO_NONINTERACTIVE=1 uv run --no-sync memo doctor 2>&1 | grep -i "token cost"`
Expected: a `[green]✓` line reading `token cost: MEMO_MCP_PROFILE=agent  ~9 tools (~2.4k tokens/connection)` — NOT `~118`.

Run: `MEMO_MCP_PROFILE=full MEMO_NONINTERACTIVE=1 uv run --no-sync memo doctor 2>&1 | grep -i "token cost"`
Expected: a `[yellow]!` warning line reading `~118 tools (~35k tokens/connection)`.

- [ ] **Step 7: Commit**

```bash
git add src/memo/surface.py src/memo/cli_doctor.py tests/test_surface_profiles.py
git commit -m "fix(doctor): report honest agent token cost (~9), resolve via mcp_profile()"
```

---

## Task 3: Dedup the live memo MCP servers (environment — controller-executed)

The environment runs **two** memo MCP servers: a manual top-level `mcpServers.memo` entry in `~/.claude.json` and the `plugin:memo` plugin. Both spawn `memo-mcp`; collapse to the plugin (it calls `memo-mcp` via PATH → the stable `~/.local/bin` shim, so it survives runtime changes). **Not a repo change, not TDD, not a subagent task** — edits live user config; the controller does it directly with a backup. (Prior incident 2026-06-25: never edit MCP config blindly — back up, validate, verify.)

- [ ] **Step 1: Snapshot the manual entry's env (so nothing is silently dropped)**

```bash
python3 -c "import json; d=json.load(open('/Users/fer/.claude.json')); print(json.dumps(d['mcpServers']['memo'], indent=2))"
```
Record the `env` keys (e.g. `MEMO_MCP_PROFILE`, `MEMO_AUTO_UPDATE`). The plugin relies on `MEMO_MCP_PROFILE`'s default (`agent`), so dropping that key is safe. If any OTHER env key here changes behavior the plugin won't get (e.g. a non-default `MEMO_AUTO_UPDATE`), note it — decide whether to set it on the plugin instead of silently losing it. Do not proceed past this step with an unexplained behavioral env key.

- [ ] **Step 2: Back up `~/.claude.json`**

```bash
cp ~/.claude.json ~/.claude.json.bak-m1dedup-$(date +%Y%m%d-%H%M%S)
```

- [ ] **Step 3: Confirm the surviving path resolves**

```bash
ls -l ~/.local/bin/memo-mcp && ~/.local/bin/memo-mcp --help >/dev/null 2>&1 && echo "memo-mcp OK"
```
Expected: the shim exists and runs. (This is what `plugin:memo` invokes via PATH.)

- [ ] **Step 4: Remove the manual `mcpServers.memo` entry (JSON-safe)**

```bash
python3 - <<'PY'
import json
p = "/Users/fer/.claude.json"
d = json.load(open(p))
removed = d.get("mcpServers", {}).pop("memo", None)
assert removed is not None, "no top-level mcpServers.memo entry found"
json.dump(d, open(p, "w"), indent=2)
print("removed manual memo entry")
PY
```

- [ ] **Step 5: Validate the JSON still parses and the plugin path is untouched**

```bash
python3 -c "import json; d=json.load(open('/Users/fer/.claude.json')); assert 'memo' not in d.get('mcpServers',{}), 'manual entry still present'; print('mcpServers now:', sorted(d.get('mcpServers',{})))"
```
Expected: valid JSON; `memo` absent from the top-level `mcpServers` list; everything else intact.

- [ ] **Step 6: Verify (user reconnects)**

Tell the user to run `/mcp` (stdio servers don't auto-respawn). After reconnect, only `plugin:memo` (`mcp__plugin_memo_memo__*`) should remain; the `mcp__memo__*` namespace is gone. Then:

```bash
MEMO_NONINTERACTIVE=1 uv run --no-sync memo doctor 2>&1 | grep -iE "MCP config|token cost"
```
Expected: the M2 MCP-config audit line stays `[green]✓` (no missing/venv-internal paths) and the token-cost line reads agent `~9`.

- [ ] **Step 7: No repo commit**

Task 3 changes only `~/.claude.json` (outside the repo). Record completion in the SDD ledger; there is nothing to commit. Keep the `.bak-m1dedup-*` backup until the user confirms the single-server setup is healthy.

---

## Final verification (after all three tasks)

- [ ] **Full suite + type + lint green**

Run: `uv run --no-sync pytest tests/ -q`
Expected: all pass (previous baseline + the new/changed surface-profile tests).

Run: `uv run --no-sync mypy src/memo/`
Expected: `Success: no issues found`.

Run: `uv run --no-sync ruff check src/`
Expected: `All checks passed!`

- [ ] **Push**

```bash
git push origin master
```

(Tasks 1-2 only; Task 3 has no commit.)

---

## Self-Review notes (author)

- **Spec correction:** the spec's premise — "el recorte `agent` no surte efecto en runtime, ~118 tools" — is FALSE. Verified (Explore, 2026-06-26): the agent profile already trims to 9 tools at runtime via `mcp_tools_to_remove()`. So M1's "verify and correct surface.py" resolves to **honesty fixes**, not a registration fix: Task 1 (definition matches runtime) + Task 2 (doctor stops lying). The spec's value claim ("~70k→2.4k") was inflated (assumed 118×2); real dedup saves one server + ~9 duplicated tools.
- **Decision logged:** agent = 9 tools (honest), per user 2026-06-26. The 4 idle/session/notification tools stay on the surface.
- **No behavior change in Task 1:** adding 4 non-CORE names to `AGENT_MCP_TOOLS` leaves `CORE − AGENT` unchanged → `mcp_tools_to_remove()` identical → runtime identical. The exact-equality test locks def↔runtime so future drift fails loudly.
- **Counts kept approximate:** core/slim stays "~25" (unchanged, surgical); only `agent` (~9) is added and the buggy `"default"` fallback is replaced by `mcp_profile()`. The stale `server.py:132` comment ("~116→~26") is out of scope (a comment, not the reported number) — noted, not touched.
- **Task 3 is ops, not code:** controller-executed with backup + JSON validation + reconnect verification, because it mutates the user's live `~/.claude.json`. No subagent, no TDD, no commit.
