# Offline Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make normal `memo-mcp` startup offline and configuration-read-only unless users explicitly opt in.

**Architecture:** Registered flags define update-check, auto-update, and self-heal policy. A small server startup helper creates only the background threads authorized by those flags; explicit update commands remain unchanged.

**Tech Stack:** Python 3.13, `threading`, memo flags, pytest monkeypatching.

## Global Constraints

- Behavioral flags are registered and read through `memo.flags`.
- Default MCP startup performs no network request and no agent-configuration mutation.
- Opted-in background failures remain non-fatal.
- Every production change starts with a focused failing test.

---

### Task 1: Opt-in flag defaults

**Files:**
- Modify: `src/memo/flags_misc.py`
- Modify: `src/memo/runtime/autoupdate.py`
- Modify: `src/memo/runtime/mcp.py`
- Modify: `.mcp.json`
- Modify: `.claude-plugin/plugin.json`
- Test: `tests/test_autoupdate.py`
- Test: `tests/test_cli_install_mcp.py`

**Interfaces:**
- Produces: `MEMO_UPDATE_CHECK_ENABLED=false`, `MEMO_AUTO_UPDATE=false`, `MEMO_STATUSLINE_SELFHEAL=false`, `MEMO_HOOK_SELFHEAL=false`.

- [ ] **Step 1: Add failing tests asserting all four defaults are disabled and generated MCP config does not force auto-update**

```python
assert flag_bool("MEMO_UPDATE_CHECK_ENABLED") is False
assert flag_bool("MEMO_AUTO_UPDATE") is False
assert flag_bool("MEMO_STATUSLINE_SELFHEAL") is False
assert flag_bool("MEMO_HOOK_SELFHEAL") is False
assert generated["env"].get("MEMO_AUTO_UPDATE") != "1"
```

- [ ] **Step 2: Run the two focused test files and observe current default-on failures**

Run: `uv run --no-sync pytest tests/test_autoupdate.py tests/test_cli_install_mcp.py -v`

- [ ] **Step 3: Register the new flag, set all defaults false, remove the raw environment special case, and stop injecting auto-update**

```python
FlagSpec("MEMO_UPDATE_CHECK_ENABLED", "bool", False, "Allow remote release notification checks."),
FlagSpec("MEMO_AUTO_UPDATE", "bool", False, "Allow background installation of tagged releases."),
```

- [ ] **Step 4: Run the focused tests and architecture-boundary test**

Run: `uv run --no-sync pytest tests/test_autoupdate.py tests/test_cli_install_mcp.py tests/test_architecture_boundaries.py -v`

- [ ] **Step 5: Commit flag and generated-config changes**

```bash
git add src/memo/flags_misc.py src/memo/runtime/autoupdate.py src/memo/runtime/mcp.py .mcp.json .claude-plugin/plugin.json tests/test_autoupdate.py tests/test_cli_install_mcp.py tests/test_architecture_boundaries.py
git commit -m "fix: make runtime updates opt in"
```

### Task 2: Gated startup threads

**Files:**
- Modify: `src/memo/server.py`
- Test: `tests/test_server_startup_policy.py`

**Interfaces:**
- Produces: `_start_background_tasks(cfg: Config) -> tuple[str, ...]` containing names of started tasks.

- [ ] **Step 1: Add failing sentinel tests for default silence and each opt-in branch**

```python
def test_background_tasks_are_silent_by_default(tmp_cfg, monkeypatch):
    started = _start_background_tasks(tmp_cfg)
    assert started == ()

@pytest.mark.parametrize("flag,expected", [
    ("MEMO_UPDATE_CHECK_ENABLED", ("update-check",)),
    ("MEMO_AUTO_UPDATE", ("update-check", "auto-update")),
    ("MEMO_STATUSLINE_SELFHEAL", ("statusline-selfheal",)),
    ("MEMO_HOOK_SELFHEAL", ("hook-selfheal",)),
])
def test_background_tasks_are_individually_opted_in(flag, expected):
    assert started == expected
```

- [ ] **Step 2: Run `uv run --no-sync pytest tests/test_server_startup_policy.py -v` and observe missing helper failure**

- [ ] **Step 3: Implement the helper with lazy imports and named daemon threads, then call it from `main()`**

```python
if flag_bool("MEMO_UPDATE_CHECK_ENABLED") or flag_bool("MEMO_AUTO_UPDATE"):
    start("update-check", notify_if_newer)
if flag_bool("MEMO_AUTO_UPDATE"):
    start("auto-update", lambda: maybe_auto_update(cfg))
```

- [ ] **Step 4: Run startup, autoupdate, hook, statusline, and runtime-isolation tests**

Run: `uv run --no-sync pytest tests/test_server_startup_policy.py tests/test_autoupdate.py tests/test_cli_hooks.py tests/test_runtime_isolation.py -v`

- [ ] **Step 5: Commit startup policy**

```bash
git add src/memo/server.py tests/test_server_startup_policy.py
git commit -m "fix: gate MCP background work by opt in"
```

### Task 3: Offline-default documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/privacy.md`
- Test: `tests/test_offline_docs.py`

**Interfaces:**
- Produces: explicit distinction between default startup and commands that intentionally use the network.

- [ ] **Step 1: Add a failing documentation contract test for the four flags and explicit-network operations**
- [ ] **Step 2: Run `uv run --no-sync pytest tests/test_offline_docs.py -v` and observe missing claims**
- [ ] **Step 3: Document defaults plus update, sync, benchmark/model download exceptions using the exact flag names**
- [ ] **Step 4: Run the documentation contract test**
- [ ] **Step 5: Commit documentation**

```bash
git add README.md docs/privacy.md tests/test_offline_docs.py
git commit -m "docs: define offline runtime defaults"
```
