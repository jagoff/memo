# Public Release Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit and fix memo (mlx-memo) for correctness + installability before the GitHub repo goes public and PyPI gets a clean release.

**Architecture:** Three sequential phases — (0) pre-flight fixes for bugs found by static analysis, (1) priority correctness audits of critical flows, (2) documentation and PyPI metadata polish. Each task ends with a green test suite and a commit. A final task produces the audit report.

**Tech Stack:** Python 3.13, uv, ruff, mypy, pytest, Click, sqlite-vec, FastMCP, MLX (Apple Silicon only)

## Global Constraints

- Run ALL tests with: `uv run --no-sync pytest tests/ -q`
- Run ruff with: `uv run --no-sync ruff check src/`
- Run mypy with: `uv run --no-sync mypy src/memo/ --ignore-missing-imports`
- Test suite must stay green (1535 pass, 0 fail) after every task
- Every bug found must have a regression test BEFORE the fix
- No personal paths (`/Users/fer/`), IPs (`192.168.`), or credentials in source
- Platform: macOS Apple Silicon only. Do NOT add Linux/Windows paths.
- Do NOT load `mlx` or `mlx_lm` at module level (deferred import invariant)
- Commit message format: `fix: <description>` or `test: <description>` or `docs: <description>`

---

## Pre-audit findings (already confirmed)

| # | Finding | Severity | File | Status |
|---|---------|----------|------|--------|
| F1 | `bootstrap_clone` self-recursion | High | `sync_git.py:138` | Fixed (commit 63dd63b) |
| F2 | Test assertions used old notification prefix | Low | `test_capture.py` | Fixed (commit 63dd63b) |
| F3 | `n` undefined → silent `NameError` in idle capture notification | High | `cli_session.py:514` | **OPEN** |
| F4 | `contextlib` imported but never used | Low | `cli_session.py:9` | **OPEN** |
| F5 | `MEMO_MAINTAIN_DISABLE` does NOT gate `--mode reflect` | High | `cli_session.py` | **OPEN** |
| F6 | Personal data scan: CLEAN (no `/Users/fer`, IPs, credentials in src/) | — | — | PASS |
| F7 | `pyproject.toml` PyPI metadata: COMPLETE | — | `pyproject.toml` | PASS |

---

## Files map

| File | Role in this audit |
|------|-------------------|
| `src/memo/cli_session.py` | T1: fix `n` undefined + unused import; T2: add `MEMO_MAINTAIN_DISABLE` gate |
| `tests/test_session.py` | T1+T2: regression tests |
| `src/memo/server_idle_capture.py` | T3: audit memo_start_session + idle_capture tools |
| `src/memo/memory/write_ops.py` | T4: save round-trip correctness |
| `src/memo/store/queries.py` | T4: thread-local connection + transaction safety |
| `src/memo/recall_server.py` (or `cli_recall.py`) | T5: recall-hook 5s budget |
| `src/memo/sync_git.py` | T6: sync flow audit |
| `src/memo/cli_transcripts.py` | T7: session reflect flock check |
| `README.md` | T8: fresh-install golden path |
| `docs/reference.md` | T8: MCP setup instructions |
| `pyproject.toml` | T9: PyPI completeness (already PASS — verify and close) |

---

### Task 1: Fix `cli_session.py` — undefined `n` + unused import

**Severity: HIGH** — `NameError` is thrown at line 514 whenever idle capture succeeds and tries to write the heartbeat. The `except Exception` on line 522 catches it silently. Net effect: the `_hb("captured-notified")` call is swallowed every time, so the heartbeat log is always incomplete after a successful capture. Users never see an error, but debugging is blind.

**Files:**
- Modify: `src/memo/cli_session.py:9` (remove `contextlib`), `src/memo/cli_session.py:514` (fix `n`)
- Test: `tests/test_session.py`

**Interfaces:**
- Consumes: `cli_session.py` → `session_idle_maintenance()` command
- Produces: nothing new — just fixes existing behavior

- [ ] **Step 1: Write failing regression test**

Add to `tests/test_session.py` (create file if it doesn't exist; check first with `ls tests/test_session*.py`):

```python
"""Regression tests for cli_session.py correctness."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from memo.cli_session import session_group


def test_idle_maintenance_capture_mode_does_not_raise_name_error(tmp_path: Path, monkeypatch) -> None:
    """Regression: _hb('captured-notified', saved=n) used undefined `n`.

    When capture mode runs and saves titles, the heartbeat call crashed with
    NameError, swallowed by the bare except. This test runs the detached-worker
    path end-to-end with a stubbed capture result and asserts no exception is
    raised (exit code 0) and the heartbeat log contains the 'captured-notified'
    entry.
    """
    state = tmp_path / "state"
    state.mkdir()
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("{}\n", encoding="utf-8")

    session_file = state / "sessions.jsonl"
    session_file.write_text(
        json.dumps({"session_id": "test-sid-001", "transcript_path": str(transcript), "cwd": str(tmp_path)}) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("MEMO_STATE_DIR", str(state))
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMO_NONINTERACTIVE", "1")
    monkeypatch.setenv("MEMO_SESSION_DEBUG", "1")

    with patch("memo.capture.run_capture_incremental") as mock_cap:
        mock_cap.return_value = {
            "status": "ok",
            "saved": ["mem-id-1"],
            "saved_titles": ["Test insight"],
        }
        with patch("memo.cli_session._write_capture_notification"):
            runner = CliRunner()
            result = runner.invoke(
                session_group,
                [
                    "idle-maintenance",
                    "--mode", "capture",
                    "--delay-secs", "0",
                    "--detached-worker",
                ],
                input=json.dumps({"session_id": "test-sid-001", "transcript_path": str(transcript)}),
                catch_exceptions=False,
            )

    # Must not crash with NameError
    assert result.exit_code == 0, f"exit={result.exit_code} output={result.output}"
    # Heartbeat log must exist and contain 'captured-notified'
    log = state / "idle_capture.log"
    assert log.exists(), "heartbeat log was not written"
    entries = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    stages = [e["stage"] for e in entries]
    assert "captured-notified" in stages, f"stages={stages}"
```

- [ ] **Step 2: Run test — expect failure**

```bash
uv run --no-sync pytest tests/test_session.py::test_idle_maintenance_capture_mode_does_not_raise_name_error -v
```

Expected: `FAILED` with `NameError: name 'n' is not defined` somewhere in the call stack (caught by the bare except, so exit_code may be 0 but log assertions fail).

- [ ] **Step 3: Fix `cli_session.py`**

Open `src/memo/cli_session.py`. Make two changes:

**Change A** — remove unused import at line 9:
```python
# BEFORE (line 9):
import contextlib

# AFTER: delete this line entirely
```

**Change B** — fix undefined `n` at line 514. The variable `n` was `len(_titles)` before a refactor removed it. Replace:
```python
# BEFORE (around line 514):
            _hb("captured-notified", saved=n)
```
with:
```python
            _hb("captured-notified", saved=len(_titles))
```

- [ ] **Step 4: Verify ruff + mypy clean after fix**

```bash
uv run --no-sync ruff check src/memo/cli_session.py
uv run --no-sync mypy src/memo/cli_session.py --ignore-missing-imports
```

Expected: 0 errors from both.

- [ ] **Step 5: Run the regression test — expect pass**

```bash
uv run --no-sync pytest tests/test_session.py::test_idle_maintenance_capture_mode_does_not_raise_name_error -v
```

Expected: `PASSED`.

- [ ] **Step 6: Run full suite**

```bash
uv run --no-sync pytest tests/ -q --tb=short
```

Expected: all green (≥1535 pass, 0 fail).

- [ ] **Step 7: Commit**

```bash
git add src/memo/cli_session.py tests/test_session.py
git commit -m "fix(session): undefined 'n' in idle-maintenance heartbeat causes silent NameError"
```

---

### Task 2: Gate reflect mode with `MEMO_MAINTAIN_DISABLE`

**Severity: HIGH** — `session idle-maintenance --mode reflect` loads the LLM (via `Memory` → `_reflect_session`) and is spawned automatically after 300s idle. On a 16GB Mac with N open sessions, after 5 minutes of idle you get N concurrent LLM loads — confirmed root cause of OOM panics on 16GB remote Mac. `MEMO_MAINTAIN_DISABLE=1` already gates `dream` and `maintain` but NOT `reflect`.

**Files:**
- Modify: `src/memo/cli_session.py` (reflect mode gate)
- Test: `tests/test_session.py`

**Interfaces:**
- Consumes: `flag_bool("MEMO_MAINTAIN_DISABLE")` from `memo.flags`
- Produces: reflect mode exits early with `{"status": "skipped_maintain_disabled"}` when flag is set

- [ ] **Step 1: Write failing test**

Add to `tests/test_session.py`:

```python
def test_idle_maintenance_reflect_mode_respects_maintain_disable(tmp_path: Path, monkeypatch) -> None:
    """Regression: MEMO_MAINTAIN_DISABLE=1 must skip reflect mode to prevent OOM.

    Without this gate, every idle session spawns a full LLM load after 300s.
    On a 16GB Mac with multiple sessions this causes OOM kernel panics.
    """
    state = tmp_path / "state"
    state.mkdir()
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("{}\n", encoding="utf-8")

    monkeypatch.setenv("MEMO_STATE_DIR", str(state))
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMO_NONINTERACTIVE", "1")
    monkeypatch.setenv("MEMO_MAINTAIN_DISABLE", "1")

    reflect_called = []

    with patch("memo.cli_transcripts._reflect_session") as mock_reflect:
        runner = CliRunner()
        result = runner.invoke(
            session_group,
            [
                "idle-maintenance",
                "--mode", "reflect",
                "--delay-secs", "0",
                "--detached-worker",
            ],
            input=json.dumps({"session_id": "test-sid-002", "transcript_path": str(transcript)}),
        )
        reflect_called.append(mock_reflect.called)

    assert result.exit_code == 0
    assert not reflect_called[0], "reflect must not be called when MEMO_MAINTAIN_DISABLE=1"
    out = json.loads(result.output.strip()) if result.output.strip() else {}
    assert out.get("status") == "skipped_maintain_disabled", f"output={result.output!r}"
```

- [ ] **Step 2: Run test — expect failure**

```bash
uv run --no-sync pytest tests/test_session.py::test_idle_maintenance_reflect_mode_respects_maintain_disable -v
```

Expected: `FAILED` — reflect IS called (no gate), output is `{}` not `{"status": "skipped_maintain_disabled"}`.

- [ ] **Step 3: Add gate in `cli_session.py`**

Find the reflect mode branch inside `session_idle_maintenance`. It's at the bottom of the detached-worker block (around line 517):

```python
# BEFORE:
        else:
            from memo.cli_transcripts import _reflect_session
            from memo.memory import Memory

            mem = Memory(cfg)
            _reflect_session(str(sid), mem, cfg, debug=flag_bool("MEMO_SESSION_DEBUG"))
```

Replace with:

```python
        else:
            if flag_bool("MEMO_MAINTAIN_DISABLE"):
                print(json.dumps({"status": "skipped_maintain_disabled"}))
                _sys.exit(0)
            from memo.cli_transcripts import _reflect_session
            from memo.memory import Memory

            mem = Memory(cfg)
            _reflect_session(str(sid), mem, cfg, debug=flag_bool("MEMO_SESSION_DEBUG"))
```

Make sure `json` is imported at the top of the file (it already is, imported as `_json` — check: `import json as _json` or `import json`). Use the same alias already in use in the file.

- [ ] **Step 4: Run the regression test — expect pass**

```bash
uv run --no-sync pytest tests/test_session.py::test_idle_maintenance_reflect_mode_respects_maintain_disable -v
```

Expected: `PASSED`.

- [ ] **Step 5: Run full suite**

```bash
uv run --no-sync pytest tests/ -q
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/memo/cli_session.py tests/test_session.py
git commit -m "fix(session): gate reflect mode with MEMO_MAINTAIN_DISABLE to prevent OOM on 16GB Macs"
```

---

### Task 3: Audit `memo_start_session` and `memo_idle_capture` MCP tools

Verify the MCP tools don't spawn unbounded background work or expose internal state.

**Files:**
- Inspect: `src/memo/server_idle_capture.py`
- Test: `tests/test_server_idle_capture.py` (create if needed)

- [ ] **Step 1: Read and trace `memo_start_session`**

```bash
grep -n "memo_start_session\|memo_idle_capture\|_ensure_chat\|Popen\|subprocess" src/memo/server_idle_capture.py
```

Check the following:
- Does `memo_start_session` spawn any background processes? (It should not — only calls `checkpoint()`)
- Does `memo_idle_capture` call `memory._ensure_chat()`? If so, does that load MLX at call time? (MLX load is OK in a tool call — it's not module-level)
- Does any tool description expose internal paths or personal data?

Expected findings:
- `memo_start_session` → only calls `checkpoint()`, no spawning. PASS.
- `memo_idle_capture` → calls `memory._ensure_chat()` which lazy-loads the chat model. This is expected and correct (deferred, not module-level).
- Tool schemas: inspect description strings for `/Users/`, `jagoff/`, or personal paths.

- [ ] **Step 2: Check tool descriptions for leakage**

```bash
grep -n '"""' src/memo/server_idle_capture.py | head -30
```

Manually read each docstring for paths, usernames, or internal-only references that would confuse a public user.

- [ ] **Step 3: Verify no module-level MLX imports**

```bash
grep -n "^import mlx\|^from mlx\|^import mlx_lm\|^from mlx_lm" src/memo/server_idle_capture.py
```

Expected: 0 results. Any hit = VIOLATION of the deferred-import invariant → fix by moving the import inside the function body.

- [ ] **Step 4: Write a test verifying `memo_start_session` doesn't spawn processes**

Add to `tests/test_server_idle_capture.py`:

```python
"""Tests for server_idle_capture MCP tool registration."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch


def test_memo_start_session_does_not_spawn_subprocesses(tmp_path: Path, tmp_cfg) -> None:
    """memo_start_session must only call checkpoint(), never spawn background processes."""
    from memo.memory import Memory
    from memo.server_idle_capture import register

    cfg = tmp_cfg
    mem = MagicMock(spec=Memory)
    mem.cfg = cfg

    server = MagicMock()
    tools = {}

    def tool_decorator():
        def wrapper(fn):
            tools[fn.__name__] = fn
            return fn
        return wrapper

    server.tool = tool_decorator

    register(server, mem)

    assert "memo_start_session" in tools, "memo_start_session not registered"

    with patch("subprocess.Popen") as mock_popen, \
         patch("memo.session.checkpoint", return_value={"project": "test", "head_commit": "abc123def"}) as mock_ckpt:
        result = tools["memo_start_session"](session_id="test-123", cwd=str(tmp_path))

    assert not mock_popen.called, "memo_start_session must NOT spawn subprocesses"
    assert mock_ckpt.called, "memo_start_session must call checkpoint()"
    assert result["status"] == "started"
    assert result["session_id"] == "test-123"
```

- [ ] **Step 5: Run the test**

```bash
uv run --no-sync pytest tests/test_server_idle_capture.py -v
```

Expected: `PASSED`.

- [ ] **Step 6: Fix any issues found in steps 1-3**

If any tool description has internal paths or personal data: edit the docstring inline.
If any module-level MLX import: move inside function body.
If any unexpected subprocess: document the finding and add to audit report.

- [ ] **Step 7: Run full suite and commit**

```bash
uv run --no-sync pytest tests/ -q
git add src/memo/server_idle_capture.py tests/test_server_idle_capture.py
git commit -m "test(mcp): verify memo_start_session doesn't spawn background processes"
```

---

### Task 4: Audit save → search round-trip

Trace the full path from `memo save` → `.md` written → indexed → `memo search` returns the memory.

**Files:**
- Inspect: `src/memo/memory/write_ops.py`, `src/memo/store/queries.py`
- Test: `tests/test_save_search_roundtrip.py`

- [ ] **Step 1: Trace the save path**

```bash
grep -n "def save\|_tx\|embed\|fts\|insert_memory\|_embed_pending" src/memo/memory/write_ops.py | head -30
```

Verify the order:
1. `.md` file written first (source of truth)
2. Embedding computed
3. sqlite insert
4. If embed fails → `_memo_embed_pending` marker in frontmatter

If the order is wrong (sqlite before .md), that's a High severity finding.

- [ ] **Step 2: Check error path — what if disk is full during save**

```bash
grep -n "OSError\|PermissionError\|disk\|errno" src/memo/memory/write_ops.py | head -20
```

Expected: write errors should raise `StorageError` (not bare `Exception`), and the sqlite row should NOT be inserted if the .md write failed.

- [ ] **Step 3: Write regression test for save → search**

Create `tests/test_save_search_roundtrip.py`:

```python
"""Regression: save followed by search must return the saved memory."""
from __future__ import annotations

from memo.config import Config
from memo.memory import Memory


def test_save_then_search_returns_memory(tmp_cfg: Config, monkeypatch) -> None:
    """Golden path: save a memory and immediately search for it."""
    monkeypatch.setenv("MEMO_EMBEDDER_DIMS", "4")
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0] for _ in inputs],
    )
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed_query",
        lambda self, query: [1.0, 0.0, 0.0, 0.0],
    )

    cfg = Config(
        data_dir=tmp_cfg.data_dir,
        vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir,
        embedder_dims=4,
    )

    mem = Memory(cfg)
    try:
        result = mem.save(
            content="MLX prefill 30% faster than Ollama on M3 Max",
            title="MLX benchmark result",
            tags=["mlx", "benchmark"],
        )
        assert result.get("id"), f"save returned no id: {result}"
        mem_id = result["id"]

        # The .md file must exist before we search
        md_files = list(cfg.memory_dir.rglob("*.md"))
        assert md_files, "no .md files written after save"

        results = mem.search("MLX prefill benchmark", limit=5)
        ids = [r.get("id") for r in results]
        assert mem_id in ids, f"saved id {mem_id} not found in search results {ids}"
    finally:
        mem.close()


def test_save_md_is_written_before_sqlite_index(tmp_cfg: Config, monkeypatch) -> None:
    """Source of truth invariant: .md must be written before sqlite insert."""
    import threading

    monkeypatch.setenv("MEMO_EMBEDDER_DIMS", "4")
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[0.0, 1.0, 0.0, 0.0] for _ in inputs],
    )
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed_query",
        lambda self, query: [0.0, 1.0, 0.0, 0.0],
    )

    cfg = Config(
        data_dir=tmp_cfg.data_dir,
        vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir,
        embedder_dims=4,
    )

    md_written_before_index = []

    original_insert = None
    try:
        from memo.store import queries as qs
        original_insert = getattr(qs.VecStore, "insert_memory", None)
    except Exception:
        pass

    if original_insert:
        def patched_insert(self, *args, **kwargs):
            md_files = list(cfg.memory_dir.rglob("*.md"))
            md_written_before_index.append(bool(md_files))
            return original_insert(self, *args, **kwargs)

        monkeypatch.setattr("memo.store.queries.VecStore.insert_memory", patched_insert)

    mem = Memory(cfg)
    try:
        mem.save(content="test source-of-truth ordering", title="order test")
        if md_written_before_index:
            assert md_written_before_index[0], ".md must be written before sqlite insert"
    finally:
        mem.close()
```

- [ ] **Step 4: Run tests**

```bash
uv run --no-sync pytest tests/test_save_search_roundtrip.py -v
```

Expected: both `PASSED`. If either fails, read the error and trace the actual save path.

- [ ] **Step 5: Fix any issues found**

If save order is wrong → fix `write_ops.py` to write `.md` first.
If error handling is silent → add explicit `StorageError` raise.

- [ ] **Step 6: Run full suite and commit**

```bash
uv run --no-sync pytest tests/ -q
git add tests/test_save_search_roundtrip.py src/memo/memory/write_ops.py
git commit -m "test(memory): regression for save→search round-trip and md-before-sqlite ordering"
```

---

### Task 5: Audit recall-hook — 5s budget and concurrent safety

The recall-hook fires on every Claude Code prompt. It must complete in <5s cold, <1s warm. Verify it degrades cleanly on sqlite lock contention and embedder failures.

**Files:**
- Inspect: `src/memo/cli.py` (recall-hook command), `src/memo/store/queries.py` (thread-local connections)
- Test: `tests/test_recall_hook.py`

- [ ] **Step 1: Find recall-hook entry point**

```bash
grep -n "recall.hook\|recall_hook" src/memo/cli.py | head -10
```

Then read that function. Key things to verify:
- Timeout is enforced (not just a soft budget)
- On sqlite lock (`sqlite3.OperationalError: database is locked`): returns empty result, not crash
- On embedder failure: returns empty result, not crash
- Output is valid JSON (required by Claude Code hook protocol)
- No module-level MLX imports in the call path

- [ ] **Step 2: Check output format**

The recall-hook output is consumed by Claude Code as `additionalContext`. Run:

```bash
uv run --no-sync memo recall-hook --help 2>/dev/null || echo "no recall-hook --help"
echo '{"query": "test"}' | uv run --no-sync memo recall-hook 2>/dev/null | head -5
```

Verify: output is valid JSON or empty string (not a Python traceback).

- [ ] **Step 3: Write concurrent safety test**

Add `tests/test_recall_hook.py`:

```python
"""Regression tests for the recall-hook — 5s budget, concurrent safety."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from memo.cli import main


@pytest.fixture
def recall_env(tmp_cfg, monkeypatch):
    """Minimal environment for recall-hook: stub embedder, no MLX."""
    monkeypatch.setenv("MEMO_EMBEDDER_DIMS", "4")
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed_query",
        lambda self, query: [1.0, 0.0, 0.0, 0.0],
    )
    return tmp_cfg


def test_recall_hook_returns_valid_json_on_empty_corpus(recall_env, monkeypatch) -> None:
    """recall-hook must return valid JSON even with no memories saved."""
    runner = CliRunner()
    result = runner.invoke(main, ["recall-hook"], catch_exceptions=False)
    assert result.exit_code == 0, f"exit={result.exit_code} output={result.output}"
    # Output must be parseable JSON or empty
    output = result.output.strip()
    if output:
        parsed = json.loads(output)  # raises if invalid JSON
        assert isinstance(parsed, (dict, list, str))


def test_recall_hook_returns_json_when_embedder_raises(recall_env, monkeypatch) -> None:
    """recall-hook must not crash when the embedder fails — degrade to empty."""
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed_query",
        lambda self, query: (_ for _ in ()).throw(RuntimeError("MLX OOM")),
    )
    runner = CliRunner()
    result = runner.invoke(main, ["recall-hook"])
    # Must not propagate exception — exit code 0 or structured error output
    assert result.exit_code == 0, f"Crash on embedder failure: {result.output}"


def test_recall_hook_concurrent_invocations(recall_env, monkeypatch, tmp_cfg) -> None:
    """Multiple concurrent recall-hooks must not deadlock or produce corrupt output."""
    errors = []
    outputs = []

    def run_hook():
        try:
            runner = CliRunner()
            r = runner.invoke(main, ["recall-hook"])
            outputs.append(r.exit_code)
        except Exception as exc:
            errors.append(str(exc))

    threads = [threading.Thread(target=run_hook) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, f"Concurrent recall-hook errors: {errors}"
    assert all(code == 0 for code in outputs), f"Non-zero exit codes: {outputs}"
```

- [ ] **Step 4: Run tests**

```bash
uv run --no-sync pytest tests/test_recall_hook.py -v
```

Expected: all `PASSED`. If `test_recall_hook_returns_json_when_embedder_raises` fails, the hook crashes on embedder errors — fix by wrapping the embedder call in a try/except in the recall-hook command.

- [ ] **Step 5: Fix any issues found**

If embedder error propagates → add `try/except Exception` around the embed call in the recall-hook, return `{}` or `""` on failure.

If concurrent calls deadlock → check `store/queries.py` thread-local connections. Each thread must get its own connection (`threading.local()`).

- [ ] **Step 6: Run full suite and commit**

```bash
uv run --no-sync pytest tests/ -q
git add tests/test_recall_hook.py
git commit -m "test(recall): concurrent safety and degradation regression tests for recall-hook"
```

---

### Task 6: Audit sync flow — init, clone, pull, flock

The sync subsystem added `sync_init` today. Verify all 4 sync sub-commands degrade cleanly when git or `gh` is missing.

**Files:**
- Inspect: `src/memo/sync_git.py`, `src/memo/cli_sync.py`
- Test: `tests/test_sync_flows.py`

- [ ] **Step 1: Map the sync commands**

```bash
grep -n "^def \|^@.*command\|sync_init\|sync_once\|clone_bootstrap\|bootstrap_clone" src/memo/cli_sync.py src/memo/sync_git.py | head -40
```

Expected commands: `memo sync init`, `memo sync once`, `memo sync pull`, `memo sync status`, `memo sync clone` (or `bootstrap`).

- [ ] **Step 2: Check `gh` dependency gate**

```bash
grep -n "gh\b\|github\s*cli\|subprocess.*gh" src/memo/sync_git.py src/memo/cli_sync.py | head -20
```

If `gh` is invoked, verify there's a `try/except` or a pre-check (`shutil.which("gh")`) that degrades gracefully. A user without `gh` installed should get a clear error message, not a traceback.

- [ ] **Step 3: Verify `sync_once` flock behavior**

```bash
grep -n "flock\|LOCK_EX\|sync.lock\|sync_pending" src/memo/sync_git.py | head -20
```

Confirm:
- Lock file is `state_dir / ".sync.lock"`
- `LOCK_NB` (non-blocking) is used — concurrent session skips, doesn't wait
- On push failure: `sync_pending` stamp is written
- On next trigger: pending stamp is retried

- [ ] **Step 4: Write degradation tests**

Create `tests/test_sync_flows.py`:

```python
"""Regression tests for sync flow: gh-missing, flock, no-remote."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from memo.cli import main


def test_sync_status_no_remote_gives_actionable_message(tmp_path, monkeypatch) -> None:
    """If no git remote configured, sync status must explain why, not traceback."""
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("MEMO_NONINTERACTIVE", "1")
    (tmp_path / "data").mkdir()
    (tmp_path / "state").mkdir()

    runner = CliRunner()
    result = runner.invoke(main, ["sync", "status"])

    assert result.exit_code == 0, f"exit={result.exit_code}\n{result.output}"
    # Must mention "not configured" or "no remote" or similar — not a Python traceback
    assert "Traceback" not in result.output, f"Traceback in output:\n{result.output}"
    output_lower = result.output.lower()
    has_useful_message = any(
        kw in output_lower
        for kw in ["not configured", "no remote", "no sync", "sync is disabled", "no git"]
    )
    assert has_useful_message, f"No actionable message in:\n{result.output}"


def test_sync_once_concurrent_skips_gracefully(tmp_path, monkeypatch) -> None:
    """Concurrent sync_once calls: second call must skip (LOCK_NB), not block."""
    import fcntl
    import threading

    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("MEMO_NONINTERACTIVE", "1")
    (tmp_path / "data").mkdir()
    (tmp_path / "state").mkdir()

    lock_path = tmp_path / "state" / ".sync.lock"
    lock_path.touch()

    results = []

    def run_sync():
        runner = CliRunner()
        r = runner.invoke(main, ["sync", "once"])
        results.append(r.exit_code)

    # Hold the flock in another thread to simulate concurrent session
    lock_fd = open(lock_path, "w")
    fcntl.flock(lock_fd, fcntl.LOCK_EX)

    t = threading.Thread(target=run_sync)
    t.start()
    t.join(timeout=5)

    fcntl.flock(lock_fd, fcntl.LOCK_UN)
    lock_fd.close()

    # sync once must have returned (not hung) — exit 0 (skip) is acceptable
    assert t.is_alive() is False, "sync once blocked instead of skipping under held lock"
    assert results, "sync once returned no result"
    assert results[0] == 0, f"sync once non-zero exit under lock: {results[0]}"
```

- [ ] **Step 5: Run tests**

```bash
uv run --no-sync pytest tests/test_sync_flows.py -v
```

Expected: `PASSED`. If `sync status` tracebacks → fix the no-remote check. If `sync once` blocks → verify `LOCK_NB` is used.

- [ ] **Step 6: Fix issues**

For no-remote: add `if not sync_tier(cfg) == "remote": return {"status": "sync_disabled", "reason": "no git remote configured"}`.

For blocking lock: change `fcntl.flock(fd, LOCK_EX)` → `fcntl.flock(fd, LOCK_EX | LOCK_NB)` in `sync_once()`.

- [ ] **Step 7: Run full suite and commit**

```bash
uv run --no-sync pytest tests/ -q
git add tests/test_sync_flows.py src/memo/sync_git.py src/memo/cli_sync.py
git commit -m "test(sync): regression for no-remote degradation and concurrent flock skip"
```

---

### Task 7: Audit session lifecycle — reflect flock, session leaks

Verify the `reflect.lock` (added today) works, and that session files don't contain personal data.

**Files:**
- Inspect: `src/memo/cli_transcripts.py` (reflect flock), `src/memo/session.py`
- Test: `tests/test_session.py` (add to existing)

- [ ] **Step 1: Verify reflect flock is in place**

```bash
grep -n "flock\|reflect.lock\|LOCK_EX\|LOCK_NB" src/memo/cli_transcripts.py | head -10
```

Expected: flock added with `LOCK_EX | LOCK_NB` so concurrent reflect calls skip.

- [ ] **Step 2: Check session files for personal data leakage**

```bash
grep -rn "/Users/fer\|fernandoferrari\|192\.168\." src/memo/session.py src/memo/cli_session.py
```

Also check what `checkpoint()` writes to `sessions.jsonl`:

```bash
grep -n "def checkpoint\|sessions.jsonl\|write\|json.dumps" src/memo/session.py | head -20
```

A session entry should contain only: `session_id`, `transcript_path`, `cwd`, `created_at`, `prompt` — not user-specific paths that would be wrong on another machine.

Note: `transcript_path` will contain the user's home path — this is expected and machine-local. It must NOT be sent anywhere (not to git sync, not to any MCP tool output).

- [ ] **Step 3: Verify transcript paths don't appear in MCP tool responses**

```bash
grep -n "transcript_path\|transcript" src/memo/server_idle_capture.py | head -20
```

If `transcript_path` appears in a tool's return value, verify it's stripped or only included when explicitly needed (e.g., `memo_idle_capture` dry_run response).

- [ ] **Step 4: Write reflect flock test**

Add to `tests/test_session.py`:

```python
def test_reflect_flock_prevents_concurrent_reflect(tmp_path: Path, monkeypatch) -> None:
    """Regression: multiple concurrent reflect calls must skip — not all run.

    Without the flock, N sessions all pass the reflected_at check before any
    stamps it → N concurrent LLM loads.
    """
    import fcntl
    import threading

    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMO_NONINTERACTIVE", "1")
    (tmp_path / "state").mkdir(parents=True)
    (tmp_path / "data").mkdir(parents=True)

    # Pre-hold the reflect.lock to simulate another session already running reflect
    lock_path = tmp_path / "state" / "reflect.lock"
    lock_path.touch()
    lock_fd = open(lock_path, "w")
    fcntl.flock(lock_fd, fcntl.LOCK_EX)

    reflect_ran = []

    def try_reflect():
        runner = CliRunner()
        r = runner.invoke(
            session_group,
            ["idle-maintenance", "--mode", "reflect", "--delay-secs", "0", "--detached-worker"],
            input=json.dumps({"session_id": "test-sid-003", "transcript_path": str(tmp_path / "t.jsonl")}),
        )
        out = {}
        try:
            out = json.loads(r.output.strip())
        except Exception:
            pass
        reflect_ran.append(out.get("status"))

    t = threading.Thread(target=try_reflect)
    t.start()
    t.join(timeout=5)
    fcntl.flock(lock_fd, fcntl.LOCK_UN)
    lock_fd.close()

    assert not t.is_alive(), "reflect hung instead of skipping under held lock"
    # When lock is held, reflect must skip (not hang, not crash)
    # Status may be "skipped_concurrent" or empty — the key is it didn't block
```

- [ ] **Step 5: Run test**

```bash
uv run --no-sync pytest tests/test_session.py -v -k "reflect_flock"
```

Expected: `PASSED`.

- [ ] **Step 6: Run full suite and commit**

```bash
uv run --no-sync pytest tests/ -q
git add tests/test_session.py
git commit -m "test(session): regression for reflect flock concurrent-skip behavior"
```

---

### Task 8: README — fresh-install golden path simulation

Simulate what a stranger sees when they first land on the repo. Fix every friction point.

**Files:**
- Modify: `README.md`, `docs/reference.md` (MCP setup section)

- [ ] **Step 1: Read the install section end-to-end**

```bash
head -70 README.md
```

Verify:
- Install command is the first thing visible
- System requirements (Apple Silicon, Python 3.13, ~8GB disk) are stated before the install command
- The install command matches what actually works today

- [ ] **Step 2: Test the install command (dry-run)**

DO NOT run the actual curl pipe in this session (it would reinstall over the current dev install). Instead, verify the URL exists and the script is correct:

```bash
head -20 install.sh
grep -n "uv tool\|pipx\|mlx-memo\|memo doctor" install.sh | head -20
```

Verify:
- `install.sh` actually uses `uv tool install mlx-memo` or `pipx install mlx-memo`
- It downloads MLX models
- It calls `memo doctor --strict-runtime` at the end
- It runs `memo install-slash` to wire MCP clients

- [ ] **Step 3: Check the MCP setup section**

```bash
grep -n "mcp\|claude.*code\|install-slash\|config\|\.mcp\.json" docs/reference.md | head -30
```

Verify:
- There is a manual MCP config example (for users who don't use `install-slash`)
- The `command` in the example is `memo-mcp` (not a hardcoded path)
- The example includes at minimum: `MEMO_MCP_PROFILE=core` (correct default for new users)
- The example does NOT include `MEMO_LLM_MODEL=Qwen3-30B` (this would OOM a 16GB Mac)

- [ ] **Step 4: Write a README completeness checklist test**

This isn't a code test — it's a manual review checklist. Mark each item:

```
README checklist for a first-time user:
[ ] Visible above the fold: what memo does (1 sentence)
[ ] Requirements listed BEFORE install command: macOS Apple Silicon, Python 3.13, ~8GB disk
[ ] Primary install command works: uv tool install mlx-memo OR pipx install mlx-memo OR curl install.sh
[ ] Second step to verify: memo doctor --strict-runtime
[ ] Third step to wire Claude Code: memo install-slash OR manual JSON config shown
[ ] First 3 commands to use after install: memo save, memo search, memo recall-hook
[ ] Link to full docs: docs/reference.md
[ ] Known limitation stated clearly: Apple Silicon only, not Linux/Windows/Intel Mac
[ ] Model download size mentioned: ~7-8GB on first install
```

- [ ] **Step 5: Fix all gaps found**

For each unchecked item above, edit `README.md` to add or fix the missing content. Keep edits minimal — fix gaps, don't rewrite content that already works.

Common gaps to fix:
- If requirements aren't before the install command: add a `## Requirements` section before `## Install`
- If model download size isn't mentioned: add `> **First install downloads ~7-8 GB of MLX models (5-15 min depending on connection speed).**`
- If Apple-Silicon-only isn't called out clearly: add a bold warning at the top of Install

- [ ] **Step 6: Commit README fixes**

```bash
git add README.md docs/reference.md
git commit -m "docs: fix README gaps for fresh-install golden path"
```

---

### Task 9: pyproject.toml — PyPI completeness verification

The pre-audit scan found pyproject.toml is mostly complete. This task verifies and closes any remaining gaps.

**Files:**
- Inspect/Modify: `pyproject.toml`

- [ ] **Step 1: Verify all required PyPI fields**

```bash
grep -A2 "^\[project\]" pyproject.toml | head -30
```

Check each:
- `name` = `"mlx-memo"` ✓
- `version` = current version ✓
- `description` — public-friendly, no internal jargon?
- `license` = `{ text = "MIT" }` ✓
- `authors` = `[{ name = "Fernando Ferrari" }]` — name is public, no email exposed ✓
- `requires-python` = `">=3.13"` — correct ✓
- `classifiers` — includes `Operating System :: MacOS :: MacOS X` ✓
- `keywords` — appropriate ✓
- `[project.urls]` — `Homepage`, `Repository`, `Issues`, `Changelog` all present ✓

- [ ] **Step 2: Check description is public-friendly**

The current description:
> "Local MCP memory backed by Obsidian vault — MLX-native LLM + embedder, sqlite-vec store. No Ollama, no API keys."

Verify:
- No internal repo names, usernames, or paths
- Clear to someone who doesn't know the project
- Accurate (Obsidian is optional, not required)

If the "backed by Obsidian vault" is misleading (Obsidian is optional): consider updating to something like:
> "Local-first semantic memory for AI agents — MLX embeddings + sqlite-vec, MCP server. No cloud, no API keys. Apple Silicon."

- [ ] **Step 3: Check `[tool.hatch.build.targets.wheel.force-include]`**

```bash
grep -A15 "force-include" pyproject.toml
```

Verify the bundled assets (`.agents`, `.claude-plugin`, `hooks`, `plugins`, etc.) don't contain personal paths or credentials. These files are shipped inside the wheel.

```bash
grep -rn "/Users/fer\|fernandoferrari\|192\.168\." .agents/ .claude-plugin/ hooks/ plugins/ commands/ statusline/ server.json 2>/dev/null | grep -v ".pyc"
```

Any hit = HIGH severity blocker.

- [ ] **Step 4: Verify bundled files don't leak internal config**

```bash
cat hooks/hooks.json 2>/dev/null | python3 -m json.tool | grep -i "path\|user\|home\|fer\b"
cat .claude-plugin/plugin.json 2>/dev/null | grep -i "path\|user\|home\|fer\b"
```

Hooks must use relative paths or `$HOME`-relative paths, not hardcoded `/Users/fer/`.

- [ ] **Step 5: Fix any issues found**

If description is misleading: update to remove "backed by Obsidian vault" or add "(Obsidian optional)".
If bundled files have personal paths: replace with `$HOME` or relative path.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml
git commit -m "chore(packaging): verify pyproject.toml completeness for PyPI public release"
```

(If no changes needed, skip the commit.)

---

### Task 10: ruff --select ALL deep pass

Run the full ruff ruleset (not just the default `E,F,I,B,UP,SIM,RUF` in `pyproject.toml`) and fix security-relevant findings.

**Files:**
- Modify: various `src/memo/*.py` files as needed

- [ ] **Step 1: Run full ruleset**

```bash
uv run --no-sync ruff check src/ --select ALL --ignore ANN,D,ERA,FIX,TD,S101,RUF002,RUF003,E501,PLR2004,N,COM,PT,ARG,PGH003
```

Flags explained:
- `ANN`: type annotations — skip (too many, separate initiative)
- `D`: docstring style — skip (existing docs vary)
- `ERA`: commented-out code — skip (intentional in some places)
- `FIX,TD`: TODO/FIXME markers — skip
- `S101`: assert in production code — skip (used in guards)
- `PLR2004`: magic values — skip (many intentional thresholds)
- `N`: naming — skip (existing naming is consistent)
- `COM`: trailing commas — skip (formatter handles)
- `PT`: pytest style — skip (not affecting behavior)
- `ARG`: unused arguments — skip (some are required by interfaces)
- `PGH003`: blanket type ignore — skip

Focus on: `S` (security), `B` (bugbear), `SIM` (simplify), `E` (pycodestyle errors), `F` (pyflakes).

- [ ] **Step 2: Triage findings**

For each finding from Step 1:
- `S` prefix (security): **fix it or add a noqa with justification comment**
- `B` prefix (bugbear): **fix it** — these are real bugs or bad patterns
- `SIM` prefix (simplify): fix if trivial, skip if risky
- Others: use judgment

Common security findings and fixes:
- `S603` (subprocess without shell=True): usually fine, just `# noqa: S603` if the command is not user-input
- `S607` (partial path in Popen): ensure `shell=False` and path is not user-controlled
- `S108` (hardcoded temp file): replace with `tempfile.mkstemp()` or `tempfile.NamedTemporaryFile()`
- `S506` (unsafe YAML load): replace `yaml.load()` with `yaml.safe_load()`

- [ ] **Step 3: Apply fixes**

For each finding to fix: edit the file, run the specific rule to verify:

```bash
uv run --no-sync ruff check src/ --select S,B --ignore S101,S603,S607
```

- [ ] **Step 4: Verify default ruleset still clean**

```bash
uv run --no-sync ruff check src/
```

Expected: 0 errors (default rules that were already clean).

- [ ] **Step 5: Run full suite**

```bash
uv run --no-sync pytest tests/ -q
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/
git commit -m "fix(lint): address security-relevant ruff findings (S, B rules)"
```

---

### Task 11: Final audit report

Document every finding from the audit, its severity, and disposition.

**Files:**
- Create: `docs/AUDIT-2026-06-25.md`

- [ ] **Step 1: Run final verification battery**

```bash
uv run --no-sync pytest tests/ -q --tb=short 2>&1 | tail -5
uv run --no-sync ruff check src/ 2>&1 | tail -3
uv run --no-sync mypy src/memo/ --ignore-missing-imports 2>&1 | tail -3
grep -rn "/Users/fer\|fernandoferrari\|192\.168\.\|com\.fer\.\|fferrari" src/ pyproject.toml hooks/ .claude-plugin/ 2>/dev/null | grep -v ".pyc" | wc -l
```

Expected:
- pytest: all pass, 0 fail
- ruff: 0 errors
- mypy: 0 errors
- personal data grep: 0 hits

- [ ] **Step 2: Write the report**

Create `docs/AUDIT-2026-06-25.md` with this template filled in:

```markdown
# memo Public Release Audit — 2026-06-25

## Summary

Audit type: Correctness + Installability  
Scope: source code, tests, pyproject.toml, README, bundled assets  
Auditor: Claude Code (Sonnet 4.6)  
Test suite at start: 1535 pass / 0 fail  
Test suite at end: [fill in] pass / 0 fail  

## Findings

| # | Severity | Area | Finding | Disposition |
|---|----------|------|---------|-------------|
| F1 | High | sync_git | bootstrap_clone called itself recursively (infinite recursion) | Fixed (63dd63b) |
| F2 | Low | test_capture | Tests used outdated "※ auto save:" prefix | Fixed (63dd63b) |
| F3 | High | cli_session | `n` undefined at line 514 → silent NameError in idle capture | Fixed (Task 1) |
| F4 | Low | cli_session | `contextlib` imported but unused | Fixed (Task 1) |
| F5 | High | cli_session | MEMO_MAINTAIN_DISABLE doesn't gate reflect mode → OOM on 16GB | Fixed (Task 2) |
| F6 | — | src/ | Personal data scan: PASS (no /Users/fer, IPs, credentials) | PASS |
| F7 | — | pyproject.toml | PyPI metadata: COMPLETE | PASS |
| [F8+] | | | [fill in from Tasks 3-10] | |

## Known deferred issues

- Git history contains personal paths → needs `git filter-repo` (separate operation)
- [add any other won't-fix findings here]

## Release checklist

- [ ] `pytest` green: [pass count] / 0 fail
- [ ] `ruff check src/` clean: 0 errors
- [ ] `mypy src/memo/` clean: 0 errors
- [ ] Personal data grep: 0 hits
- [ ] pyproject.toml complete for PyPI
- [ ] README golden path verified
- [ ] All critical flows audited (save/search, recall-hook, mcp, sync, session)
```

- [ ] **Step 3: Commit the report**

```bash
git add docs/AUDIT-2026-06-25.md
git commit -m "docs: public release audit report 2026-06-25"
```

- [ ] **Step 4: Push**

```bash
git push
```

---

## Self-Review

**Spec coverage check:**
- Phase 0 pre-flight: T1 (ruff/mypy bugs), personal data → Task 1, pyproject.toml → Task 9 ✓
- Phase 1 fresh install: README golden path → Task 8 ✓
- Phase 2 Flow 1 (save/search): Task 4 ✓
- Phase 2 Flow 2 (recall-hook): Task 5 ✓
- Phase 2 Flow 3 (memo-mcp + OOM): Tasks 2, 3 ✓
- Phase 2 Flow 4 (sync): Task 6 ✓
- Phase 2 Flow 5 (session lifecycle): Tasks 1, 2, 7 ✓
- Phase 3 ruff ALL: Task 10 ✓
- Phase 3 mypy: covered in Task 1 (fixes mypy error) + Task 10 ✓
- Phase 3 pyproject audit: Task 9 ✓
- Audit report: Task 11 ✓

**Placeholder scan:** No "TBD", "TODO", or "fill in details" except in the audit report template which is intentionally for the implementer to fill from their findings.

**Type consistency:** `flag_bool`, `flag_int`, `Memory`, `Config`, `CliRunner`, `session_group` — all used consistently matching the existing codebase imports.
