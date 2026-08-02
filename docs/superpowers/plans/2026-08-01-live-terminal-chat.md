# Live Terminal Chat Implementation Plan

> **Status (2026-08-02): superseded; do not execute this plan.** The direct
> terminal-input implementation now fails closed: CLI/MCP `send` and `enter`
> mutators are removed, automatic registration is disabled, and legacy
> registrations cannot receive input. The unchecked tasks below are retained
> only as implementation history. Any replacement requires a receiver-bound
> API with explicit destination authority and a new reviewed plan.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add safe, immediate, bidirectional prompt and Return delivery between explicitly registered local Memo agent terminals.

**Architecture:** A focused `TerminalBridge` owns a local SQLite registry and receipts. A separate presenter validates and injects bytes into an exact TTY, while thin CLI and MCP modules expose the same service and the runtime shim performs registration.

**Tech Stack:** Python 3.11+, SQLite, Click, FastMCP/Pydantic, POSIX TTY ioctls, macOS AppleScript fallbacks, pytest.

## Global Constraints

- Only same-UID local character-device TTYs may register or receive input.
- Every delivery must revalidate PID start marker, TTY association, and foreground process group.
- Message bodies are UTF-8, at most 16 KiB, and cannot contain terminal control sequences or carriage returns.
- Duplicate idempotency keys never inject input twice.
- Tests use isolated `Config` state and pseudo-terminals, never the user's real vault or terminal.
- New CLI and MCP wiring stays thin and lives in domain modules.

---

### Task 1: Registry, validation, and receipts

**Files:**
- Create: `src/memo/terminal_live.py`
- Create: `tests/test_terminal_live.py`
- Modify: `src/memo/errors.py`

**Interfaces:**
- Produces: `TerminalBridge(cfg, presenter=None)`, `register(agent, tty, pid, terminal_app, project)`, `list()`, `send(target, message, sender=None, submit=True, message_id=None)`, `enter(target, sender=None, message_id=None)`, and `history(limit=50)`.
- Produces immutable `TerminalRegistration` and `TerminalReceipt` dataclasses serializable with `dataclasses.asdict`.

- [ ] **Step 1: Write failing registry tests**

```python
def test_register_persists_same_uid_tty_and_prunes_stale_process(tmp_cfg, pty_agent):
    bridge = TerminalBridge(tmp_cfg)
    registration = bridge.register(agent="codex", tty=pty_agent.tty, pid=pty_agent.pid)
    assert bridge.list()[0].id == registration.id
    pty_agent.stop()
    assert bridge.list() == []
```

- [ ] **Step 2: Run the registry test and confirm it fails because `memo.terminal_live` does not exist**

Run: `uv run --no-sync pytest tests/test_terminal_live.py::test_register_persists_same_uid_tty_and_prunes_stale_process -v`

- [ ] **Step 3: Implement the schema, process probe, same-UID/TTY validation, registration, listing, and stale pruning**

Use SQLite transactions, a random `term-<hex>` id, mode 0600 database file,
and `ps -p <pid> -o uid=,tty=,lstart=,pgid=,tpgid=,command=` for a portable
process snapshot. Raise `TerminalValidationError` for rejected targets.

- [ ] **Step 4: Add failing sanitization, foreground, idempotency, and receipt tests**

```python
def test_send_strips_escape_sequences_and_is_idempotent(registered_bridge, presenter):
    first = registered_bridge.send("term-a", "hello\x1b[31m!\r", message_id="msg-1")
    second = registered_bridge.send("term-a", "different", message_id="msg-1")
    assert presenter.payloads == [b"hello!\r"]
    assert second.receipt_id == first.receipt_id
```

- [ ] **Step 5: Implement bounded sanitization, foreground validation, delivery receipts, and idempotency**

The original message body must never appear in exceptions or persisted receipt
errors. `enter()` calls the presenter with `b"\r"` and no body.

- [ ] **Step 6: Run all terminal bridge tests**

Run: `uv run --no-sync pytest tests/test_terminal_live.py -v`

### Task 2: Real TTY presenter

**Files:**
- Create: `src/memo/terminal_presenter.py`
- Create: `tests/test_terminal_presenter.py`

**Interfaces:**
- Produces: `deliver_input(tty: Path, payload: bytes, *, terminal_app: str) -> str`, returning the transport name.
- Consumes only an already validated canonical TTY and sanitized payload.

- [ ] **Step 1: Write a failing PTY end-to-end test**

```python
def test_deliver_input_places_text_and_carriage_return_in_real_pty(pty_reader):
    transport = deliver_input(pty_reader.tty, b"hello\r", terminal_app="")
    assert pty_reader.read_exact(6) == b"hello\r"
    assert transport == "tiocsti"
```

- [ ] **Step 2: Run the PTY test and confirm it fails because the presenter is missing**

Run: `uv run --no-sync pytest tests/test_terminal_presenter.py::test_deliver_input_places_text_and_carriage_return_in_real_pty -v`

- [ ] **Step 3: Implement bytewise `TIOCSTI` delivery with descriptor cleanup**

Open only the exact path with `O_RDWR|O_NOCTTY`; close it in `finally`; convert
no bytes because the bridge already supplies the single submit CR.

- [ ] **Step 4: Write failing mocked fallback tests for Terminal, iTerm2, and Ghostty exact-TTY routing**

Each test forces `TIOCSTI` to raise, records the bounded `osascript` argv, and
asserts that the target TTY and payload are positional arguments rather than
interpolated script text. The Ghostty test also asserts a separate Return event.

- [ ] **Step 5: Implement bounded `osascript` fallbacks without shell execution**

Use `subprocess.run([...], timeout=5, check=False, capture_output=True,
text=True)`. Refuse unknown terminal applications when TIOCSTI fails.

- [ ] **Step 6: Run presenter tests on macOS and the bridge tests together**

Run: `uv run --no-sync pytest tests/test_terminal_presenter.py tests/test_terminal_live.py -v`

### Task 3: CLI, MCP, and shim wiring

**Files:**
- Create: `src/memo/cli_terminal.py`
- Create: `src/memo/server_terminal.py`
- Create: `tests/test_cli_terminal.py`
- Create: `tests/test_server_terminal.py`
- Modify: `src/memo/cli.py`
- Modify: `src/memo/server.py`
- Modify: `src/memo/surface.py`
- Modify: `src/memo/runtime/shims.py`
- Modify: related CLI/MCP surface and shim tests when their expected public inventory changes.

**Interfaces:**
- Produces Click group `terminal_group` and MCP tools `memo_terminal_list`, `memo_terminal_send`, `memo_terminal_enter`.
- Consumes `TerminalBridge(memory.cfg)` in MCP and `TerminalBridge(Config.from_env())` in CLI.

- [ ] **Step 1: Write failing CLI user-flow tests**

```python
def test_terminal_send_cli_returns_delivery_receipt(runner, isolated_env, registered_target):
    result = runner.invoke(cli, ["terminal", "send", "--to", registered_target, "--message", "ping", "--json"], env=isolated_env)
    assert result.exit_code == 0
    assert json.loads(result.output)["status"] == "delivered"
```

- [ ] **Step 2: Implement thin Click commands and register `terminal_group` in `cli.py`**

All domain failures become `click.ClickException`; JSON output uses
`dataclasses.asdict` and contains no message body.

- [ ] **Step 3: Write failing MCP contract tests**

Call each tool through FastMCP's in-memory client. Assert an agent profile lists
all three tools, a send returns a receipt, a duplicate returns the same receipt,
and an invalid target yields structured `status="failed"` without a traceback.

- [ ] **Step 4: Implement `server_terminal.register(server, memory)` and keep the tools on every MCP profile**

`memo_terminal_send` resolves the sender registration from inherited
`MEMO_AGENT_TTY`, wraps the body in a delimited live-message envelope with a
reply target, and delegates to `TerminalBridge`.

- [ ] **Step 5: Write a failing executable shim test**

Run an installed shim against a fake downstream agent and fake Memo executable;
assert registration receives the captured TTY, shell PID, agent, terminal app,
and project before the downstream `exec`.

- [ ] **Step 6: Wire best-effort registration into the shim and run all focused tests**

Run: `uv run --no-sync pytest tests/test_cli_terminal.py tests/test_server_terminal.py tests/test_runtime_shims.py tests/test_cli_mcp_surface_smoke.py -v`

### Task 4: End-user and repository verification

**Files:**
- Modify: user docs or command reference only if existing surface tests expose an undocumented command.

**Interfaces:**
- Consumes the finished CLI/MCP/shim bridge.
- Produces verification evidence and a controlled receipt from a separate Codex terminal.

- [ ] **Step 1: Run focused tests with warnings promoted to errors**

Run: `uv run --no-sync pytest tests/test_terminal_live.py tests/test_terminal_presenter.py tests/test_cli_terminal.py tests/test_server_terminal.py -W error -v`

- [ ] **Step 2: Run CI order**

Run: `uv run --no-sync ruff check src/ tests/`

Run: `uv run --no-sync mypy src/memo`

Run: `uv run --no-sync pytest -m "not slow" -n auto --timeout=120 --cov=memo --cov-report=term-missing`

- [ ] **Step 3: Install the branch runtime and run a controlled local exchange**

Register a known Codex PID/TTY pair, send a message with `--submit`, obtain its
reply receipt, and use `memo terminal enter` only against that exact id if the
terminal is visibly waiting for input.

- [ ] **Step 4: Commit, push, open a PR, wait for every required check, merge to protected `master`, and reinstall from the merged SHA**

Verify `memo doctor --strict-runtime --agent codex --agent claude-code --json`
reports the package path inside the isolated installed runtime.
