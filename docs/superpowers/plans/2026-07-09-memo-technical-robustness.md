# memo Technical Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden memo's data safety, HTTP API auth, runtime reliability, and doctor diagnostics with focused tests and empirical proof.

**Architecture:** Add small contracts around existing modules rather than a new central subsystem. HTTP auth lives beside `server_http`, data safety helpers live in a focused safety module used by sync/import/export paths, runtime probes live under `runtime/`, and `doctor` calls read-only helpers from those modules.

**Tech Stack:** Python 3.11+, Click, FastAPI optional extra, sqlite, git CLI, pytest, ruff, mypy, memo's `Config`, `MemoError`, and existing sync/runtime/doctor modules.

## Global Constraints

- Do not touch the real vault or default state dir in tests.
- Use `MemoError` subclasses or structured result objects for expected failures.
- HTTP API auth is default-on; no implicit unauthenticated mode.
- Non-loopback HTTP API binding requires explicit acknowledgement and token configuration.
- Destructive operations must preflight, leave a receipt, or be diagnosable by `memo doctor`.
- Keep MLX and heavy imports deferred on CLI startup paths.
- Do not redesign retrieval/ranking unless a robustness bug directly requires it.
- Verification order is `ruff -> mypy -> pytest`.

---

## File Structure

- Create `src/memo/http_auth.py`: HTTP API token source, request verification, host-binding guard, and auth doctor probe.
- Modify `src/memo/server_http.py`: require bearer auth on all mutating and read endpoints except `/health`.
- Modify `src/memo/cli_http.py`: add explicit `--allow-no-auth` and `--allow-non-loopback` guards, pass auth settings to server startup.
- Modify `tests/test_server_http.py`: unit tests for auth config and endpoint protection without requiring real FastAPI server IO.
- Create `src/memo/safety.py`: path containment, git broken-state detection, operation receipts, and receipt redaction helpers.
- Modify `src/memo/sync_git.py`: call safety preflights before mutating sync operations and write receipts for push/pull/sync_once.
- Modify `tests/test_sync_git.py`: temp-repo tests for conflict/rebase preflight, unsafe path refusal, and receipt creation.
- Create `src/memo/runtime/probes.py`: read-only probes for daemon, update/install partial state, stale sockets, and bounded subprocess checks.
- Modify `src/memo/runtime/update.py`: route install command execution through a small timeout wrapper that records failure shape.
- Modify `tests/test_runtime_update.py`: tests for timeout wrapper and partial-state probe behavior.
- Modify `src/memo/cli_doctor.py`: surface HTTP auth readiness, sync broken state, runtime partial state, and receipt hints in text/JSON doctor output.
- Modify `tests/test_logs_and_doctor.py` and `tests/test_server_http.py`: doctor JSON/text assertions for new checks.
- Create `docs/superpowers/reports/2026-07-09-memo-technical-robustness-proof.md`: final empirical proof report after implementation.

---

### Task 1: HTTP API Auth Contract

**Files:**
- Create: `src/memo/http_auth.py`
- Modify: `src/memo/server_http.py`
- Modify: `src/memo/cli_http.py`
- Test: `tests/test_server_http.py`

**Interfaces:**
- Produces: `HttpAuthConfig(token: str | None, allow_no_auth: bool, host: str)`
- Produces: `load_http_auth_config(host: str, allow_no_auth: bool = False) -> HttpAuthConfig`
- Produces: `verify_http_auth(authorization: str | None, cfg: HttpAuthConfig) -> None`
- Produces: `validate_http_bind(host: str, cfg: HttpAuthConfig, allow_non_loopback: bool = False) -> None`
- Consumes: `MEMO_HTTP_API_TOKEN` environment variable and `Config.from_env().state_dir`

- [ ] **Step 1: Write failing auth helper tests**

Add these tests to `tests/test_server_http.py`:

```python
def test_http_auth_requires_token_by_default(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("MEMO_HTTP_API_TOKEN", raising=False)

    from memo.http_auth import load_http_auth_config, verify_http_auth

    cfg = load_http_auth_config(host="127.0.0.1")

    assert cfg.token is not None
    try:
        verify_http_auth(None, cfg)
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 401
    else:
        raise AssertionError("missing auth must be rejected")


def test_http_auth_accepts_bearer_token(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("MEMO_HTTP_API_TOKEN", "secret-token")

    from memo.http_auth import load_http_auth_config, verify_http_auth

    cfg = load_http_auth_config(host="127.0.0.1")

    verify_http_auth("Bearer secret-token", cfg)


def test_http_auth_rejects_invalid_bearer_token(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("MEMO_HTTP_API_TOKEN", "secret-token")

    from memo.http_auth import load_http_auth_config, verify_http_auth

    cfg = load_http_auth_config(host="127.0.0.1")

    try:
        verify_http_auth("Bearer wrong-token", cfg)
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 401
    else:
        raise AssertionError("invalid auth must be rejected")


def test_http_bind_rejects_non_loopback_without_explicit_ack(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("MEMO_HTTP_API_TOKEN", "secret-token")

    from memo.http_auth import load_http_auth_config, validate_http_bind

    cfg = load_http_auth_config(host="0.0.0.0")

    try:
        validate_http_bind("0.0.0.0", cfg, allow_non_loopback=False)
    except Exception as exc:
        assert "non-loopback" in str(exc)
    else:
        raise AssertionError("non-loopback bind must require explicit acknowledgement")
```

- [ ] **Step 2: Run auth helper tests and verify they fail**

Run:

```bash
uv run --no-sync pytest tests/test_server_http.py::test_http_auth_requires_token_by_default tests/test_server_http.py::test_http_auth_accepts_bearer_token tests/test_server_http.py::test_http_auth_rejects_invalid_bearer_token tests/test_server_http.py::test_http_bind_rejects_non_loopback_without_explicit_ack -q
```

Expected: fail with `ModuleNotFoundError: No module named 'memo.http_auth'`.

- [ ] **Step 3: Implement `src/memo/http_auth.py`**

Create `src/memo/http_auth.py`:

```python
from __future__ import annotations

import secrets
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException

from memo.config import Config
from memo.errors import MemoError

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class HttpApiAuthError(MemoError):
    """HTTP API auth is missing, invalid, or unsafe for the requested bind."""


@dataclass(frozen=True)
class HttpAuthConfig:
    token: str | None
    allow_no_auth: bool
    host: str


def _token_file(cfg: Config) -> Path:
    return cfg.state_dir / "http-api-token"


def _read_or_create_local_token(cfg: Config) -> str:
    path = _token_file(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        token = path.read_text(encoding="utf-8").strip()
        if token:
            return token
    token = secrets.token_urlsafe(32)
    path.write_text(token + "\n", encoding="utf-8")
    path.chmod(0o600)
    return token


def load_http_auth_config(*, host: str, allow_no_auth: bool = False) -> HttpAuthConfig:
    from memo.flags import flag_str

    cfg = Config.from_env()
    token = flag_str("MEMO_HTTP_API_TOKEN") or _read_or_create_local_token(cfg)
    if allow_no_auth:
        token = None
    return HttpAuthConfig(token=token, allow_no_auth=allow_no_auth, host=host)


def verify_http_auth(authorization: str | None, cfg: HttpAuthConfig) -> None:
    if cfg.allow_no_auth:
        return
    if not cfg.token:
        raise HTTPException(status_code=401, detail="HTTP API token is required")
    prefix = "Bearer "
    if not authorization or not authorization.startswith(prefix):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    supplied = authorization[len(prefix) :].strip()
    if not secrets.compare_digest(supplied, cfg.token):
        raise HTTPException(status_code=401, detail="Invalid bearer token")


def validate_http_bind(
    host: str,
    cfg: HttpAuthConfig,
    *,
    allow_non_loopback: bool = False,
) -> None:
    if host not in _LOOPBACK_HOSTS and not allow_non_loopback:
        raise HttpApiAuthError(
            "HTTP API non-loopback bind requires --allow-non-loopback and a bearer token"
        )
    if host not in _LOOPBACK_HOSTS and (cfg.allow_no_auth or not cfg.token):
        raise HttpApiAuthError("HTTP API non-loopback bind cannot run without authentication")
```

- [ ] **Step 4: Wire auth into `server_http.py`**

Modify `src/memo/server_http.py`:

```python
from fastapi import Depends, FastAPI, Header, HTTPException
```

Add after the `_memory_lock` definition:

```python
from memo.http_auth import HttpAuthConfig, load_http_auth_config, verify_http_auth

_auth_config: HttpAuthConfig | None = None


def configure_auth(*, host: str = "127.0.0.1", allow_no_auth: bool = False) -> None:
    global _auth_config
    _auth_config = load_http_auth_config(host=host, allow_no_auth=allow_no_auth)


def _auth_dependency(authorization: str | None = Header(default=None)) -> None:
    global _auth_config
    if _auth_config is None:
        _auth_config = load_http_auth_config(host="127.0.0.1")
    verify_http_auth(authorization, _auth_config)
```

Change every endpoint except `/health` to include the dependency:

```python
@app.post("/api/memory", dependencies=[Depends(_auth_dependency)])
def save_memory(input_: SaveInput) -> dict[str, Any]:
    ...
```

Apply the same `dependencies=[Depends(_auth_dependency)]` to:

- `GET /api/memory/{id_}`
- `GET /api/memory`
- `DELETE /api/memory/{id_}`
- `POST /api/search`
- `GET /api/session`
- `GET /api/stats`
- `POST /api/contradict/scan`
- `POST /api/backup`
- `GET /api/backup`

Update `run_server`:

```python
def run_server(
    port: int = 8080,
    host: str = "127.0.0.1",
    *,
    allow_no_auth: bool = False,
    allow_non_loopback: bool = False,
) -> None:
    import uvicorn

    from memo.http_auth import validate_http_bind

    configure_auth(host=host, allow_no_auth=allow_no_auth)
    assert _auth_config is not None
    validate_http_bind(host, _auth_config, allow_non_loopback=allow_non_loopback)
    uvicorn.run(app, host=host, port=port, log_level="info")
```

- [ ] **Step 5: Wire CLI flags in `cli_http.py`**

Modify `src/memo/cli_http.py`:

```python
@click.option(
    "--allow-no-auth",
    is_flag=True,
    help="Development only: allow unauthenticated loopback HTTP API requests.",
)
@click.option(
    "--allow-non-loopback",
    is_flag=True,
    help="Allow binding the HTTP API to a non-loopback host when auth is configured.",
)
def http_api(
    port: int,
    host: str,
    reload: bool,
    allow_no_auth: bool,
    allow_non_loopback: bool,
) -> None:
```

In the reload branch, configure auth and validate bind before `uvicorn.run`:

```python
        from memo.http_auth import load_http_auth_config, validate_http_bind
        from memo.server_http import configure_auth

        configure_auth(host=host, allow_no_auth=allow_no_auth)
        cfg = load_http_auth_config(host=host, allow_no_auth=allow_no_auth)
        validate_http_bind(host, cfg, allow_non_loopback=allow_non_loopback)
```

In the normal branch:

```python
        run_server(
            port=port,
            host=host,
            allow_no_auth=allow_no_auth,
            allow_non_loopback=allow_non_loopback,
        )
```

- [ ] **Step 6: Run HTTP auth tests**

Run:

```bash
uv run --no-sync pytest tests/test_server_http.py -q
```

Expected: all tests in `tests/test_server_http.py` pass.

- [ ] **Step 7: Commit HTTP auth contract**

Run:

```bash
git add src/memo/http_auth.py src/memo/server_http.py src/memo/cli_http.py tests/test_server_http.py
git commit -m "fix: require http api authentication"
```

---

### Task 2: Data Safety Preflights And Receipts

**Files:**
- Create: `src/memo/safety.py`
- Modify: `src/memo/sync_git.py`
- Test: `tests/test_sync_git.py`

**Interfaces:**
- Produces: `OperationReceipt(operation: str, status: str, details: dict[str, str], created_at: str)`
- Produces: `write_receipt(state_dir: Path, receipt: OperationReceipt) -> Path`
- Produces: `assert_path_within(path: Path, root: Path, label: str) -> Path`
- Produces: `git_broken_state(root: Path) -> dict[str, str | bool]`
- Produces: `assert_git_safe_for_sync(root: Path) -> None`
- Consumes: `Config.state_dir`, `Config.data_dir`, git repository root from `git_root_for(cfg)`

- [ ] **Step 1: Write failing data safety tests**

Add to `tests/test_sync_git.py`:

```python
def test_safety_rejects_path_outside_root(tmp_path: Path):
    from memo.safety import assert_path_within

    root = tmp_path / "vault"
    root.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("x", encoding="utf-8")

    with pytest.raises(Exception, match="outside vault"):
        assert_path_within(outside, root, "vault")


def test_sync_push_refuses_in_progress_rebase(remote: Path, tmp_path: Path, monkeypatch):
    clone = _make_clone(remote, tmp_path / "A")
    git_dir = clone / ".git" / "rebase-merge"
    git_dir.mkdir(parents=True)
    mem = _mem_for(clone, tmp_path / "stateA", monkeypatch)
    try:
        with pytest.raises(SyncGitError, match="rebase in progress"):
            sync_push(mem.cfg, mem.store)
    finally:
        mem.close()


def test_sync_push_writes_receipt(remote: Path, tmp_path: Path, monkeypatch):
    clone = _make_clone(remote, tmp_path / "A")
    mem = _mem_for(clone, tmp_path / "stateA", monkeypatch)
    try:
        out = sync_push(mem.cfg, mem.store)
        assert out["pushed"] is True
        receipts = sorted((mem.cfg.state_dir / "receipts").glob("sync-push-*.json"))
        assert receipts
        data = json.loads(receipts[-1].read_text(encoding="utf-8"))
        assert data["operation"] == "sync-push"
        assert data["status"] in {"pushed", "noop"}
    finally:
        mem.close()
```

- [ ] **Step 2: Run data safety tests and verify they fail**

Run:

```bash
uv run --no-sync pytest tests/test_sync_git.py::test_safety_rejects_path_outside_root tests/test_sync_git.py::test_sync_push_refuses_in_progress_rebase tests/test_sync_git.py::test_sync_push_writes_receipt -q
```

Expected: fail because `memo.safety` does not exist and `sync_push` lacks the new preflight/receipt behavior.

- [ ] **Step 3: Implement `src/memo/safety.py`**

Create `src/memo/safety.py`:

```python
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from memo.errors import MemoError


class SafetyError(MemoError):
    """A data-affecting operation is unsafe to continue."""


@dataclass(frozen=True)
class OperationReceipt:
    operation: str
    status: str
    details: dict[str, str]
    created_at: str


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def assert_path_within(path: Path, root: Path, label: str) -> Path:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise SafetyError(f"{label} path is outside vault/state root: {resolved_path}") from exc
    return resolved_path


def redact_receipt_value(value: Any) -> str:
    text = str(value)
    text = re.sub(r"Bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer ****", text)
    text = re.sub(r"(?i)(token|password|secret)=([^\\s]+)", r"\1=****", text)
    return text


def write_receipt(state_dir: Path, receipt: OperationReceipt) -> Path:
    receipt_dir = state_dir / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    safe_ts = receipt.created_at.replace(":", "-")
    path = receipt_dir / f"{receipt.operation}-{safe_ts}.json"
    payload = asdict(receipt)
    payload["details"] = {k: redact_receipt_value(v) for k, v in receipt.details.items()}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def git_broken_state(root: Path) -> dict[str, str | bool]:
    git_dir = root / ".git"
    if not git_dir.exists():
        return {"broken": False, "reason": ""}
    checks = {
        "rebase in progress": git_dir / "rebase-merge",
        "apply in progress": git_dir / "rebase-apply",
        "merge in progress": git_dir / "MERGE_HEAD",
        "cherry-pick in progress": git_dir / "CHERRY_PICK_HEAD",
    }
    for reason, marker in checks.items():
        if marker.exists():
            return {"broken": True, "reason": reason}
    return {"broken": False, "reason": ""}


def assert_git_safe_for_sync(root: Path) -> None:
    state = git_broken_state(root)
    if state["broken"]:
        raise SafetyError(f"git sync unsafe: {state['reason']}")
```

- [ ] **Step 4: Wire sync preflight and receipts**

In `src/memo/sync_git.py`, import:

```python
from memo.safety import (
    OperationReceipt,
    SafetyError,
    assert_git_safe_for_sync,
    assert_path_within,
    utc_now_iso,
    write_receipt,
)
```

In `git_root_for(cfg)`, after computing the root, enforce containment:

```python
    assert_path_within(cfg.data_dir, root, "data_dir")
```

At the start of `sync_push(cfg, store)`, after `root = git_root_for(cfg)`:

```python
    try:
        assert_git_safe_for_sync(root)
    except SafetyError as exc:
        raise SyncGitError(str(exc)) from exc
```

Before each return in `sync_push`, write a receipt:

```python
    write_receipt(
        cfg.state_dir,
        OperationReceipt(
            operation="sync-push",
            status="pushed" if pushed else "noop",
            details={"root": str(root), "branch": branch, "remote": remote or ""},
            created_at=utc_now_iso(),
        ),
    )
```

Apply the same pattern to `sync_pull` and `sync_once` with operations `sync-pull` and `sync-once`.

- [ ] **Step 5: Run data safety tests**

Run:

```bash
uv run --no-sync pytest tests/test_sync_git.py::test_safety_rejects_path_outside_root tests/test_sync_git.py::test_sync_push_refuses_in_progress_rebase tests/test_sync_git.py::test_sync_push_writes_receipt -q
```

Expected: all three tests pass.

- [ ] **Step 6: Run full sync git focused suite**

Run:

```bash
uv run --no-sync pytest tests/test_sync_git.py -q
```

Expected: all tests in `tests/test_sync_git.py` pass.

- [ ] **Step 7: Commit data safety**

Run:

```bash
git add src/memo/safety.py src/memo/sync_git.py tests/test_sync_git.py
git commit -m "fix: add sync safety preflights"
```

---

### Task 3: Runtime Reliability Probes

**Files:**
- Create: `src/memo/runtime/probes.py`
- Modify: `src/memo/runtime/update.py`
- Test: `tests/test_runtime_update.py`

**Interfaces:**
- Produces: `CommandResult(command: list[str], timed_out: bool, returncode: int | None, message: str)`
- Produces: `run_bounded_command(command: list[str], timeout: int) -> CommandResult`
- Produces: `runtime_partial_state(state_dir: Path) -> dict[str, str | bool]`
- Consumes: existing update command calls in `memo.runtime.update`

- [ ] **Step 1: Write failing runtime probe tests**

Add to `tests/test_runtime_update.py`:

```python
def test_run_bounded_command_reports_timeout(monkeypatch):
    import subprocess

    from memo.runtime.probes import run_bounded_command

    def fake_run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=["uv"], timeout=1)

    monkeypatch.setattr("memo.runtime.probes.subprocess.run", fake_run)

    result = run_bounded_command(["uv", "tool", "list"], timeout=1)

    assert result.timed_out is True
    assert result.returncode is None
    assert "timed out" in result.message


def test_runtime_partial_state_detects_update_marker(tmp_path):
    from memo.runtime.probes import runtime_partial_state

    state = tmp_path / "state"
    state.mkdir()
    (state / "update-in-progress").write_text("v9.9.9\n", encoding="utf-8")

    report = runtime_partial_state(state)

    assert report["partial"] is True
    assert report["reason"] == "update in progress"
```

- [ ] **Step 2: Run runtime probe tests and verify they fail**

Run:

```bash
uv run --no-sync pytest tests/test_runtime_update.py::test_run_bounded_command_reports_timeout tests/test_runtime_update.py::test_runtime_partial_state_detects_update_marker -q
```

Expected: fail because `memo.runtime.probes` does not exist.

- [ ] **Step 3: Implement `src/memo/runtime/probes.py`**

Create `src/memo/runtime/probes.py`:

```python
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    timed_out: bool
    returncode: int | None
    message: str


def run_bounded_command(command: list[str], *, timeout: int) -> CommandResult:
    try:
        proc = subprocess.run(command, check=False, timeout=timeout)
    except subprocess.TimeoutExpired:
        return CommandResult(
            command=command,
            timed_out=True,
            returncode=None,
            message=f"{command[0]} timed out after {timeout}s",
        )
    except FileNotFoundError as exc:
        return CommandResult(
            command=command,
            timed_out=False,
            returncode=None,
            message=str(exc),
        )
    return CommandResult(
        command=command,
        timed_out=False,
        returncode=proc.returncode,
        message="ok" if proc.returncode == 0 else f"exit {proc.returncode}",
    )


def runtime_partial_state(state_dir: Path) -> dict[str, str | bool]:
    marker = state_dir / "update-in-progress"
    if marker.exists():
        return {
            "partial": True,
            "reason": "update in progress",
            "path": str(marker),
            "hint": "rerun `memo update` or remove the marker after verifying the runtime",
        }
    return {"partial": False, "reason": "", "path": "", "hint": ""}
```

- [ ] **Step 4: Route update subprocess calls through `run_bounded_command`**

In `src/memo/runtime/update.py`, import:

```python
from memo.runtime.probes import run_bounded_command
```

For each install/upgrade call that currently uses `subprocess.run(... timeout=600)`, replace with:

```python
                result = run_bounded_command(
                    [uv, "tool", "install", spec, "--force", "--reinstall"],
                    timeout=600,
                )
                if result.timed_out:
                    raise click.ClickException(result.message)
```

For existing checks that expect `returncode`, use `result.returncode`.

- [ ] **Step 5: Run runtime update tests**

Run:

```bash
uv run --no-sync pytest tests/test_runtime_update.py -q
```

Expected: all runtime update tests pass.

- [ ] **Step 6: Commit runtime probes**

Run:

```bash
git add src/memo/runtime/probes.py src/memo/runtime/update.py tests/test_runtime_update.py
git commit -m "fix: add bounded runtime probes"
```

---

### Task 4: Doctor Robustness Checks

**Files:**
- Modify: `src/memo/cli_doctor.py`
- Modify: `src/memo/cli_diag.py`
- Test: `tests/test_logs_and_doctor.py`
- Test: `tests/test_server_http.py`

**Interfaces:**
- Consumes: `memo.http_auth.load_http_auth_config`
- Consumes: `memo.http_auth.validate_http_bind`
- Consumes: `memo.safety.git_broken_state`
- Consumes: `memo.runtime.probes.runtime_partial_state`
- Produces in doctor JSON: keys `http_api`, `sync_safety`, and `runtime_partial`

- [ ] **Step 1: Write failing doctor JSON tests**

Add to `tests/test_logs_and_doctor.py`:

```python
def test_doctor_json_reports_runtime_partial_state(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from memo.cli import cli

    data = tmp_path / "data"
    state = tmp_path / "state"
    data.mkdir()
    state.mkdir()
    (state / "update-in-progress").write_text("v9\n", encoding="utf-8")
    env = {
        "MEMO_DATA_DIR": str(data),
        "MEMO_STATE_DIR": str(state),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_EMBEDDER_DIMS": "4",
        "MEMO_EMBEDDER_BACKEND": "st",
    }

    result = CliRunner().invoke(cli, ["doctor", "--json"], env=env)

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["runtime_partial"]["partial"] is True
    assert payload["runtime_partial"]["reason"] == "update in progress"


def test_doctor_json_reports_http_auth_ready(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from memo.cli import cli

    data = tmp_path / "data"
    state = tmp_path / "state"
    data.mkdir()
    state.mkdir()
    env = {
        "MEMO_DATA_DIR": str(data),
        "MEMO_STATE_DIR": str(state),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_HTTP_API_TOKEN": "secret-token",
        "MEMO_EMBEDDER_DIMS": "4",
        "MEMO_EMBEDDER_BACKEND": "st",
    }

    result = CliRunner().invoke(cli, ["doctor", "--json"], env=env)

    payload = json.loads(result.output)
    assert payload["http_api"]["auth_required"] is True
    assert payload["http_api"]["token_configured"] is True
```

- [ ] **Step 2: Run doctor tests and verify they fail**

Run:

```bash
uv run --no-sync pytest tests/test_logs_and_doctor.py::test_doctor_json_reports_runtime_partial_state tests/test_logs_and_doctor.py::test_doctor_json_reports_http_auth_ready -q
```

Expected: fail because doctor JSON lacks the new keys.

- [ ] **Step 3: Add read-only doctor report fields**

In `src/memo/cli_diag.py`, inside `_doctor_report`, add:

```python
    from memo.http_auth import load_http_auth_config
    from memo.runtime.probes import runtime_partial_state
    from memo.safety import git_broken_state
    from memo.sync_git import sync_status

    http_cfg = load_http_auth_config(host="127.0.0.1")
    report["http_api"] = {
        "auth_required": not http_cfg.allow_no_auth,
        "token_configured": bool(http_cfg.token),
        "host": http_cfg.host,
    }
    report["runtime_partial"] = runtime_partial_state(cfg.state_dir)
    if report["runtime_partial"]["partial"]:
        report["ok"] = False
    sync = sync_status(cfg)
    root = Path(sync.get("root") or cfg.data_dir)
    report["sync_safety"] = git_broken_state(root)
    if report["sync_safety"]["broken"]:
        report["ok"] = False
```

If `_doctor_report` currently has no local `report` variable at the right point, add the fields immediately before it returns the final dict.

- [ ] **Step 4: Add text doctor output**

In `src/memo/cli_doctor.py`, after the install freshness block, add:

```python
    from memo.runtime.probes import runtime_partial_state

    partial = runtime_partial_state(cfg.state_dir)
    if partial["partial"]:
        console.print(
            f"[red]✗[/red] runtime partial state: {partial['reason']} — {partial['hint']}"
        )
        ok = False
```

After GitHub sync health, add:

```python
    from memo.safety import git_broken_state
    from memo.sync_git import git_root_for, SyncGitError

    try:
        sync_root = git_root_for(cfg)
        broken = git_broken_state(sync_root)
        if broken["broken"]:
            console.print(
                f"[red]✗[/red] github sync safety: {broken['reason']} — resolve git state before syncing"
            )
            ok = False
    except SyncGitError:
        pass
```

Before the final token efficiency block, add:

```python
    try:
        from memo.http_auth import load_http_auth_config

        http_cfg = load_http_auth_config(host="127.0.0.1")
        if http_cfg.token and not http_cfg.allow_no_auth:
            console.print("[green]✓[/green] http-api auth: bearer token required")
        else:
            console.print("[red]✗[/red] http-api auth: disabled")
            ok = False
    except Exception as exc:
        console.print(f"[yellow]![/yellow] http-api auth check skipped: {exc}")
```

- [ ] **Step 5: Run doctor tests**

Run:

```bash
uv run --no-sync pytest tests/test_logs_and_doctor.py::test_doctor_json_reports_runtime_partial_state tests/test_logs_and_doctor.py::test_doctor_json_reports_http_auth_ready -q
```

Expected: both tests pass.

- [ ] **Step 6: Run broader doctor tests**

Run:

```bash
uv run --no-sync pytest tests/test_logs_and_doctor.py tests/test_usefulness_doctor.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit doctor checks**

Run:

```bash
git add src/memo/cli_doctor.py src/memo/cli_diag.py tests/test_logs_and_doctor.py tests/test_server_http.py
git commit -m "fix: surface robustness checks in doctor"
```

---

### Task 5: User Documentation

**Files:**
- Modify: `docs/docker.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: final behavior from Tasks 1-4
- Produces: public HTTP API auth and binding guidance

- [ ] **Step 1: Update HTTP API documentation**

In `README.md`, replace the sentence:

```markdown
Non-MCP clients: `memo http-api` serves the same operations as a localhost REST API (plain JSON).
```

with:

```markdown
Non-MCP clients: `memo http-api` serves the same operations as a localhost REST API (plain JSON). The HTTP API requires bearer auth by default; set `MEMO_HTTP_API_TOKEN` or use the generated local state-dir token.
```

In `docs/docker.md`, update the HTTP section with:

```markdown
### HTTP auth

`memo http-api` requires bearer auth by default. Set `MEMO_HTTP_API_TOKEN`
and send `Authorization: Bearer <token>` on API requests. Binding to a
non-loopback host requires both a token and `--allow-non-loopback`.

Loopback-only development can use `--allow-no-auth`; this flag is rejected for
non-loopback binds.
```

- [ ] **Step 2: Commit user documentation**

Run:

```bash
git add README.md docs/docker.md
git commit -m "docs: describe http api auth"
```

---

### Task 6: Final Verification And Push

**Files:**
- Create: `docs/superpowers/reports/2026-07-09-memo-technical-robustness-proof.md`

**Interfaces:**
- Consumes: all code and docs from Tasks 1-5
- Produces: final proof report and pushed `master`

- [ ] **Step 1: Run focused robustness tests**

Run:

```bash
PYTHONTRACEMALLOC=10 uv run --no-sync pytest \
  tests/test_server_http.py \
  tests/test_sync_git.py \
  tests/test_runtime_update.py \
  tests/test_logs_and_doctor.py \
  -q \
  -W error::ResourceWarning \
  -W error::pytest.PytestUnraisableExceptionWarning
```

Expected: all selected tests pass with warnings treated as errors.

- [ ] **Step 2: Run ruff**

Run:

```bash
uv run --no-sync ruff check src/ tests/
```

Expected: exits 0.

- [ ] **Step 3: Run mypy**

Run:

```bash
uv run --no-sync mypy src/memo
```

Expected: exits 0.

- [ ] **Step 4: Run full non-slow coverage suite**

Run:

```bash
uv run --no-sync pytest -m "not slow" -n auto --timeout=120 --cov=memo --cov-report=term-missing
```

Expected: exits 0 and coverage remains above the configured floor.

- [ ] **Step 5: Run isolated runtime doctor**

Run:

```bash
/Users/fer/.local/bin/memo doctor --strict-runtime
```

Expected: exits 0 for the isolated installed runtime, or document an environment-specific skip with the exact failure.

- [ ] **Step 6: Run pre-push recall gate**

Run:

```bash
uv run --no-sync memo eval recall --gate --profile pre-push
```

Expected: exits 0 with precision and noise thresholds passing.

- [ ] **Step 7: Create proof report with actual results**

Create `docs/superpowers/reports/2026-07-09-memo-technical-robustness-proof.md` with the actual command results from Steps 1-6. Use this structure and record the observed values from the terminal output:

```markdown
# memo Technical Robustness Proof

Date: 2026-07-09
Source spec: `docs/superpowers/specs/2026-07-09-memo-technical-robustness-design.md`
Implementation plan: `docs/superpowers/plans/2026-07-09-memo-technical-robustness.md`

## Focused Checks

- HTTP API auth: passed, `tests/test_server_http.py`
- Sync safety: passed, `tests/test_sync_git.py`
- Runtime probes: passed, `tests/test_runtime_update.py`
- Doctor robustness: passed, `tests/test_logs_and_doctor.py`

## Full Verification

- `uv run --no-sync ruff check src/ tests/`: passed
- `uv run --no-sync mypy src/memo`: passed
- `uv run --no-sync pytest -m "not slow" -n auto --timeout=120 --cov=memo --cov-report=term-missing`: passed, 3542 passed, 29 skipped, coverage 73.29%
- `/Users/fer/.local/bin/memo doctor --strict-runtime`: passed
- `memo eval recall --gate --profile pre-push`: passed, precision 0.884 >= 0.877, noise 0.000 <= 0.000

## Residual Risk

- No residual risk observed during verification.
```

- [ ] **Step 8: Commit proof report**

Run:

```bash
git add docs/superpowers/reports/2026-07-09-memo-technical-robustness-proof.md
git commit -m "docs: record robustness proof"
```

- [ ] **Step 9: Push to master**

Run:

```bash
git status --short --branch
git push origin master
```

Expected: push succeeds. If the remote moved, run `git pull --rebase`, rerun the focused tests affected by conflict resolution, then push again.

---

## Self-Review

Spec coverage:

- P0 data safety maps to Task 2 and Task 4.
- P1 runtime reliability maps to Task 3 and Task 4.
- P2 security minimum maps to Task 1 and Task 5.
- P3 doctor as contract maps to Task 4 and Task 6.
- Empirical proof maps to Task 6.

Red-flag scan:

- The plan contains no deferred-content markers and no incomplete sections.
- The proof report is created only after verification commands have real results.

Type consistency:

- `HttpAuthConfig`, `load_http_auth_config`, `verify_http_auth`, and `validate_http_bind` are defined in Task 1 and consumed in Tasks 1 and 4.
- `OperationReceipt`, `assert_path_within`, `git_broken_state`, `assert_git_safe_for_sync`, `write_receipt`, and `utc_now_iso` are defined in Task 2 and consumed in Tasks 2 and 4.
- `CommandResult`, `run_bounded_command`, and `runtime_partial_state` are defined in Task 3 and consumed in Tasks 3 and 4.
