# M2 — Release/Runtime Guardrails in `memo doctor` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three guardrails that prevent the two runtime gotchas seen on 2026-06-25 — a stale install hidden behind an unchanged version number, and MCP configs pointing at a deleted venv-internal binary — plus a `memo release --bump` helper that keeps the four version files in sync.

**Architecture:** Two new pure-Python modules under `src/memo/runtime/` (`freshness.py`, `mcp_config_audit.py`) expose testable functions; the existing `memo doctor` text path wires them in as two new check blocks. A new `src/memo/cli_release.py` adds a `memo release --bump {major,minor,patch}` command that edits `pyproject.toml`, `.claude-plugin/plugin.json`, `server.json`, and `CHANGELOG.md`. All logic lives in pure functions so it unit-tests without touching the real environment.

**Tech Stack:** Python 3.13, Click, Rich, `tomllib` (stdlib), pytest, mypy, ruff. Build backend is hatchling (no build hooks added — freshness is computed at runtime by content hash, not a build stamp).

## Global Constraints

- Python `requires-python = ">=3.13"` — `tomllib` is available in stdlib.
- Files stay **< 800 lines**; new modules are small and single-purpose.
- Run tests with `uv run --no-sync pytest tests/...`; type-check with `uv run --no-sync mypy src/memo/`; lint with `uv run --no-sync ruff check src/`.
- Never read env via raw `os.environ` — register a flag in `src/memo/flags_*.py` and use `flag_str(...)`.
- Version is synchronized across **four** files: `pyproject.toml` `[project].version`, `.claude-plugin/plugin.json` `"version"`, `server.json` (two `"version"` occurrences), `CHANGELOG.md`.
- CLI commands live in `src/memo/cli_<domain>.py` and register in `src/memo/cli.py` via `cli.add_command(...)`.
- Doctor text checks use `console.print` with markers `[green]✓[/green]` / `[yellow]![/yellow]` / `[red]✗[/red]`; a fatal check sets `ok = False`. These three guardrails are **warnings only** (do not flip `ok`), matching the existing non-fatal advisories.

---

## File Structure

- `src/memo/runtime/freshness.py` (new) — install-freshness comparison (installed package vs dev repo). Pure functions.
- `src/memo/runtime/mcp_config_audit.py` (new) — scan known MCP config files for memo command paths and classify problems. Pure functions.
- `src/memo/cli_release.py` (new) — `memo release --bump` command + pure version/file helpers.
- `src/memo/flags_misc.py` (modify) — register `MEMO_DEV_REPO`.
- `src/memo/cli_doctor.py` (modify) — wire the two new check blocks into the text path.
- `src/memo/cli.py` (modify) — import + register `release_cmd`.
- `tests/test_runtime_freshness.py` (new)
- `tests/test_mcp_config_audit.py` (new)
- `tests/test_cli_release.py` (new)

Tasks are independent and may be implemented in any order; recommended order is Task 1 → Task 2 → Task 3.

---

## Task 1: Install-freshness guardrail

Detects "same version, different content" — the exact failure where `1.0.12` shipped two different builds.

**Files:**
- Create: `src/memo/runtime/freshness.py`
- Create: `tests/test_runtime_freshness.py`
- Modify: `src/memo/flags_misc.py` (register `MEMO_DEV_REPO`)
- Modify: `src/memo/cli_doctor.py` (wire check into text path)

**Interfaces:**
- Produces:
  - `package_content_hash(pkg_dir: Path) -> str`
  - `read_pyproject_version(repo_dir: Path) -> str | None`
  - `check_install_freshness(installed_version: str, installed_pkg_dir: Path, repo_dir: Path | None) -> dict[str, str]` returning `{"status": str, "message": str}` where `status ∈ {"skipped","repo-ahead","fresh","stale"}`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_runtime_freshness.py`:

```python
from __future__ import annotations

from pathlib import Path

from memo.runtime.freshness import (
    check_install_freshness,
    package_content_hash,
    read_pyproject_version,
)


def _make_repo(tmp_path: Path, version: str, body: str) -> Path:
    repo = tmp_path / "repo"
    pkg = repo / "src" / "memo"
    pkg.mkdir(parents=True)
    (repo / "pyproject.toml").write_text(
        f'[project]\nname = "mlx-memo"\nversion = "{version}"\n', encoding="utf-8"
    )
    (pkg / "__init__.py").write_text(body, encoding="utf-8")
    return repo


def _make_installed(tmp_path: Path, body: str) -> Path:
    pkg = tmp_path / "site-packages" / "memo"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(body, encoding="utf-8")
    return pkg


def test_read_pyproject_version(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, "1.0.13", "x = 1\n")
    assert read_pyproject_version(repo) == "1.0.13"


def test_content_hash_changes_with_content(tmp_path: Path) -> None:
    a = _make_installed(tmp_path / "a", "x = 1\n")
    b = _make_installed(tmp_path / "b", "x = 2\n")
    assert package_content_hash(a) != package_content_hash(b)


def test_fresh_when_same_version_same_content(tmp_path: Path) -> None:
    body = "VALUE = 1\n"
    repo = _make_repo(tmp_path, "1.0.13", body)
    installed = _make_installed(tmp_path, body)
    out = check_install_freshness("1.0.13", installed, repo)
    assert out["status"] == "fresh"


def test_stale_when_same_version_different_content(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, "1.0.13", "VALUE = 2\n")
    installed = _make_installed(tmp_path, "VALUE = 1\n")
    out = check_install_freshness("1.0.13", installed, repo)
    assert out["status"] == "stale"
    assert "reinstall" in out["message"]


def test_repo_ahead_when_versions_differ(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, "1.0.14", "VALUE = 9\n")
    installed = _make_installed(tmp_path, "VALUE = 1\n")
    out = check_install_freshness("1.0.13", installed, repo)
    assert out["status"] == "repo-ahead"


def test_skipped_when_no_repo() -> None:
    out = check_install_freshness("1.0.13", Path("/nonexistent"), None)
    assert out["status"] == "skipped"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_runtime_freshness.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'memo.runtime.freshness'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/memo/runtime/freshness.py`:

```python
"""Install-freshness check: catch a stale install hidden behind an unchanged
version number (a build whose number matches the repo but whose bytes don't).

Pure functions; the doctor command supplies real paths. No build hooks — the
signal is a runtime content hash of the installed package vs the dev repo.
"""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path


def package_content_hash(pkg_dir: Path) -> str:
    """sha256 over sorted (relative-path, file-bytes) of every ``*.py`` under
    ``pkg_dir``. Order-independent across machines; ignores caches by globbing
    only source files."""
    h = hashlib.sha256()
    for p in sorted(pkg_dir.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        h.update(p.relative_to(pkg_dir).as_posix().encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def read_pyproject_version(repo_dir: Path) -> str | None:
    """Return ``[project].version`` from ``repo_dir/pyproject.toml`` or None."""
    pp = repo_dir / "pyproject.toml"
    if not pp.exists():
        return None
    data = tomllib.loads(pp.read_text(encoding="utf-8"))
    version = data.get("project", {}).get("version")
    return str(version) if version else None


def check_install_freshness(
    installed_version: str,
    installed_pkg_dir: Path,
    repo_dir: Path | None,
) -> dict[str, str]:
    """Compare the installed package against a dev repo.

    status:
      - "skipped":    no usable dev repo (not configured / missing files)
      - "repo-ahead": repo version != installed version (normal during dev)
      - "fresh":      same version AND identical content
      - "stale":      same version BUT content differs -> reinstall needed
    """
    if repo_dir is None:
        return {"status": "skipped", "message": "no dev repo configured (MEMO_DEV_REPO)"}
    repo_pkg = repo_dir / "src" / "memo"
    repo_version = read_pyproject_version(repo_dir)
    if repo_version is None or not repo_pkg.is_dir():
        return {"status": "skipped", "message": f"dev repo not usable: {repo_dir}"}
    if repo_version != installed_version:
        return {
            "status": "repo-ahead",
            "message": f"repo {repo_version} != installed {installed_version}",
        }
    if package_content_hash(installed_pkg_dir) == package_content_hash(repo_pkg):
        return {"status": "fresh", "message": f"installed matches repo @ {installed_version}"}
    return {
        "status": "stale",
        "message": (
            f"installed {installed_version} differs from repo at same version — "
            f"reinstall: uv tool install --reinstall {repo_dir}"
        ),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/test_runtime_freshness.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Register the `MEMO_DEV_REPO` flag**

In `src/memo/flags_misc.py`, add this spec inside the `SPECS` tuple, right after the `MEMO_AGENT_TTY` spec (search for `"MEMO_AGENT_TTY"`):

```python
    _spec(
        "MEMO_DEV_REPO",
        "str",
        "",
        "misc",
        "Absolute path to the memo source repo (e.g. ~/repos/memo). When set, "
        "`memo doctor` compares the installed package against this repo and warns "
        "if the installed build is stale despite an identical version number.",
    ),
```

- [ ] **Step 6: Wire the check into `memo doctor` text path**

In `src/memo/cli_doctor.py`, add these imports near the top (after the existing `from memo.config import Config` line). `import importlib.metadata` goes with the stdlib imports at the very top:

```python
import importlib.metadata
from memo.flags import flag_str
from memo.runtime.freshness import check_install_freshness
```

Then, inside `doctor(...)`, immediately after the runtime-report block (after the lines that call `_print_runtime_install_report(runtime_report)` and handle `strict_runtime`), insert. `Path(__file__).parent` is the installed `memo` package dir (this file lives inside it), so no self-import is needed:

```python
    _dev_repo = flag_str("MEMO_DEV_REPO")
    freshness = check_install_freshness(
        importlib.metadata.version("mlx-memo"),
        Path(__file__).parent,
        Path(_dev_repo).expanduser() if _dev_repo else None,
    )
    if freshness["status"] == "stale":
        console.print(f"[yellow]![/yellow] install freshness: {freshness['message']}")
    elif freshness["status"] == "fresh":
        console.print(f"[green]✓[/green] install freshness: {freshness['message']}")
```

- [ ] **Step 7: Verify lint, types, and a manual smoke run**

Run: `uv run --no-sync ruff check src/memo/runtime/freshness.py src/memo/cli_doctor.py src/memo/flags_misc.py`
Expected: `All checks passed!`

Run: `uv run --no-sync mypy src/memo/runtime/freshness.py`
Expected: `Success: no issues found`.

Run: `MEMO_DEV_REPO=$PWD MEMO_NONINTERACTIVE=1 uv run --no-sync memo doctor 2>&1 | grep -i freshness`
Expected: a line containing `install freshness:` (status `fresh` when the working tree matches the installed build, or `repo-ahead` if the repo version is higher).

- [ ] **Step 8: Commit**

```bash
git add src/memo/runtime/freshness.py tests/test_runtime_freshness.py src/memo/flags_misc.py src/memo/cli_doctor.py
git commit -m "feat(doctor): warn when installed build is stale vs dev repo at same version"
```

---

## Task 2: MCP config path audit

Detects MCP configs whose memo command points at a missing or venv-internal binary — the breakage caused by removing the pipx install.

**Files:**
- Create: `src/memo/runtime/mcp_config_audit.py`
- Create: `tests/test_mcp_config_audit.py`
- Modify: `src/memo/cli_doctor.py` (wire check into text path)

**Interfaces:**
- Produces:
  - `extract_memo_command_paths(text: str) -> list[str]`
  - `classify_command_path(path: str, exists: bool) -> str | None` returning `"venv-internal"`, `"missing"`, or `None` (ok)
  - `audit_config_file(path: Path) -> list[dict[str, str]]`
  - `audit_mcp_configs(config_paths: tuple[str, ...] = DEFAULT_CONFIG_PATHS) -> list[dict[str, str]]` — each dict has keys `config`, `command`, `issue`.
  - `DEFAULT_CONFIG_PATHS: tuple[str, ...]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_mcp_config_audit.py`:

```python
from __future__ import annotations

from pathlib import Path

from memo.runtime.mcp_config_audit import (
    audit_config_file,
    classify_command_path,
    extract_memo_command_paths,
)


def test_extract_paths_from_json_and_yaml_text() -> None:
    text = (
        '{"command": "/Users/x/.local/bin/memo-mcp"}\n'
        "command: /Users/x/.local/pipx/venvs/mlx-memo/bin/memo\n"
        'other: "/usr/bin/python"\n'
    )
    paths = extract_memo_command_paths(text)
    assert "/Users/x/.local/bin/memo-mcp" in paths
    assert "/Users/x/.local/pipx/venvs/mlx-memo/bin/memo" in paths
    assert "/usr/bin/python" not in paths


def test_classify_venv_internal() -> None:
    assert (
        classify_command_path("/Users/x/.local/pipx/venvs/mlx-memo/bin/memo-mcp", True)
        == "venv-internal"
    )


def test_classify_missing() -> None:
    assert classify_command_path("/no/such/memo-mcp", False) == "missing"


def test_classify_ok_shim() -> None:
    assert classify_command_path("/Users/x/.local/bin/memo-mcp", True) is None


def test_audit_config_file_flags_missing(tmp_path: Path) -> None:
    cfg = tmp_path / "claude.json"
    cfg.write_text('{"command": "/no/such/path/memo-mcp"}', encoding="utf-8")
    issues = audit_config_file(cfg)
    assert len(issues) == 1
    assert issues[0]["issue"] == "missing"
    assert issues[0]["command"] == "/no/such/path/memo-mcp"


def test_audit_config_file_ok_when_path_exists(tmp_path: Path) -> None:
    real = tmp_path / "bin" / "memo-mcp"
    real.parent.mkdir()
    real.write_text("#!/bin/sh\n", encoding="utf-8")
    cfg = tmp_path / "claude.json"
    cfg.write_text(f'{{"command": "{real}"}}', encoding="utf-8")
    assert audit_config_file(cfg) == []


def test_audit_missing_config_file_is_empty(tmp_path: Path) -> None:
    assert audit_config_file(tmp_path / "nope.json") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_mcp_config_audit.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'memo.runtime.mcp_config_audit'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/memo/runtime/mcp_config_audit.py`:

```python
"""Audit known MCP client configs for memo command paths that won't survive a
runtime change. Catches the 2026-06-25 breakage where configs hardcoded a
pipx venv-internal path that vanished when pipx was uninstalled.

Format-agnostic: extracts absolute paths ending in ``/memo`` or ``/memo-mcp``
from the raw config text (works for JSON, JSONC, and YAML alike).
"""

from __future__ import annotations

import re
from pathlib import Path

# Absolute paths whose final component is exactly ``memo`` or ``memo-mcp``.
# ``memo-mcp`` is listed first so the alternation prefers the longer match.
_CMD_RE = re.compile(r"(/[^\s\"']+/(?:memo-mcp|memo))(?=[\"'\s]|$)")

DEFAULT_CONFIG_PATHS: tuple[str, ...] = (
    "~/.claude.json",
    "~/.config/devin/config.json",
    "~/.config/opencode/opencode.jsonc",
    "~/.config/mcp-gateway/gateway.yaml",
)


def extract_memo_command_paths(text: str) -> list[str]:
    """Return the unique absolute memo/memo-mcp command paths mentioned in text."""
    return sorted(set(_CMD_RE.findall(text)))


def classify_command_path(path: str, exists: bool) -> str | None:
    """Return an issue tag, or None when the path is a stable choice.

    "venv-internal" — points inside a pipx/uv/site-packages venv (fragile;
    breaks on reinstall). "missing" — the file does not exist.
    """
    if "/pipx/venvs/" in path or "/.venv/" in path or "/site-packages/" in path:
        return "venv-internal"
    if not exists:
        return "missing"
    return None


def audit_config_file(path: Path) -> list[dict[str, str]]:
    """Audit one config file. Returns one dict per problematic command path."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    issues: list[dict[str, str]] = []
    for command in extract_memo_command_paths(text):
        issue = classify_command_path(command, Path(command).exists())
        if issue:
            issues.append({"config": str(path), "command": command, "issue": issue})
    return issues


def audit_mcp_configs(
    config_paths: tuple[str, ...] = DEFAULT_CONFIG_PATHS,
) -> list[dict[str, str]]:
    """Audit all known MCP config files, expanding ``~``."""
    issues: list[dict[str, str]] = []
    for raw in config_paths:
        issues.extend(audit_config_file(Path(raw).expanduser()))
    return issues
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/test_mcp_config_audit.py -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Wire the audit into `memo doctor` text path**

In `src/memo/cli_doctor.py`, add the import near the other runtime imports:

```python
from memo.runtime.mcp_config_audit import audit_mcp_configs
```

Then, inside `doctor(...)`, immediately after the freshness block added in Task 1, insert:

```python
    mcp_issues = audit_mcp_configs()
    if not mcp_issues:
        console.print(
            "[green]✓[/green] MCP config paths: memo commands resolve to a stable binary"
        )
    else:
        for issue in mcp_issues:
            shim = f"~/.local/bin/{Path(issue['command']).name}"
            console.print(
                f"[yellow]![/yellow] MCP config {issue['config']}: {issue['issue']} "
                f"command {issue['command']} → repoint to {shim}"
            )
```

- [ ] **Step 6: Verify lint, types, and a manual smoke run**

Run: `uv run --no-sync ruff check src/memo/runtime/mcp_config_audit.py src/memo/cli_doctor.py`
Expected: `All checks passed!`

Run: `uv run --no-sync mypy src/memo/runtime/mcp_config_audit.py`
Expected: `Success: no issues found`.

Run: `MEMO_NONINTERACTIVE=1 uv run --no-sync memo doctor 2>&1 | grep -i "MCP config"`
Expected: a line about `MCP config paths` (the `✓` form when all configs are clean).

- [ ] **Step 7: Commit**

```bash
git add src/memo/runtime/mcp_config_audit.py tests/test_mcp_config_audit.py src/memo/cli_doctor.py
git commit -m "feat(doctor): audit MCP configs for missing/venv-internal memo command paths"
```

---

## Task 3: `memo release --bump` helper

Bumps the version across the four source-of-truth files in one command, so content changes can't ship under a reused number.

**Files:**
- Create: `src/memo/cli_release.py`
- Create: `tests/test_cli_release.py`
- Modify: `src/memo/cli.py` (import + register `release_cmd`)

**Interfaces:**
- Produces:
  - `bump_version(current: str, level: str) -> str`
  - `read_current_version(repo: Path) -> str`
  - `apply_bump(repo: Path, old: str, new: str, date: str, *, dry_run: bool) -> list[str]` — returns a human-readable list of changes; writes files unless `dry_run`.
  - `release_cmd` — Click command registered as `memo release`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli_release.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from memo.cli_release import apply_bump, bump_version, read_current_version


def _fake_repo(tmp_path: Path, version: str) -> Path:
    repo = tmp_path / "repo"
    (repo / ".claude-plugin").mkdir(parents=True)
    (repo / "pyproject.toml").write_text(
        f'[project]\nname = "mlx-memo"\nversion = "{version}"\ndescription = "x"\n',
        encoding="utf-8",
    )
    (repo / ".claude-plugin" / "plugin.json").write_text(
        f'{{\n  "name": "memo",\n  "version": "{version}"\n}}\n', encoding="utf-8"
    )
    (repo / "server.json").write_text(
        f'{{\n  "version": "{version}",\n  "packages": [\n'
        f'    {{ "version": "{version}" }}\n  ]\n}}\n',
        encoding="utf-8",
    )
    (repo / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [Unreleased]\n\n## [%s] - 2026-06-01\n" % version,
        encoding="utf-8",
    )
    return repo


@pytest.mark.parametrize(
    "current,level,expected",
    [
        ("1.0.13", "patch", "1.0.14"),
        ("1.0.13", "minor", "1.1.0"),
        ("1.0.13", "major", "2.0.0"),
    ],
)
def test_bump_version(current: str, level: str, expected: str) -> None:
    assert bump_version(current, level) == expected


def test_bump_version_rejects_unknown_level() -> None:
    with pytest.raises(ValueError):
        bump_version("1.0.13", "huge")


def test_read_current_version(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path, "1.0.13")
    assert read_current_version(repo) == "1.0.13"


def test_apply_bump_updates_all_four_files(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path, "1.0.13")
    changes = apply_bump(repo, "1.0.13", "1.0.14", "2026-06-25", dry_run=False)

    assert 'version = "1.0.14"' in (repo / "pyproject.toml").read_text()
    assert '"version": "1.0.14"' in (repo / ".claude-plugin" / "plugin.json").read_text()
    # server.json has TWO version occurrences; both must move
    assert (repo / "server.json").read_text().count('"version": "1.0.14"') == 2
    changelog = (repo / "CHANGELOG.md").read_text()
    assert "## [1.0.14] - 2026-06-25" in changelog
    # new section sits directly under Unreleased
    assert changelog.index("## [1.0.14]") < changelog.index("## [1.0.13]")
    assert any("CHANGELOG" in c for c in changes)


def test_apply_bump_dry_run_writes_nothing(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path, "1.0.13")
    apply_bump(repo, "1.0.13", "1.0.14", "2026-06-25", dry_run=True)
    assert 'version = "1.0.13"' in (repo / "pyproject.toml").read_text()
    assert "## [1.0.14]" not in (repo / "CHANGELOG.md").read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_cli_release.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'memo.cli_release'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/memo/cli_release.py`:

```python
"""`memo release --bump` — synchronize the version across the four
source-of-truth files in one step so content changes can't ship under a reused
number (the 2026-06-25 1.0.12 stale-build trap).

Only runs from the memo source repo (it edits tracked files); guards otherwise.
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path

import click

from memo.cli_common import console

_PYPROJECT_VERSION_RE = re.compile(r'(?m)^version = "(\d+\.\d+\.\d+)"$')


def bump_version(current: str, level: str) -> str:
    """Return the next semver string for ``level`` in {major, minor, patch}."""
    major, minor, patch = (int(part) for part in current.split("."))
    if level == "major":
        return f"{major + 1}.0.0"
    if level == "minor":
        return f"{major}.{minor + 1}.0"
    if level == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(f"unknown bump level: {level!r}")


def read_current_version(repo: Path) -> str:
    """Read ``[project].version`` from ``repo/pyproject.toml``."""
    match = _PYPROJECT_VERSION_RE.search((repo / "pyproject.toml").read_text(encoding="utf-8"))
    if not match:
        raise click.ClickException("could not find `version = \"x.y.z\"` in pyproject.toml")
    return match.group(1)


def apply_bump(repo: Path, old: str, new: str, date: str, *, dry_run: bool) -> list[str]:
    """Replace ``old`` with ``new`` across the four version files and add a
    CHANGELOG section. Returns a human-readable change list; writes unless
    ``dry_run``."""
    changes: list[str] = []

    # (relative path, exact needle, replacement, replace-all?)
    edits = [
        ("pyproject.toml", f'version = "{old}"', f'version = "{new}"', False),
        (".claude-plugin/plugin.json", f'"version": "{old}"', f'"version": "{new}"', False),
        ("server.json", f'"version": "{old}"', f'"version": "{new}"', True),
    ]
    for rel, needle, repl, replace_all in edits:
        path = repo / rel
        text = path.read_text(encoding="utf-8")
        count = text.count(needle)
        if count == 0:
            continue
        new_text = text.replace(needle, repl) if replace_all else text.replace(needle, repl, 1)
        moved = count if replace_all else 1
        changes.append(f"{rel}: {moved}x {old} -> {new}")
        if not dry_run:
            path.write_text(new_text, encoding="utf-8")

    changelog = repo / "CHANGELOG.md"
    ctext = changelog.read_text(encoding="utf-8")
    anchor = "## [Unreleased]\n"
    if anchor in ctext and f"## [{new}]" not in ctext:
        section = (
            f"## [Unreleased]\n\n"
            f"## [{new}] - {date}\n\n"
            f"### Changed\n\n"
            f"<!-- describe changes for {new} -->\n"
        )
        new_ctext = ctext.replace(anchor, section, 1)
        changes.append(f"CHANGELOG.md: add [{new}] section")
        if not dry_run:
            changelog.write_text(new_ctext, encoding="utf-8")

    return changes


@click.command(name="release")
@click.option(
    "--bump",
    "level",
    type=click.Choice(["major", "minor", "patch"]),
    required=True,
    help="Which semver component to increment.",
)
@click.option("--date", default=None, help="Release date YYYY-MM-DD (default: today).")
@click.option("--dry-run", is_flag=True, help="Show the changes without writing files.")
def release_cmd(level: str, date: str | None, dry_run: bool) -> None:
    """Bump the version across pyproject.toml, plugin.json, server.json, CHANGELOG.md."""
    repo = Path(__file__).resolve().parents[2]
    if not (repo / "pyproject.toml").exists():
        raise click.ClickException("run `memo release` from the memo source repo")
    old = read_current_version(repo)
    new = bump_version(old, level)
    when = date or datetime.date.today().isoformat()
    changes = apply_bump(repo, old, new, when, dry_run=dry_run)
    prefix = "[dim]dry-run[/dim] " if dry_run else ""
    console.print(f"{prefix}[green]{old} → {new}[/green]")
    for change in changes:
        console.print(f"  {change}")
    if not dry_run:
        console.print("[dim]review the diff, fill the CHANGELOG section, then commit + tag.[/dim]")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/test_cli_release.py -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Register the command in `cli.py`**

In `src/memo/cli.py`, add the import alongside the other `from memo.cli_<domain> import ...` lines (keep alphabetical with neighbors, e.g. after the `cli_recall_daemon`/`cli_query` group):

```python
from memo.cli_release import release_cmd
```

Then add the registration alongside the other `cli.add_command(...)` calls (near `cli.add_command(version_group)`):

```python
cli.add_command(release_cmd)
```

- [ ] **Step 6: Verify lint, types, registration, and a dry-run smoke**

Run: `uv run --no-sync ruff check src/memo/cli_release.py src/memo/cli.py`
Expected: `All checks passed!`

Run: `uv run --no-sync mypy src/memo/cli_release.py`
Expected: `Success: no issues found`.

Run: `MEMO_NONINTERACTIVE=1 uv run --no-sync memo release --bump patch --dry-run`
Expected: a `dry-run X → Y` line listing the four files; verify with `git status --short` that NO files changed.

- [ ] **Step 7: Commit**

```bash
git add src/memo/cli_release.py tests/test_cli_release.py src/memo/cli.py
git commit -m "feat(release): add 'memo release --bump' to sync version across the 4 files"
```

---

## Final verification (after all three tasks)

- [ ] **Full suite + type + lint green**

Run: `uv run --no-sync pytest tests/ -q`
Expected: all pass (previous baseline 1554 passed / 25 skipped, plus the new tests).

Run: `uv run --no-sync mypy src/memo/`
Expected: `Success: no issues found`.

Run: `uv run --no-sync ruff check src/`
Expected: `All checks passed!`

- [ ] **Push**

```bash
git push origin master
```

---

## Self-Review notes (author)

- **Spec coverage:** M2's three deliverables map to Task 1 (installed≠source), Task 2 (MCP config path validation), Task 3 (`memo release --bump`). All covered.
- **Scope:** single subsystem (doctor guardrails + release helper); three independently testable tasks — appropriately sized for one plan.
- **Build-stamp vs content-hash:** the spec said "version + content-hash/build-stamp". This plan implements the content-hash variant (no hatchling build hook) because it is testable without build-system changes and directly catches the same-version/different-bytes case. A build stamp can be a later enhancement if cross-machine comparison (without the dev repo present) is needed.
- **Non-fatal:** all three doctor additions are warnings; they do not flip `ok` / exit code, matching existing advisory checks. If `--strict-runtime` should fail on a stale install, that is a one-line follow-up, intentionally out of scope here.
