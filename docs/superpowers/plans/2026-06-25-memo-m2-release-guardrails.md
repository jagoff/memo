# M2 — Release/Runtime Guardrails in `memo doctor` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three guardrails that catch the two runtime failures seen on 2026-06-25 — a stale install hiding behind an unchanged version number, and MCP configs pointing at a deleted venv-internal binary — plus a `memo release bump` helper that keeps the four version files in sync.

**Architecture:** Two new pure-Python modules under `src/memo/runtime/` (`freshness.py`, `mcp_config.py`) expose side-effect-free check functions; `cli_doctor.py` wires them into the existing `doctor` text output and `cli_diag.py:_doctor_report` into the JSON report. A new `cli_release.py` adds the `memo release bump` command group, registered in `cli.py`. All version/repo paths are resolved through a single new flag, `MEMO_DEV_REPO`.

**Tech Stack:** Python 3.13, Click, Rich, `tomllib` (stdlib), `hashlib`/`re` (stdlib). Tests: `uv run --no-sync pytest`. Types: `uv run --no-sync mypy src/memo/`. Lint: `uv run --no-sync ruff check src/`.

## Global Constraints

- Python floor: `requires-python = ">=3.13"` — `tomllib` is always available; do not add a `tomli` fallback.
- Files stay < 800 lines (repo rule). New modules are small and focused.
- `MEMO_*` env vars MUST be registered in the flags registry (`src/memo/flags*.py`) via `_spec(...)`; never read `os.environ` inline. Use `flag_str(name)`. `memo config validate` parses every set flag.
- Version source-of-truth is **four** files, kept identical: `pyproject.toml` `[project].version`, `.claude-plugin/plugin.json` `"version"`, `server.json` (`"version"` appears **twice** — top-level + package), `CHANGELOG.md` (Keep-a-Changelog: a `## [Unreleased]` section sits above the newest release section).
- MCP `doctor` checks must never raise — wrap I/O in best-effort guards and degrade to "skipped".
- New CLI command files follow the `cli_<domain>.py` pattern: a `@click.group(...)`/command imported into `cli.py` and registered with `cli.add_command(...)`.
- Rich console comes from `from memo.cli_common import console`. Status glyphs used by `doctor`: `[green]✓[/green]`, `[yellow]![/yellow]`, `[red]✗[/red]`.

---

### Task 1: Install-freshness check (`runtime/freshness.py`) + `MEMO_DEV_REPO` flag + doctor wiring

Detects "installed package differs from the dev repo at the **same** version" — the exact failure where `1.0.12` shipped two different contents. Pure functions compare a content hash of the installed `memo` package against the repo's `src/memo`, gated by the new `MEMO_DEV_REPO` flag.

**Files:**
- Create: `src/memo/runtime/freshness.py`
- Create: `tests/test_runtime_freshness.py`
- Modify: `src/memo/flags_misc.py` (add `MEMO_DEV_REPO` spec)
- Modify: `src/memo/cli_doctor.py` (wire text output)
- Modify: `src/memo/cli_diag.py` (add `freshness` key to `_doctor_report`)

**Interfaces:**
- Produces:
  - `check_install_freshness(*, installed_version: str, installed_pkg_dir: Path | None, repo_root: Path | None) -> dict[str, str]` — returns `{"status": one of "fresh"|"stale"|"repo-ahead"|"skipped", "message": str}`.
  - `installed_package_dir() -> Path | None` — directory of the importable `memo` package.
  - Flag `MEMO_DEV_REPO` (str, default `""`), read via `flag_str("MEMO_DEV_REPO")`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_runtime_freshness.py`:

```python
from __future__ import annotations

from pathlib import Path

from memo.runtime.freshness import check_install_freshness


def _make_pkg(root: Path, version: str, body: str) -> Path:
    """Build a fake repo at `root` with src/memo/{__init__.py,sample.py} + pyproject."""
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "mlx-memo"\nversion = "{version}"\n', encoding="utf-8"
    )
    pkg = root / "src" / "memo"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("# memo\n", encoding="utf-8")
    (pkg / "sample.py").write_text(body, encoding="utf-8")
    return pkg


def test_fresh_when_version_and_content_match(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    pkg = _make_pkg(repo, "1.2.3", "X = 1\n")
    # Installed dir = a byte-identical copy of the repo package.
    installed = tmp_path / "site" / "memo"
    installed.mkdir(parents=True)
    (installed / "__init__.py").write_text("# memo\n", encoding="utf-8")
    (installed / "sample.py").write_text("X = 1\n", encoding="utf-8")

    out = check_install_freshness(
        installed_version="1.2.3", installed_pkg_dir=installed, repo_root=repo
    )
    assert out["status"] == "fresh"


def test_stale_when_same_version_different_content(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_pkg(repo, "1.2.3", "X = 2\n")  # repo has new content
    installed = tmp_path / "site" / "memo"
    installed.mkdir(parents=True)
    (installed / "__init__.py").write_text("# memo\n", encoding="utf-8")
    (installed / "sample.py").write_text("X = 1\n", encoding="utf-8")  # old content

    out = check_install_freshness(
        installed_version="1.2.3", installed_pkg_dir=installed, repo_root=repo
    )
    assert out["status"] == "stale"
    assert "reinstall" in out["message"]


def test_repo_ahead_when_versions_differ(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_pkg(repo, "1.3.0", "X = 1\n")
    out = check_install_freshness(
        installed_version="1.2.3", installed_pkg_dir=None, repo_root=repo
    )
    assert out["status"] == "repo-ahead"


def test_skipped_when_no_repo(tmp_path: Path) -> None:
    out = check_install_freshness(
        installed_version="1.2.3", installed_pkg_dir=None, repo_root=None
    )
    assert out["status"] == "skipped"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/repos/memo && uv run --no-sync pytest tests/test_runtime_freshness.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'memo.runtime.freshness'`

- [ ] **Step 3: Write minimal implementation**

Create `src/memo/runtime/freshness.py`:

```python
"""Install-freshness check: catch a stale install hiding behind an unchanged
version number (e.g. 1.0.12 shipped twice with different content).

Pure functions: the caller supplies the installed version, the installed
package directory, and the dev repo root (from MEMO_DEV_REPO). Nothing here
performs I/O beyond reading files under the paths it is handed.
"""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path


def _package_content_hash(pkg_dir: Path) -> str:
    """Stable sha256 over every ``*.py`` under ``pkg_dir`` (sorted relpath + bytes)."""
    h = hashlib.sha256()
    for path in sorted(pkg_dir.rglob("*.py")):
        rel = path.relative_to(pkg_dir).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _read_pyproject_version(repo_root: Path) -> str | None:
    pyproject = repo_root / "pyproject.toml"
    if not pyproject.exists():
        return None
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    version = data.get("project", {}).get("version")
    return version if isinstance(version, str) else None


def installed_package_dir() -> Path | None:
    """Directory of the importable ``memo`` package, or None if not locatable."""
    import importlib.util

    spec = importlib.util.find_spec("memo")
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).parent


def check_install_freshness(
    *,
    installed_version: str,
    installed_pkg_dir: Path | None,
    repo_root: Path | None,
) -> dict[str, str]:
    """Compare the installed package against a dev repo.

    Status values:
      - "skipped":    no dev repo configured / not locatable
      - "repo-ahead": repo version != installed (normal during development)
      - "fresh":      same version AND byte-identical content
      - "stale":      same version but DIFFERENT content -> reinstall needed
    """
    if repo_root is None or not repo_root.exists():
        return {"status": "skipped", "message": "no dev repo configured (set MEMO_DEV_REPO)"}
    repo_version = _read_pyproject_version(repo_root)
    if repo_version is None:
        return {"status": "skipped", "message": f"no pyproject version at {repo_root}"}
    if repo_version != installed_version:
        return {
            "status": "repo-ahead",
            "message": f"repo {repo_version} != installed {installed_version} (expected during dev)",
        }
    repo_pkg = repo_root / "src" / "memo"
    if installed_pkg_dir is None or not installed_pkg_dir.exists() or not repo_pkg.exists():
        return {"status": "skipped", "message": "package dir not locatable"}
    if _package_content_hash(installed_pkg_dir) == _package_content_hash(repo_pkg):
        return {"status": "fresh", "message": f"installed matches repo at {installed_version}"}
    return {
        "status": "stale",
        "message": (
            f"installed {installed_version} differs from repo source at the SAME version — "
            f"stale build; run `uv tool install --reinstall .` from {repo_root}"
        ),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/repos/memo && uv run --no-sync pytest tests/test_runtime_freshness.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Register the `MEMO_DEV_REPO` flag**

In `src/memo/flags_misc.py`, add this spec inside the `SPECS` tuple, immediately after the `MEMO_AGENT_TTY` spec (search for `"MEMO_AGENT_TTY"`):

```python
    _spec(
        "MEMO_DEV_REPO",
        "str",
        "",
        "session",
        "Path to the memo source checkout (e.g. ~/repos/memo). When set, "
        "`memo doctor` compares the installed package against this repo and "
        "warns if they differ at the SAME version (stale build), and "
        "`memo release bump` targets this repo. Empty = checks skipped.",
    ),
```

- [ ] **Step 6: Verify the flag is recognized**

Run: `cd ~/repos/memo && MEMO_DEV_REPO=/tmp MEMO_NONINTERACTIVE=1 uv run --no-sync memo config validate`
Expected: output contains `all valid` (MEMO_DEV_REPO no longer reported as unknown).

- [ ] **Step 7: Wire the check into `doctor` text output**

In `src/memo/cli_doctor.py`, add these imports near the existing `from memo.config import Config` import:

```python
from memo import __version__ as _installed_version
from memo.flags import flag_str
from memo.runtime.freshness import check_install_freshness, installed_package_dir
```

Then, inside `doctor(...)`, immediately after the runtime-install block (after the lines `if strict_runtime and runtime_report["warnings"]:` / `    ok = False`), insert:

```python
    _dev_repo = flag_str("MEMO_DEV_REPO")
    _fresh = check_install_freshness(
        installed_version=_installed_version,
        installed_pkg_dir=installed_package_dir(),
        repo_root=Path(_dev_repo).expanduser() if _dev_repo else None,
    )
    if _fresh["status"] == "stale":
        console.print(f"[yellow]![/yellow] install freshness: {_fresh['message']}")
    elif _fresh["status"] in ("fresh", "repo-ahead"):
        console.print(f"[green]✓[/green] install freshness: {_fresh['message']}")
```

- [ ] **Step 8: Add `freshness` to the JSON report**

In `src/memo/cli_diag.py`, inside `_doctor_report(...)`, add the same computation and a report key. Near the top of the function body add:

```python
    from memo import __version__ as _installed_version
    from memo.flags import flag_str
    from memo.runtime.freshness import check_install_freshness, installed_package_dir

    _dev_repo = flag_str("MEMO_DEV_REPO")
    _freshness = check_install_freshness(
        installed_version=_installed_version,
        installed_pkg_dir=installed_package_dir(),
        repo_root=Path(_dev_repo).expanduser() if _dev_repo else None,
    )
```

Then, where the function assembles its return dict, add the key `"freshness": _freshness,`. (If `Path` is not already imported in `cli_diag.py`, add `from pathlib import Path` to its imports.)

- [ ] **Step 9: Run the full guard set**

Run: `cd ~/repos/memo && uv run --no-sync ruff check src/ && uv run --no-sync mypy src/memo/ && uv run --no-sync pytest tests/test_runtime_freshness.py -q`
Expected: ruff "All checks passed!", mypy "Success", pytest PASS.

- [ ] **Step 10: Smoke-test against the real repo**

Run: `cd ~/repos/memo && MEMO_DEV_REPO=$HOME/repos/memo MEMO_NONINTERACTIVE=1 uv run --no-sync memo doctor 2>&1 | grep "install freshness"`
Expected: a line `✓ install freshness: ...` (running from source → "fresh" or "repo-ahead").

- [ ] **Step 11: Commit**

```bash
cd ~/repos/memo
git add src/memo/runtime/freshness.py tests/test_runtime_freshness.py src/memo/flags_misc.py src/memo/cli_doctor.py src/memo/cli_diag.py
git commit -m "feat(doctor): warn when installed build is stale vs dev repo at same version"
```

---

### Task 2: MCP config path scan (`runtime/mcp_config.py`) + doctor wiring

Detects MCP configs whose `memo`/`memo-mcp` launch path is missing or points inside a venv (the failure where uninstalling pipx orphaned four configs). Format-agnostic: scans raw config text with a regex, so it works for JSON, JSONC, and YAML without parsers.

**Files:**
- Create: `src/memo/runtime/mcp_config.py`
- Create: `tests/test_runtime_mcp_config.py`
- Modify: `src/memo/cli_doctor.py` (wire text output)
- Modify: `src/memo/cli_diag.py` (add `mcp_config_issues` key to `_doctor_report`)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces:
  - `extract_memo_command_paths(text: str) -> list[str]`
  - `classify_command_path(path: str) -> str | None` — `"missing"`, `"venv-internal"`, or `None` (ok)
  - `scan_mcp_configs(config_paths: tuple[str, ...] = KNOWN_MCP_CONFIGS, *, shim_dir: str = "~/.local/bin") -> list[dict[str, str]]` — each finding: `{"config", "command", "issue", "suggestion"}`
  - `KNOWN_MCP_CONFIGS: tuple[str, ...]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_runtime_mcp_config.py`:

```python
from __future__ import annotations

from pathlib import Path

from memo.runtime.mcp_config import (
    classify_command_path,
    extract_memo_command_paths,
    scan_mcp_configs,
)


def test_extract_finds_memo_and_memo_mcp_paths() -> None:
    text = '{"command": "/Users/x/.local/bin/memo-mcp", "env": "/Users/x/.local/bin/memo"}'
    assert extract_memo_command_paths(text) == [
        "/Users/x/.local/bin/memo",
        "/Users/x/.local/bin/memo-mcp",
    ]


def test_classify_venv_internal(tmp_path: Path) -> None:
    p = tmp_path / "venv-like"
    # The path string itself signals a venv-internal location.
    assert classify_command_path("/Users/x/.local/pipx/venvs/mlx-memo/bin/memo-mcp") == "venv-internal"


def test_classify_missing(tmp_path: Path) -> None:
    assert classify_command_path(str(tmp_path / "does-not-exist" / "memo-mcp")) == "missing"


def test_classify_ok_for_existing_shim(tmp_path: Path) -> None:
    binp = tmp_path / "memo-mcp"
    binp.write_text("#!/bin/sh\n", encoding="utf-8")
    assert classify_command_path(str(binp)) is None


def test_scan_reports_dead_path(tmp_path: Path) -> None:
    cfg = tmp_path / "claude.json"
    cfg.write_text(
        '{"command": "/Users/x/.local/pipx/venvs/mlx-memo/bin/memo-mcp"}', encoding="utf-8"
    )
    findings = scan_mcp_configs((str(cfg),))
    assert len(findings) == 1
    assert findings[0]["issue"] == "venv-internal"
    assert findings[0]["suggestion"].endswith("/memo-mcp")


def test_scan_skips_missing_config(tmp_path: Path) -> None:
    assert scan_mcp_configs((str(tmp_path / "nope.json"),)) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/repos/memo && uv run --no-sync pytest tests/test_runtime_mcp_config.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'memo.runtime.mcp_config'`

- [ ] **Step 3: Write minimal implementation**

Create `src/memo/runtime/mcp_config.py`:

```python
"""Scan known MCP config files for fragile memo-mcp launch paths.

A config that hardcodes a venv-internal binary path (pipx/uv site) breaks the
moment the runtime is reinstalled or switched. The stable target is the shim
``~/.local/bin/<bin>`` or the bare name on PATH. Format-agnostic: matches the
raw text, so JSON, JSONC, and YAML all work without a parser.
"""

from __future__ import annotations

import re
from pathlib import Path

# Configs that commonly launch memo-mcp as a stdio MCP server.
KNOWN_MCP_CONFIGS: tuple[str, ...] = (
    "~/.claude.json",
    "~/.config/devin/config.json",
    "~/.config/opencode/opencode.jsonc",
    "~/.config/mcp-gateway/gateway.yaml",
)

# Absolute path ending in /memo or /memo-mcp (the launched binary).
_MEMO_BIN_PATH = re.compile(r"(/[^\s\"':,]*?/(?:memo-mcp|memo))(?=[\s\"':,]|$)")

# Fragile: points inside a managed venv instead of the stable shim.
_VENV_INTERNAL = re.compile(r"/(?:pipx/venvs|\.venv|site-packages)/")


def extract_memo_command_paths(text: str) -> list[str]:
    """All absolute ``/…/memo`` or ``/…/memo-mcp`` paths mentioned in a config file."""
    return sorted({m.group(1) for m in _MEMO_BIN_PATH.finditer(text)})


def classify_command_path(path: str) -> str | None:
    """Return an issue label, or None if the path is fine.

    - "venv-internal": points inside a venv (breaks on reinstall/runtime change)
    - "missing":       file does not exist on disk
    """
    if _VENV_INTERNAL.search(path):
        return "venv-internal"
    if not Path(path).exists():
        return "missing"
    return None


def scan_mcp_configs(
    config_paths: tuple[str, ...] = KNOWN_MCP_CONFIGS,
    *,
    shim_dir: str = "~/.local/bin",
) -> list[dict[str, str]]:
    """Inspect known MCP config files for fragile/broken memo-mcp command paths."""
    findings: list[dict[str, str]] = []
    shim = str(Path(shim_dir).expanduser())
    for cfg_str in config_paths:
        cfg_path = Path(cfg_str).expanduser()
        if not cfg_path.exists():
            continue
        try:
            text = cfg_path.read_text(encoding="utf-8")
        except OSError:
            continue
        for cmd in extract_memo_command_paths(text):
            issue = classify_command_path(cmd)
            if issue is None:
                continue
            bin_name = cmd.rsplit("/", 1)[-1]
            findings.append(
                {
                    "config": str(cfg_path),
                    "command": cmd,
                    "issue": issue,
                    "suggestion": f"{shim}/{bin_name}",
                }
            )
    return findings
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/repos/memo && uv run --no-sync pytest tests/test_runtime_mcp_config.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Wire the check into `doctor` text output**

In `src/memo/cli_doctor.py`, add the import near the other `from memo.runtime...` imports:

```python
from memo.runtime.mcp_config import scan_mcp_configs
```

Then, inside `doctor(...)`, immediately after the install-freshness block added in Task 1, insert:

```python
    _mcp_issues = scan_mcp_configs()
    if not _mcp_issues:
        console.print("[green]✓[/green] mcp config paths: stable")
    else:
        for _f in _mcp_issues:
            console.print(
                f"[yellow]![/yellow] mcp config: {_f['config']} → {_f['command']} "
                f"({_f['issue']}); use {_f['suggestion']}"
            )
```

- [ ] **Step 6: Add `mcp_config_issues` to the JSON report**

In `src/memo/cli_diag.py`, inside `_doctor_report(...)`, add near the freshness computation from Task 1:

```python
    from memo.runtime.mcp_config import scan_mcp_configs

    _mcp_config_issues = scan_mcp_configs()
```

Then add the key `"mcp_config_issues": _mcp_config_issues,` to the returned dict.

- [ ] **Step 7: Run the full guard set**

Run: `cd ~/repos/memo && uv run --no-sync ruff check src/ && uv run --no-sync mypy src/memo/ && uv run --no-sync pytest tests/test_runtime_mcp_config.py -q`
Expected: ruff "All checks passed!", mypy "Success", pytest PASS.

- [ ] **Step 8: Smoke-test against the real machine**

Run: `cd ~/repos/memo && MEMO_NONINTERACTIVE=1 uv run --no-sync memo doctor 2>&1 | grep "mcp config"`
Expected: `✓ mcp config paths: stable` (the configs were repointed to `~/.local/bin` on 2026-06-25). If any line shows `venv-internal`/`missing`, that is a real finding to repoint.

- [ ] **Step 9: Commit**

```bash
cd ~/repos/memo
git add src/memo/runtime/mcp_config.py tests/test_runtime_mcp_config.py src/memo/cli_doctor.py src/memo/cli_diag.py
git commit -m "feat(doctor): flag MCP configs that launch memo-mcp from a venv-internal or dead path"
```

---

### Task 3: `memo release bump` helper (`cli_release.py`) + registration

Bumps the version across all four source-of-truth files and seeds a CHANGELOG section, so a content change can't ship under a stale number. Repo is resolved from `MEMO_DEV_REPO` (falling back to the running checkout), which keeps the CLI testable against a temp repo.

**Files:**
- Create: `src/memo/cli_release.py`
- Create: `tests/test_cli_release.py`
- Modify: `src/memo/cli.py` (import + `add_command`)

**Interfaces:**
- Consumes: flag `MEMO_DEV_REPO` from Task 1 (`flag_str`).
- Produces:
  - `bump_version(current: str, level: str) -> str` — `level` in `{"major","minor","patch"}`
  - `plan_release_edits(repo: Path, old: str, new: str, date: str) -> dict[Path, str]` — pure; maps each file Path to its new full content
  - Click group `release_group` with subcommand `bump` (`memo release bump <level> [--dry-run] [--date YYYY-MM-DD]`)

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli_release.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from memo.cli_release import bump_version, plan_release_edits, release_group


def test_bump_version_levels() -> None:
    assert bump_version("1.2.3", "patch") == "1.2.4"
    assert bump_version("1.2.3", "minor") == "1.3.0"
    assert bump_version("1.2.3", "major") == "2.0.0"


def test_bump_version_rejects_non_semver() -> None:
    with pytest.raises(ValueError):
        bump_version("1.2", "patch")


def _fake_repo(root: Path, version: str) -> Path:
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "mlx-memo"\nversion = "{version}"\n', encoding="utf-8"
    )
    (root / ".claude-plugin").mkdir()
    (root / ".claude-plugin" / "plugin.json").write_text(
        f'{{\n  "name": "memo",\n  "version": "{version}"\n}}\n', encoding="utf-8"
    )
    (root / "server.json").write_text(
        f'{{\n  "version": "{version}",\n  "packages": [\n    {{\n      "version": "{version}"\n    }}\n  ]\n}}\n',
        encoding="utf-8",
    )
    (root / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [Unreleased]\n\n## [%s] - 2026-01-01\n\n- prior\n" % version,
        encoding="utf-8",
    )
    return root


def test_plan_release_edits_syncs_four_files(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path, "1.2.3")
    edits = plan_release_edits(repo, "1.2.3", "1.2.4", "2026-06-25")

    assert 'version = "1.2.4"' in edits[repo / "pyproject.toml"]
    assert '"version": "1.2.4"' in edits[repo / ".claude-plugin" / "plugin.json"]
    # server.json has TWO version occurrences — both must move.
    assert edits[repo / "server.json"].count('"version": "1.2.4"') == 2
    assert edits[repo / "server.json"].count('"version": "1.2.3"') == 0
    # CHANGELOG gains a new section right under Unreleased, above the old one.
    cl = edits[repo / "CHANGELOG.md"]
    assert "## [1.2.4] - 2026-06-25" in cl
    assert cl.index("## [1.2.4]") < cl.index("## [1.2.3]")


def test_release_bump_dry_run_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _fake_repo(tmp_path, "1.2.3")
    monkeypatch.setenv("MEMO_DEV_REPO", str(repo))
    result = CliRunner().invoke(release_group, ["bump", "patch", "--dry-run"])
    assert result.exit_code == 0
    assert '1.2.3" ' not in (repo / "pyproject.toml").read_text()  # unchanged
    assert 'version = "1.2.3"' in (repo / "pyproject.toml").read_text()


def test_release_bump_writes_all_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _fake_repo(tmp_path, "1.2.3")
    monkeypatch.setenv("MEMO_DEV_REPO", str(repo))
    result = CliRunner().invoke(release_group, ["bump", "minor", "--date", "2026-06-25"])
    assert result.exit_code == 0, result.output
    assert 'version = "1.3.0"' in (repo / "pyproject.toml").read_text()
    assert (repo / "server.json").read_text().count('"version": "1.3.0"') == 2
    assert "## [1.3.0] - 2026-06-25" in (repo / "CHANGELOG.md").read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/repos/memo && uv run --no-sync pytest tests/test_cli_release.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'memo.cli_release'`

- [ ] **Step 3: Write minimal implementation**

Create `src/memo/cli_release.py`:

```python
"""`memo release` — version bump helper.

Synchronizes the four version source-of-truth files and seeds a CHANGELOG
section so a content change can't ship under a stale version number.
Registered in cli.py via `cli.add_command(release_group)`.
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path

import click

from memo.cli_common import console
from memo.flags import flag_str

# src/memo/cli_release.py -> repo root when running from a source checkout.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_repo() -> Path:
    """Dev repo to operate on: MEMO_DEV_REPO if set, else the running checkout."""
    dev = flag_str("MEMO_DEV_REPO")
    return Path(dev).expanduser() if dev else _REPO_ROOT


def bump_version(current: str, level: str) -> str:
    """Return the next semver for ``level`` in {major, minor, patch}."""
    parts = current.split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise ValueError(f"non-semver version: {current!r}")
    major, minor, patch = (int(p) for p in parts)
    if level == "major":
        return f"{major + 1}.0.0"
    if level == "minor":
        return f"{major}.{minor + 1}.0"
    if level == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(f"unknown level: {level!r}")


def _read_current_version(repo: Path) -> str:
    text = (repo / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^version = "([^"]+)"', text, re.MULTILINE)
    if not m:
        raise ValueError("could not find version in pyproject.toml")
    return m.group(1)


def _sub_exact(text: str, pattern: str, repl: str, *, count: int) -> str:
    new, n = re.subn(pattern, repl, text, count=count)
    if n != count:
        raise ValueError(f"expected {count} match(es) for {pattern!r}, got {n}")
    return new


def plan_release_edits(repo: Path, old: str, new: str, date: str) -> dict[Path, str]:
    """Compute new file contents for all four source-of-truth files. Pure."""
    edits: dict[Path, str] = {}

    pp = repo / "pyproject.toml"
    edits[pp] = _sub_exact(
        pp.read_text(encoding="utf-8"),
        rf'^version = "{re.escape(old)}"',
        f'version = "{new}"',
        count=1,
    )

    plugin = repo / ".claude-plugin" / "plugin.json"
    edits[plugin] = _sub_exact(
        plugin.read_text(encoding="utf-8"),
        rf'"version": "{re.escape(old)}"',
        f'"version": "{new}"',
        count=1,
    )

    server = repo / "server.json"
    edits[server] = _sub_exact(
        server.read_text(encoding="utf-8"),
        rf'"version": "{re.escape(old)}"',
        f'"version": "{new}"',
        count=2,
    )

    changelog = repo / "CHANGELOG.md"
    section = f"## [{new}] - {date}\n\n### Fixed\n\n- TODO: describe changes\n\n"
    edits[changelog] = _sub_exact(
        changelog.read_text(encoding="utf-8"),
        r"## \[Unreleased\]\n\n",
        f"## [Unreleased]\n\n{section}",
        count=1,
    )
    return edits


@click.group(name="release")
def release_group() -> None:
    """Release helpers — keep version numbers in sync."""


@release_group.command(name="bump")
@click.argument("level", type=click.Choice(["major", "minor", "patch"]))
@click.option("--dry-run", is_flag=True, help="Show what would change without writing.")
@click.option("--date", default=None, help="CHANGELOG date (YYYY-MM-DD); default today.")
def release_bump(level: str, dry_run: bool, date: str | None) -> None:
    """Bump version across pyproject, plugin.json, server.json, CHANGELOG."""
    repo = _resolve_repo()
    old = _read_current_version(repo)
    new = bump_version(old, level)
    when = date or datetime.date.today().isoformat()
    edits = plan_release_edits(repo, old, new, when)
    console.print(f"[bold]{old} → {new}[/bold] ({level})")
    for path in edits:
        verb = "would update" if dry_run else "updated"
        console.print(f"  {verb}: {path.relative_to(repo)}")
    if dry_run:
        console.print("[dim]dry-run: no files written[/dim]")
        return
    for path, content in edits.items():
        path.write_text(content, encoding="utf-8")
    console.print("[green]✓[/green] version synced; edit the CHANGELOG TODO before committing")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/repos/memo && uv run --no-sync pytest tests/test_cli_release.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Register the command group in `cli.py`**

In `src/memo/cli.py`, add the import alongside the other `from memo.cli_<domain> import ...` lines (keep alphabetical-ish ordering near `cli_recall`/`cli_query`):

```python
from memo.cli_release import release_group
```

Then, alongside the other `cli.add_command(...)` calls (near `cli.add_command(version_group)`), add:

```python
cli.add_command(release_group)
```

- [ ] **Step 6: Verify the command is wired**

Run: `cd ~/repos/memo && MEMO_NONINTERACTIVE=1 uv run --no-sync memo release bump --help`
Expected: help text for `bump` listing `--dry-run` and `--date`.

- [ ] **Step 7: Dry-run against the real repo (no writes)**

Run: `cd ~/repos/memo && MEMO_DEV_REPO=$HOME/repos/memo MEMO_NONINTERACTIVE=1 uv run --no-sync memo release bump patch --dry-run`
Expected: prints `1.0.13 → 1.0.14 (patch)` and four `would update:` lines; `git status --short` shows no changes.

- [ ] **Step 8: Run the full guard set**

Run: `cd ~/repos/memo && uv run --no-sync ruff check src/ && uv run --no-sync mypy src/memo/ && uv run --no-sync pytest tests/test_cli_release.py -q`
Expected: ruff "All checks passed!", mypy "Success", pytest PASS.

- [ ] **Step 9: Commit**

```bash
cd ~/repos/memo
git add src/memo/cli_release.py tests/test_cli_release.py src/memo/cli.py
git commit -m "feat(release): add 'memo release bump' to sync the four version files + CHANGELOG"
```

---

### Task 4: Document + release the guardrails

Record the new behavior and ship a real version bump (dogfooding Task 3).

**Files:**
- Modify: `src/memo/CLAUDE.md` is NOT touched; instead the four version files via `memo release bump` and `CHANGELOG.md` content.

- [ ] **Step 1: Run the full suite to confirm green**

Run: `cd ~/repos/memo && uv run --no-sync pytest tests/ -q -p no:cacheprovider`
Expected: all pass (prior baseline: 1554 passed, 25 skipped — now higher with the new tests).

- [ ] **Step 2: Bump the version with the new tool (dogfood)**

Run: `cd ~/repos/memo && MEMO_DEV_REPO=$HOME/repos/memo MEMO_NONINTERACTIVE=1 uv run --no-sync memo release bump minor`
Expected: `1.0.13 → 1.1.0`; four files updated.

- [ ] **Step 3: Fill in the CHANGELOG TODO**

In `CHANGELOG.md`, replace the `- TODO: describe changes` line under the new section with:

```markdown
### Added

- **`memo doctor` install-freshness guard.** With `MEMO_DEV_REPO` set, doctor warns when the installed package differs from the repo source at the same version (the stale-build trap from 1.0.12).
- **`memo doctor` MCP config path check.** Flags MCP configs whose `memo`/`memo-mcp` command points inside a venv or at a missing file, and suggests the stable `~/.local/bin` shim.
- **`memo release bump <level>`.** Syncs the four version source-of-truth files (`pyproject.toml`, `.claude-plugin/plugin.json`, `server.json`, `CHANGELOG.md`) and seeds a changelog section.
```

(Remove the auto-inserted `### Fixed` / TODO block.)

- [ ] **Step 4: Reinstall and verify the guardrails live**

```bash
cd ~/repos/memo
uv tool install --reinstall .
MEMO_DEV_REPO=$HOME/repos/memo MEMO_NONINTERACTIVE=1 memo doctor 2>&1 | grep -E "install freshness|mcp config"
```
Expected: `✓ install freshness: ...` and `✓ mcp config paths: stable` (or real findings).

- [ ] **Step 5: Commit + tag + push**

```bash
cd ~/repos/memo
git add pyproject.toml .claude-plugin/plugin.json server.json CHANGELOG.md
git commit -m "chore(release): 1.1.0 — doctor release/runtime guardrails"
git tag -a v1.1.0 -m "memo 1.1.0 — doctor install-freshness + MCP path checks + release bump"
git push origin master
git push origin v1.1.0
```

---

## Self-Review

**Spec coverage (M2 from the design doc):**
- (1) doctor check `installed ≠ source` → Task 1 (version + content-hash via `check_install_freshness`). ✅
- (2) MCP config path validation (dead / venv-internal, prefer `~/.local/bin`) → Task 2 (`scan_mcp_configs`). ✅
- (3) `memo release --bump` syncing the 4 files + CHANGELOG → Task 3 (`memo release bump`, `plan_release_edits`). ✅ (Implemented as the subcommand `release bump`; the design's `--bump` intent is satisfied.)

**Placeholder scan:** The only literal `TODO` strings are (a) the CHANGELOG seed text the tool intentionally writes for the human to fill, and (b) Task 4 Step 3 which fills it. No plan-step placeholders. ✅

**Type consistency:** `check_install_freshness` / `installed_package_dir` / `scan_mcp_configs` / `extract_memo_command_paths` / `classify_command_path` / `bump_version` / `plan_release_edits` names and signatures are identical across their definition, wiring, and tests. `MEMO_DEV_REPO` is defined once (Task 1) and reused (Task 3). server.json `count=2` matches its two `"version"` occurrences. ✅

**Note on scope:** Tasks 1–3 are independently shippable; Task 4 dogfoods Task 3 to cut the release. If `release bump` is not yet trusted, Task 4 Step 2 can be replaced with manual edits to the four files — but the whole point of M2 is to use the tool.
```
