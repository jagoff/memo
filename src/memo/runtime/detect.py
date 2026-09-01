from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any


def _safe_resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path


def _resolve_command(
    name: str,
    *,
    prefer_invoked: bool = False,
    sibling_of: Path | None = None,
) -> tuple[Path | None, Path | None]:
    """Resolve an executable, preferring the active install when known."""
    candidates: list[Path] = []

    if prefer_invoked:
        invoked = Path(sys.argv[0])
        if invoked.name == name and invoked.exists():
            candidates.append(invoked)

    if sibling_of is not None:
        sibling = sibling_of.with_name(name)
        if sibling.exists():
            candidates.append(sibling)

    raw = shutil.which(name)
    if raw:
        candidates.append(Path(raw))

    for candidate in candidates:
        resolved = _safe_resolve(candidate)
        if resolved.exists():
            return candidate, resolved
    return None, None


def _env_root_for_bin(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.parent.name == "bin":
        return path.parent.parent
    return None


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _install_mode(root: Path | None) -> str:
    if root is None:
        return "unknown"
    pipx_home = os.environ.get("PIPX_HOME")
    if pipx_home:
        pipx_venvs = _safe_resolve(Path(pipx_home).expanduser() / "venvs")
        if root.parent == pipx_venvs:
            return "pipx"
    uv_tool_dir = os.environ.get("UV_TOOL_DIR")
    if uv_tool_dir:
        uv_tools = _safe_resolve(Path(uv_tool_dir).expanduser())
        if root.parent == uv_tools:
            return "uv tool"
    parts = set(root.parts)
    root_s = str(root)
    if "pipx" in parts and "venvs" in parts:
        return "pipx"
    if "uv" in parts and "tools" in parts:
        return "uv tool"
    if "Cellar" in parts or root_s.startswith("/opt/homebrew/"):
        return "homebrew"
    if root.name in {".venv", "venv"} or (root / "pyvenv.cfg").is_file():
        return "venv"
    return "unknown"


def is_homebrew_install() -> bool:
    """True if the running interpreter is a Homebrew (Cellar) install.

    Fast, no subprocess — the update path and banner use it to pick the channel
    the corresponds (`brew upgrade` vs `memo update`). Mirrors the Cellar/opt
    markers `_install_mode` classifies as ``homebrew``.
    """
    exe = sys.executable or ""
    return (
        "/Cellar/" in exe
        or exe.startswith("/opt/homebrew/")
        or exe.startswith("/usr/local/Cellar/")
    )


def launchagent_runtime_warnings(agents_dir: Path, primary_root: Path | None) -> list[str]:
    """`com.memo.*` LaunchAgents whose program is NOT in the primary runtime.

    `--strict-runtime` compared `memo` against `memo-mcp` and stopped there,
    so a daemon launched from a third runtime was invisible to it. Measured
    2026-09-01: `com.memo.proxy`'s plist hard-coded
    `~/.local/share/memo/proxy-runtime/`, a venv nothing in this repo creates
    or upgrades. It was running memo 4.14.3 against a 4.15.0 tool install —
    twelve releases of proxy fixes that had never executed, with every gate
    green the whole time, because the agent is the thing that serves traffic
    and nothing compared it to anything.

    Reads the plist as text rather than importing a parser: the value needed
    is the first `<string>` of `ProgramArguments`, and a doctor check must not
    fail because a plist is malformed. Silent when the directory is absent
    (non-macOS, or no agents installed).
    """
    if primary_root is None or not agents_dir.is_dir():
        return []
    import re

    out: list[str] = []
    for path in sorted(agents_dir.glob("com.memo.*.plist")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        m = re.search(r"<key>ProgramArguments</key>\s*<array>\s*<string>([^<]+)</string>", text)
        if not m:
            continue
        program = _safe_resolve(Path(m.group(1).strip()))
        # Only agents that launch a memo BINARY are comparable. `com.memo.nightly`
        # runs a shell wrapper (`memo-nightly.sh`), which has no environment root
        # of its own — reporting it would be a false positive, and a check that
        # cries wolf on a healthy install is one nobody reads.
        if program is None or program.name not in {"memo", "memo-mcp"}:
            continue
        root = _env_root_for_bin(program)
        if root is not None and root != primary_root:
            out.append(
                f"{path.stem} runs from {root}, not the active runtime "
                f"{primary_root} — `memo ops install {path.stem.rsplit('.', 1)[-1]}` repoints it"
            )
    return out


def _runtime_install_report(
    cwd: Path | None = None, *, package_file: Path | None = None
) -> dict[str, Any]:
    cwd = _safe_resolve(cwd or Path.cwd())
    package_path = _safe_resolve(package_file or Path(__file__))
    memo_cmd, memo_resolved = _resolve_command("memo", prefer_invoked=True)
    mcp_cmd, mcp_resolved = _resolve_command("memo-mcp", sibling_of=memo_resolved)
    py_resolved = _safe_resolve(Path(sys.executable))

    memo_root = _env_root_for_bin(memo_resolved)
    mcp_root = _env_root_for_bin(mcp_resolved)
    py_root = _env_root_for_bin(py_resolved)
    primary_root = memo_root or mcp_root or py_root
    mode = _install_mode(primary_root)

    warnings: list[str] = []
    warnings.extend(
        launchagent_runtime_warnings(Path.home() / "Library" / "LaunchAgents", primary_root)
    )
    if memo_resolved is None:
        warnings.append("`memo` is not on PATH")
    if mcp_resolved is None:
        warnings.append("`memo-mcp` is not on PATH; MCP clients cannot start it")
    if memo_root is not None and mcp_root is not None and memo_root != mcp_root:
        warnings.append(
            "`memo` and `memo-mcp` resolve to different environments; "
            "reinstall with `pipx install --force mlx-memo` or `uv tool install --force mlx-memo`"
        )
    if mode == "venv" and primary_root is not None:
        if _path_is_relative_to(primary_root, cwd):
            warnings.append(
                f"running from project venv {primary_root}; prefer an isolated "
                "tool install so MCP is not tied to this repo"
            )
        else:
            warnings.append(
                f"running from venv {primary_root}; verify this is memo's own "
                "dedicated environment, not another project's venv"
            )
    elif mode == "unknown":
        warnings.append(
            "install mode is unknown; recommended: `pipx install mlx-memo` "
            "or `uv tool install mlx-memo`"
        )
    if primary_root is not None and not _path_is_relative_to(package_path, primary_root):
        warnings.append(
            f"memo Python package loaded from {package_path}, outside the isolated runtime "
            f"{primary_root}; clear PYTHONPATH or reinstall without an editable source link"
        )

    return {
        "mode": mode,
        "root": str(primary_root) if primary_root else None,
        "memo_cmd": str(memo_cmd) if memo_cmd else None,
        "memo_resolved": str(memo_resolved) if memo_resolved else None,
        "mcp_cmd": str(mcp_cmd) if mcp_cmd else None,
        "mcp_resolved": str(mcp_resolved) if mcp_resolved else None,
        "python": str(py_resolved),
        "package_path": str(package_path),
        "warnings": warnings,
    }


def _resolved_memo_mcp() -> Path | None:
    _, memo_resolved = _resolve_command("memo", prefer_invoked=True)
    if memo_resolved is not None:
        sibling = memo_resolved.with_name("memo-mcp")
        if sibling.exists():
            return sibling
    _, resolved = _resolve_command("memo-mcp")
    if resolved is not None:
        return resolved
    return None
