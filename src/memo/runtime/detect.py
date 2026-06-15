from __future__ import annotations

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


def _runtime_install_report(cwd: Path | None = None) -> dict[str, Any]:
    cwd = _safe_resolve(cwd or Path.cwd())
    memo_cmd, memo_resolved = _resolve_command("memo", prefer_invoked=True)
    mcp_cmd, mcp_resolved = _resolve_command("memo-mcp", sibling_of=memo_resolved)
    py_resolved = _safe_resolve(Path(sys.executable))

    memo_root = _env_root_for_bin(memo_resolved)
    mcp_root = _env_root_for_bin(mcp_resolved)
    py_root = _env_root_for_bin(py_resolved)
    primary_root = memo_root or mcp_root or py_root
    mode = _install_mode(primary_root)

    warnings: list[str] = []
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

    return {
        "mode": mode,
        "root": str(primary_root) if primary_root else None,
        "memo_cmd": str(memo_cmd) if memo_cmd else None,
        "memo_resolved": str(memo_resolved) if memo_resolved else None,
        "mcp_cmd": str(mcp_cmd) if mcp_cmd else None,
        "mcp_resolved": str(mcp_resolved) if mcp_resolved else None,
        "python": str(py_resolved),
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
