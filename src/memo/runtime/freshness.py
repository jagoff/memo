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


def _package_content_hash(pkg_dir: Path) -> str | None:
    """Stable sha256 over every ``*.py`` under ``pkg_dir`` (sorted relpath + bytes).

    Returns ``None`` when any file cannot be read (e.g. ``PermissionError`` or
    the file vanishes between the ``rglob`` scan and the ``read_bytes`` call).
    """
    h = hashlib.sha256()
    try:
        paths = sorted(pkg_dir.rglob("*.py"))
        for path in paths:
            rel = path.relative_to(pkg_dir).as_posix()
            h.update(rel.encode("utf-8"))
            h.update(b"\0")
            h.update(path.read_bytes())
            h.update(b"\0")
    except OSError:
        return None
    return h.hexdigest()


def _read_pyproject_version(repo_root: Path) -> str | None:
    pyproject = repo_root / "pyproject.toml"
    if not pyproject.exists():
        return None
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    version = data.get("project", {}).get("version")
    return version if isinstance(version, str) else None


def installed_package_dir() -> Path | None:
    """Directory of the importable ``memo`` package, or None if not locatable."""
    import importlib.util

    try:
        spec = importlib.util.find_spec("memo")
    except (ValueError, ImportError):
        return None
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
    installed_hash = _package_content_hash(installed_pkg_dir)
    repo_hash = _package_content_hash(repo_pkg)
    if installed_hash is None or repo_hash is None:
        return {"status": "skipped", "message": "I/O error reading package"}
    if installed_hash == repo_hash:
        return {"status": "fresh", "message": f"installed matches repo at {installed_version}"}
    return {
        "status": "stale",
        "message": (
            f"installed {installed_version} differs from repo source at the SAME version — "
            f"stale build; run `uv tool install --reinstall .` from {repo_root}"
        ),
    }
