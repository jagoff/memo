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


def _sub_exact(text: str, pattern: str, repl: str, *, count: int, flags: int = 0) -> str:
    new, n = re.subn(pattern, repl, text, flags=flags)
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
        flags=re.MULTILINE,
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
    if dry_run:
        for path in edits:
            console.print(f"  would update: {path.relative_to(repo)}")
        console.print("[dim]dry-run: no files written[/dim]")
        return
    for path, content in edits.items():
        path.write_text(content, encoding="utf-8")
        console.print(f"  updated: {path.relative_to(repo)}")
    console.print("[green]✓[/green] version synced; edit the CHANGELOG TODO before committing")
