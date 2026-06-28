"""`memo release` — version bump helper.

Synchronizes the four version source-of-truth files and seeds a CHANGELOG
section so a content change can't ship under a stale version number.
Registered in cli.py via `cli.add_command(release_group)`.
"""

from __future__ import annotations

import datetime
import json
import os
import re
from dataclasses import dataclass
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
    pyproject = repo / "pyproject.toml"
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError as exc:
        raise click.ClickException(
            f"no pyproject.toml at {repo} — run from the memo checkout or set MEMO_DEV_REPO"
        ) from exc
    m = re.search(r'^version = "([^"]+)"', text, re.MULTILINE)
    if not m:
        raise click.ClickException("could not find version in pyproject.toml")
    return m.group(1)


@dataclass
class ReleaseCheckReport:
    version: str
    versions: dict[str, str]
    issues: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "version": self.version,
            "versions": self.versions,
            "issues": self.issues,
            "warnings": self.warnings,
        }


def _json_file(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {path.name}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return raw


def _add_doc_drift(
    destination: list[str],
    *,
    path: Path,
    expected: str,
    found: str,
) -> None:
    destination.append(f"{path.name} references {found}, expected {expected}")


def release_check_report(repo: Path, *, strict_docs: bool = False) -> ReleaseCheckReport:
    """Validate that the checkout is release-ready.

    The hard gate checks the authoritative release files: pyproject,
    plugin manifest, MCP server manifest, and the CHANGELOG section for the
    current version. Optional docs/formula drift is a warning unless
    ``strict_docs`` is set because formulas may intentionally lag until a
    tarball hash exists.
    """
    issues: list[str] = []
    warnings: list[str] = []
    versions: dict[str, str] = {}

    try:
        version = _read_current_version(repo)
    except click.ClickException as exc:
        return ReleaseCheckReport("", versions, [str(exc)], warnings)
    versions["pyproject.toml"] = version

    try:
        plugin_version = str(_json_file(repo / ".claude-plugin" / "plugin.json").get("version") or "")
        versions[".claude-plugin/plugin.json"] = plugin_version
        if plugin_version != version:
            issues.append(
                f".claude-plugin/plugin.json version {plugin_version!r} != pyproject {version!r}"
            )
    except ValueError as exc:
        issues.append(str(exc))

    try:
        server = _json_file(repo / "server.json")
        server_version = str(server.get("version") or "")
        versions["server.json"] = server_version
        if server_version != version:
            issues.append(f"server.json version {server_version!r} != pyproject {version!r}")
        packages = server.get("packages") or []
        if isinstance(packages, list):
            for i, pkg in enumerate(packages):
                if not isinstance(pkg, dict) or "version" not in pkg:
                    continue
                pkg_version = str(pkg.get("version") or "")
                key = f"server.json packages[{i}]"
                versions[key] = pkg_version
                if pkg_version != version:
                    issues.append(f"{key} version {pkg_version!r} != pyproject {version!r}")
    except ValueError as exc:
        issues.append(str(exc))

    changelog = repo / "CHANGELOG.md"
    try:
        cl_text = changelog.read_text(encoding="utf-8")
    except OSError as exc:
        issues.append(f"could not read CHANGELOG.md: {exc}")
    else:
        section = re.search(
            rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}\n(?P<body>.*?)(?=^## \[|\Z)",
            cl_text,
            flags=re.MULTILINE | re.DOTALL,
        )
        if section is None:
            issues.append(f"CHANGELOG.md is missing a dated section for {version}")
        else:
            body = section.group("body")
            if re.search(r"\bTODO\b|describe changes", body, flags=re.IGNORECASE):
                issues.append(f"CHANGELOG.md section for {version} still contains TODO text")

    doc_drifts = issues if strict_docs else warnings
    formula = repo / "docs" / "homebrew" / "mlx-memo.rb"
    if formula.exists():
        try:
            text = formula.read_text(encoding="utf-8")
        except OSError as exc:
            doc_drifts.append(f"could not read {formula.relative_to(repo)}: {exc}")
        else:
            m = re.search(r"/archive/refs/tags/v([^/]+)\.tar\.gz", text) or re.search(
                r"mlx_memo-([0-9][^/\"']*)\.tar\.gz", text
            )
            if m and m.group(1) != version:
                _add_doc_drift(
                    doc_drifts,
                    path=formula,
                    expected=version,
                    found=m.group(1),
                )

    return ReleaseCheckReport(version, versions, issues, warnings)


def _sub_exact(text: str, pattern: str, repl: str, *, count: int, flags: int = 0) -> str:
    new, n = re.subn(pattern, repl, text, flags=flags)
    if n != count:
        raise ValueError(f"expected {count} match(es) for {pattern!r}, got {n}")
    return new


def _atomic_write_all(edits: dict[Path, str]) -> None:
    """Write every file atomically. Stage all contents to sibling ``*.tmp``
    files first, so a failure while staging touches no real file; then
    ``os.replace`` each temp into place (each replace is atomic on POSIX).
    Cleans up staged temps if staging fails."""
    staged: dict[Path, Path] = {}
    try:
        for path, content in edits.items():
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(content, encoding="utf-8")
            staged[path] = tmp
    except OSError:
        for tmp in staged.values():
            tmp.unlink(missing_ok=True)
        raise
    for path, tmp in staged.items():
        os.replace(tmp, path)


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
    cl_text = changelog.read_text(encoding="utf-8")
    cl_new, cl_n = re.subn(
        r"## \[Unreleased\]\n\n",
        lambda _m: f"## [Unreleased]\n\n{section}",
        cl_text,
    )
    if cl_n != 1:
        raise ValueError(f"expected 1 Unreleased section, got {cl_n}")
    edits[changelog] = cl_new
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
    try:
        edits = plan_release_edits(repo, old, new, when)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    console.print(f"[bold]{old} → {new}[/bold] ({level})")
    if dry_run:
        for path in edits:
            console.print(f"  would update: {path.relative_to(repo)}")
        console.print("[dim]dry-run: no files written[/dim]")
        return
    _atomic_write_all(edits)
    for path in edits:
        console.print(f"  updated: {path.relative_to(repo)}")
    console.print("[green]✓[/green] version synced; edit the CHANGELOG TODO before committing")


@release_group.command(name="check")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
@click.option(
    "--strict-docs",
    is_flag=True,
    help="Treat docs/formula version drift as a release-blocking issue.",
)
def release_check(as_json: bool, strict_docs: bool) -> None:
    """Verify version files and CHANGELOG are release-ready."""
    repo = _resolve_repo()
    report = release_check_report(repo, strict_docs=strict_docs)

    if as_json:
        click.echo(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        raise click.exceptions.Exit(0 if report.ok else 1)

    if report.ok:
        console.print(f"[green]✓[/green] release check passed for {report.version}")
    else:
        console.print(f"[red]✗[/red] release check failed for {report.version or repo}")
        for issue in report.issues:
            console.print(f"  [red]-[/red] {issue}")
    if report.warnings:
        console.print("[yellow]Warnings:[/yellow]")
        for warning in report.warnings:
            console.print(f"  [yellow]-[/yellow] {warning}")
    raise click.exceptions.Exit(0 if report.ok else 1)
