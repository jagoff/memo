"""`memo release` — version bump helper.

Synchronizes the package, plugin, and MCP manifest versions and seeds a
CHANGELOG section so a content change can't ship under a stale version number.
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
_CLAUDE_PLUGIN = Path(".claude-plugin/plugin.json")
_CODEX_PLUGIN = Path("plugins/memo/.codex-plugin/plugin.json")


@dataclass(frozen=True)
class VersionJsonPath:
    path: tuple[str | int, ...]
    label: str | None = None


@dataclass(frozen=True)
class VersionTarget:
    rel_path: Path
    pattern: str
    replacements: int
    flags: int = 0
    json_paths: tuple[VersionJsonPath, ...] = ()


_VERSION_TARGETS = (
    VersionTarget(
        Path("pyproject.toml"),
        r'(^version\s*=\s*")([^"]+)(")',
        1,
        flags=re.MULTILINE,
    ),
    VersionTarget(
        _CLAUDE_PLUGIN,
        r'("version"\s*:\s*")([^"]+)(")',
        1,
        json_paths=(VersionJsonPath(("version",)),),
    ),
    VersionTarget(
        _CODEX_PLUGIN,
        r'("version"\s*:\s*")([^"]+)(")',
        1,
        json_paths=(VersionJsonPath(("version",)),),
    ),
    VersionTarget(
        Path("server.json"),
        r'("version"\s*:\s*")([^"]+)(")',
        2,
        json_paths=(
            VersionJsonPath(("version",)),
            VersionJsonPath(("packages", 0, "version"), "server.json packages[0]"),
        ),
    ),
)


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


def _json_path_value(raw: object, path: tuple[str | int, ...], *, label: str) -> object:
    current = raw
    for part in path:
        if isinstance(part, str):
            if not isinstance(current, dict) or part not in current:
                raise ValueError(f"{label} is missing {'.'.join(map(str, path))}")
            current = current[part]
            continue
        if not isinstance(current, list) or part >= len(current):
            raise ValueError(f"{label} is missing {'.'.join(map(str, path))}")
        current = current[part]
    return current


def _check_json_version_targets(
    repo: Path,
    target: VersionTarget,
    *,
    expected: str,
    versions: dict[str, str],
    issues: list[str],
) -> None:
    try:
        raw = _json_file(repo / target.rel_path)
    except ValueError as exc:
        issues.append(str(exc))
        return

    for json_path in target.json_paths:
        key = json_path.label or target.rel_path.as_posix()
        try:
            found = str(_json_path_value(raw, json_path.path, label=key) or "")
        except ValueError as exc:
            issues.append(str(exc))
            continue
        versions[key] = found
        if found != expected:
            issues.append(f"{key} version {found!r} != pyproject {expected!r}")


def _add_doc_drift(
    destination: list[str],
    *,
    path: Path,
    expected: str,
    found: str,
) -> None:
    destination.append(f"{path.name} references {found}, expected {expected}")


def _check_mcpb_manifest(
    repo: Path, *, expected: str, versions: dict[str, str], issues: list[str]
) -> None:
    manifest = repo / "packaging" / "mcpb" / "manifest.json"
    if not manifest.exists():
        return
    key = manifest.relative_to(repo).as_posix()
    try:
        raw = _json_file(manifest)
    except ValueError as exc:
        issues.append(f"{key}: {exc}")
        return

    found = str(raw.get("version") or "")
    versions[key] = found
    if found != expected:
        issues.append(f"{key} version {found!r} != pyproject {expected!r}")

    try:
        args = _json_path_value(raw, ("server", "mcp_config", "args"), label=key)
    except ValueError as exc:
        issues.append(str(exc))
        return
    if not isinstance(args, list):
        issues.append(f"{key} server.mcp_config.args must be a list")
        return

    for arg in args:
        if not isinstance(arg, str):
            continue
        match = re.fullmatch(r"mlx-memo(?:==|>=)([^,\s]+)", arg)
        if match is None:
            continue
        package_key = f"{key} mlx-memo"
        package_version = match.group(1)
        versions[package_key] = package_version
        if package_version != expected:
            issues.append(f"{package_key} version {package_version!r} != pyproject {expected!r}")


def release_check_report(repo: Path, *, strict_docs: bool = False) -> ReleaseCheckReport:
    """Validate that the checkout is release-ready.

    The hard gate checks the authoritative release files: pyproject, plugin
    manifests, MCP server manifest, and the CHANGELOG section for the current
    version. Optional docs/formula drift is a warning unless ``strict_docs`` is
    set because formulas may intentionally lag until a tarball hash exists.
    """
    issues: list[str] = []
    warnings: list[str] = []
    versions: dict[str, str] = {}

    try:
        version = _read_current_version(repo)
    except click.ClickException as exc:
        return ReleaseCheckReport("", versions, [str(exc)], warnings)
    versions["pyproject.toml"] = version

    for target in _VERSION_TARGETS:
        if not target.json_paths:
            continue
        _check_json_version_targets(
            repo, target, expected=version, versions=versions, issues=issues
        )

    # server.json may grow more packages; _VERSION_TARGETS pins only index 0,
    # so validate the rest dynamically.
    try:
        server_raw = _json_file(repo / "server.json")
    except ValueError:
        pass  # already reported by the targets loop
    else:
        packages = server_raw.get("packages")
        if isinstance(packages, list):
            for i, pkg in enumerate(packages[1:], start=1):
                key = f"server.json packages[{i}]"
                found = str(pkg.get("version") or "") if isinstance(pkg, dict) else ""
                versions[key] = found
                if found != version:
                    issues.append(f"{key} version {found!r} != pyproject {version!r}")

    _check_mcpb_manifest(repo, expected=version, versions=versions, issues=issues)

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


def _replace_target_version(text: str, target: VersionTarget, version: str) -> str:
    # No `count=` cap: n must count ALL matches so a surplus version field
    # (e.g. a second server.json package) fails loudly instead of shipping stale.
    new, n = re.subn(
        target.pattern,
        lambda m: f"{m.group(1)}{version}{m.group(3)}",
        text,
        flags=target.flags,
    )
    if n != target.replacements:
        raise ValueError(
            f"expected {target.replacements} version match(es) in "
            f"{target.rel_path.as_posix()}, got {n}"
        )
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
    """Compute new file contents for all version source-of-truth files. Pure."""
    edits: dict[Path, str] = {}
    del old

    for target in _VERSION_TARGETS:
        path = repo / target.rel_path
        edits[path] = _replace_target_version(path.read_text(encoding="utf-8"), target, new)

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


def plan_release_sync_edits(repo: Path, version: str) -> dict[Path, str]:
    """Compute edits that realign versioned release files to ``version``."""
    edits: dict[Path, str] = {}
    for target in _VERSION_TARGETS:
        path = repo / target.rel_path
        current = path.read_text(encoding="utf-8")
        updated = _replace_target_version(current, target, version)
        if updated != current:
            edits[path] = updated
    return edits


@click.group(name="release")
def release_group() -> None:
    """Release helpers — keep version numbers in sync."""


@release_group.command(name="bump")
@click.argument("level", type=click.Choice(["major", "minor", "patch"]))
@click.option("--dry-run", is_flag=True, help="Show what would change without writing.")
@click.option("--date", default=None, help="CHANGELOG date (YYYY-MM-DD); default today.")
def release_bump(level: str, dry_run: bool, date: str | None) -> None:
    """Bump version across release manifests and CHANGELOG."""
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


@release_group.command(name="sync")
@click.option("--dry-run", is_flag=True, help="Show what would change without writing.")
def release_sync(dry_run: bool) -> None:
    """Realign versioned release files to the pyproject version."""
    repo = _resolve_repo()
    version = _read_current_version(repo)
    try:
        edits = plan_release_sync_edits(repo, version)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if not edits:
        console.print(f"[green]✓[/green] version files already aligned for {version}")
        return
    if dry_run:
        for path in edits:
            console.print(f"  would update: {path.relative_to(repo)}")
        console.print("[dim]dry-run: no files written[/dim]")
        return

    _atomic_write_all(edits)
    for path in edits:
        console.print(f"  updated: {path.relative_to(repo)}")
    console.print(f"[green]✓[/green] versions synced to {version}")


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
