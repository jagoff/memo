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
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import click

from memo.cli_common import console
from memo.flags import flag_str
from memo.release_mcpb import (
    MCPB_MEMBERS,
    MCPB_NODE_MEMBERS,
    build_mcpb,
    build_mcpb_node,
)

# src/memo/cli_release.py -> repo root when running from a source checkout.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CLAUDE_PLUGIN = Path(".claude-plugin/plugin.json")
_CODEX_PLUGIN = Path("plugins/memo/.codex-plugin/plugin.json")
_PINNED_INSTALL_PATHS = {
    Path("install.sh"),
    Path("README.md"),
    Path("docs/install-new-mac.md"),
    Path("docs/reference.md"),
    Path("docs/docker.md"),
}


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
    optional: bool = False
    # Substitute the supported-release LINE (`X.Y.x`) rather than the exact
    # version. SECURITY.md names the line that still gets fixes, so it changes
    # on a minor bump and stays put on a patch one.
    release_line: bool = False


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
    VersionTarget(
        # Checked by _check_mcpb_manifest (which also parses the uvx args), so
        # no json_paths here — this target only makes `bump`/`sync` edit it.
        # Optional to match the check's exists() tolerance (older checkouts).
        Path("packaging/mcpb/manifest.json"),
        r'("version"\s*:\s*"|"mlx-memo(?:==|>=))([^"]+)(")',
        2,
        optional=True,
    ),
    VersionTarget(
        # Node bundle manifest: its version IS the bootstrap install pin
        # (bootstrap.js readPin), no mlx-memo arg to rewrite. Sync is gated by
        # tests/test_release_mcpb_node.py::test_pin_chain_in_sync.
        Path("packaging/mcpb-node/manifest.json"),
        r'("version"\s*:\s*")([^"]+)(")',
        1,
        optional=True,
    ),
    VersionTarget(
        Path("install.sh"),
        r'(^DEFAULT_VERSION=")([^"]+)(")',
        1,
        flags=re.MULTILINE,
        optional=True,
    ),
    VersionTarget(
        Path("README.md"),
        r"(raw\.githubusercontent\.com/jagoff/memo/v)([^/]+)(/install\.sh)",
        3,
        optional=True,
    ),
    VersionTarget(
        Path("docs/install-new-mac.md"),
        r"(raw\.githubusercontent\.com/jagoff/memo/v)([^/]+)(/install\.sh)",
        3,
        optional=True,
    ),
    VersionTarget(
        Path("docs/reference.md"),
        r"(raw\.githubusercontent\.com/jagoff/memo/v)([^/]+)(/install\.sh)",
        7,
        optional=True,
    ),
    VersionTarget(
        Path("docs/docker.md"),
        r"(raw\.githubusercontent\.com/jagoff/memo/v)([^/]+)(/install\.sh)",
        1,
        optional=True,
    ),
    # The supported-release line, asserted by
    # tests/test_supply_chain.py::test_security_policy_matches_current_release_
    # and_opt_in_surfaces. Left out of this table, `release bump minor` produced
    # a tree that failed CI on a test the bump itself invalidated (4.9.3 ->
    # 4.10.0, 2026-08-13).
    VersionTarget(
        Path("SECURITY.md"),
        r"(currently the `)([^`]+)(` line)",
        1,
        optional=True,
        release_line=True,
    ),
)


def _is_memo_checkout(candidate: Path) -> bool:
    """Whether ``candidate`` is a memo source checkout, not just any project."""
    pyproject = candidate / "pyproject.toml"
    if not pyproject.is_file() or not (candidate / "src" / "memo").is_dir():
        return False
    try:
        return 'name = "mlx-memo"' in pyproject.read_text(encoding="utf-8")
    except OSError:
        return False


def _checkout_containing_cwd() -> Path | None:
    """The memo checkout the caller is standing in, if any."""
    try:
        start = Path.cwd().resolve()
    except OSError:
        return None
    for candidate in (start, *start.parents):
        if _is_memo_checkout(candidate):
            return candidate
    return None


def _resolve_repo() -> Path:
    """Dev repo to operate on: MEMO_DEV_REPO, else the checkout holding the cwd,
    else the one this module was imported from.

    The cwd step is what keeps a release cut from an isolated worktree — the
    procedure CLAUDE.md prescribes — from rewriting the shared working tree the
    module happens to be imported from.
    """
    dev = flag_str("MEMO_DEV_REPO")
    if dev:
        return Path(dev).expanduser()
    return _checkout_containing_cwd() or _REPO_ROOT


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

    dependency_found = False
    for arg in args:
        if not isinstance(arg, str):
            continue
        match = re.fullmatch(r"mlx-memo(==|>=)([^,\s]+)", arg)
        if match is None:
            continue
        dependency_found = True
        package_key = f"{key} mlx-memo"
        operator = match.group(1)
        package_version = match.group(2)
        versions[package_key] = package_version
        if operator != "==":
            issues.append(f"{package_key} must use exact mlx-memo=={expected}")
        if package_version != expected:
            issues.append(f"{package_key} version {package_version!r} != pyproject {expected!r}")
    if not dependency_found:
        issues.append(f"{key} must include an exact mlx-memo=={expected} dependency")


def _check_mcpb_node_manifest(
    repo: Path, *, expected: str, versions: dict[str, str], issues: list[str]
) -> None:
    """Mirror of _check_mcpb_manifest for the Node bundle. Its version IS the
    bootstrap install pin (bootstrap.js readPin), so there is no mlx-memo arg
    to parse. Tolerant when missing (same optional criterion as the bump)."""
    manifest = repo / "packaging" / "mcpb-node" / "manifest.json"
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


def _check_mcpb_archive(
    repo: Path,
    *,
    expected: str,
    versions: dict[str, str],
    issues: list[str],
    archive_name: str = "memo.mcpb",
    source_dir_name: str = "mcpb",
    members: tuple[str, ...] = MCPB_MEMBERS,
    fallback_source_dir_name: str | None = None,
) -> None:
    archive_path = repo / "packaging" / archive_name
    key = archive_path.relative_to(repo).as_posix()
    if not archive_path.is_file():
        issues.append(f"{key} is missing; run `memo release mcpb`")
        return

    try:
        with zipfile.ZipFile(archive_path) as archive:
            names = tuple(sorted(archive.namelist()))
            expected_names = tuple(sorted(members))
            if names != expected_names:
                missing = sorted(set(expected_names) - set(names))
                unexpected = sorted(set(names) - set(expected_names))
                if missing:
                    issues.append(f"{key} missing member(s): {', '.join(missing)}")
                if unexpected:
                    issues.append(f"{key} has unexpected member(s): {', '.join(unexpected)}")

            archived: dict[str, bytes] = {}
            for member in members:
                try:
                    archived[member] = archive.read(member)
                except KeyError:
                    continue
    except (OSError, zipfile.BadZipFile) as exc:
        issues.append(f"could not read {key}: {exc}")
        return

    manifest_bytes = archived.get("manifest.json")
    if manifest_bytes is not None:
        try:
            archived_manifest = json.loads(manifest_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            issues.append(f"{key} manifest.json is invalid JSON: {exc}")
        else:
            if not isinstance(archived_manifest, dict):
                issues.append(f"{key} manifest.json must contain a JSON object")
            else:
                found = str(archived_manifest.get("version") or "")
                versions[f"{key} manifest.json"] = found
                if found != expected:
                    issues.append(
                        f"{key} manifest.json version {found!r} != pyproject {expected!r}"
                    )

    source_dir = repo / "packaging" / source_dir_name
    fallback_source_dir = (
        repo / "packaging" / fallback_source_dir_name if fallback_source_dir_name else None
    )
    for member, archived_bytes in archived.items():
        source_path = source_dir / member
        if not source_path.exists() and fallback_source_dir is not None:
            source_path = fallback_source_dir / member
        try:
            source_bytes = source_path.read_bytes()
        except OSError as exc:
            issues.append(f"could not read {source_path.relative_to(repo)}: {exc}")
            continue
        if archived_bytes != source_bytes:
            source_key = source_path.relative_to(repo).as_posix()
            issues.append(f"{key} member {member} differs from {source_key}")


def _check_install_pins(
    repo: Path, version: str, versions: dict[str, str], issues: list[str]
) -> None:
    for target in _VERSION_TARGETS:
        if target.rel_path not in _PINNED_INSTALL_PATHS:
            continue
        path = repo / target.rel_path
        if target.optional and not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            issues.append(f"could not read {target.rel_path.as_posix()}: {exc}")
            continue
        pins = [match.group(2) for match in re.finditer(target.pattern, text, target.flags)]
        versions[f"{target.rel_path.as_posix()} install pins"] = ",".join(pins)
        if len(pins) != target.replacements or any(pin != version for pin in pins):
            issues.append(
                f"{target.rel_path.as_posix()} install pin(s) {pins!r} != pyproject {version!r}"
            )


def _check_additional_server_packages(
    repo: Path, version: str, versions: dict[str, str], issues: list[str]
) -> None:
    try:
        server_raw = _json_file(repo / "server.json")
    except ValueError:
        return  # the authoritative target loop already reported the error
    packages = server_raw.get("packages")
    if not isinstance(packages, list):
        return
    for index, package in enumerate(packages[1:], start=1):
        key = f"server.json packages[{index}]"
        found = str(package.get("version") or "") if isinstance(package, dict) else ""
        versions[key] = found
        if found != version:
            issues.append(f"{key} version {found!r} != pyproject {version!r}")


def _check_changelog(repo: Path, version: str, issues: list[str]) -> None:
    try:
        text = (repo / "CHANGELOG.md").read_text(encoding="utf-8")
    except OSError as exc:
        issues.append(f"could not read CHANGELOG.md: {exc}")
        return
    section = re.search(
        rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}\n(?P<body>.*?)(?=^## \[|\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if section is None:
        issues.append(f"CHANGELOG.md is missing a dated section for {version}")
    elif re.search(r"\bTODO\b|describe changes", section.group("body"), flags=re.IGNORECASE):
        issues.append(f"CHANGELOG.md section for {version} still contains TODO text")


def _check_formula_runtime_contracts(
    *, repo: Path, formula: Path, text: str, destination: list[str]
) -> None:
    if "virtualenv_create(" in text and not re.search(
        r"^\s*include\s+Language::Python::Virtualenv\s*$", text, flags=re.MULTILINE
    ):
        destination.append(
            f"{formula.relative_to(repo)} must include Language::Python::Virtualenv "
            "before calling virtualenv_create"
        )

    if re.search(r"\.\s*pip_install(?:_and_link)?\s+buildpath\b", text):
        destination.append(
            f"{formula.relative_to(repo)} must not install buildpath with Homebrew's "
            "virtualenv pip helpers because they pass --no-deps"
        )

    if "--python=#{libexec}/bin/python" in text and not re.search(
        r"^\s*preserve_rpath\s*$", text, flags=re.MULTILINE
    ):
        destination.append(
            f"{formula.relative_to(repo)} must preserve_rpath for binary Python wheels"
        )

    if re.search(r'system\s+bin/"memo-mcp",\s*"--help"', text):
        destination.append(
            f"{formula.relative_to(repo)} must not start memo-mcp --help in its test "
            "because the stdio server blocks waiting for input"
        )


def _check_formula(
    repo: Path, version: str, *, strict_docs: bool, issues: list[str], warnings: list[str]
) -> None:
    destination = issues if strict_docs else warnings
    formula = repo / "docs" / "homebrew" / "mlx-memo.rb"
    if not formula.exists():
        return
    try:
        text = formula.read_text(encoding="utf-8")
    except OSError as exc:
        destination.append(f"could not read {formula.relative_to(repo)}: {exc}")
        return
    match = re.search(r"/archive/refs/tags/v([^/]+)\.tar\.gz", text) or re.search(
        r"mlx_memo-([0-9][^/\"']*)\.tar\.gz", text
    )
    if match and match.group(1) != version:
        _add_doc_drift(destination, path=formula, expected=version, found=match.group(1))

    url_match = re.search(r'^\s*url\s+"(?P<url>[^"]+)"', text, flags=re.MULTILINE)
    if url_match:
        url = url_match.group("url")
        if not re.fullmatch(
            rf"https://files\.pythonhosted\.org/packages/(?!source/)[^\s]+/"
            rf"mlx_memo-{re.escape(version)}\.tar\.gz",
            url,
        ):
            destination.append(
                f"{formula.relative_to(repo)} should use the exact PyPI source distribution URL"
            )

    arch_dependency = text.find("depends_on arch:")
    macos_dependency = text.find("depends_on :macos")
    if arch_dependency >= 0 and macos_dependency >= 0 and arch_dependency > macos_dependency:
        destination.append(
            f"{formula.relative_to(repo)} dependency order should put arch before macos"
        )

    _check_formula_runtime_contracts(repo=repo, formula=formula, text=text, destination=destination)


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

    _check_install_pins(repo, version, versions, issues)

    # server.json may grow more packages; _VERSION_TARGETS pins only index 0,
    # so validate the rest dynamically.
    _check_additional_server_packages(repo, version, versions, issues)

    _check_mcpb_manifest(repo, expected=version, versions=versions, issues=issues)
    _check_mcpb_node_manifest(repo, expected=version, versions=versions, issues=issues)
    _check_mcpb_archive(repo, expected=version, versions=versions, issues=issues)
    if (repo / "packaging" / "mcpb-node" / "manifest.json").exists():
        _check_mcpb_archive(
            repo,
            expected=version,
            versions=versions,
            issues=issues,
            archive_name="memo-node.mcpb",
            source_dir_name="mcpb-node",
            members=MCPB_NODE_MEMBERS,
            fallback_source_dir_name="mcpb",
        )

    _check_changelog(repo, version, issues)
    _check_formula(
        repo,
        version,
        strict_docs=strict_docs,
        issues=issues,
        warnings=warnings,
    )

    # Surfaces that carry no version but ship with the release, and whose drift
    # is silent: a hook firing a subcommand the CLI dropped (hooks soft-fail by
    # design), an `.mcp.json` whose embedder dims stopped matching its model
    # (MLX invariant 3), a manifest pointing at a file that is gone.
    from memo.adapter_matrix import adapter_issues

    issues.extend(adapter_issues(repo))

    return ReleaseCheckReport(version, versions, issues, warnings)


def _replace_target_version(text: str, target: VersionTarget, version: str) -> str:
    written = ".".join(version.split(".")[:2]) + ".x" if target.release_line else version
    # No `count=` cap: n must count ALL matches so a surplus version field
    # (e.g. a second server.json package) fails loudly instead of shipping stale.
    new, n = re.subn(
        target.pattern,
        lambda m: f"{m.group(1)}{written}{m.group(3)}",
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
    """Write every file atomically through private, unpredictable siblings."""
    staged: dict[Path, Path] = {}
    try:
        for path, content in edits.items():
            if path.is_symlink():
                raise ValueError(f"release target must be a regular file: {path}")
            if path.exists():
                metadata = path.lstat()
                if not stat.S_ISREG(metadata.st_mode):
                    raise ValueError(f"release target must be a regular file: {path}")
                mode = stat.S_IMODE(metadata.st_mode)
            else:
                mode = 0o644
            descriptor, temporary_name = tempfile.mkstemp(
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
            )
            tmp = Path(temporary_name)
            staged[path] = tmp
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    descriptor = -1
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(tmp, mode)
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
    except BaseException:
        for tmp in staged.values():
            tmp.unlink(missing_ok=True)
        raise
    try:
        for path, tmp in staged.items():
            os.replace(tmp, path)
    finally:
        for tmp in staged.values():
            tmp.unlink(missing_ok=True)


def plan_release_edits(repo: Path, old: str, new: str, date: str) -> dict[Path, str]:
    """Compute new file contents for all version source-of-truth files. Pure."""
    edits: dict[Path, str] = {}
    del old

    for target in _VERSION_TARGETS:
        path = repo / target.rel_path
        if target.optional and not path.exists():
            continue
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
        if target.optional and not path.exists():
            continue
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


@release_group.command(name="mcpb")
@click.option(
    "--output",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Archive destination; defaults to packaging/memo.mcpb.",
)
@click.option(
    "--node-output",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Node archive destination; defaults to packaging/memo-node.mcpb.",
)
def release_mcpb(output: Path | None, node_output: Path | None) -> None:
    """Build both deterministic MCPB archives from their source directories."""
    repo = _resolve_repo()
    try:
        destinations = (build_mcpb(repo, output), build_mcpb_node(repo, node_output))
    except OSError as exc:
        raise click.ClickException(f"could not build MCPB: {exc}") from exc
    for destination in destinations:
        # Artifact paths are intended to be copyable and machine-readable.
        # Rich's default wrapping can split a filename itself when the absolute
        # path is wider than the terminal (for example ``memo.mc\npb``).
        console.print(f"[green]✓[/green] built {destination}", soft_wrap=True)


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
