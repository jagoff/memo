#!/usr/bin/env python3
"""adapter_matrix.py — deterministic drift gate over memo's published surfaces.

memo ships the same release through five manifests, two MCP configs, a hook
graph, and a CLI. Nothing today proves they agree, so a rename or a partial
version bump lands green and breaks a surface nobody runs locally: a hook
firing a command the CLI no longer registers fails silently (every hook is
soft-fail by design), and an `.mcp.json` whose ``MEMO_EMBEDDER_DIMS`` drifts
from its ``MEMO_EMBEDDER_MODEL`` corrupts the vec0 table on first write.

Each check is a pure file/registry comparison — same commit, same verdict.

Usage:
    python3 scripts/adapter_matrix.py            # print the matrix
    python3 scripts/adapter_matrix.py --check    # exit 1 on any drift
    python3 scripts/adapter_matrix.py --json     # machine-readable

Exit codes: 0 = no drift, 1 = drift found (--check only), 2 = usage/IO error.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

# MLX invariant 3 (CLAUDE.md): dims must match the Qwen3-Embedding model size.
# A mismatch corrupts the vec0 table or trips the dims guard in store/queries.py.
MODEL_DIMS = {"0.6B": 1024, "4B": 2560, "8B": 4096}


@dataclass
class Check:
    check_id: str
    surface: str
    description: str
    verification: str
    ok: bool = True
    skipped: bool = False
    findings: list[str] = field(default_factory=list)

    def fail(self, message: str) -> None:
        self.ok = False
        self.findings.append(message)

    def as_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "surface": self.surface,
            "description": self.description,
            "verification": self.verification,
            "status": "skipped" if self.skipped else ("pass" if self.ok else "fail"),
            "findings": self.findings,
        }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# --- check 1: version parity -------------------------------------------------


def _pyproject_version(path: Path) -> str | None:
    # Only the [project] table's version; a [tool.*] version must not match here.
    in_project = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_project = stripped == "[project]"
            continue
        if in_project:
            match = re.match(r'version\s*=\s*"([^"]+)"', stripped)
            if match:
                return match.group(1)
    return None


def _changelog_latest(path: Path) -> str | None:
    # First released heading; `## [Unreleased]` is skipped by the version regex.
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"##\s*\[(\d+\.\d+\.\d+)\]", line.strip())
        if match:
            return match.group(1)
    return None


def check_version_parity(root: Path) -> Check:
    check = Check(
        check_id="version-parity",
        surface="release manifests",
        description="every published manifest declares the same version",
        verification="python3 scripts/adapter_matrix.py --check",
    )
    found: dict[str, str | None] = {}

    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        found["pyproject.toml [project].version"] = _pyproject_version(pyproject)
    changelog = root / "CHANGELOG.md"
    if changelog.is_file():
        found["CHANGELOG.md latest release"] = _changelog_latest(changelog)

    for rel, keys in (
        (".claude-plugin/plugin.json", ("version",)),
        ("plugins/memo/.codex-plugin/plugin.json", ("version",)),
        ("server.json", ("version",)),
    ):
        path = root / rel
        if not path.is_file():
            check.fail(f"{rel} is missing")
            continue
        data = _read_json(path)
        for key in keys:
            found[f"{rel} .{key}"] = data.get(key)

    server = root / "server.json"
    if server.is_file():
        for index, package in enumerate(_read_json(server).get("packages") or []):
            found[f"server.json .packages[{index}].version"] = package.get("version")

    for label, value in found.items():
        if not value:
            check.fail(f"{label} has no version")
    versions = {v for v in found.values() if v}
    if len(versions) > 1:
        detail = ", ".join(f"{label}={value}" for label, value in sorted(found.items()))
        check.fail(f"versions disagree ({len(versions)} distinct): {detail}")
    return check


# --- check 2: embedder dims parity -------------------------------------------


def check_embedder_dims(root: Path) -> Check:
    check = Check(
        check_id="embedder-dims-parity",
        surface="MCP configs",
        description="a config pinning MEMO_EMBEDDER_MODEL pins matching MEMO_EMBEDDER_DIMS",
        verification="python3 scripts/adapter_matrix.py --check",
    )
    configs = sorted(root.glob(".mcp.json")) + sorted(root.glob("plugins/*/.mcp.json"))
    if not configs:
        check.skipped = True
        check.findings.append("no .mcp.json found")
        return check

    for path in configs:
        rel = path.relative_to(root)
        for name, server in (_read_json(path).get("mcpServers") or {}).items():
            env = server.get("env") or {}
            model = env.get("MEMO_EMBEDDER_MODEL")
            dims = env.get("MEMO_EMBEDDER_DIMS")
            if not model:
                # Shipping a config without a pinned model is deliberate: the
                # installed index is self-describing and adopts its own profile.
                if dims:
                    check.fail(f"{rel} [{name}] pins DIMS={dims} with no MEMO_EMBEDDER_MODEL")
                continue
            if not dims:
                check.fail(f"{rel} [{name}] pins a model but no MEMO_EMBEDDER_DIMS")
                continue
            sizes = [size for size in MODEL_DIMS if size in model]
            if len(sizes) != 1:
                check.fail(
                    f"{rel} [{name}] model {model!r} matches no known size {list(MODEL_DIMS)}"
                )
                continue
            expected = MODEL_DIMS[sizes[0]]
            if str(dims) != str(expected):
                check.fail(
                    f"{rel} [{name}] MEMO_EMBEDDER_DIMS={dims} but {sizes[0]} model needs {expected}"
                )
    return check


# --- check 3: hook commands resolve against the CLI --------------------------

_MEMO_INVOCATION = re.compile(r"(?:^|\s)memo\s+(.+)$")


def _hook_commands(data: Any) -> list[str]:
    """Every ``command`` string in the hook graph, at any nesting depth."""
    found: list[str] = []
    if isinstance(data, dict):
        command = data.get("command")
        if isinstance(command, str):
            found.append(command)
        for value in data.values():
            found.extend(_hook_commands(value))
    elif isinstance(data, list):
        for item in data:
            found.extend(_hook_commands(item))
    return found


def _invoked_path(command: str) -> list[str]:
    """The `memo` subcommand path in a hook command, minus env prefix and flags."""
    # Last match: env assignments are uppercase, so a lowercase `memo` is the
    # binary — but an interpolated value could still contain one.
    matches = list(_MEMO_INVOCATION.finditer(command))
    if not matches:
        return []
    path: list[str] = []
    for token in matches[-1].group(1).split():
        if token.startswith("-"):
            break
        path.append(token)
    return path


def check_hook_commands(root: Path) -> Check:
    check = Check(
        check_id="hook-commands-resolve",
        surface="hooks/hooks.json ↔ CLI",
        description="every memo subcommand a hook fires is registered in the CLI",
        verification="python3 scripts/adapter_matrix.py --check",
    )
    hooks_path = root / "hooks" / "hooks.json"
    if not hooks_path.is_file():
        check.fail("hooks/hooks.json is missing")
        return check

    try:
        from memo.cli import cli as memo_cli
    except ImportError as exc:
        # A drift gate must not masquerade as passing when it could not run.
        check.skipped = True
        check.findings.append(f"memo package not importable, CLI not verified: {exc}")
        return check

    for command in sorted(set(_hook_commands(_read_json(hooks_path)))):
        path = _invoked_path(command)
        if not path:
            continue
        node: Any = memo_cli
        for depth, token in enumerate(path):
            get_command = getattr(node, "get_command", None)
            if get_command is None:
                break  # a leaf command; remaining tokens are its arguments
            child = get_command(None, token)
            if child is None:
                resolved = " ".join(path[:depth]) or "memo"
                check.fail(f"`memo {' '.join(path)}` — {token!r} is not a command of `{resolved}`")
                break
            node = child
    return check


# --- check 4: referenced paths exist -----------------------------------------


def check_referenced_paths(root: Path) -> Check:
    check = Check(
        check_id="referenced-paths-exist",
        surface="plugin manifests",
        description="every path a manifest points at is present in the repo",
        verification="python3 scripts/adapter_matrix.py --check",
    )

    codex = root / "plugins" / "memo" / ".codex-plugin" / "plugin.json"
    if codex.is_file():
        reference = _read_json(codex).get("mcpServers")
        if isinstance(reference, str):
            # Resolved against the plugin root (parent of .codex-plugin/).
            target = (codex.parent.parent / reference).resolve()
            if not target.is_file():
                check.fail(
                    f"plugins/memo/.codex-plugin/plugin.json mcpServers={reference!r} "
                    f"does not resolve ({target} is missing)"
                )

    marketplace = root / ".claude-plugin" / "marketplace.json"
    if marketplace.is_file():
        for plugin in _read_json(marketplace).get("plugins") or []:
            source = plugin.get("source")
            if not isinstance(source, str):
                continue
            manifest = (
                marketplace.parent.parent / source / ".claude-plugin" / "plugin.json"
            ).resolve()
            if not manifest.is_file():
                check.fail(
                    f".claude-plugin/marketplace.json source={source!r} has no "
                    f"plugin manifest ({manifest} is missing)"
                )
    return check


CHECKS = (
    check_version_parity,
    check_embedder_dims,
    check_hook_commands,
    check_referenced_paths,
)


def run(root: Path) -> list[Check]:
    return [check(root) for check in CHECKS]


def _render(checks: list[Check]) -> str:
    lines = ["memo adapter matrix", ""]
    for check in checks:
        mark = "SKIP" if check.skipped else ("ok  " if check.ok else "FAIL")
        lines.append(f"[{mark}] {check.check_id}  ({check.surface})")
        lines.append(f"        {check.description}")
        for finding in check.findings:
            lines.append(f"        → {finding}")
    failed = [c for c in checks if not c.ok]
    skipped = [c for c in checks if c.skipped]
    lines.append("")
    lines.append(
        f"{len(checks) - len(failed) - len(skipped)} passed, "
        f"{len(failed)} failed, {len(skipped)} skipped"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--check", action="store_true", help="exit non-zero when any check fails")
    parser.add_argument("--json", dest="as_json", action="store_true", help="emit JSON")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repo root to audit")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    if not root.is_dir():
        print(f"adapter_matrix: {root} is not a directory", file=sys.stderr)
        return 2

    try:
        checks = run(root)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"adapter_matrix: could not audit {root}: {exc}", file=sys.stderr)
        return 2

    if args.as_json:
        print(
            json.dumps(
                {
                    "schema_version": "memo.adapter_matrix.v1",
                    "root": str(root),
                    "checks": [c.as_dict() for c in checks],
                    "ok": all(c.ok for c in checks),
                },
                indent=2,
            )
        )
    else:
        print(_render(checks))

    return 1 if (args.check and not all(c.ok for c in checks)) else 0


if __name__ == "__main__":
    sys.exit(main())
