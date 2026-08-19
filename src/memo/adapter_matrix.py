"""Drift checks over the surfaces memo publishes alongside its version.

`memo release check` already owns **version parity** — it validates pyproject,
both plugin manifests, server.json (including extra packages), the mcpb
manifests and archive, install pins, the Homebrew formula and the CHANGELOG
section. This module deliberately does NOT re-check any of that; duplicating it
here would only add a weaker second opinion that can disagree with the real one.

What it covers is the drift `release check` does not see, and that fails
*silently* when it happens:

- **hook-commands-resolve** — `hooks/hooks.json` OR the nightly LaunchAgent
  script firing a `memo` subcommand the CLI does not register. Every hook is
  soft-fail by design, so a rename stops recall, capture or sync with no error
  anywhere; the nightly script logs `Error: No such command` into a file nobody
  reads and skips that pass (observed for four consecutive nights when
  `ops gc-emitted-ledgers` shipped in the template before the binary).
- **embedder-dims-parity** — an `.mcp.json` pinning `MEMO_EMBEDDER_MODEL` whose
  `MEMO_EMBEDDER_DIMS` does not match the model size. That is MLX invariant 3:
  a mismatch corrupts the vec0 table on first write.
- **referenced-paths-exist** — a manifest pointing at a file that is not there.

Each check is a pure file/registry comparison: same commit, same verdict.
Surfaced through `memo release check`; `tests/test_adapter_matrix.py` proves
each one fails on real drift rather than only passing on a clean tree.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# MLX invariant 3 (CLAUDE.md): dims must match the Qwen3-Embedding model size.
# A mismatch corrupts the vec0 table or trips the dims guard in store/queries.py.
MODEL_DIMS = {"0.6B": 1024, "4B": 2560, "8B": 4096}


@dataclass
class Check:
    check_id: str
    surface: str
    description: str
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
            "status": "skipped" if self.skipped else ("pass" if self.ok else "fail"),
            "findings": self.findings,
        }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# --- hook commands resolve against the CLI -----------------------------------

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


def _script_memo_commands(script: str) -> list[str]:
    """Every `memo <subcommand>` invocation in a shell script.

    The nightly template calls the binary through a `__MEMO_BIN__` placeholder
    (rendered at install time), so match that form as well as a literal `memo`.
    """
    found: list[str] = []
    for raw in script.splitlines():
        line = raw.strip()
        if line.startswith("#"):
            continue
        for match in re.finditer(r'"?(?:__MEMO_BIN__|\bmemo)"?\s+([a-z][\w -]*)', line):
            found.append(f"memo {match.group(1).strip()}")
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
    )
    hooks_path = root / "hooks" / "hooks.json"
    nightly_path = root / "launchd" / "memo-nightly.sh"
    invoked: list[tuple[str, str]] = []
    if hooks_path.is_file():
        invoked += [("hooks/hooks.json", c) for c in _hook_commands(_read_json(hooks_path))]
    if nightly_path.is_file():
        invoked += [
            ("launchd/memo-nightly.sh", c)
            for c in _script_memo_commands(nightly_path.read_text(encoding="utf-8"))
        ]
    if not invoked:
        # Nothing to resolve — vacuously clean, the same way the other checks
        # no-op on an absent surface. Whether the repo *should* ship these files
        # is not this check's question, and treating absence as drift makes the
        # gate fire on every partial tree.
        return check

    try:
        from memo.cli import cli as memo_cli
    except ImportError as exc:
        # A drift gate must not masquerade as passing when it could not run.
        check.skipped = True
        check.findings.append(f"memo package not importable, CLI not verified: {exc}")
        return check

    for surface, command in sorted(set(invoked)):
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
                check.fail(
                    f"{surface}: `memo {' '.join(path)}` — "
                    f"{token!r} is not a command of `{resolved}`"
                )
                break
            node = child
    return check


# --- embedder dims parity ----------------------------------------------------


def check_embedder_dims(root: Path) -> Check:
    check = Check(
        check_id="embedder-dims-parity",
        surface="MCP configs",
        description="a config pinning MEMO_EMBEDDER_MODEL pins matching MEMO_EMBEDDER_DIMS",
    )
    configs = sorted(root.glob(".mcp.json")) + sorted(root.glob("plugins/*/.mcp.json"))
    # No config is not a skip: there is simply nothing to check. `skipped` is
    # reserved for a real inability to verify a surface that IS there, because
    # `adapter_issues` reports that as an issue rather than a silent pass.
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


# --- referenced paths exist --------------------------------------------------


def check_referenced_paths(root: Path) -> Check:
    check = Check(
        check_id="referenced-paths-exist",
        surface="plugin manifests",
        description="every path a manifest points at is present in the repo",
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
    check_hook_commands,
    check_embedder_dims,
    check_referenced_paths,
)


def run(root: Path) -> list[Check]:
    """Run every adapter check against `root`. Raises on unreadable/invalid JSON."""
    return [check(root) for check in CHECKS]


def adapter_issues(root: Path) -> list[str]:
    """Flat issue strings for `memo release check`.

    A check that could not run is reported as an issue too, never silently
    treated as a pass — the whole point is that these drifts are invisible.
    """
    try:
        checks = run(root)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return [f"adapter matrix could not run: {exc}"]

    issues: list[str] = []
    for check in checks:
        if check.skipped:
            issues.extend(f"{check.check_id} did not run: {f}" for f in check.findings)
        elif not check.ok:
            issues.extend(f"{check.check_id}: {f}" for f in check.findings)
    return issues
