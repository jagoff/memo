"""Fail-closed extraction of the immutable Synapse operation surface.

The extractor deliberately reads syntax instead of importing the snapshot.  A
retirement snapshot is evidence, not code that this process is allowed to run.
"""

from __future__ import annotations

import ast
import json
import os
import stat
from collections.abc import Iterable
from pathlib import Path

from memo.operational_event import canonical_json_bytes
from tools.memflow_absorption.schemas import SynapseOperation


class SynapseCatalogError(RuntimeError):
    """A snapshot cannot be admitted as canonical Synapse source evidence."""


_DAEMON_OPERATIONS: tuple[tuple[str, str, tuple[str, ...], str | None], ...] = (
    ("runtime.py", "synapse.runtime.loop", ("runtime_loop",), "self_audit"),
    ("watcher.py", "synapse.watcher.event", ("_emit",), None),
    ("morning_digest.py", "synapse.morning_digest.run", ("run_morning_digest",), None),
    (
        "whatsapp_live.py",
        "synapse.whatsapp_live.message",
        ("last_messages", "last_messages_multi"),
        None,
    ),
    ("vault_archive.py", "synapse.vault_archive.move", ("move_to_archive",), None),
)


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise SynapseCatalogError(f"Synapse snapshot contains symlink component: {current}")


def _canonical_source(snapshot: Path) -> tuple[Path, str]:
    root = _absolute(snapshot)
    _reject_symlink_components(root)
    if not root.is_dir():
        raise SynapseCatalogError("Synapse snapshot is not a directory")
    source = root / "source.json"
    if source.is_symlink() or not source.is_file():
        raise SynapseCatalogError("Synapse snapshot lacks a regular source.json")
    try:
        encoded = source.read_bytes()
        record = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SynapseCatalogError("Synapse source record is unreadable") from exc
    if canonical_json_bytes(record) != encoded:
        raise SynapseCatalogError("Synapse source record is not canonical JSON")
    commit = record.get("source_commit") if isinstance(record, dict) else None
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise SynapseCatalogError("Synapse source commit is invalid")
    return root, commit


def _regular_files(root: Path) -> tuple[str, ...]:
    files: list[str] = []
    for directory, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        parent = Path(directory)
        for name in directory_names:
            if (parent / name).is_symlink():
                raise SynapseCatalogError(
                    f"Synapse snapshot contains symlink: {(parent / name).relative_to(root)}"
                )
        for name in file_names:
            path = parent / name
            if path.is_symlink():
                raise SynapseCatalogError(
                    f"Synapse snapshot contains symlink: {path.relative_to(root)}"
                )
            if path.is_file():
                files.append(path.relative_to(root).as_posix())
    return tuple(sorted(files))


def _parse(path: Path) -> ast.Module:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        raise SynapseCatalogError(f"canonical Synapse source is not parseable: {path}") from exc


def _keyword(call: ast.Call, name: str) -> str | None:
    for item in call.keywords:
        if (
            item.arg == name
            and isinstance(item.value, ast.Constant)
            and isinstance(item.value.value, str)
        ):
            return item.value.value
    return None


def _mcp_operations(root: Path, fixtures: tuple[str, ...]) -> list[SynapseOperation]:
    relative = "src/synapse/mcp_catalog.py"
    path = root / relative
    if not path.is_file():
        raise SynapseCatalogError("Synapse snapshot lacks canonical MCP catalog")
    rows: list[SynapseOperation] = []
    for node in ast.walk(_parse(path)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "McpToolManifest":
            continue
        operation = _keyword(node, "tool_id")
        mcp_name = _keyword(node, "mcp_name")
        if operation is None or mcp_name is None or not operation.startswith("synapse."):
            raise SynapseCatalogError("canonical MCP catalog has an incomplete tool row")
        rows.append(
            SynapseOperation(
                source_operation=operation,
                source_files=(relative,),
                source_symbols=(operation,),
                consumers=(f"mcp:{mcp_name}",),
                daemon_routes=(),
                exclusion_reason=None,
                fixture_paths=_fixtures_for(operation, fixtures),
            )
        )
    if not rows:
        raise SynapseCatalogError("canonical MCP catalog has no tools")
    return rows


def _cli_operations(root: Path, fixtures: tuple[str, ...]) -> list[SynapseOperation]:
    relative = "src/synapse/cli/parser.py"
    path = root / relative
    if not path.is_file():
        raise SynapseCatalogError("Synapse snapshot lacks canonical CLI parser")
    verbs: set[str] = set()
    for node in ast.walk(_parse(path)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_parser"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            verbs.add(node.args[0].value)
    if not verbs:
        raise SynapseCatalogError("canonical CLI parser has no verbs")
    return [
        SynapseOperation(
            source_operation=f"synapse.cli.{verb.replace('-', '_')}",
            source_files=(relative,),
            source_symbols=("build_parser",),
            consumers=(f"cli:{verb}",),
            daemon_routes=(),
            exclusion_reason=None,
            fixture_paths=_fixtures_for(verb, fixtures),
        )
        for verb in sorted(verbs)
    ]


def _fixtures_for(name: str, fixtures: Iterable[str]) -> tuple[str, ...]:
    tokens = tuple(token for token in name.replace(".", "_").split("_") if len(token) > 2)
    return tuple(sorted(path for path in fixtures if any(token in path for token in tokens)))


def _daemon_operations(root: Path, fixtures: tuple[str, ...]) -> list[SynapseOperation]:
    rows: list[SynapseOperation] = []
    for filename, operation, symbols, exclusion in _DAEMON_OPERATIONS:
        relative = f"src/synapse/{filename}"
        path = root / relative
        if not path.is_file():
            raise SynapseCatalogError(f"Synapse snapshot lacks daemon source: {relative}")
        tree = _parse(path)
        declared = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        missing = set(symbols) - declared
        if missing:
            raise SynapseCatalogError(f"daemon source lacks canonical symbol: {relative}")
        route = operation.removeprefix("synapse.").replace(".", ":")
        rows.append(
            SynapseOperation(
                source_operation=operation,
                source_files=(relative,),
                source_symbols=symbols,
                consumers=(f"daemon:{operation.split('.')[1]}",),
                daemon_routes=(route,),
                exclusion_reason=exclusion,
                fixture_paths=_fixtures_for(operation, fixtures),
            )
        )
    return rows


def discover_synapse_operations(snapshot: Path) -> tuple[SynapseOperation, ...]:
    """Return the complete canonical operation set without executing snapshot code."""

    root, _commit = _canonical_source(snapshot)
    files = _regular_files(root)
    fixtures = tuple(path for path in files if path.startswith(("tests/", "eval/", "launchd/")))
    rows = [
        *_mcp_operations(root, fixtures),
        *_cli_operations(root, fixtures),
        *_daemon_operations(root, fixtures),
    ]
    names = [row.source_operation for row in rows]
    if len(names) != len(set(names)):
        raise SynapseCatalogError("canonical Synapse operation names overlap")
    return tuple(sorted(rows, key=lambda row: row.source_operation))


__all__ = ["SynapseCatalogError", "discover_synapse_operations"]
