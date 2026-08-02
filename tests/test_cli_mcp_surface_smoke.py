"""Smoke coverage for the registered CLI and MCP surfaces.

These tests intentionally do not assert command behavior. Domain behavior lives
in focused test modules; this file catches wiring/import/help/schema breakage
across the whole exposed surface.
"""

from __future__ import annotations

import ast
import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest
from click import Command, Group
from click.testing import CliRunner

from memo.cli import cli
from memo.config import Config
from memo.memory import Memory
from memo.server import build_server


def _walk_cli_commands(cmd: Command, path: tuple[str, ...] = ()) -> Iterator[tuple[str, ...]]:
    if not isinstance(cmd, Group):
        return
    for name, subcommand in sorted(cmd.commands.items()):
        subpath = (*path, name)
        yield subpath
        yield from _walk_cli_commands(subcommand, subpath)


CLI_COMMAND_PATHS = tuple(_walk_cli_commands(cli))

OPERATIONAL_TOOL_NAMES = frozenset(
    {
        "memo_attention_ack",
        "memo_attention_add",
        "memo_conflict_open",
        "memo_conflict_resolve",
        "memo_evidence_pack",
        "memo_federation_preview",
        "memo_focus_clear",
        "memo_focus_set",
        "memo_handoff_consume",
        "memo_handoff_create",
        "memo_journal_verify",
        "memo_operational_state",
        "memo_outcome_record",
        "memo_procedure_candidates",
        "memo_procedure_promote",
        "memo_signal_list",
        "memo_signal_remember",
    }
)


@pytest.mark.parametrize("command_path", CLI_COMMAND_PATHS, ids=lambda p: " ".join(p))
def test_cli_command_help_smoke(command_path: tuple[str, ...], tmp_path: Path) -> None:
    env = {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_VAULT_PATH": str(tmp_path / "vault"),
        "MEMO_EMBEDDER_VIA_DAEMON": "0",
        "MEMO_SKIP_MODEL_VERSION_CHECK": "1",
    }

    result = CliRunner().invoke(cli, [*command_path, "--help"], env=env)

    assert result.exit_code == 0, result.output
    assert "Usage:" in result.output


def _decorated_server_tool_names() -> set[str]:
    names: set[str] = set()
    for path in sorted(Path("src/memo").glob("server*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for decorator in node.decorator_list:
                decorator_name = ""
                target = decorator.func if isinstance(decorator, ast.Call) else decorator
                if isinstance(target, ast.Name):
                    decorator_name = target.id
                elif isinstance(target, ast.Attribute):
                    decorator_name = target.attr
                if decorator_name in {"annotated_tool", "tool"}:
                    names.add(node.name)
    return names


def _mcp_tool_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: str) -> set[str]:
    monkeypatch.setenv("MEMO_NONINTERACTIVE", "1")
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / profile / "data"))
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / profile / "state"))
    monkeypatch.setenv("MEMO_VAULT_PATH", str(tmp_path / profile / "vault"))
    monkeypatch.setenv("MEMO_EMBEDDER_DIMS", "4")
    monkeypatch.setenv("MEMO_EMBEDDER_VIA_DAEMON", "0")
    monkeypatch.setenv("MEMO_SKIP_MODEL_VERSION_CHECK", "1")
    monkeypatch.setenv("MEMO_MCP_PROFILE", profile)
    monkeypatch.delenv("MEMO_MCP_SLIM", raising=False)

    cfg = Config(
        data_dir=tmp_path / profile / "data",
        vault_path=tmp_path / profile / "vault",
        state_dir=tmp_path / profile / "state",
        embedder_dims=4,
    )
    mem = Memory(cfg)
    try:
        mem.embedder.embed = lambda inputs: [[1.0, 0.0, 0.0, 0.0] for _ in inputs]
        server = build_server(memory=mem)
        tools = asyncio.run(server.list_tools())
        return {tool.name for tool in tools}
    finally:
        mem.close()


def test_mcp_full_profile_registers_every_decorated_server_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = _decorated_server_tool_names()

    registered = _mcp_tool_names(tmp_path, monkeypatch, "full")

    assert expected <= registered
    assert len(registered) == 164


@pytest.mark.parametrize(
    ("profile", "expected_count"),
    [
        ("agent", 43),
        ("core", 60),
        ("full", 164),
    ],
)
def test_mcp_profile_tool_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: str, expected_count: int
) -> None:
    names = _mcp_tool_names(tmp_path, monkeypatch, profile)

    assert len(names) == expected_count
    assert "memo_version" in names
    assert "memo_graph" in names
    assert names >= OPERATIONAL_TOOL_NAMES
