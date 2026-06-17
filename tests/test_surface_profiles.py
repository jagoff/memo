from __future__ import annotations

import asyncio

from click.testing import CliRunner

from memo.cli import cli
from memo.config import Config
from memo.memory import Memory
from memo.server import build_server


def _isolated_env(tmp_path, **extra: str) -> dict[str, str]:
    data = tmp_path / "data"
    state = tmp_path / "state"
    vault = tmp_path / "vault"
    data.mkdir()
    state.mkdir()
    vault.mkdir()
    return {
        "MEMO_DATA_DIR": str(data),
        "MEMO_STATE_DIR": str(state),
        "MEMO_VAULT_PATH": str(vault),
        "MEMO_CONFIG_FILE": str(tmp_path / "memo-config.toml"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_EMBEDDER_VIA_DAEMON": "0",
        **extra,
    }


def test_cli_core_profile_hides_experimental_commands(tmp_path) -> None:
    runner = CliRunner()
    env = _isolated_env(tmp_path, MEMO_CLI_PROFILE="core")

    result = runner.invoke(cli, ["--help"], env=env)

    assert result.exit_code == 0
    assert "save" in result.output
    assert "search" in result.output
    assert "briefing" in result.output
    assert "graph" not in result.output
    assert "collaborative" not in result.output
    assert "multimodal" not in result.output


def test_cli_core_profile_rejects_hidden_command(tmp_path) -> None:
    runner = CliRunner()
    env = _isolated_env(tmp_path, MEMO_CLI_PROFILE="core")

    result = runner.invoke(cli, ["graph", "--help"], env=env)

    assert result.exit_code != 0
    assert "No such command" in result.output


def test_mcp_core_profile_hides_advanced_tools(tmp_path, monkeypatch) -> None:
    cfg = Config(
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        vault_path=tmp_path / "vault",
        embedder_dims=4,
    )
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0] for _ in inputs],
    )
    monkeypatch.setenv("MEMO_MCP_PROFILE", "core")
    mem = Memory(cfg)
    try:
        server = build_server(memory=mem)

        assert asyncio.run(server.get_tool("memory_save")) is not None
        assert asyncio.run(server.get_tool("memory_search")) is not None
        assert asyncio.run(server.get_tool("memory_graph_path")) is None
        assert asyncio.run(server.get_tool("memory_collaborative_connections")) is None
    finally:
        mem.close()
