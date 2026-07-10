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

        assert asyncio.run(server.get_tool("memo_save")) is not None
        assert asyncio.run(server.get_tool("memo_search")) is not None
        assert asyncio.run(server.get_tool("memo_graph_path")) is None
        assert asyncio.run(server.get_tool("memo_collaborative_connections")) is None
    finally:
        mem.close()


def test_mcp_core_profile_tools_have_descriptions(tmp_path, monkeypatch) -> None:
    cfg = Config(
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        vault_path=tmp_path / "vault",
        embedder_dims=4,
    )
    monkeypatch.setenv("MEMO_MCP_PROFILE", "core")
    mem = Memory(cfg)
    try:
        tools = asyncio.run(build_server(memory=mem).list_tools())
        missing = sorted(tool.name for tool in tools if not (tool.description or "").strip())
        assert missing == []
    finally:
        mem.close()


def test_mcp_agent_profile_is_default_and_exposes_core_tools(tmp_path, monkeypatch) -> None:
    from memo.surface import AGENT_MCP_TOOLS

    cfg = Config(
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        vault_path=tmp_path / "vault",
        embedder_dims=4,
    )
    monkeypatch.delenv("MEMO_MCP_PROFILE", raising=False)
    monkeypatch.delenv("MEMO_MCP_SLIM", raising=False)
    mem = Memory(cfg)
    try:
        tools = asyncio.run(build_server(memory=mem).list_tools())
        tool_names = {tool.name for tool in tools}
        # Definition must equal runtime — no silent drift either way.
        assert tool_names == set(AGENT_MCP_TOOLS)
        # Advanced tools the agent profile must NOT have
        assert "memo_graph_path" not in tool_names
        assert "memo_contradict_scan" not in tool_names
    finally:
        mem.close()


def test_mcp_full_profile_keeps_advanced_tools(tmp_path, monkeypatch) -> None:
    cfg = Config(
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        vault_path=tmp_path / "vault",
        embedder_dims=4,
    )
    monkeypatch.setenv("MEMO_MCP_PROFILE", "full")
    mem = Memory(cfg)
    try:
        server = build_server(memory=mem)
        assert asyncio.run(server.get_tool("memo_graph_path")) is not None
        assert asyncio.run(server.get_tool("memo_collaborative_connections")) is not None
    finally:
        mem.close()


def test_agent_tools_definition_is_14_and_excludes_idle_from_removal(monkeypatch) -> None:
    monkeypatch.setenv("MEMO_MCP_PROFILE", "agent")
    monkeypatch.delenv("MEMO_MCP_SLIM", raising=False)
    from memo.surface import AGENT_MCP_TOOLS, mcp_tools_to_remove

    removed = mcp_tools_to_remove()
    assert len(AGENT_MCP_TOOLS) == 14
    for name in (
        "memo_idle_capture",
        "memo_pop_notification",
        "memo_start_session",
        "memo_save_text",
        "memo_graph",
    ):
        assert name in AGENT_MCP_TOOLS
        assert name not in removed


def test_token_cost_recognizes_agent() -> None:
    from memo.surface import mcp_profile_token_cost

    count, cost, reduced = mcp_profile_token_cost("agent")
    assert count == "14"
    assert cost == "~1.4k"
    assert reduced is True


def test_token_cost_core_and_slim_are_reduced() -> None:
    from memo.surface import mcp_profile_token_cost

    for profile in ("core", "slim"):
        count, cost, reduced = mcp_profile_token_cost(profile)
        assert count == "34"
        assert cost == "~3.0k"
        assert reduced is True


def test_token_cost_full_is_not_reduced() -> None:
    from memo.surface import mcp_profile_token_cost

    for profile in ("full", "default"):
        count, cost, reduced = mcp_profile_token_cost(profile)
        assert count == "126"
        assert cost == "~15k"
        assert reduced is False


def test_token_cost_active_default_resolves_to_agent(monkeypatch) -> None:
    monkeypatch.delenv("MEMO_MCP_PROFILE", raising=False)
    monkeypatch.delenv("MEMO_MCP_SLIM", raising=False)
    from memo.surface import mcp_profile_token_cost

    count, cost, reduced = mcp_profile_token_cost()
    assert count == "14"
    assert cost == "~1.4k"
    assert reduced is True
