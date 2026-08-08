from __future__ import annotations

import asyncio

import pytest
from click.testing import CliRunner

from memo.cli import cli
from memo.config import Config
from memo.errors import ValidationError
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


def test_mcp_profile_rejects_unknown_value(monkeypatch) -> None:
    monkeypatch.setenv("MEMO_MCP_PROFILE", "typo")

    from memo.surface import mcp_profile

    with pytest.raises(ValidationError, match=r"MEMO_MCP_PROFILE.*typo"):
        mcp_profile()


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


def test_agent_tools_definition_is_41_and_excludes_idle_from_removal(monkeypatch) -> None:
    monkeypatch.setenv("MEMO_MCP_PROFILE", "agent")
    monkeypatch.delenv("MEMO_MCP_SLIM", raising=False)
    from memo.surface import AGENT_MCP_TOOLS, mcp_tools_to_remove

    removed = mcp_tools_to_remove()
    assert len(AGENT_MCP_TOOLS) == 41
    for name in (
        "memo_idle_capture",
        "memo_pop_notification",
        "memo_start_session",
        "memo_save_text",
        "memo_graph",
        "memo_evidence_pack",
        "memo_operational_state",
        "memo_outcome_record",
        "memo_signal_list",
        "memo_signal_remember",
    ):
        assert name in AGENT_MCP_TOOLS
        assert name not in removed
    assert "memo_terminal_list" not in removed


def test_token_cost_recognizes_agent() -> None:
    from memo.surface import mcp_profile_token_cost

    count, cost, reduced = mcp_profile_token_cost("agent")
    assert count == "41"
    assert cost == "~9.4k"
    assert reduced is True


def test_token_cost_core_and_slim_are_reduced() -> None:
    from memo.surface import mcp_profile_token_cost

    for profile in ("core", "slim"):
        count, cost, reduced = mcp_profile_token_cost(profile)
        assert count == "58"
        assert cost == "~12.9k"
        assert reduced is True


def test_token_cost_full_is_not_reduced() -> None:
    from memo.surface import mcp_profile_token_cost

    for profile in ("full", "default"):
        count, cost, reduced = mcp_profile_token_cost(profile)
        assert count == "164"
        assert cost == "~30.4k"
        assert reduced is False


def test_token_cost_active_default_resolves_to_agent(monkeypatch) -> None:
    monkeypatch.delenv("MEMO_MCP_PROFILE", raising=False)
    monkeypatch.delenv("MEMO_MCP_SLIM", raising=False)
    from memo.surface import mcp_profile_token_cost

    count, cost, reduced = mcp_profile_token_cost()
    assert count == "41"
    assert cost == "~9.4k"
    assert reduced is True


@pytest.mark.parametrize("profile", ["core", "full"])
def test_token_cost_count_matches_real_server(tmp_path, monkeypatch, profile) -> None:
    """The hand-maintained _PROFILE_TOKEN_COST tool counts must equal the real
    build_server()+list_tools() count for that profile, so the advisory table
    can't silently desync from the registered surface. (The `agent` count is
    already cross-checked by test_mcp_agent_profile_is_default_and_exposes_core_tools.)
    """
    from memo.surface import mcp_profile_token_cost

    cfg = Config(
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        vault_path=tmp_path / "vault",
        embedder_dims=4,
    )
    monkeypatch.setenv("MEMO_MCP_PROFILE", profile)
    monkeypatch.delenv("MEMO_MCP_SLIM", raising=False)
    mem = Memory(cfg)
    try:
        tools = asyncio.run(build_server(memory=mem).list_tools())
    finally:
        mem.close()

    count_label, _cost, _reduced = mcp_profile_token_cost(profile)
    assert len(tools) == int(count_label)


def _measure_wire_tokens(tmp_path, monkeypatch, profile: str) -> tuple[int, int]:
    """Measure the real `tools/list` payload for `profile`.

    Returns ``(tool_count, estimated_tokens)``. The payload is what a client
    actually pays for on every connection — ``name`` + ``description`` +
    ``inputSchema`` per tool, serialized as compact JSON — sized with memo's
    own chars/4 estimator so the advisory label speaks the same units as the
    rest of memo's token accounting.
    """
    import json

    from memo.eval_tokens import count_tokens

    cfg = Config(
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        vault_path=tmp_path / "vault",
        embedder_dims=4,
    )
    monkeypatch.setenv("MEMO_MCP_PROFILE", profile)
    monkeypatch.delenv("MEMO_MCP_SLIM", raising=False)
    mem = Memory(cfg)
    try:
        tools = asyncio.run(build_server(memory=mem).list_tools())
    finally:
        mem.close()

    payload = []
    for tool in tools:
        mcp_tool = tool.to_mcp_tool()
        payload.append(
            {
                "name": mcp_tool.name,
                "description": mcp_tool.description or "",
                "inputSchema": mcp_tool.inputSchema,
            }
        )
    wire = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    return len(tools), count_tokens(wire)


# The label is an advisory rounded to 0.1k, so it must not red on byte-level
# churn in one tool description — but it must red on the ~2x understatement
# that let "~18.1k" survive against a 30k surface.
_LABEL_TOLERANCE = 0.05


@pytest.mark.parametrize("profile", ["agent", "core", "full"])
def test_token_cost_label_matches_measured_wire_payload(tmp_path, monkeypatch, profile) -> None:
    """`_PROFILE_TOKEN_COST` is hand-maintained; this is what re-derives it.

    Run this test to get the current number: on a mismatch the failure message
    carries the exact label to paste in. Keep `docs/`, `README.md`, and the
    `install-mcp`/`doctor` advisories in step with it — they quote the same
    figures.
    """
    from memo.surface import mcp_profile_token_cost

    count, measured = _measure_wire_tokens(tmp_path, monkeypatch, profile)
    count_label, cost_label, _reduced = mcp_profile_token_cost(profile)

    assert count == int(count_label)

    claimed = float(cost_label.removeprefix("~").removesuffix("k")) * 1000
    assert abs(claimed - measured) <= _LABEL_TOLERANCE * measured, (
        f"{profile} profile advertises {cost_label} but its {count} tool schemas "
        f"measure {measured} tokens — update the label to ~{measured / 1000:.1f}k"
    )
