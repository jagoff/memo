def test_mcp_command_profile_follows_the_registry(monkeypatch, tmp_path) -> None:
    """`mcp-command`, `setup` and `doctor --agent` must agree on the profile.

    MEMO_MCP_PROFILE has a registry default of "agent", so the old
    `flag_str(...) or "agent"` fallback never reached the client-specific value:
    `memo mcp-command --client codex` emitted "agent" while `memo setup codex`
    wrote "core" and the doctor check asserted "core".
    """
    from memo.runtime.agent_registry import AGENT_REGISTRY
    from memo.runtime.mcp import _resolved_mcp_profile

    monkeypatch.delenv("MEMO_MCP_PROFILE", raising=False)
    monkeypatch.setenv("MEMO_CONFIG_DIR", str(tmp_path))

    assert _resolved_mcp_profile("codex") == AGENT_REGISTRY["codex"].mcp_profile
    assert _resolved_mcp_profile("claude-code") == AGENT_REGISTRY["claude-code"].mcp_profile
    assert _resolved_mcp_profile(None) == "agent"
    assert _resolved_mcp_profile("not-a-client") == "agent"

    # An explicit setting still wins over the registry default.
    monkeypatch.setenv("MEMO_MCP_PROFILE", "full")
    assert _resolved_mcp_profile("codex") == "full"
