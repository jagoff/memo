"""Tests for `memo install-mcp` (contract-backed MCP installer wrapper)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

pytest.importorskip("consciousness_contracts")

from memo import cli_install_mcp
from memo.cli import cli


def test_resolve_isolated_rejects_venv(monkeypatch, tmp_path):
    """A .venv memo-mcp must never be chosen — it's the documented footgun."""
    # No isolated candidate exists; PATH + fallback both point at a .venv.
    monkeypatch.setattr(cli_install_mcp, "Path", Path)  # keep real Path
    monkeypatch.setattr(cli_install_mcp.shutil, "which", lambda _x: str(tmp_path / ".venv/bin/memo-mcp"))
    monkeypatch.setattr(cli_install_mcp, "_resolved_memo_mcp", lambda: tmp_path / ".venv/bin/memo-mcp")
    # Point HOME at an empty dir so the known isolated candidates don't exist.
    monkeypatch.setattr(cli_install_mcp.Path, "home", staticmethod(lambda: tmp_path))
    assert cli_install_mcp._resolve_isolated_memo_mcp() is None


def test_resolve_isolated_prefers_local_bin(monkeypatch, tmp_path):
    iso = tmp_path / ".local" / "bin" / "memo-mcp"
    iso.parent.mkdir(parents=True)
    iso.write_text("#!/bin/sh\n")
    monkeypatch.setattr(cli_install_mcp.Path, "home", staticmethod(lambda: tmp_path))
    assert cli_install_mcp._resolve_isolated_memo_mcp() == iso


def test_install_mcp_dry_run_reports_per_agent(monkeypatch, tmp_path):
    iso = tmp_path / ".local" / "bin" / "memo-mcp"
    iso.parent.mkdir(parents=True)
    iso.write_text("#!/bin/sh\n")
    monkeypatch.setattr(cli_install_mcp.Path, "home", staticmethod(lambda: tmp_path))

    res = CliRunner().invoke(cli, ["install-mcp", "--agent", "windsurf", "--agent", "codex"])
    assert res.exit_code == 0, res.output
    assert "memo MCP → " in res.output and "/.local/bin/memo-mcp" in res.output
    assert "windsurf" in res.output
    assert "codex mcp add memo" in res.output  # cli strategy argv shown
    assert "dry-run" in res.output


def test_install_mcp_with_mandate_targets_selected_agents(monkeypatch, tmp_path):
    iso = tmp_path / ".local" / "bin" / "memo-mcp"
    iso.parent.mkdir(parents=True)
    iso.write_text("#!/bin/sh\n")
    monkeypatch.setattr(cli_install_mcp.Path, "home", staticmethod(lambda: tmp_path))

    res = CliRunner().invoke(
        cli,
        ["install-mcp", "--agent", "devin", "--agent", "opencode", "--with-mandate"],
    )

    assert res.exit_code == 0, res.output
    assert "AGENTS.md" in res.output
    assert res.output.count("AGENTS.md") == 1


# ---------------------------------------------------------------------------
# Profile tests
# ---------------------------------------------------------------------------

def _make_iso(tmp_path: Path) -> Path:
    iso = tmp_path / ".local" / "bin" / "memo-mcp"
    iso.parent.mkdir(parents=True)
    iso.write_text("#!/bin/sh\n")
    return iso


def _captured_servers(monkeypatch, tmp_path: Path, cli_args: list[str]) -> list[Any]:
    """Run install-mcp and return the list of AgentMcpServer objects passed to register_agent_mcp."""
    from consciousness_contracts import register_agent_mcp as real_register
    from memo.runtime.mcp import _MCP_ENV_FORWARD_KEYS

    _make_iso(tmp_path)
    monkeypatch.setattr(cli_install_mcp.Path, "home", staticmethod(lambda: tmp_path))
    for key in _MCP_ENV_FORWARD_KEYS:
        monkeypatch.delenv(key, raising=False)

    captured: list[Any] = []

    def fake_register(agent, server, *, write=False, preset=None):
        captured.append(server)
        return {"ok": True, "agent": agent, "action": "dry-run", "strategy": "cli", "argv": ["memo", "mcp", "add", "memo", "--", str(server.command)]}

    # Patch at the import site inside install_mcp (it imports inside the function).
    import consciousness_contracts as cc
    monkeypatch.setattr(cc, "register_agent_mcp", fake_register)

    res = CliRunner().invoke(cli, ["install-mcp"] + cli_args)
    assert res.exit_code == 0, res.output
    return captured


def test_explicit_profile_core_injects_env(monkeypatch, tmp_path):
    """--profile core → MEMO_MCP_PROFILE=core in server env."""
    captured = _captured_servers(monkeypatch, tmp_path, ["--agent", "claude-code", "--profile", "core"])
    assert len(captured) == 1
    assert captured[0].env.get("MEMO_MCP_PROFILE") == "core"


def test_auto_profile_codex(monkeypatch, tmp_path):
    """No --profile, agent=codex → MEMO_MCP_PROFILE=core (auto)."""
    captured = _captured_servers(monkeypatch, tmp_path, ["--agent", "codex"])
    assert len(captured) == 1
    assert captured[0].env.get("MEMO_MCP_PROFILE") == "core"


def test_auto_profile_opencode(monkeypatch, tmp_path):
    """No --profile, agent=opencode → MEMO_MCP_PROFILE=core (auto)."""
    captured = _captured_servers(monkeypatch, tmp_path, ["--agent", "opencode"])
    assert len(captured) == 1
    assert captured[0].env.get("MEMO_MCP_PROFILE") == "core"


def test_no_auto_profile_claude_code(monkeypatch, tmp_path):
    """No --profile, agent=claude-code → MEMO_MCP_PROFILE stays at base 'agent' (not 'core')."""
    captured = _captured_servers(monkeypatch, tmp_path, ["--agent", "claude-code"])
    assert len(captured) == 1
    assert captured[0].env.get("MEMO_MCP_PROFILE") == "agent"


def test_explicit_profile_default_no_injection(monkeypatch, tmp_path):
    """--profile default → MEMO_MCP_PROFILE stays at base 'agent' (default = no override to core)."""
    captured = _captured_servers(monkeypatch, tmp_path, ["--agent", "codex", "--profile", "default"])
    assert len(captured) == 1
    assert captured[0].env.get("MEMO_MCP_PROFILE") == "agent"


def test_profile_annotation_in_header_output(monkeypatch, tmp_path):
    """--profile core → output header shows [profile: core]."""
    _make_iso(tmp_path)
    monkeypatch.setattr(cli_install_mcp.Path, "home", staticmethod(lambda: tmp_path))

    res = CliRunner().invoke(cli, ["install-mcp", "--agent", "claude-code", "--profile", "core"])
    assert res.exit_code == 0, res.output
    assert "[profile: core]" in res.output


def test_profile_annotation_omitted_for_default(monkeypatch, tmp_path):
    """--profile default → output header does NOT show profile annotation."""
    _make_iso(tmp_path)
    monkeypatch.setattr(cli_install_mcp.Path, "home", staticmethod(lambda: tmp_path))

    res = CliRunner().invoke(cli, ["install-mcp", "--agent", "claude-code", "--profile", "default"])
    assert res.exit_code == 0, res.output
    assert "[profile:" not in res.output
