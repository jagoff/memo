"""Tests for `memo install-mcp` (contract-backed MCP installer wrapper)."""

from __future__ import annotations

from pathlib import Path

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
