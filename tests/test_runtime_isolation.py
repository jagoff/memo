from __future__ import annotations

import sys
from pathlib import Path

from click.testing import CliRunner

import memo.cli as cli_mod
from memo.cli import cli


def test_runtime_report_accepts_pipx_install(monkeypatch, tmp_path):
    root = tmp_path / ".local" / "pipx" / "venvs" / "mlx-memo"
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True)

    def fake_which(name: str) -> str:
        return str(bin_dir / name)

    monkeypatch.setattr(cli_mod.shutil, "which", fake_which)
    monkeypatch.setattr(sys, "executable", str(bin_dir / "python"))

    report = cli_mod._runtime_install_report(cwd=tmp_path / "repo")

    assert report["mode"] == "pipx"
    assert report["root"] == str(root)
    assert report["warnings"] == []


def test_runtime_report_warns_for_project_venv(monkeypatch, tmp_path):
    root = tmp_path / "rag" / ".venv"
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True)

    def fake_which(name: str) -> str:
        return str(bin_dir / name)

    monkeypatch.setattr(cli_mod.shutil, "which", fake_which)
    monkeypatch.setattr(sys, "executable", str(bin_dir / "python"))

    report = cli_mod._runtime_install_report(cwd=tmp_path / "rag")

    assert report["mode"] == "venv"
    assert any("project venv" in warning for warning in report["warnings"])


def test_mcp_command_pins_resolved_memo_mcp(monkeypatch):
    monkeypatch.setattr(
        cli_mod,
        "_resolved_memo_mcp",
        lambda: Path("/Users/USER/.local/pipx/venvs/mlx-memo/bin/memo-mcp"),
    )

    result = CliRunner().invoke(cli, ["mcp-command"])

    assert result.exit_code == 0
    assert (
        "claude mcp add memo -s user "
        "/Users/USER/.local/pipx/venvs/mlx-memo/bin/memo-mcp"
    ) in result.output


def test_mcp_command_json(monkeypatch):
    monkeypatch.setattr(
        cli_mod,
        "_resolved_memo_mcp",
        lambda: Path("/Users/USER/.local/pipx/venvs/mlx-memo/bin/memo-mcp"),
    )

    result = CliRunner().invoke(cli, ["mcp-command", "--client", "json"])

    assert result.exit_code == 0
    assert '"memo"' in result.output
    assert '"command": "/Users/USER/.local/pipx/venvs/mlx-memo/bin/memo-mcp"' in result.output
