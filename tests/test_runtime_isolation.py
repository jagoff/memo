from __future__ import annotations

import json
import sys
from pathlib import Path

from click.testing import CliRunner

import memo.cli_runtime as cli_mod
import memo.runtime.install as install_mod
from memo.cli import cli


def test_runtime_report_accepts_pipx_install(monkeypatch, tmp_path):
    root = tmp_path / ".local" / "pipx" / "venvs" / "mlx-memo"
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True)
    for name in ("memo", "memo-mcp", "python"):
        (bin_dir / name).touch()

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
    for name in ("memo", "memo-mcp", "python"):
        (bin_dir / name).touch()

    def fake_which(name: str) -> str:
        return str(bin_dir / name)

    monkeypatch.setattr(cli_mod.shutil, "which", fake_which)
    monkeypatch.setattr(sys, "executable", str(bin_dir / "python"))

    report = cli_mod._runtime_install_report(cwd=tmp_path / "rag")

    assert report["mode"] == "venv"
    assert any("project venv" in warning for warning in report["warnings"])


def test_runtime_report_prefers_invoked_memo_over_path(monkeypatch, tmp_path):
    active_root = tmp_path / "active" / "pipx" / "venvs" / "mlx-memo"
    active_bin = active_root / "bin"
    active_bin.mkdir(parents=True)
    for name in ("memo", "memo-mcp", "python"):
        (active_bin / name).touch()

    path_root = tmp_path / "path" / "pipx" / "venvs" / "mlx-memo"
    path_bin = path_root / "bin"
    path_bin.mkdir(parents=True)
    for name in ("memo", "memo-mcp"):
        (path_bin / name).touch()

    monkeypatch.setattr(sys, "argv", [str(active_bin / "memo"), "doctor"])
    monkeypatch.setattr(sys, "executable", str(active_bin / "python"))
    monkeypatch.setattr(cli_mod.shutil, "which", lambda name: str(path_bin / name))

    report = cli_mod._runtime_install_report(cwd=tmp_path / "repo")

    assert report["root"] == str(active_root)
    assert report["memo_resolved"] == str(active_bin / "memo")
    assert report["mcp_resolved"] == str(active_bin / "memo-mcp")
    assert report["warnings"] == []


def test_resolved_memo_mcp_prefers_invoked_memo_sibling(monkeypatch, tmp_path):
    active_root = tmp_path / "active" / "pipx" / "venvs" / "mlx-memo"
    active_bin = active_root / "bin"
    active_bin.mkdir(parents=True)
    (active_bin / "memo").touch()
    (active_bin / "memo-mcp").touch()

    path_root = tmp_path / "path" / "pipx" / "venvs" / "mlx-memo"
    path_bin = path_root / "bin"
    path_bin.mkdir(parents=True)
    (path_bin / "memo-mcp").touch()

    monkeypatch.setattr(sys, "argv", [str(active_bin / "memo"), "mcp-command"])
    monkeypatch.setattr(cli_mod.shutil, "which", lambda _name: str(path_bin / "memo-mcp"))

    assert cli_mod._resolved_memo_mcp() == active_bin / "memo-mcp"


def test_mcp_command_pins_resolved_memo_mcp(monkeypatch):
    _clear_memo_env(monkeypatch)
    monkeypatch.setattr(
        install_mod,
        "_resolved_memo_mcp",
        lambda: Path("/opt/test-pipx/venvs/mlx-memo/bin/memo-mcp"),
    )

    result = CliRunner().invoke(cli, ["mcp-command"])

    assert result.exit_code == 0
    assert "claude mcp add-json -s user memo" in result.output
    assert '"MEMO_NONINTERACTIVE":"1"' in result.output


def test_backup_group_keeps_portable_out_option():
    result = CliRunner().invoke(cli, ["backup", "--help"], env={"MEMO_NONINTERACTIVE": "1"})

    assert result.exit_code == 0
    assert "--out" in result.output
    assert "create" in result.output
    assert "restore" in result.output


def test_mcp_command_json(monkeypatch):
    _clear_memo_env(monkeypatch)
    monkeypatch.setattr(
        install_mod,
        "_resolved_memo_mcp",
        lambda: Path("/opt/test-pipx/venvs/mlx-memo/bin/memo-mcp"),
    )

    result = CliRunner().invoke(cli, ["mcp-command", "--client", "json"])

    assert result.exit_code == 0
    assert '"memo"' in result.output
    assert '"command": "/opt/test-pipx/venvs/mlx-memo/bin/memo-mcp"' in result.output
    assert '"MEMO_NONINTERACTIVE": "1"' in result.output


def test_mcp_command_codex(monkeypatch):
    _clear_memo_env(monkeypatch)
    monkeypatch.setattr(
        install_mod,
        "_resolved_memo_mcp",
        lambda: Path("/opt/test-pipx/venvs/mlx-memo/bin/memo-mcp"),
    )

    result = CliRunner().invoke(cli, ["mcp-command", "--client", "codex"])

    assert result.exit_code == 0
    assert "codex mcp add memo" in result.output
    assert "--env MEMO_NONINTERACTIVE=1" in result.output
    assert "--env MEMO_SOURCE=codex" in result.output
    assert "/opt/test-pipx/venvs/mlx-memo/bin/memo-mcp" in result.output


def test_mcp_command_devin(monkeypatch):
    _clear_memo_env(monkeypatch)
    monkeypatch.setattr(
        install_mod,
        "_resolved_memo_mcp",
        lambda: Path("/opt/test-pipx/venvs/mlx-memo/bin/memo-mcp"),
    )

    result = CliRunner().invoke(cli, ["mcp-command", "--client", "devin"])

    assert result.exit_code == 0
    assert "devin mcp add -s user" in result.output
    assert "-e MEMO_NONINTERACTIVE=1" in result.output
    assert "-e MEMO_SOURCE=devin" in result.output
    assert "/opt/test-pipx/venvs/mlx-memo/bin/memo-mcp" in result.output


def test_mcp_command_windsurf(monkeypatch):
    _clear_memo_env(monkeypatch)
    monkeypatch.setattr(
        install_mod,
        "_resolved_memo_mcp",
        lambda: Path("/opt/test-pipx/venvs/mlx-memo/bin/memo-mcp"),
    )

    result = CliRunner().invoke(cli, ["mcp-command", "--client", "windsurf"])

    assert result.exit_code == 0
    assert '"memo"' in result.output
    assert '"command": "/opt/test-pipx/venvs/mlx-memo/bin/memo-mcp"' in result.output
    assert '"MEMO_NONINTERACTIVE": "1"' in result.output
    assert '"type": "stdio"' not in result.output


def test_mcp_command_forwards_model_env(monkeypatch):
    _clear_memo_env(monkeypatch)
    monkeypatch.setenv("MEMO_EMBEDDER_MODEL", "mlx-community/Qwen3-Embedding-4B-4bit-DWQ")
    monkeypatch.setenv("MEMO_EMBEDDER_DIMS", "2560")
    monkeypatch.setattr(
        install_mod,
        "_resolved_memo_mcp",
        lambda: Path("/opt/test-pipx/venvs/mlx-memo/bin/memo-mcp"),
    )

    result = CliRunner().invoke(cli, ["mcp-command", "--client", "codex"])

    assert result.exit_code == 0
    assert "--env MEMO_EMBEDDER_MODEL=mlx-community/Qwen3-Embedding-4B-4bit-DWQ" in result.output
    assert "--env MEMO_EMBEDDER_DIMS=2560" in result.output


def test_install_slash_dry_run(monkeypatch):
    _clear_memo_env(monkeypatch)
    repo = Path.cwd()
    monkeypatch.setattr(
        install_mod,
        "_resolved_memo_mcp",
        lambda: Path("/opt/test-pipx/venvs/mlx-memo/bin/memo-mcp"),
    )

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli,
            ["install-slash", "--client", "devin", "--dry-run", "--repo", str(repo)],
        )

    assert result.exit_code == 0
    assert "copy" in result.output
    assert "devin mcp add -s user" in result.output
    assert "-e MEMO_NONINTERACTIVE=1" in result.output
    assert "-e MEMO_SOURCE=devin" in result.output
    assert "Mandate" in result.output
    assert "AGENTS.md" in result.output
    assert "would write" in result.output


def test_install_slash_opencode_dry_run(monkeypatch):
    _clear_memo_env(monkeypatch)
    monkeypatch.setattr(
        install_mod,
        "_resolved_memo_mcp",
        lambda: Path("/opt/test-pipx/venvs/mlx-memo/bin/memo-mcp"),
    )

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli,
            ["install-slash", "--client", "opencode", "--dry-run"],
        )

    assert result.exit_code == 0
    assert "opencode mcp add memo --env MEMO_NONINTERACTIVE=1 --" in result.output
    assert "Mandate" in result.output
    assert "AGENTS.md" in result.output
    assert "would write" in result.output


def test_install_slash_codex_installs_plugin_and_mcp(monkeypatch):
    _clear_memo_env(monkeypatch)
    monkeypatch.setenv("CODEX_HOME", "/tmp/codex-home")
    monkeypatch.setattr(
        install_mod,
        "_resolved_memo_mcp",
        lambda: Path("/opt/test-pipx/venvs/mlx-memo/bin/memo-mcp"),
    )

    result = CliRunner().invoke(
        cli,
        ["install-slash", "--client", "codex", "--dry-run", "--repo", str(Path.cwd())],
    )

    assert result.exit_code == 0
    assert "copy" in result.output
    assert "/tmp/codex-home/skills/memo/SKILL.md  # /memo" in result.output
    assert "codex app-server --listen stdio:// --enable plugins" in result.output
    assert "codex mcp remove memo" in result.output
    assert "codex mcp add memo --env MEMO_NONINTERACTIVE=1 --" in result.output
    assert "slash menu currently lists only built-in slash commands" in result.output


def test_install_slash_claude_uses_add_json(monkeypatch):
    _clear_memo_env(monkeypatch)
    monkeypatch.setattr(
        install_mod,
        "_resolved_memo_mcp",
        lambda: Path("/opt/test-pipx/venvs/mlx-memo/bin/memo-mcp"),
    )

    result = CliRunner().invoke(
        cli,
        ["install-slash", "--client", "claude-code", "--dry-run", "--repo", str(Path.cwd())],
    )

    assert result.exit_code == 0
    assert "claude mcp remove -s user memo" in result.output
    assert "claude mcp add-json -s user memo" in result.output
    assert '"MEMO_NONINTERACTIVE":"1"' in result.output


def test_install_slash_windsurf_writes_mcp_config(monkeypatch, tmp_path):
    _clear_memo_env(monkeypatch)
    cfg_path = tmp_path / "mcp_config.json"
    cfg_path.write_text(
        '{"mcpServers":{"existing":{"command":"node","args":["server.js"]}}}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("WINDSURF_MCP_CONFIG", str(cfg_path))
    monkeypatch.setenv("MEMO_EMBEDDER_DIMS", "2560")
    monkeypatch.setattr(
        install_mod,
        "_resolved_memo_mcp",
        lambda: Path("/opt/test-pipx/venvs/mlx-memo/bin/memo-mcp"),
    )

    result = CliRunner().invoke(cli, ["install-slash", "--client", "windsurf"])

    assert result.exit_code == 0
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert data["mcpServers"]["existing"]["command"] == "node"
    memo = data["mcpServers"]["memo"]
    assert memo["command"] == "/opt/test-pipx/venvs/mlx-memo/bin/memo-mcp"
    assert memo["args"] == []
    assert memo["env"]["MEMO_NONINTERACTIVE"] == "1"
    assert memo["env"]["MEMO_EMBEDDER_DIMS"] == "2560"
    assert "type" not in memo


def test_mcp_command_opencode(monkeypatch):
    _clear_memo_env(monkeypatch)
    monkeypatch.setenv("MEMO_EMBEDDER_DIMS", "2560")
    monkeypatch.setattr(
        install_mod,
        "_resolved_memo_mcp",
        lambda: Path("/opt/test-pipx/venvs/mlx-memo/bin/memo-mcp"),
    )

    result = CliRunner().invoke(cli, ["mcp-command", "--client", "opencode"])

    assert result.exit_code == 0
    assert "opencode mcp add memo" in result.output
    assert "--env MEMO_NONINTERACTIVE=1" in result.output
    assert "--env MEMO_SOURCE=opencode" in result.output
    assert "--env MEMO_EMBEDDER_DIMS=2560" in result.output


def _clear_memo_env(monkeypatch):
    for key in cli_mod._MCP_ENV_FORWARD_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("WINDSURF_MCP_CONFIG", raising=False)
