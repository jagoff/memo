from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

import memo.cli_runtime as cli_mod
import memo.runtime.install as install_mod
import memo.runtime.mcp as mcp_mod
import memo.runtime.shims as shims_mod
from memo.cli import cli


@pytest.fixture(autouse=True)
def _sandbox_home(monkeypatch, tmp_path_factory):
    """Keep install-slash side effects off the developer's machine.

    Non-dry-run `install-slash` writes startup-banner shims (+ a PATH snippet
    into ~/.zshrc) and AGENTS.md mandates at cwd. Point every HOME-derived
    path at a per-test sandbox: `_DEFAULT_BIN_DIR` is a module constant bound
    at import time, so patching HOME alone is not enough. Mandate writes are
    redirected into the sandbox (their cwd is the real checkout when pytest
    runs from the repo root); dry runs stay untouched so output asserts hold.
    """
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(shims_mod, "_DEFAULT_BIN_DIR", home / ".memo" / "bin")

    real_write = install_mod.write_mandates_for_clients

    def _sandboxed_write(*args, **kwargs):
        if not kwargs.get("dry_run"):
            kwargs["cwd"] = home
        return real_write(*args, **kwargs)

    monkeypatch.setattr(install_mod, "write_mandates_for_clients", _sandboxed_write)
    return home


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

    report = cli_mod._runtime_install_report(
        cwd=tmp_path / "repo", package_file=root / "lib" / "python" / "memo" / "detect.py"
    )

    assert report["mode"] == "pipx"
    assert report["root"] == str(root)
    assert report["warnings"] == []


def test_runtime_report_warns_when_pythonpath_shadows_isolated_install(monkeypatch, tmp_path):
    root = tmp_path / ".local" / "share" / "uv" / "tools" / "mlx-memo"
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True)
    for name in ("memo", "memo-mcp", "python"):
        (bin_dir / name).touch()

    source_file = tmp_path / "repo" / "src" / "memo" / "runtime" / "detect.py"
    source_file.parent.mkdir(parents=True)
    source_file.touch()

    monkeypatch.setattr(cli_mod.shutil, "which", lambda name: str(bin_dir / name))
    monkeypatch.setattr(sys, "executable", str(bin_dir / "python"))
    report = cli_mod._runtime_install_report(cwd=tmp_path / "repo", package_file=source_file)

    assert report["mode"] == "uv tool"
    assert report["package_path"] == str(source_file)
    assert any("outside the isolated runtime" in warning for warning in report["warnings"])
    assert any("PYTHONPATH" in warning for warning in report["warnings"])


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

    report = cli_mod._runtime_install_report(
        cwd=tmp_path / "repo",
        package_file=active_root / "lib" / "python" / "memo" / "detect.py",
    )

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
    assert '"MEMO_MCP_PROFILE":"agent"' in result.output


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


def test_mcp_server_env_clears_pythonpath(monkeypatch):
    _clear_memo_env(monkeypatch)
    monkeypatch.setenv("PYTHONPATH", "src")
    monkeypatch.setattr(mcp_mod, "_actual_embedder_config", lambda: {})

    env = mcp_mod._mcp_server_env()

    assert env["PYTHONPATH"] == ""


def test_mcp_server_env_forwards_custom_markdown_config_dir(monkeypatch, tmp_path):
    _clear_memo_env(monkeypatch)
    config_dir = tmp_path / "memo-config"
    monkeypatch.setenv("MEMO_CONFIG_DIR", str(config_dir))
    monkeypatch.setattr(mcp_mod, "_actual_embedder_config", lambda: {})

    env = mcp_mod._mcp_server_env()

    assert env["MEMO_CONFIG_DIR"] == str(config_dir)


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
    assert "--env MEMO_MCP_PROFILE=agent" in result.output
    assert "--env MEMO_SOURCE=codex" in result.output
    assert result.output.rstrip().endswith("/opt/test-pipx/venvs/mlx-memo/bin/memo-mcp")


def test_mcp_command_forwards_explicit_full_profile(monkeypatch):
    _clear_memo_env(monkeypatch)
    monkeypatch.setenv("MEMO_MCP_PROFILE", "full")
    monkeypatch.setattr(
        install_mod,
        "_resolved_memo_mcp",
        lambda: Path("/opt/test-pipx/venvs/mlx-memo/bin/memo-mcp"),
    )

    result = CliRunner().invoke(cli, ["mcp-command", "--client", "codex"])

    assert result.exit_code == 0
    assert "--env MEMO_MCP_PROFILE=full" in result.output
    assert "--env MEMO_SOURCE=codex" in result.output


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
    assert "-e MEMO_MCP_PROFILE=agent" in result.output
    assert "MEMO_SOURCE=devin" in result.output
    assert result.output.rstrip().endswith("/opt/test-pipx/venvs/mlx-memo/bin/memo-mcp")


def test_mcp_command_devin_desktop(monkeypatch):
    _clear_memo_env(monkeypatch)
    monkeypatch.setattr(
        install_mod,
        "_resolved_memo_mcp",
        lambda: Path("/opt/test-pipx/venvs/mlx-memo/bin/memo-mcp"),
    )

    result = CliRunner().invoke(cli, ["mcp-command", "--client", "devin-desktop"])

    assert result.exit_code == 0
    assert '"memo"' in result.output
    assert '"command": "/opt/test-pipx/venvs/mlx-memo/bin/memo-mcp"' in result.output
    assert '"MEMO_NONINTERACTIVE": "1"' in result.output
    assert '"MEMO_SOURCE": "devin-desktop"' in result.output
    assert '"type": "stdio"' in result.output


def test_mcp_command_forwards_model_env(monkeypatch):
    _clear_memo_env(monkeypatch)
    # Model/dims come from the live index via _actual_embedder_config(), never
    # from env — stub it so the test doesn't depend on this machine's memvec.db.
    monkeypatch.setattr(
        mcp_mod,
        "_actual_embedder_config",
        lambda: {
            "MEMO_EMBEDDER_MODEL": "mlx-community/Qwen3-Embedding-4B-4bit-DWQ",
            "MEMO_EMBEDDER_DIMS": "2560",
        },
    )
    monkeypatch.setattr(
        install_mod,
        "_resolved_memo_mcp",
        lambda: Path("/opt/test-pipx/venvs/mlx-memo/bin/memo-mcp"),
    )

    result = CliRunner().invoke(cli, ["mcp-command", "--client", "codex"])

    assert result.exit_code == 0
    assert "--env MEMO_EMBEDDER_MODEL=mlx-community/Qwen3-Embedding-4B-4bit-DWQ" in result.output
    assert "--env MEMO_EMBEDDER_DIMS=2560" in result.output


def test_mcp_command_forwards_st_revision(monkeypatch):
    _clear_memo_env(monkeypatch)
    monkeypatch.setenv("MEMO_EMBEDDER_BACKEND", "st")
    monkeypatch.setenv("MEMO_ST_EMBEDDER_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    revision = "d" * 40
    monkeypatch.setenv("MEMO_ST_EMBEDDER_REVISION", revision)
    monkeypatch.setattr(mcp_mod, "_actual_embedder_config", lambda: {})
    monkeypatch.setattr(
        install_mod,
        "_resolved_memo_mcp",
        lambda: Path("/opt/test-pipx/venvs/mlx-memo/bin/memo-mcp"),
    )

    result = CliRunner().invoke(cli, ["mcp-command", "--client", "codex"])

    assert result.exit_code == 0
    assert "--env MEMO_EMBEDDER_BACKEND=st" in result.output
    assert "--env MEMO_ST_EMBEDDER_MODEL=Qwen/Qwen3-Embedding-0.6B" in result.output
    assert f"--env MEMO_ST_EMBEDDER_REVISION={revision}" in result.output


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
    assert "-e MEMO_MCP_PROFILE=agent" in result.output
    assert "MEMO_SOURCE=devin" in result.output
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


def _fake_run_factory(calls: list[list[str]], remove_stderr: str):
    """Subprocess stub: a `mcp remove` exits non-zero (no prior entry on a clean
    machine), everything else succeeds. Records every argv for assertions."""

    def fake_run(args, check, capture_output, text, **kwargs):
        argv = [str(a) for a in args]
        calls.append(argv)
        if "remove" in argv:
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr=remove_stderr)
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    return fake_run


def test_install_slash_claude_proceeds_to_add_when_remove_fails(monkeypatch):
    """Clean machine: `claude mcp remove` fails (no prior entry) but the client
    must still run `mcp add` and report as configured, not skipped."""
    _clear_memo_env(monkeypatch)
    repo = Path.cwd()
    monkeypatch.setattr(
        install_mod,
        "_resolved_memo_mcp",
        lambda: Path("/opt/test-pipx/venvs/mlx-memo/bin/memo-mcp"),
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(
        mcp_mod.subprocess,
        "run",
        _fake_run_factory(calls, 'No MCP server named "memo" in user scope'),
    )

    runner = CliRunner()
    with runner.isolated_filesystem():
        # invoked WITHOUT --best-effort: this is the default fresh-install path
        result = runner.invoke(
            cli, ["install-slash", "--client", "claude-code", "--repo", str(repo)]
        )

    assert result.exit_code == 0, result.output
    assert any("remove" in argv for argv in calls)
    assert any("add-json" in argv for argv in calls)
    assert "skipped clients" not in result.output
    assert "agent-client install complete" in result.output


def test_install_slash_devin_proceeds_to_add_when_remove_fails(monkeypatch, tmp_path):
    """Clean machine: `devin mcp remove` fails with Devin's distinct message but
    the client must still run `mcp add` and report as configured, not skipped."""
    _clear_memo_env(monkeypatch)
    repo = Path.cwd()
    monkeypatch.setattr(install_mod.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(
        install_mod,
        "_resolved_memo_mcp",
        lambda: Path("/opt/test-pipx/venvs/mlx-memo/bin/memo-mcp"),
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(
        mcp_mod.subprocess,
        "run",
        _fake_run_factory(
            calls,
            "Error: Server 'memo' is not in the user config. It is configured by Claude.",
        ),
    )

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["install-slash", "--client", "devin", "--repo", str(repo)])

    assert result.exit_code == 0, result.output
    assert any("remove" in argv for argv in calls)
    assert any("add" in argv and "remove" not in argv for argv in calls)
    assert "skipped clients" not in result.output
    assert "agent-client install complete" in result.output


def test_install_slash_devin_desktop_writes_mcp_config(monkeypatch, tmp_path):
    _clear_memo_env(monkeypatch)
    cfg_path = tmp_path / "mcp_config.json"
    cfg_path.write_text(
        '{"mcpServers":{"existing":{"command":"node","args":["server.js"]}}}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("DEVIN_DESKTOP_MCP_CONFIG", str(cfg_path))
    # MEMO_EMBEDDER_DIMS is deliberately NOT forwarded from env — it comes from
    # the live index via _actual_embedder_config(). Stub it so the assertion
    # doesn't depend on whatever memvec.db this machine happens to have.
    monkeypatch.setattr(mcp_mod, "_actual_embedder_config", lambda: {"MEMO_EMBEDDER_DIMS": "2560"})
    monkeypatch.setattr(
        install_mod,
        "_resolved_memo_mcp",
        lambda: Path("/opt/test-pipx/venvs/mlx-memo/bin/memo-mcp"),
    )

    result = CliRunner().invoke(cli, ["install-slash", "--client", "devin-desktop"])

    assert result.exit_code == 0
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert data["mcpServers"]["existing"]["command"] == "node"
    memo = data["mcpServers"]["memo"]
    assert memo["command"] == "/opt/test-pipx/venvs/mlx-memo/bin/memo-mcp"
    assert memo["args"] == []
    assert memo["env"]["MEMO_NONINTERACTIVE"] == "1"
    assert memo["env"]["MEMO_EMBEDDER_DIMS"] == "2560"
    assert memo["env"]["MEMO_SOURCE"] == "devin-desktop"
    assert memo["type"] == "stdio"
    assert "Startup-banner shims" not in result.output


def test_install_slash_default_installs_same_shims_as_explicit_all(monkeypatch):
    """No --client defaults to all clients, so it must install the same shim
    set as an explicit `--client all` (gemini/blackbox were silently skipped)."""
    _clear_memo_env(monkeypatch)
    monkeypatch.setenv("CODEX_HOME", "/tmp/codex-home")
    monkeypatch.setattr(
        install_mod,
        "_resolved_memo_mcp",
        lambda: Path("/opt/test-pipx/venvs/mlx-memo/bin/memo-mcp"),
    )
    shim_calls: list[tuple[str, ...]] = []

    def _spy_install_shims(agents, bin_dir, *, dry_run=False):
        shim_calls.append(tuple(agents))
        return [f"wrote:{bin_dir}/{a}" for a in agents]

    monkeypatch.setattr(shims_mod, "install_shims", _spy_install_shims)

    repo = str(Path.cwd())
    r_default = CliRunner().invoke(cli, ["install-slash", "--dry-run", "--repo", repo])
    r_all = CliRunner().invoke(
        cli, ["install-slash", "--client", "all", "--dry-run", "--repo", repo]
    )

    assert r_default.exit_code == 0, r_default.output
    assert r_all.exit_code == 0, r_all.output
    assert shim_calls == [
        ("codex", "devin", "opencode", "gemini", "blackbox"),
        ("codex", "devin", "opencode", "gemini", "blackbox"),
    ]


def test_install_watcher_rejects_autodetected_project_venv(monkeypatch, tmp_path):
    """An auto-detected `memo` inside a project .venv must be refused: a
    KeepAlive=true plist baked with it crash-loops launchd once the venv is
    removed. Explicit --bin stays untouched."""
    import shutil as _shutil

    monkeypatch.setattr(sys, "platform", "darwin")
    venv_memo = tmp_path / "proj" / ".venv" / "bin" / "memo"
    venv_memo.parent.mkdir(parents=True)
    venv_memo.touch()
    monkeypatch.setattr(_shutil, "which", lambda name: str(venv_memo))

    result = CliRunner().invoke(cli, ["install-watcher"])

    assert result.exit_code != 0
    assert "project venv" in result.output
    assert "--bin" in result.output


def test_install_watcher_explicit_bin_is_untouched(monkeypatch, tmp_path, _sandbox_home):
    """Explicit --bin is the user's call — even a venv path is baked as-is
    (--no-load keeps launchctl out of the test)."""
    monkeypatch.setattr(sys, "platform", "darwin")
    venv_memo = tmp_path / "proj" / ".venv" / "bin" / "memo"
    venv_memo.parent.mkdir(parents=True)
    venv_memo.touch()

    result = CliRunner().invoke(cli, ["install-watcher", "--bin", str(venv_memo), "--no-load"])

    assert result.exit_code == 0, result.output
    plist = _sandbox_home / "Library" / "LaunchAgents" / "com.memo.watch.plist"
    assert plist.is_file()
    assert str(venv_memo) in plist.read_text(encoding="utf-8")


def test_codex_plugin_manifest_does_not_duplicate_user_skill() -> None:
    manifest = json.loads(
        Path("plugins/memo/.codex-plugin/plugin.json").read_text(encoding="utf-8")
    )

    assert "skills" not in manifest


def test_mcp_command_opencode(monkeypatch):
    _clear_memo_env(monkeypatch)
    monkeypatch.setattr(mcp_mod, "_actual_embedder_config", lambda: {"MEMO_EMBEDDER_DIMS": "2560"})
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
    monkeypatch.delenv("DEVIN_DESKTOP_MCP_CONFIG", raising=False)
