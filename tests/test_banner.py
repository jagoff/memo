"""Tests for Memo-owned startup banners and agent shims."""

from __future__ import annotations

import base64
import importlib.metadata
import json
import os
import pty
import stat
import subprocess

from click.testing import CliRunner

from memo.cli import cli
from memo.runtime.shims import install_path_snippet, install_shims


def _env(tmp_cfg) -> dict[str, str]:
    home = tmp_cfg.state_dir / "home"
    home.mkdir(parents=True, exist_ok=True)
    return {
        "HOME": str(home),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_cfg.data_dir),
        "MEMO_STATE_DIR": str(tmp_cfg.state_dir),
    }


def test_startup_banner_reports_memo_version_and_agent(tmp_cfg) -> None:
    result = CliRunner().invoke(
        cli,
        ["startup-banner", "--agent", "opencode"],
        env=_env(tmp_cfg),
    )

    assert result.exit_code == 0
    assert "[Memo " in result.output
    assert "opencode" in result.output


def test_update_command_is_channel_aware(monkeypatch) -> None:
    from memo import cli_banner

    monkeypatch.setattr("memo.runtime.detect.is_homebrew_install", lambda: True)
    assert cli_banner._update_command() == "brew upgrade mlx-memo"

    monkeypatch.setattr("memo.runtime.detect.is_homebrew_install", lambda: False)
    assert cli_banner._update_command() == "memo update"


def test_startup_banner_offer_uses_brew_on_homebrew(tmp_cfg, monkeypatch) -> None:
    from memo import cli_banner

    monkeypatch.setattr(cli_banner, "_pending_update_tag", lambda: "v9.9.9")
    monkeypatch.setattr("memo.runtime.detect.is_homebrew_install", lambda: True)

    result = CliRunner().invoke(cli, ["startup-banner", "--agent", "codex"], env=_env(tmp_cfg))

    assert result.exit_code == 0, result.output
    assert "v9.9.9 available" in result.output
    assert "brew upgrade mlx-memo" in result.output


def test_install_shims_writes_executable_self_contained_script(tmp_cfg, tmp_path) -> None:
    bin_dir = tmp_path / "bin"
    result = CliRunner().invoke(
        cli,
        ["install-shims", "--agents", "codex,opencode", "--bin-dir", str(bin_dir)],
        env=_env(tmp_cfg),
    )

    assert result.exit_code == 0, result.output
    assert (bin_dir / "codex").stat().st_mode & stat.S_IEXEC
    content = (bin_dir / "codex").read_text(encoding="utf-8")
    assert "# memo-shim" in content
    assert 'startup-banner --agent "$_AGENT"' in content
    assert 'codex-badge --agent "$_AGENT"' in content
    assert "memflow" not in content.lower()
    assert "synapse" not in content.lower()


def test_shim_skips_itself_via_symlinked_path_entry(tmp_path) -> None:
    shim_dir = tmp_path / "shim"
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    real = real_dir / "codex"
    real.write_text("#!/usr/bin/env bash\necho real-codex-ran\n", encoding="utf-8")
    real.chmod(0o755)
    install_shims(("codex",), shim_dir)
    link_dir = tmp_path / "link"
    link_dir.symlink_to(shim_dir)

    proc = subprocess.run(
        [str(link_dir / "codex")],
        env={
            **os.environ,
            "PATH": os.pathsep.join([str(link_dir), str(real_dir), os.environ.get("PATH", "")]),
            "MEMO_STARTUP_BANNER": "0",
            "MEMO_CODEX_BADGE": "0",
        },
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert proc.returncode == 0, proc.stderr
    assert "real-codex-ran" in proc.stdout


def test_codex_shim_prints_startup_banner_once_for_nested_call(tmp_path) -> None:
    shim_dir = tmp_path / "shim"
    tools_dir = tmp_path / "tools"
    real_dir = tmp_path / "real"
    log_path = tmp_path / "memo-calls.log"
    tools_dir.mkdir()
    real_dir.mkdir()
    memo = tools_dir / "memo"
    memo.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$MEMO_TEST_LOG"\n',
        encoding="utf-8",
    )
    real = real_dir / "codex"
    real.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'if [ "${1:-}" = "child" ]; then exit 0; fi\n'
        "codex child\n",
        encoding="utf-8",
    )
    memo.chmod(0o755)
    real.chmod(0o755)
    install_shims(("codex",), shim_dir)
    master_fd, slave_fd = pty.openpty()
    try:
        proc = subprocess.run(
            [str(shim_dir / "codex")],
            env={
                **os.environ,
                "PATH": os.pathsep.join(
                    [str(shim_dir), str(tools_dir), str(real_dir), os.environ.get("PATH", "")]
                ),
                "MEMO_TEST_LOG": str(log_path),
                "MEMO_CODEX_BADGE": "0",
                "MEMO_STARTUP_BANNER_SHOWN": "0",
            },
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=slave_fd,
            text=True,
            timeout=5,
        )
    finally:
        os.close(slave_fd)
        os.close(master_fd)

    assert proc.returncode == 0
    assert log_path.read_text(encoding="utf-8").splitlines() == ["startup-banner --agent codex"]


def test_codex_badge_uses_memo_notify_protocol(tmp_cfg, tmp_path, monkeypatch) -> None:
    tty = tmp_path / "tty"
    tty.touch()
    monkeypatch.setenv("MEMO_AGENT_TTY", str(tty))
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "9.8.7" if name == "mlx-memo" else "0",
    )

    result = CliRunner().invoke(
        cli,
        ["codex-badge", "--agent", "codex"],
        env=_env(tmp_cfg),
    )

    assert result.exit_code == 0
    raw = tty.read_bytes()
    title = base64.b64encode(b"[Memo 9.8.7]").decode("ascii")
    assert raw.startswith(b"\x1b]3008;start=codex;kind=notify;")
    assert f"title={title}".encode() in raw
    assert raw.endswith(b"\x1b\\")


def _opencode_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    path = tmp_path / "config" / "opencode"
    path.mkdir(parents=True)
    return path


def test_refresh_opencode_username_is_idempotent_and_preserves_keys(tmp_path, monkeypatch) -> None:
    from memo.cli_banner import _opencode_config_path, _refresh_opencode_username

    opencode_dir = _opencode_dir(tmp_path, monkeypatch)
    (opencode_dir / "opencode.json").write_text(
        json.dumps({"lsp": True, "username": "fer · [Memo 1.0.0]"}),
        encoding="utf-8",
    )

    assert _refresh_opencode_username("4.0.0") == "fer · [Memo 4.0.0]"
    before = _opencode_config_path().read_text(encoding="utf-8")
    assert _refresh_opencode_username("4.0.0") is None
    assert _opencode_config_path().read_text(encoding="utf-8") == before
    assert json.loads(before)["lsp"] is True


def test_install_shims_no_clobber_and_dry_run(tmp_cfg, tmp_path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    existing = bin_dir / "codex"
    existing.write_text('#!/bin/bash\nexec real-codex "$@"\n', encoding="utf-8")

    result = CliRunner().invoke(
        cli,
        ["install-shims", "--agents", "codex", "--bin-dir", str(bin_dir)],
        env=_env(tmp_cfg),
    )
    dry_dir = tmp_path / "dry"
    dry = CliRunner().invoke(
        cli,
        ["install-shims", "--dry-run", "--agents", "codex", "--bin-dir", str(dry_dir)],
        env=_env(tmp_cfg),
    )

    assert result.exit_code == 0
    assert "skip" in result.output
    assert "real-codex" in existing.read_text(encoding="utf-8")
    assert dry.exit_code == 0
    assert not dry_dir.exists()


def test_install_path_snippet_is_memo_owned_and_idempotent(monkeypatch, tmp_path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("SHELL", "/bin/zsh")
    memo_bin = home / ".memo" / "bin"

    first = install_path_snippet(memo_bin)
    second = install_path_snippet(memo_bin)
    text = (home / ".zshrc").read_text(encoding="utf-8")

    assert first.startswith("written:")
    assert second == "already"
    assert f'export PATH="{memo_bin}:$PATH"' in text
    assert "memflow" not in text.lower()
    assert "synapse" not in text.lower()
