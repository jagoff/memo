"""Tests for memo startup-banner and install-shims commands."""

from __future__ import annotations

import base64
import importlib.metadata
import os
import pty
import subprocess

from click.testing import CliRunner

from memo.cli import cli
from memo.runtime.shims import install_path_snippet, install_shims


def _env(tmp_cfg) -> dict:
    home = tmp_cfg.state_dir / "home"
    home.mkdir(parents=True, exist_ok=True)
    return {
        "HOME": str(home),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_cfg.data_dir),
        "MEMO_STATE_DIR": str(tmp_cfg.state_dir),
    }


def test_startup_banner_exits_zero(tmp_cfg):
    result = CliRunner().invoke(cli, ["startup-banner", "--agent", "codex"], env=_env(tmp_cfg))
    assert result.exit_code == 0


def test_startup_banner_contains_memo_version(tmp_cfg):
    result = CliRunner().invoke(cli, ["startup-banner", "--agent", "opencode"], env=_env(tmp_cfg))
    assert "[Memo " in result.output
    assert "[MEMO " not in result.output


def test_startup_banner_contains_agent_name(tmp_cfg):
    result = CliRunner().invoke(cli, ["startup-banner", "--agent", "blackbox"], env=_env(tmp_cfg))
    assert "blackbox" in result.output


def test_install_shims_writes_executable_scripts(tmp_cfg, tmp_path):
    bin_dir = tmp_path / "bin"
    result = CliRunner().invoke(
        cli,
        ["install-shims", "--agents", "codex,opencode", "--bin-dir", str(bin_dir)],
        env=_env(tmp_cfg),
    )
    assert result.exit_code == 0, result.output
    assert (bin_dir / "codex").is_file()
    assert (bin_dir / "opencode").is_file()
    # Must be executable
    import stat as _stat

    mode = (bin_dir / "codex").stat().st_mode
    assert mode & _stat.S_IEXEC


def test_install_shims_contains_memo_shim_marker(tmp_cfg, tmp_path):
    bin_dir = tmp_path / "bin"
    CliRunner().invoke(
        cli, ["install-shims", "--agents", "codex", "--bin-dir", str(bin_dir)], env=_env(tmp_cfg)
    )
    content = (bin_dir / "codex").read_text()
    assert "memo-shim" in content
    assert "MEMFLOW_STARTUP_BANNER" in content
    assert "_MEMFLOW_BANNER_DEFAULT=0" in content
    assert "grep -qF" not in content
    assert 'startup-banner --agent "$_AGENT"' in content
    assert 'startup-banner --agent "$_AGENT" 2>/dev/null' not in content
    assert 'codex-badge --agent "$_AGENT"' in content
    assert "MEMO_CODEX_BADGE_DELAY:-1" in content
    assert "exec" in content


def test_shim_skips_itself_via_symlinked_path_entry(tmp_path):
    """Regression: a PATH entry that is a symlink to the shim dir defeats the
    string compare against `pwd -P`, so the shim exec'd ITSELF forever. The
    inode identity guard (-ef) must skip it and fall through to the real
    binary."""
    shim_dir = tmp_path / "shim"
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    (real_dir / "codex").write_text("#!/usr/bin/env bash\necho real-codex-ran\n", encoding="utf-8")
    (real_dir / "codex").chmod(0o755)
    install_shims(("codex",), shim_dir)
    link_dir = tmp_path / "link"
    link_dir.symlink_to(shim_dir)

    env = {
        **os.environ,
        # The symlinked entry comes first: its string differs from the shim's
        # resolved `pwd -P` dir, but it IS the same shim on disk.
        "PATH": os.pathsep.join([str(link_dir), str(real_dir), os.environ.get("PATH", "")]),
        "MEMO_STARTUP_BANNER": "0",
        "MEMO_CODEX_BADGE": "0",
    }
    proc = subprocess.run(
        [str(link_dir / "codex")],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=5,  # infinite exec recursion shows up as a timeout here
    )

    assert proc.returncode == 0, proc.stderr
    assert "real-codex-ran" in proc.stdout


def test_codex_shim_prints_startup_banner_once_for_nested_codex_call(tmp_path):
    shim_dir = tmp_path / "shim"
    tools_dir = tmp_path / "tools"
    real_dir = tmp_path / "real"
    log_path = tmp_path / "memo-calls.log"
    tools_dir.mkdir()
    real_dir.mkdir()

    (tools_dir / "memo").write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$MEMO_TEST_LOG"\n',
        encoding="utf-8",
    )
    (real_dir / "codex").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'if [ "${1:-}" = "child" ]; then exit 0; fi\n'
        "codex child\n",
        encoding="utf-8",
    )
    (tools_dir / "memo").chmod(0o755)
    (real_dir / "codex").chmod(0o755)
    install_shims(("codex",), shim_dir)

    env = {
        **os.environ,
        "PATH": os.pathsep.join(
            [str(shim_dir), str(tools_dir), str(real_dir), os.environ.get("PATH", "")]
        ),
        "MEMO_TEST_LOG": str(log_path),
        "MEMO_CODEX_BADGE": "0",
        "MEMO_STARTUP_BANNER_SHOWN": "0",
        "MEMO_CODEX_BADGE_SHOWN": "0",
    }
    master_fd, slave_fd = pty.openpty()
    try:
        proc = subprocess.run(
            [str(shim_dir / "codex")],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=slave_fd,
            text=True,
            timeout=5,
        )
    finally:
        os.close(slave_fd)
        os.close(master_fd)

    assert proc.returncode == 0, proc.stderr
    assert log_path.read_text(encoding="utf-8").splitlines() == ["startup-banner --agent codex"]


def test_codex_shim_prints_startup_banner_before_memflow_shim_by_default(tmp_path):
    shim_dir = tmp_path / "shim"
    tools_dir = tmp_path / "tools"
    memflow_dir = tmp_path / ".memflow" / "bin"
    memo_log = tmp_path / "memo-calls.log"
    memflow_log = tmp_path / "memflow-ran.log"
    tools_dir.mkdir()
    memflow_dir.mkdir(parents=True)

    (tools_dir / "memo").write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$MEMO_TEST_LOG"\n',
        encoding="utf-8",
    )
    (memflow_dir / "codex").write_text(
        '#!/usr/bin/env bash\nprintf "memflow-ran\\n" >> "$MEMFLOW_TEST_LOG"\n',
        encoding="utf-8",
    )
    (tools_dir / "memo").chmod(0o755)
    (memflow_dir / "codex").chmod(0o755)
    install_shims(("codex",), shim_dir)

    env = {
        **os.environ,
        "PATH": os.pathsep.join(
            [str(shim_dir), str(tools_dir), str(memflow_dir), os.environ.get("PATH", "")]
        ),
        "MEMO_TEST_LOG": str(memo_log),
        "MEMFLOW_TEST_LOG": str(memflow_log),
        "MEMO_CODEX_BADGE": "0",
        "MEMO_STARTUP_BANNER_SHOWN": "0",
        "MEMO_CODEX_BADGE_SHOWN": "0",
    }
    master_fd, slave_fd = pty.openpty()
    try:
        proc = subprocess.run(
            [str(shim_dir / "codex")],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=slave_fd,
            text=True,
            timeout=5,
        )
    finally:
        os.close(slave_fd)
        os.close(master_fd)

    assert proc.returncode == 0, proc.stderr
    assert memflow_log.read_text(encoding="utf-8").splitlines() == ["memflow-ran"]
    assert memo_log.read_text(encoding="utf-8").splitlines() == ["startup-banner --agent codex"]


def test_codex_shim_delegates_startup_banner_to_memflow_shim_when_enabled(tmp_path):
    shim_dir = tmp_path / "shim"
    tools_dir = tmp_path / "tools"
    memflow_dir = tmp_path / ".memflow" / "bin"
    memo_log = tmp_path / "memo-calls.log"
    memflow_log = tmp_path / "memflow-ran.log"
    tools_dir.mkdir()
    memflow_dir.mkdir(parents=True)

    (tools_dir / "memo").write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$MEMO_TEST_LOG"\n',
        encoding="utf-8",
    )
    (memflow_dir / "codex").write_text(
        '#!/usr/bin/env bash\nprintf "memflow-ran\\n" >> "$MEMFLOW_TEST_LOG"\n',
        encoding="utf-8",
    )
    (tools_dir / "memo").chmod(0o755)
    (memflow_dir / "codex").chmod(0o755)
    install_shims(("codex",), shim_dir)

    env = {
        **os.environ,
        "PATH": os.pathsep.join(
            [str(shim_dir), str(tools_dir), str(memflow_dir), os.environ.get("PATH", "")]
        ),
        "MEMO_TEST_LOG": str(memo_log),
        "MEMFLOW_TEST_LOG": str(memflow_log),
        "MEMFLOW_STARTUP_BANNER": "1",
        "MEMO_CODEX_BADGE": "0",
        "MEMO_STARTUP_BANNER_SHOWN": "0",
        "MEMO_CODEX_BADGE_SHOWN": "0",
    }
    master_fd, slave_fd = pty.openpty()
    try:
        proc = subprocess.run(
            [str(shim_dir / "codex")],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=slave_fd,
            text=True,
            timeout=5,
        )
    finally:
        os.close(slave_fd)
        os.close(master_fd)

    assert proc.returncode == 0, proc.stderr
    assert memflow_log.read_text(encoding="utf-8").splitlines() == ["memflow-ran"]
    assert not memo_log.exists()


def test_codex_badge_uses_memo_version_notify_protocol(tmp_cfg, tmp_path, monkeypatch):
    tty = tmp_path / "tty"
    tty.touch()
    monkeypatch.setenv("MEMO_AGENT_TTY", str(tty))
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "9.8.7" if name == "mlx-memo" else "0",
    )

    result = CliRunner().invoke(cli, ["codex-badge", "--agent", "codex"], env=_env(tmp_cfg))

    assert result.exit_code == 0, result.output
    raw = tty.read_bytes()
    title = base64.b64encode(b"[Memo 9.8.7]").decode("ascii")
    assert raw.startswith(b"\x1b]3008;start=codex;kind=notify;")
    assert f"title={title}".encode("ascii") in raw
    assert b"body=" in raw
    assert raw.endswith(b"\x1b\\")


def _opencode_dir(tmp_path, monkeypatch):
    """Point XDG_CONFIG_HOME at a tmp dir and create an opencode/ config dir."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    oc = tmp_path / "config" / "opencode"
    oc.mkdir(parents=True)
    return oc


def test_refresh_opencode_username_stamps_version_badge(tmp_path, monkeypatch):
    import getpass

    from memo.cli_banner import _refresh_opencode_username

    _opencode_dir(tmp_path, monkeypatch)
    monkeypatch.setattr(getpass, "getuser", lambda: "fer")

    result = _refresh_opencode_username("2.4.3")

    assert result == "fer · [Memo 2.4.3]"
    import json

    data = json.loads((tmp_path / "config" / "opencode" / "opencode.json").read_text())
    assert data["username"] == "fer · [Memo 2.4.3]"


def test_refresh_opencode_username_is_idempotent(tmp_path, monkeypatch):
    import getpass

    from memo.cli_banner import _opencode_config_path, _refresh_opencode_username

    _opencode_dir(tmp_path, monkeypatch)
    monkeypatch.setattr(getpass, "getuser", lambda: "fer")

    assert _refresh_opencode_username("2.4.3") == "fer · [Memo 2.4.3]"
    before = _opencode_config_path().read_text()
    # Second call with same version is a no-op (no rewrite).
    assert _refresh_opencode_username("2.4.3") is None
    assert _opencode_config_path().read_text() == before


def test_refresh_opencode_username_restamps_old_badge_and_preserves_keys(tmp_path, monkeypatch):
    import json

    from memo.cli_banner import _refresh_opencode_username

    oc = _opencode_dir(tmp_path, monkeypatch)
    (oc / "opencode.json").write_text(
        json.dumps({"$schema": "x", "lsp": True, "username": "bob · [Memo 1.0.0]"})
    )

    result = _refresh_opencode_username("9.9.9")

    assert result == "bob · [Memo 9.9.9]"
    data = json.loads((oc / "opencode.json").read_text())
    assert data["username"] == "bob · [Memo 9.9.9]"
    # Non-conflicting keys survive the rewrite.
    assert data["lsp"] is True
    assert data["$schema"] == "x"


def test_refresh_opencode_username_noop_when_opencode_absent(tmp_path, monkeypatch):
    from memo.cli_banner import _refresh_opencode_username

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))  # no opencode/ dir

    assert _refresh_opencode_username("2.4.3") is None
    assert not (tmp_path / "config" / "opencode").exists()


def test_startup_banner_opencode_writes_username(tmp_cfg, tmp_path, monkeypatch):
    import json

    _opencode_dir(tmp_path, monkeypatch)
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "3.2.1" if name == "mlx-memo" else "0",
    )

    result = CliRunner().invoke(cli, ["startup-banner", "--agent", "opencode"], env=_env(tmp_cfg))

    assert result.exit_code == 0, result.output
    data = json.loads((tmp_path / "config" / "opencode" / "opencode.json").read_text())
    assert data["username"].endswith("[Memo 3.2.1]")


def test_startup_banner_codex_does_not_touch_opencode_config(tmp_cfg, tmp_path, monkeypatch):
    _opencode_dir(tmp_path, monkeypatch)

    CliRunner().invoke(cli, ["startup-banner", "--agent", "codex"], env=_env(tmp_cfg))

    # Only opencode refreshes the username; other agents leave it absent.
    assert not (tmp_path / "config" / "opencode" / "opencode.json").exists()


def test_install_shims_skips_non_memo_file(tmp_cfg, tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "codex").write_text('#!/bin/bash\nexec real-codex "$@"\n')
    result = CliRunner().invoke(
        cli, ["install-shims", "--agents", "codex", "--bin-dir", str(bin_dir)], env=_env(tmp_cfg)
    )
    assert result.exit_code == 0
    assert "skip" in result.output
    # File must be unchanged
    assert "real-codex" in (bin_dir / "codex").read_text()


def test_install_shims_overwrites_own_shim(tmp_cfg, tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # Write a stale memo shim
    (bin_dir / "codex").write_text('#!/bin/bash\n# memo-shim\nexec old "$@"\n')
    result = CliRunner().invoke(
        cli, ["install-shims", "--agents", "codex", "--bin-dir", str(bin_dir)], env=_env(tmp_cfg)
    )
    assert result.exit_code == 0
    assert "wrote" in result.output
    assert "old" not in (bin_dir / "codex").read_text()


def test_install_shims_dry_run_writes_nothing(tmp_cfg, tmp_path):
    bin_dir = tmp_path / "bin"
    result = CliRunner().invoke(
        cli,
        ["install-shims", "--dry-run", "--bin-dir", str(bin_dir), "--agents", "codex"],
        env=_env(tmp_cfg),
    )
    assert result.exit_code == 0
    assert not bin_dir.exists() or not any(bin_dir.iterdir())


def test_install_path_snippet_keeps_memo_before_downstream_wrappers(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    rc = home / ".zshrc"
    rc.write_text(
        'export PATH="$HOME/.memflow/bin:$PATH"\n'
        "# memo-shims PATH\n"
        'export PATH="/old/memo/bin:$PATH"  # memo-shims PATH\n',
        encoding="utf-8",
    )
    memo_bin = home / ".memo" / "bin"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("SHELL", "/bin/zsh")

    status = install_path_snippet(memo_bin)

    assert status.startswith(("written:", "upgraded:"))
    text = rc.read_text(encoding="utf-8")
    assert str(memo_bin) in text
    assert text.rindex(str(memo_bin)) > text.rindex(".memflow/bin")
