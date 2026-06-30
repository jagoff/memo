"""Tests for memo startup-banner and install-shims commands."""
from __future__ import annotations

from click.testing import CliRunner

from memo.cli import cli


def _env(tmp_cfg) -> dict:
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_cfg.data_dir),
        "MEMO_STATE_DIR": str(tmp_cfg.state_dir),
    }


def test_startup_banner_exits_zero(tmp_cfg):
    result = CliRunner().invoke(
        cli, ["startup-banner", "--agent", "codex"], env=_env(tmp_cfg)
    )
    assert result.exit_code == 0


def test_startup_banner_contains_memo_version(tmp_cfg):
    result = CliRunner().invoke(
        cli, ["startup-banner", "--agent", "opencode"], env=_env(tmp_cfg)
    )
    assert "[Memo " in result.output
    assert "[MEMO " not in result.output


def test_startup_banner_contains_agent_name(tmp_cfg):
    result = CliRunner().invoke(
        cli, ["startup-banner", "--agent", "blackbox"], env=_env(tmp_cfg)
    )
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
    assert "exec" in content


def test_install_shims_skips_non_memo_file(tmp_cfg, tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "codex").write_text("#!/bin/bash\nexec real-codex \"$@\"\n")
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
    (bin_dir / "codex").write_text("#!/bin/bash\n# memo-shim\nexec old \"$@\"\n")
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
