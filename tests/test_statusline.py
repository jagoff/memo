"""Tests for `memo install-statusline` chain-aware wiring + the bundled script.

The systemic guarantee under test: installing the memo statusline makes the
``[MEMO <ver>]`` badge appear on ANY machine, *coexisting* with whatever
statusline was already configured (caveman, memflow, a hand-rolled one) instead
of silently skipping it. See ``src/memo/cli_statusline.py``.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from click.testing import CliRunner

from memo.cli import cli


def _env(tmp_path: Path) -> dict:
    # Point CLAUDE_CONFIG_DIR at a temp dir so the real ~/.claude is never touched.
    return {
        "MEMO_NONINTERACTIVE": "1",
        "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
    }


def _settings(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "claude" / "settings.json").read_text())


# ── install: fresh machine (no prior statusLine) ──────────────────────────────


def test_install_fresh_sets_standalone_command(tmp_path):
    result = CliRunner().invoke(cli, ["install-statusline"], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    cmd = _settings(tmp_path)["statusLine"]["command"]
    dest = tmp_path / "claude" / "memo-statusline.sh"
    assert cmd == f'bash "{dest}"'
    assert "--wrap" not in cmd


def test_install_writes_executable_script(tmp_path):
    CliRunner().invoke(cli, ["install-statusline"], env=_env(tmp_path))
    dest = tmp_path / "claude" / "memo-statusline.sh"
    assert dest.is_file()
    import stat as _stat

    assert dest.stat().st_mode & _stat.S_IEXEC


# ── install: foreign statusLine present → WRAP it, do not skip ─────────────────


def test_install_wraps_foreign_statusline(tmp_path):
    claude = tmp_path / "claude"
    claude.mkdir(parents=True)
    foreign = 'bash "/Users/x/.claude/hooks/caveman-statusline.sh"'
    (claude / "settings.json").write_text(
        json.dumps({"statusLine": {"type": "command", "command": foreign}})
    )
    result = CliRunner().invoke(cli, ["install-statusline"], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    cmd = _settings(tmp_path)["statusLine"]["command"]
    dest = claude / "memo-statusline.sh"
    # memo wrapper drives, foreign command preserved as the --wrap target
    assert f'bash "{dest}" --wrap ' in cmd
    assert "caveman-statusline.sh" in cmd


def test_install_wrap_is_idempotent(tmp_path):
    claude = tmp_path / "claude"
    claude.mkdir(parents=True)
    foreign = 'bash "/Users/x/.claude/hooks/caveman-statusline.sh"'
    (claude / "settings.json").write_text(
        json.dumps({"statusLine": {"type": "command", "command": foreign}})
    )
    env = _env(tmp_path)
    CliRunner().invoke(cli, ["install-statusline"], env=env)
    CliRunner().invoke(cli, ["install-statusline"], env=env)
    cmd = _settings(tmp_path)["statusLine"]["command"]
    # exactly one wrap layer — no memo-around-memo recursion
    assert cmd.count("--wrap") == 1
    assert cmd.count("caveman-statusline.sh") == 1


def test_install_does_not_wrap_own_standalone(tmp_path):
    env = _env(tmp_path)
    CliRunner().invoke(cli, ["install-statusline"], env=env)  # standalone
    CliRunner().invoke(cli, ["install-statusline"], env=env)  # re-run
    cmd = _settings(tmp_path)["statusLine"]["command"]
    assert "--wrap" not in cmd


def test_force_drops_wrap_to_standalone(tmp_path):
    claude = tmp_path / "claude"
    claude.mkdir(parents=True)
    foreign = 'bash "/Users/x/.claude/hooks/caveman-statusline.sh"'
    (claude / "settings.json").write_text(
        json.dumps({"statusLine": {"type": "command", "command": foreign}})
    )
    result = CliRunner().invoke(cli, ["install-statusline", "--force"], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    cmd = _settings(tmp_path)["statusLine"]["command"]
    dest = claude / "memo-statusline.sh"
    assert cmd == f'bash "{dest}"'
    assert "caveman" not in cmd


def test_install_preserves_other_settings_keys(tmp_path):
    claude = tmp_path / "claude"
    claude.mkdir(parents=True)
    (claude / "settings.json").write_text(json.dumps({"model": "opus", "env": {"A": "1"}}))
    CliRunner().invoke(cli, ["install-statusline"], env=_env(tmp_path))
    s = _settings(tmp_path)
    assert s["model"] == "opus"
    assert s["env"] == {"A": "1"}
    assert "statusLine" in s


# ── the bundled script: wrap mode prepends the MEMO badge to inner output ──────


def _bundled_script() -> Path:
    from memo.cli_statusline import _bundled_statusline

    return _bundled_statusline()


def test_script_wrap_prepends_memo_badge(tmp_path):
    script = _bundled_script()
    assert script.is_file()
    # Force a deterministic version via the .memo-version fallback file.
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    (cfg / ".memo-version").write_text("9.9.9")
    env = {**os.environ, "CLAUDE_CONFIG_DIR": str(cfg)}
    proc = subprocess.run(
        ["bash", str(script), "--wrap", "echo INNER_LINE"],
        input='{"workspace":{"current_dir":"/tmp"},"model":{"display_name":"Opus"}}',
        capture_output=True,
        text=True,
        env=env,
    )
    out = proc.stdout
    assert "INNER_LINE" in out
    assert "[MEMO" in out
    # badge precedes the inner output
    assert out.index("[MEMO") < out.index("INNER_LINE")


def test_script_standalone_still_emits_badge(tmp_path):
    script = _bundled_script()
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    (cfg / ".memo-version").write_text("9.9.9")
    env = {**os.environ, "CLAUDE_CONFIG_DIR": str(cfg)}
    proc = subprocess.run(
        ["bash", str(script)],
        input='{"workspace":{"current_dir":"/tmp"},"model":{"display_name":"Opus"}}',
        capture_output=True,
        text=True,
        env=env,
    )
    assert "[MEMO" in proc.stdout
    assert "Opus" in proc.stdout
