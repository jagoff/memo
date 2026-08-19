"""Tests for `memo install-statusline` chain-aware wiring + the bundled script.

The systemic guarantee under test: installing the memo statusline makes the
``[Memo <ver>]`` badge appear on ANY machine, *coexisting* with whatever
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


# ── the bundled script: wrap mode prepends the Memo badge to inner output ──────


def _bundled_script() -> Path:
    from memo.cli_statusline import _bundled_statusline

    return _bundled_statusline()


def test_script_wrap_prepends_memo_badge(tmp_path):
    script = _bundled_script()
    assert script.is_file()
    # Force a deterministic version via the .memo-version fallback file.
    home = tmp_path / "home"
    home.mkdir()
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    (cfg / ".memo-version").write_text("9.9.9")
    # Pin activity off: ambient presence (recalls/saves) must never alter the
    # exact ``[Memo <ver>]`` bracket this test asserts.
    env = {
        **os.environ,
        "HOME": str(home),
        "CLAUDE_CONFIG_DIR": str(cfg),
        "MEMO_STATUSLINE_ACTIVITY": "0",
    }
    proc = subprocess.run(
        ["bash", str(script), "--wrap", "echo INNER_LINE"],
        input='{"workspace":{"current_dir":"/tmp"},"model":{"display_name":"Opus"}}',
        capture_output=True,
        text=True,
        env=env,
    )
    out = proc.stdout
    assert "INNER_LINE" in out
    assert "[Memo 9.9.9]" in out
    assert "[MEMO " not in out
    # badge precedes the inner output
    assert out.index("[Memo") < out.index("INNER_LINE")


def test_script_wrap_does_not_duplicate_existing_legacy_memo_badge(tmp_path):
    script = _bundled_script()
    home = tmp_path / "home"
    home.mkdir()
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    (cfg / ".memo-version").write_text("9.9.9")
    env = {**os.environ, "HOME": str(home), "CLAUDE_CONFIG_DIR": str(cfg)}
    proc = subprocess.run(
        ["bash", str(script), "--wrap", "echo '[MEMO 1.2.3] INNER_LINE'"],
        input='{"workspace":{"current_dir":"/tmp"},"model":{"display_name":"Opus"}}',
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.stdout == "[MEMO 1.2.3] INNER_LINE"


def test_script_standalone_still_emits_badge(tmp_path):
    script = _bundled_script()
    home = tmp_path / "home"
    home.mkdir()
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    (cfg / ".memo-version").write_text("9.9.9")
    # Pin activity off: ambient presence (recalls/saves) must never alter the
    # exact ``[Memo <ver>]`` bracket this test asserts.
    env = {
        **os.environ,
        "HOME": str(home),
        "CLAUDE_CONFIG_DIR": str(cfg),
        "MEMO_STATUSLINE_ACTIVITY": "0",
    }
    proc = subprocess.run(
        ["bash", str(script)],
        input='{"workspace":{"current_dir":"/tmp"},"model":{"display_name":"Opus"}}',
        capture_output=True,
        text=True,
        env=env,
    )
    assert "[Memo 9.9.9]" in proc.stdout
    assert "[MEMO " not in proc.stdout
    assert "Opus" in proc.stdout


# ── activity badge (presence_today.json) ─────────────────────────────────────


def _run_statusline(input_json: dict, env: dict) -> str:
    script = _bundled_script()
    merged = {**os.environ, **env}
    proc = subprocess.run(
        ["bash", str(script)],
        input=json.dumps(input_json),
        capture_output=True,
        text=True,
        env=merged,
    )
    return proc.stdout


def test_activity_badge_from_presence_file(tmp_path) -> None:
    from datetime import date as _date

    (tmp_path / ".memo-version").write_text("9.9.9", encoding="utf-8")
    (tmp_path / "presence_today.json").write_text(
        json.dumps({"date": _date.today().isoformat(), "recalls": 12, "saves": 3}),
        encoding="utf-8",
    )
    out = _run_statusline(
        {"model": {"display_name": "X"}},
        env={"MEMO_STATE_DIR": str(tmp_path), "CLAUDE_CONFIG_DIR": str(tmp_path)},
    )
    assert "🧠12" in out
    assert "💾3" in out


def test_activity_badge_skips_stale_date(tmp_path) -> None:
    (tmp_path / ".memo-version").write_text("9.9.9", encoding="utf-8")
    (tmp_path / "presence_today.json").write_text(
        json.dumps({"date": "2020-01-01", "recalls": 12, "saves": 3}),
        encoding="utf-8",
    )
    out = _run_statusline(
        {"model": {"display_name": "X"}},
        env={"MEMO_STATE_DIR": str(tmp_path), "CLAUDE_CONFIG_DIR": str(tmp_path)},
    )
    assert "🧠" not in out


def test_activity_badge_all_zero_counters_plain_badge(tmp_path) -> None:
    """All-zero counters (recalls=0, saves=0) leave the badge as plain
    [Memo <ver>] with no 🧠/💾/tok segment — the "tok" segment (a hardcoded
    tokens_saved estimate) was removed outright, not just at zero."""
    from datetime import date as _date

    (tmp_path / ".memo-version").write_text("9.9.9", encoding="utf-8")
    (tmp_path / "presence_today.json").write_text(
        json.dumps({"date": _date.today().isoformat(), "recalls": 0, "saves": 0}),
        encoding="utf-8",
    )
    out = _run_statusline(
        {"model": {"display_name": "X"}},
        env={"MEMO_STATE_DIR": str(tmp_path), "CLAUDE_CONFIG_DIR": str(tmp_path)},
    )
    assert "[Memo " in out  # badge present in some form
    assert "🧠" not in out
    assert "💾" not in out
    assert "tok" not in out


def test_activity_badge_never_renders_a_tok_segment(tmp_path) -> None:
    """Round-2: even with real recalls/saves activity, no 'tok' segment
    renders — the presence tokens_saved counter (grounded*350 + consults*200)
    was removed outright."""
    from datetime import date as _date

    (tmp_path / ".memo-version").write_text("9.9.9", encoding="utf-8")
    (tmp_path / "presence_today.json").write_text(
        json.dumps({"date": _date.today().isoformat(), "recalls": 12, "saves": 3}),
        encoding="utf-8",
    )
    out = _run_statusline(
        {"model": {"display_name": "X"}},
        env={"MEMO_STATE_DIR": str(tmp_path), "CLAUDE_CONFIG_DIR": str(tmp_path)},
    )
    assert "tok" not in out


def test_activity_badge_disabled_by_env(tmp_path) -> None:
    from datetime import date as _date

    (tmp_path / ".memo-version").write_text("9.9.9", encoding="utf-8")
    (tmp_path / "presence_today.json").write_text(
        json.dumps({"date": _date.today().isoformat(), "recalls": 12, "saves": 0}),
        encoding="utf-8",
    )
    out = _run_statusline(
        {"model": {"display_name": "X"}},
        env={
            "MEMO_STATE_DIR": str(tmp_path),
            "CLAUDE_CONFIG_DIR": str(tmp_path),
            "MEMO_STATUSLINE_ACTIVITY": "0",
        },
    )
    assert "🧠" not in out
