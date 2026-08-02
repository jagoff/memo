"""Read-only CLI diagnostics for disabled legacy terminal coordination."""

from __future__ import annotations

import json

from click.testing import CliRunner

from memo.cli import cli


def _env(tmp_cfg) -> dict[str, str]:
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_cfg.data_dir),
        "MEMO_STATE_DIR": str(tmp_cfg.state_dir),
    }


def test_terminal_command_exposes_only_read_only_diagnostics(tmp_cfg) -> None:
    result = CliRunner().invoke(
        cli,
        ["terminal", "--help"],
        env=_env(tmp_cfg),
    )

    assert result.exit_code == 0, result.output
    assert all(command in result.output for command in ("list", "history"))
    assert all(f"\n  {command}" not in result.output for command in ("register", "send", "enter"))


def test_terminal_list_and_history_are_empty_by_default(tmp_cfg) -> None:
    runner = CliRunner()

    listed = runner.invoke(cli, ["terminal", "list", "--json"], env=_env(tmp_cfg))
    history = runner.invoke(cli, ["terminal", "history", "--json"], env=_env(tmp_cfg))

    assert listed.exit_code == 0, listed.output
    assert history.exit_code == 0, history.output
    assert json.loads(listed.output) == []
    assert json.loads(history.output) == []
