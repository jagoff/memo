"""`memo init` + first-run gate — picker mocked, asserts config file written."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from memo.cli import cli
from memo.setup.picker import PickerResult


@pytest.fixture
def runner_env(tmp_path: Path) -> dict[str, str]:
    """CliRunner env: isolated config + non-interactive guard for the gate.

    `MEMO_NONINTERACTIVE=1` ensures the *gate* doesn't fire during tests
    that aren't testing the gate itself. Tests that ARE testing the gate
    selectively unset it via `runner.invoke(..., env={**env, "MEMO_NONINTERACTIVE": ""})`.
    """
    return {
        "MEMO_CONFIG_FILE": str(tmp_path / "memo-config.toml"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),  # avoid touching ~/Documents/memo
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }


def test_init_writes_config_file(tmp_path: Path, runner_env):
    runner = CliRunner()
    target_data = tmp_path / "chosen"
    fake_result = PickerResult(data_dir=target_data, vault_path=None)
    with patch("memo.cli.run_picker", return_value=fake_result):
        result = runner.invoke(cli, ["init", "--force"], env=runner_env)
    assert result.exit_code == 0, result.output
    cfg_path = Path(runner_env["MEMO_CONFIG_FILE"])
    assert cfg_path.is_file()
    body = cfg_path.read_text(encoding="utf-8")
    assert str(target_data) in body
    assert target_data.is_dir()


def test_init_writes_vault_path_when_obsidian_branch(tmp_path: Path, runner_env):
    runner = CliRunner()
    vault = tmp_path / "Notes"
    vault.mkdir()
    data_dir = vault / "AI" / "memory"
    fake = PickerResult(data_dir=data_dir, vault_path=vault)
    with patch("memo.cli.run_picker", return_value=fake):
        result = runner.invoke(cli, ["init", "--force"], env=runner_env)
    assert result.exit_code == 0, result.output
    body = Path(runner_env["MEMO_CONFIG_FILE"]).read_text(encoding="utf-8")
    assert f'data_dir = "{data_dir}"' in body
    assert f'vault_path = "{vault}"' in body


def test_init_prompts_to_overwrite_existing(tmp_path: Path, runner_env):
    """Without --force, init asks before overwriting."""
    cfg_file = Path(runner_env["MEMO_CONFIG_FILE"])
    cfg_file.parent.mkdir(parents=True, exist_ok=True)
    cfg_file.write_text('[storage]\ndata_dir = "/tmp/old"\n', encoding="utf-8")
    runner = CliRunner()
    fake = PickerResult(data_dir=tmp_path / "new", vault_path=None)
    with patch("memo.cli.run_picker", return_value=fake):
        # Decline the overwrite.
        result = runner.invoke(cli, ["init"], input="n\n", env=runner_env)
    assert result.exit_code == 0
    # Old config preserved.
    assert "/tmp/old" in cfg_file.read_text(encoding="utf-8")


def test_doctor_doesnt_trigger_picker(tmp_path: Path):
    """Doctor must always run, even on fresh installs without a config file."""
    runner = CliRunner()
    env = {
        "MEMO_CONFIG_FILE": str(tmp_path / "missing.toml"),
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        # Note: no MEMO_NONINTERACTIVE — doctor is on the skip-list.
    }
    with patch("memo.cli.run_picker") as picker_mock:
        result = runner.invoke(cli, ["doctor"], env=env)
    picker_mock.assert_not_called()
    # Doctor itself may report some checks failed (no MLX on test box,
    # etc.) but should not crash trying to fire the picker.
    assert "data_dir" in result.output


def test_gate_skips_when_noninteractive_env_set(tmp_path: Path, runner_env):
    """The gate must never block hooks (which set MEMO_NONINTERACTIVE=1)."""
    runner = CliRunner()
    with patch("memo.cli.run_picker") as picker_mock:
        # `stats` is a non-skip-listed command; would fire the picker
        # on first-run in interactive mode.
        runner.invoke(cli, ["stats"], env=runner_env)
    picker_mock.assert_not_called()
