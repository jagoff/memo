"""Markdown-backed ``memo config`` CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from memo.cli import cli


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "MEMO_CONFIG_DIR": str(tmp_path / "memo-home"),
        "MEMO_CONFIG_FILE": str(tmp_path / "legacy.toml"),
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_NONINTERACTIVE": "1",
    }


def test_config_init_creates_markdown_files(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(cli, ["config", "init"], env=_env(tmp_path))

    assert result.exit_code == 0, result.output
    assert (tmp_path / "memo-home" / "memo-config.md").is_file()
    assert (tmp_path / "memo-home" / "config" / "storage-config.md").is_file()
    assert "created" in result.output.lower()


def test_config_path_prints_markdown_and_legacy_paths(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(cli, ["config", "path"], env=_env(tmp_path))

    assert result.exit_code == 0, result.output
    assert "memo-config.md" in result.output
    assert "legacy.toml" in result.output


def test_config_set_show_and_unset(tmp_path: Path) -> None:
    runner = CliRunner()
    env = _env(tmp_path)
    assert runner.invoke(cli, ["config", "init"], env=env).exit_code == 0

    set_result = runner.invoke(cli, ["config", "set", "recall.top_k", "9"], env=env)
    assert set_result.exit_code == 0, set_result.output

    show_result = runner.invoke(cli, ["config", "show", "--effective"], env=env)
    assert show_result.exit_code == 0, show_result.output
    assert "MEMO_RECALL_TOP_K" in show_result.output
    assert "9" in show_result.output

    unset_result = runner.invoke(cli, ["config", "unset", "recall.top_k"], env=env)
    assert unset_result.exit_code == 0, unset_result.output


def test_config_show_effective_uses_environment_over_markdown(tmp_path: Path) -> None:
    runner = CliRunner()
    env = {**_env(tmp_path), "MEMO_RECALL_TOP_K": "2"}
    assert runner.invoke(cli, ["config", "init"], env=env).exit_code == 0

    result = runner.invoke(cli, ["config", "show", "--effective", "--json"], env=env)

    assert result.exit_code == 0, result.output
    rows = {row["key"]: row for row in json.loads(result.output)}
    assert rows["MEMO_RECALL_TOP_K"] == {
        "key": "MEMO_RECALL_TOP_K",
        "value": "2",
        "source": "env",
        "env": "MEMO_RECALL_TOP_K",
    }


def test_config_migrate_reads_legacy_toml(tmp_path: Path) -> None:
    legacy = tmp_path / "config.toml"
    legacy.write_text(f'[storage]\ndata_dir = "{tmp_path / "legacy-data"}"\n', encoding="utf-8")
    runner = CliRunner()
    env = {**_env(tmp_path), "MEMO_CONFIG_FILE": str(legacy)}

    result = runner.invoke(cli, ["config", "migrate"], env=env)

    assert result.exit_code == 0, result.output
    assert legacy.with_suffix(".toml.pre-md-config.bak").is_file()
    body = (tmp_path / "memo-home" / "config" / "storage-config.md").read_text(encoding="utf-8")
    assert str(tmp_path / "legacy-data") in body
