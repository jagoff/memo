"""Markdown-backed ``memo config`` CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from rich.console import Console

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


def test_config_path_prints_markdown_and_legacy_paths_without_wrapping(tmp_path: Path) -> None:
    runner = CliRunner()

    with patch("memo.cli_config.console", Console(width=20, force_terminal=False)):
        result = runner.invoke(cli, ["config", "path"], env=_env(tmp_path))

    assert result.exit_code == 0, result.output
    assert "memo-config.md" in result.output
    assert "legacy.toml" in result.output
    assert "legacy.\ntoml" not in result.output


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


def test_config_set_rolls_back_when_validation_fails(tmp_path: Path) -> None:
    """An invalid value must not stay live on disk (daemons/MCP read the
    markdown config the moment it is written)."""
    from memo.config_md import ConfigProblem

    runner = CliRunner()
    env = _env(tmp_path)
    assert runner.invoke(cli, ["config", "init"], env=env).exit_code == 0
    assert runner.invoke(cli, ["config", "set", "recall.top_k", "9"], env=env).exit_code == 0

    with patch(
        "memo.config_md.validate_markdown_config",
        return_value=[ConfigProblem(file="x.md", key="recall.top_k", value="3", error="boom")],
    ):
        result = runner.invoke(cli, ["config", "set", "recall.top_k", "3"], env=env)
    assert result.exit_code != 0
    assert "boom" in result.output

    # The prior value is restored, not the rejected one.
    show = runner.invoke(cli, ["config", "show", "--effective"], env=env)
    assert show.exit_code == 0, show.output
    assert "9" in show.output
    assert "3" not in show.output.split("MEMO_RECALL_TOP_K")[-1].splitlines()[0]


def test_config_set_and_unset_reject_runtime_only_keys(tmp_path: Path) -> None:
    runner = CliRunner()
    env = _env(tmp_path)

    for command in (
        ["config", "set", "misc.noninteractive", "on"],
        ["config", "unset", "misc.noninteractive"],
    ):
        result = runner.invoke(cli, command, env=env)

        assert result.exit_code != 0
        assert "runtime-only" in result.output


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


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("MEMO_MCP_TRANSPORT", "htpp"),
        ("MEMO_MCP_PROFILE", "typo"),
        ("MEMO_MCP_PORT", "0"),
        ("MEMO_MCP_PORT", "70000"),
    ],
)
def test_config_validate_rejects_invalid_mcp_values(
    tmp_path: Path,
    name: str,
    value: str,
) -> None:
    result = CliRunner().invoke(
        cli,
        ["config", "validate"],
        env={**_env(tmp_path), name: value},
    )

    assert result.exit_code == 1
    assert name in result.output


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


def test_bare_config_launches_tui_on_tty(tmp_path: Path) -> None:
    runner = CliRunner()
    env = {**_env(tmp_path), "MEMO_NONINTERACTIVE": ""}
    with (
        patch("memo.cli_config._terminal_is_interactive", return_value=True),
        patch("memo.tui.config.run_config_tui", return_value=0) as run,
    ):
        result = runner.invoke(cli, ["config"], env=env)

    assert result.exit_code == 0, result.output
    run.assert_called_once()


def test_bare_config_prints_help_without_tty(tmp_path: Path) -> None:
    runner = CliRunner()
    with patch("memo.tui.config.run_config_tui") as run:
        result = runner.invoke(cli, ["config"], env=_env(tmp_path))

    assert result.exit_code == 0
    assert "show" in result.output
    run.assert_not_called()


def test_noninteractive_blocks_tui_even_with_tty(tmp_path: Path) -> None:
    runner = CliRunner()
    with (
        patch("memo.cli_config._terminal_is_interactive", return_value=True),
        patch("memo.tui.config.run_config_tui") as run,
    ):
        result = runner.invoke(cli, ["config"], env=_env(tmp_path))

    assert result.exit_code == 0
    assert "show" in result.output
    run.assert_not_called()


def test_config_flags_reports_the_effective_value_and_its_source(tmp_path, monkeypatch):
    """`config flags` must answer "what is in effect", not "what is in the env".

    The `active` column was `active_flags()` — env vars only — so every flag a
    user had pinned through `memo config set` (the *recommended* channel, the
    only one that reaches daemons and hooks) rendered blank, which reads as OFF.
    That is the exact misreading the Markdown config exists to prevent.
    """
    from click.testing import CliRunner

    from memo.cli_config import config_group

    cfg_dir = tmp_path / "memo-home"
    (cfg_dir / "config").mkdir(parents=True)
    (cfg_dir / "config" / "graph-config.md").write_text(
        '# Graph config\n\n```toml\n[graph]\nreason_enabled = "on"\n```\n',
        encoding="utf-8",
    )

    env = {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_CONFIG_DIR": str(cfg_dir),
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }
    for key in ("MEMO_GRAPH_REASON_ENABLED",):
        monkeypatch.delenv(key, raising=False)

    result = CliRunner().invoke(config_group, ["flags", "--group", "graph", "--json"], env=env)
    assert result.exit_code == 0, result.output
    rows = {r["flag"]: r for r in json.loads(result.output)}

    row = rows["MEMO_GRAPH_REASON_ENABLED"]
    assert row["effective"] is True, "a flag pinned in Markdown config reads as OFF"
    assert row["source"] == "config"


def test_config_flags_active_filter_includes_markdown_config(tmp_path, monkeypatch):
    """`--active` means "explicitly configured", not "exported in this shell"."""
    from click.testing import CliRunner

    from memo.cli_config import config_group

    cfg_dir = tmp_path / "memo-home"
    (cfg_dir / "config").mkdir(parents=True)
    (cfg_dir / "config" / "graph-config.md").write_text(
        '# Graph config\n\n```toml\n[graph]\nreason_enabled = "on"\n```\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("MEMO_GRAPH_REASON_ENABLED", raising=False)

    result = CliRunner().invoke(
        config_group,
        ["flags", "--group", "graph", "--active", "--json"],
        env={
            "MEMO_NONINTERACTIVE": "1",
            "MEMO_CONFIG_DIR": str(cfg_dir),
            "MEMO_DATA_DIR": str(tmp_path / "data"),
            "MEMO_STATE_DIR": str(tmp_path / "state"),
        },
    )
    assert result.exit_code == 0, result.output
    names = {r["flag"] for r in json.loads(result.output)}
    assert "MEMO_GRAPH_REASON_ENABLED" in names
