"""Main Textual configuration center behavior."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import Input, Switch

from memo.tui.config.app import ConfigApp
from memo.tui.config.session import ConfigSession, ValueSource
from memo.tui.config.widgets import SettingRow, SourceBadge


def _session(tmp_path: Path, **overrides: str) -> ConfigSession:
    home = tmp_path / "memo-home"
    home.mkdir(exist_ok=True)
    (home / "memo-config.md").write_text("# Memo config\n", encoding="utf-8")
    env = {
        "MEMO_CONFIG_DIR": str(home),
        "MEMO_CONFIG_FILE": str(tmp_path / "legacy.toml"),
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        **overrides,
    }
    return ConfigSession.open(env)


@pytest.mark.asyncio
async def test_boolean_setting_uses_switch(tmp_path: Path) -> None:
    app = ConfigApp(_session(tmp_path))

    async with app.run_test(size=(120, 36)):
        row = app.query_one("#setting-recall-disable", SettingRow)

        assert row.query_one(Switch).value is False


@pytest.mark.asyncio
async def test_search_finds_folded_advanced_setting(tmp_path: Path) -> None:
    app = ConfigApp(_session(tmp_path))

    async with app.run_test(size=(120, 36)) as pilot:
        search = app.query_one("#setting-search", Input)
        search.value = "intra dedup threshold"
        await pilot.pause()

        assert app.query(SettingRow).first().setting_key == "recall.intra_dedup_threshold"


@pytest.mark.asyncio
async def test_env_override_badge_keeps_markdown_editable(tmp_path: Path) -> None:
    config_dir = tmp_path / "memo-home" / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "recall-config.md").write_text(
        "```toml\n[recall]\ntop_k = 7\n```\n",
        encoding="utf-8",
    )
    app = ConfigApp(_session(tmp_path, MEMO_RECALL_TOP_K="2"))

    async with app.run_test(size=(120, 36)):
        row = app.query_one("#setting-recall-top-k", SettingRow)

        assert row.query_one(SourceBadge).source is ValueSource.ENV
        assert row.query_one(Input).disabled is False
        assert row.query_one(Input).value == "2"


@pytest.mark.asyncio
async def test_input_change_updates_session_draft(tmp_path: Path) -> None:
    app = ConfigApp(_session(tmp_path))

    async with app.run_test(size=(120, 36)) as pilot:
        row = app.query_one("#setting-recall-top-k", SettingRow)
        control = row.query_one(Input)
        app.set_focus(control)
        await pilot.press("end", "backspace", "9")
        await pilot.pause()

        assert app.session.state("recall.top_k").pending_value == 9


@pytest.mark.asyncio
async def test_ctrl_c_quits_even_when_input_has_focus(tmp_path: Path) -> None:
    app = ConfigApp(_session(tmp_path))

    async with app.run_test(size=(120, 36)) as pilot:
        app.set_focus(app.query_one("#setting-search", Input))
        await pilot.press("ctrl+c")
        await pilot.pause()

        assert not app.is_running


@pytest.mark.asyncio
async def test_too_small_terminal_shows_requirement(tmp_path: Path) -> None:
    app = ConfigApp(_session(tmp_path))

    async with app.run_test(size=(70, 20)):
        warning = app.query_one("#size-warning")

        assert warning.display is True


@pytest.mark.parametrize("terminal_size", [(80, 24), (100, 30), (140, 45)])
def test_config_center_snapshots(
    snap_compare, tmp_path: Path, terminal_size: tuple[int, int]
) -> None:
    assert snap_compare(ConfigApp(_session(tmp_path)), terminal_size=terminal_size)
