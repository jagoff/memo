"""Typed Textual controls for configuration settings."""

from __future__ import annotations

from pathlib import Path

from textual.widgets import Input, Select, Switch

from memo.tui.config.controls import control_for
from memo.tui.config.session import ConfigSession


def _session(tmp_path: Path) -> ConfigSession:
    return ConfigSession.open(
        {
            "MEMO_CONFIG_DIR": str(tmp_path / "memo-home"),
            "MEMO_CONFIG_FILE": str(tmp_path / "legacy.toml"),
            "MEMO_DATA_DIR": str(tmp_path / "data"),
            "MEMO_STATE_DIR": str(tmp_path / "state"),
        }
    )


def test_boolean_setting_uses_switch(tmp_path: Path) -> None:
    assert isinstance(control_for(_session(tmp_path).state("recall.disable")), Switch)


def test_choice_setting_uses_select(tmp_path: Path) -> None:
    assert isinstance(control_for(_session(tmp_path).state("models.model_profile")), Select)


def test_numeric_and_path_settings_use_inputs(tmp_path: Path) -> None:
    session = _session(tmp_path)

    assert isinstance(control_for(session.state("recall.top_k")), Input)
    assert isinstance(control_for(session.state("storage.data_dir")), Input)
