"""Tests for the verbatim turn-level FTS5 index (flags + TurnStore config property)."""
from __future__ import annotations

from pathlib import Path

from memo.config import Config


def test_verbatim_flags_registered_defaults():
    """All three verbatim flags are registered with correct defaults."""
    from memo.flags import REGISTRY

    assert REGISTRY["MEMO_VERBATIM_INDEX"].default is False
    assert REGISTRY["MEMO_VERBATIM_MAX_DAYS"].default == 90
    assert REGISTRY["MEMO_VERBATIM_MIN_CHARS"].default == 20


def test_verbatim_db_separate_when_single_db_off(tmp_path: Path):
    """When single_db is False (default), verbatim_db is a separate file in state_dir."""
    cfg = Config(data_dir=tmp_path / "d", state_dir=tmp_path / "s")
    assert cfg.single_db is False
    assert cfg.verbatim_db == cfg.state_dir / "verbatim.db"
    assert cfg.verbatim_db != cfg.db_path


def test_verbatim_db_collapses_with_single_db_true(tmp_path: Path):
    """When single_db is True, verbatim_db collapses onto db_path."""
    cfg = Config(data_dir=tmp_path / "d", state_dir=tmp_path / "s", single_db=True)
    assert cfg.single_db is True
    assert cfg.verbatim_db == cfg.db_path


def test_verbatim_db_from_env(monkeypatch, tmp_path: Path):
    """MEMO_SINGLE_DB env var controls verbatim_db collapse."""
    monkeypatch.setenv("MEMO_CONFIG_FILE", str(tmp_path / "none.toml"))
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "s"))
    monkeypatch.setenv("MEMO_SINGLE_DB", "1")
    cfg = Config.from_env()
    assert cfg.single_db is True
    assert cfg.verbatim_db == cfg.db_path
