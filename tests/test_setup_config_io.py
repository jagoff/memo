"""TOML config round-trip — write, read, override location via env."""

from __future__ import annotations

from pathlib import Path

from memo.setup.config_io import load_config_file, write_config_file


def test_round_trip_minimal(tmp_path: Path):
    cfg_path = tmp_path / "config.toml"
    written = write_config_file(data_dir=Path("/tmp/memo"), path=cfg_path)
    assert written == cfg_path
    loaded = load_config_file(path=cfg_path)
    assert loaded == {"storage": {"data_dir": "/tmp/memo"}}


def test_round_trip_with_vault_path(tmp_path: Path):
    cfg_path = tmp_path / "config.toml"
    write_config_file(
        data_dir=Path("/tmp/memo"),
        vault_path=Path("/tmp/Notes"),
        path=cfg_path,
    )
    loaded = load_config_file(path=cfg_path)
    assert loaded == {
        "storage": {"data_dir": "/tmp/memo", "vault_path": "/tmp/Notes"},
    }


def test_load_returns_none_when_file_missing(tmp_path: Path):
    assert load_config_file(path=tmp_path / "missing.toml") is None


def test_load_returns_none_on_corrupt_file(tmp_path: Path, capsys):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[storage\n broken = ", encoding="utf-8")
    assert load_config_file(path=cfg_path) is None
    # And prints a warning so the user notices.
    assert "warning" in capsys.readouterr().err.lower()


def test_write_creates_parent_dirs(tmp_path: Path):
    """`mkdir(parents=True)` covers fresh installs where ~/.config/memo doesn't exist."""
    cfg_path = tmp_path / "deep" / "nested" / "config.toml"
    write_config_file(data_dir=Path("/tmp/memo"), path=cfg_path)
    assert cfg_path.is_file()


def test_memo_config_file_env_var_overrides_default(tmp_path: Path, monkeypatch):
    """Setting MEMO_CONFIG_FILE makes the defaultless variants point there."""
    target = tmp_path / "alt-config.toml"
    monkeypatch.setenv("MEMO_CONFIG_FILE", str(target))
    # Re-import to re-resolve. The internal helper resolves the env at
    # call time, so this also tests that.
    from memo.setup.config_io import _resolve_config_path
    assert _resolve_config_path() == target


def test_atomic_write_doesnt_corrupt_existing(tmp_path: Path):
    """The .tmp + replace pattern means an existing file is replaced atomically."""
    cfg_path = tmp_path / "config.toml"
    write_config_file(data_dir=Path("/tmp/v1"), path=cfg_path)
    # Overwrite with a new value.
    write_config_file(data_dir=Path("/tmp/v2"), path=cfg_path)
    loaded = load_config_file(path=cfg_path)
    assert loaded["storage"]["data_dir"] == "/tmp/v2"
    # No leftover .tmp sidecar.
    assert not (cfg_path.with_suffix(cfg_path.suffix + ".tmp")).exists()
