"""Config — env loading, default paths, derived properties.

Resolution order under test (highest first):

1. Explicit kwargs.
2. `MEMO_*` env vars.
3. `~/.config/memo/config.toml` `[storage]` section.
4. Legacy back-compat: `MEMO_VAULT_PATH` + `MEMO_MEMORY_SUBDIR` derive `data_dir`.
5. Hardcoded defaults.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from memo.config import Config


def test_defaults_resolve_paths():
    cfg = Config()
    assert cfg.data_dir.is_absolute()
    assert cfg.state_dir.is_absolute()
    # vault_path is optional now; default is None.
    assert cfg.vault_path is None
    assert cfg.embedder_dims == 1024
    assert cfg.llm_model.startswith("mlx-community/")


def test_from_env_picks_up_overrides(monkeypatch, tmp_path: Path):
    # Disable config-file lookup so this test isolates env-var behaviour.
    monkeypatch.setenv("MEMO_CONFIG_FILE", str(tmp_path / "non-existent-config.toml"))
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "s"))
    monkeypatch.setenv("MEMO_LLM_MODEL", "mlx-community/My-Model-X")
    monkeypatch.setenv("MEMO_SEARCH_DEFAULT_LIMIT", "5")
    cfg = Config.from_env()
    assert cfg.data_dir == (tmp_path / "d").resolve()
    assert cfg.state_dir == (tmp_path / "s").resolve()
    assert cfg.llm_model == "mlx-community/My-Model-X"
    assert cfg.search_default_limit == 5


def test_legacy_env_vars_derive_data_dir(monkeypatch, tmp_path: Path):
    """`MEMO_VAULT_PATH` + `MEMO_MEMORY_SUBDIR` (legacy install) derive
    `data_dir = vault_path / memory_subdir` when `MEMO_DATA_DIR` is not set."""
    monkeypatch.setenv("MEMO_CONFIG_FILE", str(tmp_path / "non-existent.toml"))
    monkeypatch.delenv("MEMO_DATA_DIR", raising=False)
    monkeypatch.setenv("MEMO_VAULT_PATH", str(tmp_path / "v"))
    monkeypatch.setenv("MEMO_MEMORY_SUBDIR", "99-obsidian/99-AI/memory")
    cfg = Config.from_env()
    expected = (tmp_path / "v" / "99-obsidian/99-AI/memory").resolve()
    assert cfg.data_dir == expected
    assert cfg.vault_path == (tmp_path / "v").resolve()


def test_data_dir_env_overrides_legacy(monkeypatch, tmp_path: Path):
    """`MEMO_DATA_DIR` wins over the legacy back-compat path."""
    monkeypatch.setenv("MEMO_CONFIG_FILE", str(tmp_path / "non-existent.toml"))
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "explicit"))
    monkeypatch.setenv("MEMO_VAULT_PATH", str(tmp_path / "v"))
    monkeypatch.setenv("MEMO_MEMORY_SUBDIR", "ignored")
    cfg = Config.from_env()
    assert cfg.data_dir == (tmp_path / "explicit").resolve()


def test_config_file_loads(monkeypatch, tmp_path: Path):
    """A `~/.config/memo/config.toml` populates fields. Env vars still win."""
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        '[storage]\n'
        f'data_dir = "{tmp_path / "from-file"}"\n'
        f'vault_path = "{tmp_path / "vault-from-file"}"\n'
    )
    monkeypatch.setenv("MEMO_CONFIG_FILE", str(cfg_file))
    monkeypatch.delenv("MEMO_DATA_DIR", raising=False)
    monkeypatch.delenv("MEMO_VAULT_PATH", raising=False)
    cfg = Config.from_env()
    assert cfg.data_dir == (tmp_path / "from-file").resolve()
    assert cfg.vault_path == (tmp_path / "vault-from-file").resolve()


def test_env_overrides_config_file(monkeypatch, tmp_path: Path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        '[storage]\n'
        f'data_dir = "{tmp_path / "from-file"}"\n'
    )
    monkeypatch.setenv("MEMO_CONFIG_FILE", str(cfg_file))
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "from-env"))
    cfg = Config.from_env()
    assert cfg.data_dir == (tmp_path / "from-env").resolve()


def test_derived_paths_compose(tmp_path: Path):
    cfg = Config(data_dir=tmp_path / "d", state_dir=tmp_path / "state")
    assert cfg.memory_dir == cfg.data_dir
    assert cfg.db_path == cfg.state_dir / "memvec.db"
    assert cfg.history_db == cfg.state_dir / "history.db"


def test_ensure_dirs_creates_state_and_data(tmp_path: Path):
    cfg = Config(data_dir=tmp_path / "d", state_dir=tmp_path / "s")
    cfg.ensure_dirs()
    assert cfg.state_dir.is_dir()
    assert cfg.data_dir.is_dir()


def test_ensure_dirs_no_longer_requires_vault_path(tmp_path: Path):
    """`vault_path` is optional now; ensure_dirs must not raise when it's None."""
    cfg = Config(data_dir=tmp_path / "d", state_dir=tmp_path / "s", vault_path=None)
    cfg.ensure_dirs()  # should not raise


def test_frozen():
    from pydantic import ValidationError
    cfg = Config()
    with pytest.raises(ValidationError):
        cfg.embedder_dims = 99  # type: ignore[misc]
