"""Config — env loading, default paths, derived properties."""

from __future__ import annotations

from pathlib import Path

import pytest

from mem_lmx.config import Config


def test_defaults_resolve_paths():
    cfg = Config()
    assert cfg.vault_path.is_absolute()
    assert cfg.state_dir.is_absolute()
    assert cfg.embedder_dims == 1024
    assert cfg.llm_model.startswith("mlx-community/")


def test_from_env_picks_up_overrides(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MEM_LMX_VAULT_PATH", str(tmp_path / "v"))
    monkeypatch.setenv("MEM_LMX_STATE_DIR", str(tmp_path / "s"))
    monkeypatch.setenv("MEM_LMX_LLM_MODEL", "mlx-community/My-Model-X")
    monkeypatch.setenv("MEM_LMX_SEARCH_DEFAULT_LIMIT", "5")
    cfg = Config.from_env()
    assert cfg.vault_path == (tmp_path / "v").resolve()
    assert cfg.state_dir == (tmp_path / "s").resolve()
    assert cfg.llm_model == "mlx-community/My-Model-X"
    assert cfg.search_default_limit == 5


def test_derived_paths_compose(tmp_path: Path):
    cfg = Config(vault_path=tmp_path / "vault", state_dir=tmp_path / "state")
    assert cfg.memory_dir == cfg.vault_path / "04-Archive/99-obsidian-system/99-AI/memory"
    assert cfg.db_path == cfg.state_dir / "memvec.db"


def test_ensure_dirs_raises_on_missing_vault(tmp_path: Path):
    cfg = Config(vault_path=tmp_path / "missing", state_dir=tmp_path / "s")
    with pytest.raises(RuntimeError, match="Vault path does not exist"):
        cfg.ensure_dirs()


def test_ensure_dirs_creates_state_and_memory(tmp_path: Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    cfg = Config(vault_path=vault, state_dir=tmp_path / "s")
    cfg.ensure_dirs()
    assert cfg.state_dir.is_dir()
    assert cfg.memory_dir.is_dir()


def test_frozen():
    cfg = Config()
    with pytest.raises(Exception):  # pydantic ValidationError on frozen mutation
        cfg.embedder_dims = 99  # type: ignore[misc]
