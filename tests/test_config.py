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


def test_from_env_picks_up_reranker_revision(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MEMO_CONFIG_FILE", str(tmp_path / "non-existent-config.toml"))
    monkeypatch.setenv("MEMO_RERANKER_REVISION", "9655b27")
    cfg = Config.from_env()
    assert cfg.reranker_revision == "9655b27"


def test_model_profile_quality_sets_model_bundle(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MEMO_CONFIG_FILE", str(tmp_path / "non-existent-config.toml"))
    monkeypatch.setenv("MEMO_MODEL_PROFILE", "quality")
    cfg = Config.from_env()

    assert cfg.model_profile == "quality"
    assert cfg.embedder_model == "mlx-community/Qwen3-Embedding-4B-4bit-DWQ"
    assert cfg.embedder_dims == 2560
    assert cfg.reranker_enabled is True


def test_model_profile_allows_specific_env_override(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MEMO_CONFIG_FILE", str(tmp_path / "non-existent-config.toml"))
    monkeypatch.setenv("MEMO_MODEL_PROFILE", "quality")
    monkeypatch.setenv("MEMO_EMBEDDER_MODEL", "mlx-community/Custom-Embedding")
    monkeypatch.setenv("MEMO_EMBEDDER_DIMS", "1024")
    cfg = Config.from_env()

    assert cfg.model_profile == "quality"
    assert cfg.embedder_model == "mlx-community/Custom-Embedding"
    assert cfg.embedder_dims == 1024


def test_legacy_env_vars_derive_data_dir(monkeypatch, tmp_path: Path):
    """`MEMO_VAULT_PATH` + `MEMO_MEMORY_SUBDIR` (legacy install) derive
    `data_dir = vault_path / memory_subdir` when `MEMO_DATA_DIR` is not set."""
    monkeypatch.setenv("MEMO_CONFIG_FILE", str(tmp_path / "non-existent.toml"))
    monkeypatch.delenv("MEMO_DATA_DIR", raising=False)
    monkeypatch.setenv("MEMO_VAULT_PATH", str(tmp_path / "v"))
    monkeypatch.setenv("MEMO_MEMORY_SUBDIR", "Obsidian/AI/memory")
    cfg = Config.from_env()
    expected = (tmp_path / "v" / "Obsidian/AI/memory").resolve()
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
        "[storage]\n"
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
    cfg_file.write_text(f'[storage]\ndata_dir = "{tmp_path / "from-file"}"\n')
    monkeypatch.setenv("MEMO_CONFIG_FILE", str(cfg_file))
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "from-env"))
    cfg = Config.from_env()
    assert cfg.data_dir == (tmp_path / "from-env").resolve()


def test_derived_paths_compose(tmp_path: Path):
    cfg = Config(data_dir=tmp_path / "d", state_dir=tmp_path / "state")
    assert cfg.memory_dir == cfg.data_dir
    assert cfg.db_path == cfg.state_dir / "memvec.db"


def _write_index_meta(
    db_path: Path, *, model: str | None, dims: int | None, vec_ddl: str | None = None
) -> None:
    """Plant a minimal self-describing index at `db_path` for adoption tests."""
    import sqlite3

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        if model is not None and dims is not None:
            conn.execute("CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            conn.executemany(
                "INSERT INTO schema_meta (key, value) VALUES (?, ?)",
                [("embedder_model", model), ("embedder_dims", str(dims))],
            )
        if vec_ddl is not None:
            conn.execute(vec_ddl)
        conn.commit()
    finally:
        conn.close()


def test_from_env_adopts_index_embedder_profile(monkeypatch, tmp_path: Path):
    """A bare launch (no embedder env) adopts the profile the index was built
    with, instead of defaulting to 0.6B/1024 and crashing on a 2560 index.

    This is the systemic fix for the opaque MCP "connection closed" handshake
    failure: MCP clients don't inherit the shell env, so the embedder must be
    recoverable from the self-describing index.
    """
    monkeypatch.setenv("MEMO_CONFIG_FILE", str(tmp_path / "non-existent.toml"))
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "s"))
    for var in ("MEMO_EMBEDDER_MODEL", "MEMO_EMBEDDER_DIMS", "MEMO_MODEL_PROFILE"):
        monkeypatch.delenv(var, raising=False)

    db_path = Config.from_env().db_path  # default profile (balanced/1024)
    _write_index_meta(db_path, model="mlx-community/Qwen3-Embedding-4B-4bit-DWQ", dims=2560)

    cfg = Config.from_env()
    assert cfg.embedder_dims == 2560
    assert cfg.embedder_model == "mlx-community/Qwen3-Embedding-4B-4bit-DWQ"


def test_explicit_embedder_env_blocks_index_adoption(monkeypatch, tmp_path: Path):
    """An explicit `MEMO_EMBEDDER_DIMS` is honoured even when it conflicts with
    the index — the operator's pin wins, preserving the reindex-or-fix signal."""
    monkeypatch.setenv("MEMO_CONFIG_FILE", str(tmp_path / "non-existent.toml"))
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "s"))
    monkeypatch.setenv("MEMO_EMBEDDER_DIMS", "1024")

    db_path = Config.from_env().db_path
    _write_index_meta(db_path, model="mlx-community/Qwen3-Embedding-4B-4bit-DWQ", dims=2560)

    cfg = Config.from_env()
    assert cfg.embedder_dims == 1024  # pin respected, no adoption


def test_index_adoption_falls_back_to_vec_table_dims(monkeypatch, tmp_path: Path):
    """A pre-`schema_meta` index (only the vec0 table records dims) still heals:
    the model is derived from the dims via MODEL_PROFILES."""
    monkeypatch.setenv("MEMO_CONFIG_FILE", str(tmp_path / "non-existent.toml"))
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "s"))
    for var in ("MEMO_EMBEDDER_MODEL", "MEMO_EMBEDDER_DIMS", "MEMO_MODEL_PROFILE"):
        monkeypatch.delenv(var, raising=False)

    db_path = Config.from_env().db_path
    _write_index_meta(
        db_path, model=None, dims=None, vec_ddl="CREATE TABLE vec (embedding FLOAT[2560])"
    )

    cfg = Config.from_env()
    assert cfg.embedder_dims == 2560
    assert cfg.embedder_model == "mlx-community/Qwen3-Embedding-4B-4bit-DWQ"
    assert cfg.history_db == cfg.state_dir / "history.db"


def test_ensure_dirs_creates_state_and_data(tmp_path: Path):
    cfg = Config(data_dir=tmp_path / "d", state_dir=tmp_path / "s")
    cfg.ensure_dirs()
    assert cfg.state_dir.is_dir()
    assert cfg.data_dir.is_dir()


def test_memories_in_vault_derives_memory_dir(tmp_path: Path):
    """When the toggle is on AND a vault is set, memorias live under the vault."""
    cfg = Config(
        data_dir=tmp_path / "d",
        state_dir=tmp_path / "s",
        vault_path=tmp_path / "vault",
        memories_in_vault=True,
    )
    # AI_SUBDIR already includes SYSTEM_DIR ("Obsidian/AI").
    assert cfg.memory_dir == (tmp_path / "vault" / "Obsidian" / "AI" / "memory").resolve()
    assert cfg.memory_dir != cfg.data_dir


def test_memories_in_vault_without_vault_falls_back_to_data_dir(tmp_path: Path):
    """The toggle is inert without a vault_path — no crash, no vault path."""
    cfg = Config(data_dir=tmp_path / "d", state_dir=tmp_path / "s", memories_in_vault=True)
    assert cfg.memory_dir == cfg.data_dir


def test_memories_in_vault_default_off(tmp_path: Path):
    """Default keeps existing installs on data_dir even with a vault configured."""
    cfg = Config(data_dir=tmp_path / "d", state_dir=tmp_path / "s", vault_path=tmp_path / "vault")
    assert cfg.memories_in_vault is False
    assert cfg.memory_dir == cfg.data_dir


def test_memories_in_vault_from_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MEMO_CONFIG_FILE", str(tmp_path / "none.toml"))
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "s"))
    monkeypatch.setenv("MEMO_VAULT_PATH", str(tmp_path / "vault"))
    monkeypatch.setenv("MEMO_MEMORIES_IN_VAULT", "1")
    cfg = Config.from_env()
    assert cfg.memories_in_vault is True
    assert cfg.memory_dir == (tmp_path / "vault" / "Obsidian" / "AI" / "memory").resolve()


def test_single_db_default_keeps_separate_sidecar_files(tmp_path: Path):
    cfg = Config(data_dir=tmp_path / "d", state_dir=tmp_path / "s")
    assert cfg.single_db is False
    assert cfg.history_db == cfg.state_dir / "history.db"
    assert cfg.graph_db == cfg.state_dir / "graph.db"
    assert cfg.crossref_db == cfg.state_dir / "crossref.db"
    assert cfg.contradictions_db == cfg.state_dir / "contradictions.db"
    assert cfg.history_db != cfg.db_path


def test_single_db_collapses_sidecars_onto_db_path(tmp_path: Path):
    cfg = Config(data_dir=tmp_path / "d", state_dir=tmp_path / "s", single_db=True)
    assert cfg.history_db == cfg.db_path
    assert cfg.graph_db == cfg.db_path
    assert cfg.crossref_db == cfg.db_path
    assert cfg.contradictions_db == cfg.db_path


def test_single_db_from_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MEMO_CONFIG_FILE", str(tmp_path / "none.toml"))
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "s"))
    monkeypatch.setenv("MEMO_SINGLE_DB", "1")
    cfg = Config.from_env()
    assert cfg.single_db is True
    assert cfg.history_db == cfg.db_path


def test_ensure_dirs_creates_vault_memory_dir(tmp_path: Path):
    """ensure_dirs must create the vault memory subtree, not just data_dir."""
    cfg = Config(
        data_dir=tmp_path / "d",
        state_dir=tmp_path / "s",
        vault_path=tmp_path / "vault",
        memories_in_vault=True,
    )
    cfg.ensure_dirs()
    assert cfg.memory_dir.is_dir()


def test_ensure_dirs_no_longer_requires_vault_path(tmp_path: Path):
    """`vault_path` is optional now; ensure_dirs must not raise when it's None."""
    cfg = Config(data_dir=tmp_path / "d", state_dir=tmp_path / "s", vault_path=None)
    cfg.ensure_dirs()  # should not raise


def test_frozen():
    from pydantic import ValidationError

    cfg = Config()
    with pytest.raises(ValidationError):
        cfg.embedder_dims = 99  # type: ignore[misc]


def test_device_id_first_run_mints_and_persists(tmp_path: Path):
    cfg = Config(data_dir=tmp_path / "d", state_dir=tmp_path / "s")
    first = cfg.device_id
    assert first
    assert not first.startswith("transient-")
    assert (tmp_path / "s" / ".device_id").read_text(encoding="utf-8").strip() == first
    # second access re-reads the same persisted id
    assert cfg.device_id == first
    # no leftover tmp files from the atomic publish
    assert list((tmp_path / "s").glob(".device_id.*.tmp")) == []


def test_device_id_lost_mint_race_adopts_winner(tmp_path: Path, monkeypatch):
    """If another first-run process publishes between our is_file() check
    and our link attempt, the FILE's id wins — not our fresh mint."""
    import os as _os

    cfg = Config(data_dir=tmp_path / "d", state_dir=tmp_path / "s")
    id_path = tmp_path / "s" / ".device_id"

    def racing_link(src, dst, *a, **kw):
        id_path.write_text("winner123456", encoding="utf-8")
        raise FileExistsError(dst)

    monkeypatch.setattr(_os, "link", racing_link)
    assert cfg.device_id == "winner123456"
    # the loser's mint was never published
    assert id_path.read_text(encoding="utf-8").strip() == "winner123456"


def test_device_id_lost_race_empty_winner_falls_back_transient(tmp_path: Path, monkeypatch, caplog):
    """A lost race against an unreadable/empty winner keeps the existing
    transient fallback (never returns '')."""
    import logging
    import os as _os

    cfg = Config(data_dir=tmp_path / "d", state_dir=tmp_path / "s")
    id_path = tmp_path / "s" / ".device_id"

    def racing_link(src, dst, *a, **kw):
        id_path.write_text("", encoding="utf-8")
        raise FileExistsError(dst)

    monkeypatch.setattr(_os, "link", racing_link)
    with caplog.at_level(logging.WARNING, logger="memo.config"):
        device_id = cfg.device_id
    assert device_id.startswith("transient-")
