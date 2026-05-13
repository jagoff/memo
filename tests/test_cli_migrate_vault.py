"""`memo migrate-vault` — copy memorias to a new data_dir, rebuild index."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from memo.cli import cli
from memo.config import Config
from memo.memory import Memory


@pytest.fixture
def seeded_old_layout(tmp_path: Path, monkeypatch):
    """Seed an 'old' data_dir with three memorias, indexed.

    Returns (cfg, memo_files) where cfg points at the seeded layout.
    """
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0] for _ in inputs],
    )
    old_data = tmp_path / "old"
    old_data.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    cfg = Config(data_dir=old_data, state_dir=state, embedder_dims=4)
    mem = Memory(cfg)
    files = []
    for i, title in enumerate(["A primero", "B segundo", "C tercero"]):
        rec = mem.save(content=f"contenido del memo {i}", title=title)
        files.append(rec.path)
    mem.store.close() if hasattr(mem.store, "close") else None
    return cfg, files


def test_migrate_copies_files_and_reindexes(
    tmp_path: Path, seeded_old_layout, monkeypatch,
):
    cfg, _files = seeded_old_layout
    new_data = tmp_path / "new"
    cfg_file = tmp_path / "memo-config.toml"

    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0] for _ in inputs],
    )

    runner = CliRunner()
    env = {
        "MEMO_CONFIG_FILE": str(cfg_file),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(cfg.data_dir),  # source
        "MEMO_STATE_DIR": str(cfg.state_dir),
        # Match the 4-dim stub embedder. Tests in this repo override
        # MLXEmbedder.embed via monkeypatch but the dim assertion in
        # `Config` is driven by env, so we have to pin it here.
        "MEMO_EMBEDDER_DIMS": "4",
        "MEMO_EMBEDDER_MODEL": "stub",
    }
    result = runner.invoke(
        cli,
        ["migrate-vault", str(new_data), "--yes"],
        env=env,
    )
    assert result.exit_code == 0, result.output

    # All 3 .md files copied to new location.
    new_files = sorted(new_data.rglob("*.md"))
    assert len(new_files) == 3

    # Old files preserved (migration is non-destructive).
    old_files = sorted(cfg.data_dir.rglob("*.md"))
    assert len(old_files) == 3

    # Config file updated to point at new dir.
    body = cfg_file.read_text(encoding="utf-8")
    assert f'data_dir = "{new_data.resolve()}"' in body

    # memvec.db rebuilt from new location.
    assert (cfg.state_dir / "memvec.db").is_file()


def test_migrate_refuses_non_empty_destination(tmp_path: Path, seeded_old_layout):
    cfg, _ = seeded_old_layout
    new_data = tmp_path / "non-empty"
    new_data.mkdir()
    (new_data / "stranger.md").write_text("not yours", encoding="utf-8")

    runner = CliRunner()
    env = {
        "MEMO_CONFIG_FILE": str(tmp_path / "config.toml"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(cfg.data_dir),
        "MEMO_STATE_DIR": str(cfg.state_dir),
        "MEMO_EMBEDDER_DIMS": "4",
        "MEMO_EMBEDDER_MODEL": "stub",
    }
    result = runner.invoke(
        cli, ["migrate-vault", str(new_data), "--yes"], env=env,
    )
    assert result.exit_code == 1
    assert "non-empty" in result.output


def test_migrate_refuses_same_src_and_dst(tmp_path: Path, seeded_old_layout):
    cfg, _ = seeded_old_layout
    runner = CliRunner()
    env = {
        "MEMO_CONFIG_FILE": str(tmp_path / "config.toml"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(cfg.data_dir),
        "MEMO_STATE_DIR": str(cfg.state_dir),
        "MEMO_EMBEDDER_DIMS": "4",
        "MEMO_EMBEDDER_MODEL": "stub",
    }
    result = runner.invoke(
        cli, ["migrate-vault", str(cfg.data_dir), "--yes"], env=env,
    )
    assert result.exit_code == 1
    assert "same" in result.output.lower()
