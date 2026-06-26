"""Regression: save followed by search must return the saved memory.

Covers two invariants from CLAUDE.md:
1. Golden path: save → search returns the memory.
2. Source-of-truth ordering: .md is written before the sqlite/vec insert.
"""
from __future__ import annotations

from memo.config import Config
from memo.memory import Memory


def _make_cfg(tmp_cfg: Config, dims: int = 4) -> Config:
    """Create an isolated Config with a fixed embedder_dims for test stubs."""
    return Config(
        data_dir=tmp_cfg.data_dir,
        vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir,
        embedder_dims=dims,
        reranker_enabled=False,
    )


def _stub_embedder(monkeypatch, dims: int = 4) -> None:
    """Patch MLXEmbedder to return a unit vector of length `dims`."""
    vec = [1.0] + [0.0] * (dims - 1)
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [vec for _ in inputs],
    )
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed_query",
        lambda self, query: vec,
    )
    monkeypatch.setenv("MEMO_EMBEDDER_DIMS", str(dims))


def test_save_then_search_returns_memory(tmp_cfg: Config, monkeypatch) -> None:
    """Golden path: save a memory and immediately search for it."""
    _stub_embedder(monkeypatch)
    cfg = _make_cfg(tmp_cfg)

    mem = Memory(cfg)
    try:
        result = mem.save(
            content="MLX prefill 30% faster than Ollama on M3 Max",
            title="MLX benchmark result",
            tags=["mlx", "benchmark"],
        )
        assert result.id, f"save returned no id: {result}"
        mem_id = result.id

        # The .md file must exist before we search
        md_files = list(cfg.memory_dir.rglob("*.md"))
        assert md_files, "no .md files written after save"

        results = mem.search("MLX prefill benchmark", limit=5)
        ids = [r.id for r in results]
        assert mem_id in ids, f"saved id {mem_id} not found in search results {ids}"
    finally:
        mem.close()


def test_save_md_is_written_before_sqlite_index(tmp_cfg: Config, monkeypatch) -> None:
    """Source-of-truth invariant: .md must be written before the sqlite/vec upsert."""
    _stub_embedder(monkeypatch)
    cfg = _make_cfg(tmp_cfg)

    md_written_before_index: list[bool] = []

    from memo.store.store import VecStore

    original_upsert = VecStore.upsert

    def patched_upsert(self, **kwargs):
        # At the moment upsert is called, the .md must already exist on disk.
        md_files = list(cfg.memory_dir.rglob("*.md"))
        md_written_before_index.append(bool(md_files))
        return original_upsert(self, **kwargs)

    monkeypatch.setattr("memo.store.store.VecStore.upsert", patched_upsert)

    mem = Memory(cfg)
    try:
        mem.save(content="test source-of-truth ordering", title="order test")
        assert md_written_before_index, "upsert was never called (save path changed?)"
        assert md_written_before_index[0], ".md must be written to disk before sqlite/vec upsert"
    finally:
        mem.close()
