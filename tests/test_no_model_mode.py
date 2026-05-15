from __future__ import annotations

import pytest

from memo.memory import Memory


def test_deferred_embed_save_is_searchable_via_bm25(tmp_cfg):
    mem = Memory(tmp_cfg)

    rec = mem.save(
        content="Decision: keep memo isolated from project venvs.",
        title="Runtime isolation",
        type_="decision",
        tags=["memo", "runtime"],
        defer_embed=True,
    )

    assert rec.extra["_memo_embed_pending"] is True
    assert mem.store.get(rec.id) is not None
    assert mem.store.has_vector(rec.id) is False

    hits = mem.search("isolated project venvs", mode="bm25", limit=5)
    assert [h.id for h in hits] == [rec.id]


def test_deferred_embed_save_does_not_touch_embedder(tmp_cfg, monkeypatch):
    mem = Memory(tmp_cfg)

    def fail_embed(_inputs):
        raise AssertionError("embedder should not run")

    monkeypatch.setattr(mem.embedder, "embed", fail_embed)

    rec = mem.save(
        content="Fact: bm25-only save does not require MLX.",
        title="BM25-only save",
        tags=["memo", "bm25"],
        defer_embed=True,
    )

    assert rec.title == "BM25-only save"


def test_reindex_fills_missing_vector_for_deferred_save(tmp_cfg, monkeypatch):
    mem = Memory(tmp_cfg)
    rec = mem.save(
        content="Fact: reindex fills vectors after deferred saves.",
        title="Deferred reindex",
        tags=["memo", "reindex"],
        defer_embed=True,
    )

    monkeypatch.setattr(mem.embedder, "embed", lambda _inputs: [[0.5] * 1024])
    counts = mem.reindex()

    assert counts["reindexed"] == 1
    assert mem.store.has_vector(rec.id) is True


def test_reindex_surfaces_bad_deferred_vector_dim(tmp_cfg, monkeypatch):
    mem = Memory(tmp_cfg)
    mem.save(
        content="Fact: invalid deferred embedding still fails loudly.",
        title="Bad vector",
        tags=["memo", "reindex"],
        defer_embed=True,
    )

    monkeypatch.setattr(mem.embedder, "embed", lambda _inputs: [[0.1]])

    with pytest.raises(ValueError):
        mem.reindex()
