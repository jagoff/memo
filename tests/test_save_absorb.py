"""C3 absorb-on-recurrence: a >=0.88 near-dup save rewrites the EXISTING
record via versioned update() instead of creating a near-copy (flag-gated)."""

from __future__ import annotations

import pytest

from memo.config import Config


@pytest.fixture
def mem_const(tmp_cfg, monkeypatch):
    """Memory with a constant-vector embedder (every pair is a cosine-1.0
    near-duplicate) and dims pinned to the stub via Config(embedder_dims=4)."""
    from memo.memory import Memory

    cfg = Config(
        data_dir=tmp_cfg.data_dir,
        vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir,
        embedder_dims=4,
        reranker_enabled=False,
    )
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0] for _ in inputs],
    )
    mem = Memory(cfg)
    yield mem
    mem.close()


class _AbsorbChat:
    def chat(self, model, messages, options=None):
        return {"message": {"content": "MERGED: puerto 8765 (confirmado 2x)"}}


class _BoomChat:
    def chat(self, *a, **k):
        raise RuntimeError("mlx down")


def test_absorb_rewrites_existing_record(mem_const, monkeypatch):
    monkeypatch.setenv("MEMO_SAVE_ABSORB", "1")
    mem_const._chat = _AbsorbChat()
    r1 = mem_const.save(content="El dashboard corre en el puerto 8765", title="Dashboard port")
    r2 = mem_const.save(content="Confirmado: dashboard en 8765", title="Dashboard port check")

    assert r2.id == r1.id  # absorbed — no new record
    rec = mem_const.get(r1.id)
    assert rec.body == "MERGED: puerto 8765 (confirmado 2x)"
    assert rec.extra["proof_count"] == 2
    assert rec.extra["last_absorbed_at"]
    assert len(mem_const.list(limit=10)) == 1  # corpus did not grow
    # corroboration counter also bumped (support_count rides along)
    assert mem_const.store.get_support_batch([r1.id]) == {r1.id: 1}


def test_absorb_is_versioned_and_rollbackable(mem_const, monkeypatch):
    monkeypatch.setenv("MEMO_SAVE_ABSORB", "1")
    mem_const._chat = _AbsorbChat()
    r1 = mem_const.save(content="cuerpo original", title="Nota")
    mem_const.save(content="cuerpo original bis", title="Nota bis")
    versions = mem_const.versioning.get_version_history(r1.id)
    assert versions  # pre-update snapshot exists → memo version rollback works


def test_absorb_off_by_default_creates_new(mem_const):
    r1 = mem_const.save(content="El dashboard corre en el puerto 8765", title="Dashboard port")
    r2 = mem_const.save(content="Confirmado: dashboard en 8765", title="Dashboard port check")
    assert r2.id != r1.id  # current warn-and-create behavior preserved


def test_absorb_falls_back_on_llm_failure(mem_const, monkeypatch):
    monkeypatch.setenv("MEMO_SAVE_ABSORB", "1")
    mem_const._chat = _BoomChat()
    r1 = mem_const.save(content="El dashboard corre en el puerto 8765", title="Dashboard port")
    r2 = mem_const.save(content="Confirmado: dashboard en 8765", title="Dashboard port check")
    assert r2.id != r1.id  # graceful fallback — the save is never blocked
