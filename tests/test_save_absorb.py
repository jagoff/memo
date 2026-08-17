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


def test_absorb_on_by_default_rewrites_existing_record(mem_const):
    """No monkeypatch.setenv here — proves the flag's own default (now True)
    drives the absorb, not just an explicit override."""
    mem_const._chat = _AbsorbChat()
    r1 = mem_const.save(content="El dashboard corre en el puerto 8765", title="Dashboard port")
    r2 = mem_const.save(content="Confirmado: dashboard en 8765", title="Dashboard port check")
    assert r2.id == r1.id


def test_absorb_can_be_disabled(mem_const, monkeypatch):
    monkeypatch.setenv("MEMO_SAVE_ABSORB", "0")
    r1 = mem_const.save(content="El dashboard corre en el puerto 8765", title="Dashboard port")
    r2 = mem_const.save(content="Confirmado: dashboard en 8765", title="Dashboard port check")
    assert r2.id != r1.id  # opt-out preserved: warn-and-create behavior


def test_absorb_falls_back_on_llm_failure(mem_const, monkeypatch):
    monkeypatch.setenv("MEMO_SAVE_ABSORB", "1")
    mem_const._chat = _BoomChat()
    r1 = mem_const.save(content="El dashboard corre en el puerto 8765", title="Dashboard port")
    r2 = mem_const.save(content="Confirmado: dashboard en 8765", title="Dashboard port check")
    assert r2.id != r1.id  # graceful fallback — the save is never blocked


def test_absorb_skips_cross_type_near_duplicate(mem_const, monkeypatch):
    """A near-duplicate of a DIFFERENT type must never absorb — the LLM merge
    rewrites the existing record's body without touching its type, so
    absorbing across types would silently blend cross-type content under the
    original type label."""
    monkeypatch.setenv("MEMO_SAVE_ABSORB", "1")
    mem_const._chat = _AbsorbChat()
    r1 = mem_const.save(
        content="El dashboard corre en el puerto 8765", title="Dashboard port", type_="fact"
    )
    r2 = mem_const.save(
        content="Confirmado: dashboard en 8765", title="Dashboard port check", type_="note"
    )
    assert r2.id != r1.id  # type mismatch — falls back to warn-and-create
    rec = mem_const.get(r1.id)
    assert rec.body == "El dashboard corre en el puerto 8765"  # untouched by the merge
    assert rec.type == "fact"


class _AbsorbChatThatDeletesTarget:
    """Stands in for the LLM merge call, and — as a side effect of that call
    being in flight — deletes the absorb target. Reproduces the real race:
    absorb's update() runs unlocked and can land after a concurrent
    consolidation merge has already archived+deleted the same record."""

    def __init__(self, mem, target_id):
        self._mem = mem
        self._target_id = target_id

    def chat(self, model, messages, options=None):
        self._mem.delete(self._target_id)
        return {"message": {"content": "MERGED: puerto 8765 (confirmado 2x)"}}


def test_absorb_warns_when_target_vanishes_mid_flight(mem_const, monkeypatch, caplog):
    import logging

    monkeypatch.setenv("MEMO_SAVE_ABSORB", "1")
    r1 = mem_const.save(content="El dashboard corre en el puerto 8765", title="Dashboard port")
    mem_const._chat = _AbsorbChatThatDeletesTarget(mem_const, r1.id)

    with caplog.at_level(logging.WARNING, logger="memo.memory.record"):
        r2 = mem_const.save(content="Confirmado: dashboard en 8765", title="Dashboard port check")

    assert r2.id != r1.id  # target vanished mid-absorb — falls back to a new record
    assert mem_const.get(r1.id) is None  # confirms the delete actually landed first
    assert "vanished before update()" in caplog.text  # the race is now logged, not silent
