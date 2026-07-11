"""recall-dedup-collapse: collapse paraphrase dups in the pool before top-K (default off).

collapse_near_dups itself is already covered by tests/test_recall_dedup.py.
These tests cover the NEW wiring: MEMO_RECALL_DEDUP_COLLAPSE gates a
pre-top-K collapse of `qualifying`, distinct from the existing post-top-K
MEMO_RECALL_INTRA_DEDUP wire on `relevant`.
"""
from __future__ import annotations

import pytest

from memo.config import Config
from memo.memory import Memory
from memo.recall_logic import _recall_logic


@pytest.fixture
def mem(tmp_cfg: Config, monkeypatch) -> Memory:
    cfg = Config(
        data_dir=tmp_cfg.data_dir,
        vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir,
        embedder_dims=64,
    )

    def _stub_embed(self, inputs):
        out = []
        for s in inputs:
            h = sum(ord(c) for c in s) % 64
            v = [0.0] * 64
            v[h] = 1.0
            out.append(v)
        return out

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _stub_embed)
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed_query",
        lambda self, q: _stub_embed(self, [q])[0],
    )
    m = Memory(cfg)
    yield m
    m.close()


def _base_env(monkeypatch) -> None:
    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.0")
    monkeypatch.setenv("MEMO_RECALL_SKIP_BELOW", "0")
    monkeypatch.setenv("MEMO_RECALL_MIN_BODY_CHARS", "0")


def test_dedup_collapse_runs_pre_topk_when_flag_on(mem: Memory, monkeypatch):
    """With MEMO_RECALL_DEDUP_COLLAPSE=1, collapse_near_dups fires on the pool
    (qualifying) BEFORE the top-K slice — distinct from MEMO_RECALL_INTRA_DEDUP,
    which runs after."""
    _base_env(monkeypatch)
    monkeypatch.setenv("MEMO_RECALL_DEDUP_COLLAPSE", "1")
    monkeypatch.setenv("MEMO_RECALL_INTRA_DEDUP_THRESHOLD", "0.5")
    monkeypatch.delenv("MEMO_RECALL_INTRA_DEDUP", raising=False)
    monkeypatch.setenv("MEMO_RECALL_TOP_K", "5")

    mem.save(content="el cutover memflow a mac-work fue ok", title="Deploy cutover mac-work", type_="fact")
    mem.save(content="el cutover memflow a mac work fue ok", title="Deploy cutover en mac-work", type_="fact")

    calls: list[tuple] = []
    original_collapse = __import__("memo.recall_logic", fromlist=["collapse_near_dups"]).collapse_near_dups

    def _spy(relevant, *, threshold):
        result = original_collapse(relevant, threshold=threshold)
        calls.append((len(relevant), len(result)))
        return result

    monkeypatch.setattr("memo.recall_logic.collapse_near_dups", _spy)

    out, _ = _recall_logic("deploy cutover mac-work", None, mem, mem.cfg)
    assert out != "{}", "expected at least one recall hit — check stub embedder / saved memories"
    assert calls, "collapse_near_dups must be called pre-top-K when MEMO_RECALL_DEDUP_COLLAPSE=1"


def test_dedup_collapse_skipped_when_flag_off(mem: Memory, monkeypatch):
    """With MEMO_RECALL_DEDUP_COLLAPSE unset, no pre-top-K collapse runs — recall
    stays byte-identical to today."""
    _base_env(monkeypatch)
    monkeypatch.delenv("MEMO_RECALL_DEDUP_COLLAPSE", raising=False)
    monkeypatch.delenv("MEMO_RECALL_INTRA_DEDUP", raising=False)

    mem.save(content="el cutover memflow a mac-work fue ok", title="Deploy cutover mac-work", type_="fact")
    mem.save(content="el cutover memflow a mac work fue ok", title="Deploy cutover en mac-work", type_="fact")

    calls: list = []

    def _spy(relevant, *, threshold):
        calls.append(True)
        return relevant

    monkeypatch.setattr("memo.recall_logic.collapse_near_dups", _spy)

    _recall_logic("deploy cutover mac-work", None, mem, mem.cfg)
    assert not calls, "collapse_near_dups must NOT be called when MEMO_RECALL_DEDUP_COLLAPSE is unset"


def test_collapse_near_dups_drops_lower_scored_paraphrase():
    """Direct helper-level check (already covered generally by test_recall_dedup.py;
    kept here as a scenario mirroring this unit's brief for traceability)."""
    from types import SimpleNamespace

    from memo import recall_logic

    def _hit(id_, title, body, score):
        return SimpleNamespace(
            id=id_, title=title, body=body, tags=[], score=score,
            type="note", updated="2026-07-10", extra={},
        )

    pool = [
        _hit("aaaa1111", "Dashboard port", "The dashboard runs on port 8765", 0.9),
        _hit("bbbb2222", "Dashboard port", "The dashboard runs on port 8765", 0.6),  # dup
        _hit("cccc3333", "Recall budget", "Recall hook must stay under 5 seconds", 0.8),
    ]
    kept = recall_logic.collapse_near_dups(pool, threshold=0.8)
    ids = [h.id for h in kept]
    assert "aaaa1111" in ids and "cccc3333" in ids
    assert "bbbb2222" not in ids  # lower-scored paraphrase collapsed
