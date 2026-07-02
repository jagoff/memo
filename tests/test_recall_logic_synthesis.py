"""Tests for _deduplicate_synthesis in recall_logic."""

from __future__ import annotations

from types import SimpleNamespace

from memo.recall_logic import _deduplicate_synthesis


def _hit(id: str, type: str = "note", extra: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(id=id, type=type, extra=extra)


def test_deduplicate_synthesis_no_synthesis() -> None:
    """No synthesis hits — list is returned unchanged."""
    hits = [_hit("aaa"), _hit("bbb"), _hit("ccc")]
    result = _deduplicate_synthesis(hits)
    assert [h.id for h in result] == ["aaa", "bbb", "ccc"]


def test_deduplicate_synthesis_removes_sources() -> None:
    """Synthesis covers two sources — sources are removed."""
    synth = _hit("synth1", type="synthesis", extra={"synthesis_sources": ["src1", "src2"]})
    src1 = _hit("src1")
    src2 = _hit("src2")
    hits = [synth, src1, src2]
    result = _deduplicate_synthesis(hits)
    assert [h.id for h in result] == ["synth1"]


def test_deduplicate_synthesis_partial() -> None:
    """Synthesis covers 2 of 3 sources; uncovered source stays."""
    synth = _hit("synth1", type="synthesis", extra={"synthesis_sources": ["src1", "src2"]})
    src1 = _hit("src1")
    src2 = _hit("src2")
    src3 = _hit("src3")
    hits = [synth, src1, src2, src3]
    result = _deduplicate_synthesis(hits)
    assert [h.id for h in result] == ["synth1", "src3"]


def test_deduplicate_synthesis_bad_extra() -> None:
    """Synthesis with extra=None must not raise; list is returned unchanged."""
    synth = _hit("synth1", type="synthesis", extra=None)
    src = _hit("src1")
    hits = [synth, src]
    result = _deduplicate_synthesis(hits)
    # extra=None → no covered_ids → nothing removed
    assert [h.id for h in result] == ["synth1", "src1"]


# --- #6 hybrid min_sim gate (gate on vec cosine, not RRF score) --------------


def _hybrid_mem(tmp_path, monkeypatch):
    """Real Memory with a 4-dim stub: text containing 'HYBRIDTARGET' embeds to
    [1,0,0,0]; everything else to [0,1,0,0]. So a query mentioning the marker
    has cosine 1.0 with the target doc and 0.0 with the rest — while hybrid RRF
    scores stay well below the 0.5 cosine-calibrated min_sim."""
    from memo.config import Config
    from memo.memory import Memory

    def _vec(text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0] if "HYBRIDTARGET" in (text or "") else [0.0, 1.0, 0.0, 0.0]

    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed", lambda self, inputs: [_vec(t) for t in inputs]
    )
    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed_query", lambda self, q: _vec(q))
    cfg = Config(
        data_dir=tmp_path / "data",
        vault_path=tmp_path / "vault",
        state_dir=tmp_path / "state",
        embedder_dims=4,
    )
    return Memory(cfg), cfg


def test_hybrid_recall_gate_uses_vec_cosine_not_rrf(tmp_path, monkeypatch):
    from memo.recall_logic import _recall_logic

    monkeypatch.setenv("MEMO_RECALL_MODE", "hybrid")
    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.5")
    monkeypatch.setenv("MEMO_RECALL_FORCE_MODE", "1")
    mem, cfg = _hybrid_mem(tmp_path, monkeypatch)
    mem.save(
        content="HYBRIDTARGET decidimos que el reranker queda en el modelo 0.6B por latencia warm aceptable",
        title="HYBRIDTARGET decision",
        type_="decision",
    )
    mem.save(
        content="algo totalmente distinto sobre otra cosa que no viene al caso aquí",
        title="otro",
        type_="note",
    )

    context, _cb = _recall_logic("HYBRIDTARGET cuál reranker", cwd=None, mem=mem, cfg=cfg)
    # Without the vec-cosine gate the hybrid RRF score (<0.5) would gate every
    # hit out and return "{}" — the fix surfaces the high-cosine match.
    assert "HYBRIDTARGET" in context
    mem.close()


def test_hybrid_recall_gate_still_drops_low_cosine(tmp_path, monkeypatch):
    from memo.recall_logic import _recall_logic

    monkeypatch.setenv("MEMO_RECALL_MODE", "hybrid")
    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.5")
    monkeypatch.setenv("MEMO_RECALL_FORCE_MODE", "1")
    mem, cfg = _hybrid_mem(tmp_path, monkeypatch)
    mem.save(content="algo totalmente distinto sobre otra cosa", title="otro", type_="note")

    # Query marker → cosine 0.0 with the only (non-marker) doc → gated out.
    context, _cb = _recall_logic("HYBRIDTARGET cuál reranker", cwd=None, mem=mem, cfg=cfg)
    assert context == "{}"
    mem.close()
