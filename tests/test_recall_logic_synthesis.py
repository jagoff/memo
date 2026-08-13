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
    [1,0,0,0]; text containing 'PARTIALTARGET' to [0.6,0.8,0,0]; everything else
    to [0,1,0,0]. So a query mentioning the HYBRIDTARGET marker has cosine 1.0
    with the target doc, 0.6 with a PARTIALTARGET doc and 0.0 with the rest —
    while hybrid RRF scores stay well below the 0.5 cosine-calibrated min_sim."""
    from memo.config import Config
    from memo.memory import Memory

    def _vec(text: str) -> list[float]:
        if "HYBRIDTARGET" in (text or ""):
            return [1.0, 0.0, 0.0, 0.0]
        if "PARTIALTARGET" in (text or ""):
            return [0.6, 0.8, 0.0, 0.0]
        return [0.0, 1.0, 0.0, 0.0]

    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed", lambda self, inputs: [_vec(t) for t in inputs]
    )
    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed_query", lambda self, q: _vec(q))
    cfg = Config(
        data_dir=tmp_path / "data",
        vault_path=tmp_path / "vault",
        state_dir=tmp_path / "state",
        embedder_dims=4,
        reranker_enabled=False,
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


# --- MEMO_RECALL_SKIP_BELOW is a cosine floor, on every mode's cosine scale ---


def test_hybrid_skip_below_floor_does_not_need_a_metadata_boost(tmp_path, monkeypatch):
    """A perfect-cosine hit surfaces in hybrid mode with NO curatorial metadata.

    ``MEMO_RECALL_SKIP_BELOW`` (0.45) is calibrated on the cosine scale, but the
    hybrid ``h.score`` is RRF-fused (~0.17 here) and can never reach it. While
    the floor compared that raw score, the only hybrid hits that survived were
    the ones whose filename/title overlap multiplied them over 0.45 — recall
    leaning on ``retrieval_boost`` to clear a gate measuring the wrong quantity.
    The title here shares no term with the query, so the boost is exactly 1.0
    and cannot mask a regression in the floor."""
    from memo.recall_logic import _recall_logic
    from memo.retrieval_boost import boost_for

    monkeypatch.setenv("MEMO_RECALL_MODE", "hybrid")
    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.5")
    monkeypatch.setenv("MEMO_RECALL_SKIP_BELOW", "0.45")
    monkeypatch.setenv("MEMO_RECALL_FORCE_MODE", "1")
    mem, cfg = _hybrid_mem(tmp_path, monkeypatch)
    mem.save(
        content="HYBRIDTARGET decidimos que el modelo queda en 0.6B por latencia warm aceptable",
        title="notas varias de la semana pasada",
        type_="decision",
    )
    prompt = "HYBRIDTARGET cuál reranker"

    hit = mem.search(prompt, limit=5, mode="hybrid", recency=True)[0]
    # The premise: zero metadata overlap, so retrieval_boost is a no-op here and
    # the RRF score stays far under the cosine floor.
    assert boost_for(query=prompt, filename=hit.path or "", title=hit.title or "") == 1.0
    assert (hit.score or 0.0) < 0.45

    context, _cb = _recall_logic(prompt, cwd=None, mem=mem, cfg=cfg)
    assert "HYBRIDTARGET" in context
    mem.close()


def test_hybrid_skip_below_floor_still_suppresses_a_weak_cosine(tmp_path, monkeypatch):
    """The floor is rescaled, not disabled: a 0.6 cosine under a 0.9 floor is
    still suppressed even though it clears min_sim and ranked first."""
    from memo.recall_logic import _recall_logic

    monkeypatch.setenv("MEMO_RECALL_MODE", "hybrid")
    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.5")
    monkeypatch.setenv("MEMO_RECALL_SKIP_BELOW", "0.9")
    monkeypatch.setenv("MEMO_RECALL_FORCE_MODE", "1")
    mem, cfg = _hybrid_mem(tmp_path, monkeypatch)
    mem.save(
        content="PARTIALTARGET decidimos que el modelo queda en 0.6B por latencia warm aceptable",
        title="notas varias de la semana pasada",
        type_="decision",
    )

    # Query embeds to [1,0,0,0], the PARTIALTARGET doc to [0.6,0.8,0,0] → cosine
    # 0.6: above min_sim 0.5 (so rank_hits keeps it), below skip_below 0.9.
    context, _cb = _recall_logic("HYBRIDTARGET cuál reranker", cwd=None, mem=mem, cfg=cfg)
    assert context == "{}"
    mem.close()
