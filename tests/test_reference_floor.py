"""Reference-tier noise floor in explicit retrieval (MEMO_REFERENCE_SEARCH_FLOOR).

The recall hook already SQL-excludes the bulk `reference` tier
(MEMO_RECALL_EXCLUDE_REFERENCE, default on), but the EXPLICIT retrieval
paths — memo_search / memo_ask / chat_ask (all routed through
`Memory.search()`) — let reference chunks compete with durable memories.
`MEMO_REFERENCE_SEARCH_FLOOR` (default 0.0 = off) requires a reference-tier
hit to clear a final-score floor to stay in results; durable-tier hits are
never affected. Implemented at the single choke point in
`memory/search_ops.py::_SearchOpsMixin.search`.
"""

from __future__ import annotations

from memo.memory import Memory

# Deterministic 4-dim vectors: the query embeds to QUERY_VEC; docs embed to
# QUERY_VEC (cosine ~1.0 with the query) or ORTHO_VEC (cosine ~0.0) depending
# on a marker token in their composed text. mem_with_stub pins
# embedder_dims=4 and disables the reranker.
QUERY_VEC = [1.0, 0.0, 0.0, 0.0]
ORTHO_VEC = [0.0, 1.0, 0.0, 0.0]

# Reference bodies must be >= 60 chars (MIN_REFERENCE_CHARS) or the
# write-path noise gate rejects them.
_PAD = " relleno de contenido para superar el umbral minimo de caracteres."


def _wire_embeddings(mem: Memory, monkeypatch) -> None:
    """Docs containing 'nearmark' embed at QUERY_VEC; the rest at ORTHO_VEC."""

    def _embed(inputs):
        return [list(QUERY_VEC) if "nearmark" in text else list(ORTHO_VEC) for text in inputs]

    monkeypatch.setattr(mem.embedder, "embed", _embed)
    monkeypatch.setattr(mem.embedder, "embed_query", lambda _q: list(QUERY_VEC))


def _save_corpus(mem: Memory) -> dict[str, str]:
    """One durable near hit, one reference near hit, one reference far hit."""
    durable = mem.save(content="nearmark decision durable" + _PAD, title="Durable", type_="fact")
    ref_hi = mem.save(content="nearmark ref chunk" + _PAD, title="RefHigh", type_="reference")
    ref_lo = mem.save(content="lejano ref chunk" + _PAD, title="RefLow", type_="reference")
    return {"durable": durable.id, "ref_hi": ref_hi.id, "ref_lo": ref_lo.id}


# --- floor semantics (vec mode: score = cosine, deterministic) ----------------


def test_reference_below_floor_is_dropped(mem_with_stub: Memory, monkeypatch):
    _wire_embeddings(mem_with_stub, monkeypatch)
    ids = _save_corpus(mem_with_stub)
    monkeypatch.setenv("MEMO_REFERENCE_SEARCH_FLOOR", "0.5")

    hits = {h.id for h in mem_with_stub.search("nearmark", mode="vec", limit=10)}

    assert ids["ref_lo"] not in hits
    assert ids["durable"] in hits


def test_reference_above_floor_is_kept(mem_with_stub: Memory, monkeypatch):
    _wire_embeddings(mem_with_stub, monkeypatch)
    ids = _save_corpus(mem_with_stub)
    monkeypatch.setenv("MEMO_REFERENCE_SEARCH_FLOOR", "0.5")

    hits = {h.id for h in mem_with_stub.search("nearmark", mode="vec", limit=10)}

    assert ids["ref_hi"] in hits


def test_durable_below_floor_is_never_affected(mem_with_stub: Memory, monkeypatch):
    _wire_embeddings(mem_with_stub, monkeypatch)
    # Durable memory embeds ORTHOGONAL to the query → score ~0.0, well below
    # the floor. It must survive: the floor applies to reference tier only.
    low_durable = mem_with_stub.save(content="lejano durable" + _PAD, title="D", type_="decision")
    monkeypatch.setenv("MEMO_REFERENCE_SEARCH_FLOOR", "0.5")

    hits = {h.id for h in mem_with_stub.search("nearmark", mode="vec", limit=10)}

    assert low_durable.id in hits


def test_default_floor_zero_is_a_noop(mem_with_stub: Memory, monkeypatch):
    _wire_embeddings(mem_with_stub, monkeypatch)
    ids = _save_corpus(mem_with_stub)
    monkeypatch.delenv("MEMO_REFERENCE_SEARCH_FLOOR", raising=False)

    envelope = mem_with_stub.search_with_trace("nearmark", mode="vec", limit=10)

    hits = {h.id for h in envelope["hits"]}
    # Low-score reference stays in results — identical to pre-flag behavior.
    assert ids["ref_lo"] in hits
    assert hits == set(ids.values())
    # And the pipeline never runs the floor stage.
    assert "reference_floor" not in [item["stage"] for item in envelope["trace"]]


def test_explicit_reference_type_filter_bypasses_floor(mem_with_stub: Memory, monkeypatch):
    _wire_embeddings(mem_with_stub, monkeypatch)
    ids = _save_corpus(mem_with_stub)
    monkeypatch.setenv("MEMO_REFERENCE_SEARCH_FLOOR", "0.5")

    # The reference tier is "searchable on demand": an explicit
    # type_="reference" search must not be nuked by the floor.
    hits = {h.id for h in mem_with_stub.search("nearmark", mode="vec", limit=10, type_="reference")}

    assert ids["ref_lo"] in hits
    assert ids["ref_hi"] in hits


def test_floor_stage_appears_in_trace(mem_with_stub: Memory, monkeypatch):
    _wire_embeddings(mem_with_stub, monkeypatch)
    _save_corpus(mem_with_stub)
    monkeypatch.setenv("MEMO_REFERENCE_SEARCH_FLOOR", "0.5")

    envelope = mem_with_stub.search_with_trace("nearmark", mode="vec", limit=10)

    stages = {item["stage"]: item for item in envelope["trace"]}
    assert "reference_floor" in stages
    assert stages["reference_floor"]["floor"] == 0.5
    assert stages["reference_floor"]["output_count"] < stages["reference_floor"]["input_count"]


# --- wiring across modes and consumer paths -----------------------------------


def test_floor_applies_in_bm25_mode(mem_with_stub: Memory, monkeypatch):
    # BM25 scores use a different scale than cosine; a floor far above any
    # score drops every reference hit while durable hits are untouched —
    # proving the filter is tier-scoped, not score-scale-dependent.
    _wire_embeddings(mem_with_stub, monkeypatch)
    ids = _save_corpus(mem_with_stub)
    monkeypatch.setenv("MEMO_REFERENCE_SEARCH_FLOOR", "1000000")

    hits = {h.id for h in mem_with_stub.search("nearmark", mode="bm25", limit=10)}

    assert ids["durable"] in hits
    assert ids["ref_hi"] not in hits
    assert ids["ref_lo"] not in hits


def test_floor_applies_in_hybrid_mode(mem_with_stub: Memory, monkeypatch):
    _wire_embeddings(mem_with_stub, monkeypatch)
    ids = _save_corpus(mem_with_stub)
    monkeypatch.setenv("MEMO_REFERENCE_SEARCH_FLOOR", "1000000")

    hits = {h.id for h in mem_with_stub.search("nearmark", mode="hybrid", limit=10)}

    assert ids["durable"] in hits
    assert ids["ref_hi"] not in hits
    assert ids["ref_lo"] not in hits


def test_floor_applies_in_ask_path(mock_memory: Memory, monkeypatch):
    """memo_ask routes through _build_ask_context → Memory.search (hybrid)."""
    mock_memory.save(content="pizza decision durable" + _PAD, title="Durable", type_="decision")
    mock_memory.save(content="pizza reference chunk" + _PAD, title="Ref", type_="reference")

    # Control: without the floor the reference chunk reaches the sources.
    monkeypatch.delenv("MEMO_REFERENCE_SEARCH_FLOOR", raising=False)
    control = mock_memory.ask("¿qué se decidió sobre pizza?", k=10)
    control_types = {s["type"] for s in control["sources"]}
    assert "reference" in control_types

    # RRF-fused hybrid scores are far below 1000000 → every reference drops.
    monkeypatch.setenv("MEMO_REFERENCE_SEARCH_FLOOR", "1000000")
    res = mock_memory.ask("¿qué se decidió sobre pizza?", k=10)
    types = {s["type"] for s in res["sources"]}
    assert "reference" not in types
    assert "decision" in types


def test_floor_applies_in_chat_path(mock_memory: Memory, monkeypatch):
    """chat_ask delegates to ask() → same search() choke point."""
    mock_memory.save(content="pizza decision durable" + _PAD, title="Durable", type_="decision")
    mock_memory.save(content="pizza reference chunk" + _PAD, title="Ref", type_="reference")
    monkeypatch.setenv("MEMO_REFERENCE_SEARCH_FLOOR", "1000000")

    res = mock_memory.chat_ask("¿qué se decidió sobre pizza?", k=10)

    types = {s["type"] for s in res["sources"]}
    assert "reference" not in types
    assert "decision" in types
