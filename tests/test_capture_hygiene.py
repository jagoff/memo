"""Capture hygiene trio — meta-commentary filter, intra-batch near-dup window,
type-confidence scoring (Q3 Mes 2).

Follows test_capture.py's conventions: stubbed embedder (no MLX), monkeypatched
`extract_insights` / `_passes_quality` / `find_near_duplicate` module globals so
`_extract_and_save` is exercised directly, isolated Config via tmp_cfg.
"""

from __future__ import annotations

import pytest

import memo.capture as capture_mod
from memo.capture import (
    dedupe_batch,
    is_meta_commentary,
    score_type_confidence,
    strip_meta_commentary,
)
from memo.config import Config
from memo.memory import Memory

# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def mem_with_stub(tmp_cfg: Config, monkeypatch) -> Memory:
    """Real Memory with a 64-dim hash-bucket embedder stub (same shape as
    test_capture.py's fixture)."""
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

    def _stub_embed_query(self, query: str):
        h = sum(ord(c) for c in query) % 64
        v = [0.0] * 64
        v[h] = 1.0
        return v

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _stub_embed)
    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed_query", _stub_embed_query)
    mem = Memory(cfg)
    yield mem
    mem.close()


def _cand(title: str, body: str, type_: str = "note", tags: list[str] | None = None) -> dict:
    return {"title": title, "body": body, "type": type_, "tags": tags or []}


def _wire(monkeypatch, candidates: list[dict]) -> None:
    """Route _extract_and_save straight at `candidates`, bypassing LLM/quality/
    store-dedup so only the hygiene passes under test are in play."""
    monkeypatch.setattr(capture_mod, "extract_insights", lambda *a, **kw: list(candidates))
    monkeypatch.setattr(capture_mod, "_passes_quality", lambda *a, **kw: True)
    monkeypatch.setattr(capture_mod, "find_near_duplicate", lambda *a, **kw: None)


# ── meta-commentary filter: pure function ────────────────────────────────────


def test_strip_meta_drops_spanish_narration_keeps_substance():
    text = "Voy a revisar el config del reranker. El root cause era un threshold alto."
    assert strip_meta_commentary(text) == "El root cause era un threshold alto."


def test_strip_meta_drops_english_narration_keeps_substance():
    text = "Let me check the config. The root cause was a stale cache in the reranker."
    assert strip_meta_commentary(text) == "The root cause was a stale cache in the reranker."


def test_strip_meta_all_narration_returns_empty():
    text = "Voy a correr los tests. Ahora voy a mirar el log. I'll update the config."
    assert strip_meta_commentary(text) == ""


def test_strip_meta_clean_text_roundtrips_byte_identical():
    text = "Decidimos usar MLX porque la latencia bajó 30%.\n\nEl fix fue truncar a 1200 chars."
    assert strip_meta_commentary(text) is text or strip_meta_commentary(text) == text


def test_strip_meta_multiline_drops_narration_lines():
    text = "Primero voy a mapear el módulo.\nBM25 folds diacritics via unicode61."
    assert strip_meta_commentary(text) == "BM25 folds diacritics via unicode61."


def test_strip_meta_no_false_positive_on_ill_formed():
    # "I'll" requires the apostrophe: "Ill-formed" is substance.
    text = "Ill-formed inputs crash the parser."
    assert strip_meta_commentary(text) == text


def test_strip_meta_no_false_positive_on_sure_enough():
    # "sure" needs the filler comma: "Sure enough, …" is a discovery.
    text = "Sure enough, the bug was in the tokenizer."
    assert strip_meta_commentary(text) == text


def test_strip_meta_filler_opener_keeps_the_substance():
    # Filler openers trim only the opener — the sentence body survives.
    text = "Okay, the fix is to use flock on the state file."
    assert strip_meta_commentary(text) == "the fix is to use flock on the state file."


def test_strip_meta_pure_filler_segment_dropped():
    assert strip_meta_commentary("Certainly!") == ""


def test_strip_meta_great_question_opener_keeps_answer():
    text = "Great question! The daemon path skips the cold load."
    assert strip_meta_commentary(text) == "The daemon path skips the cold load."


def test_strip_meta_keeps_i_will_preference():
    # Bare "I will …" is a preference statement, not process narration.
    text = "I will always use uv for this repo."
    assert strip_meta_commentary(text) == text


def test_is_meta_commentary_flags_narration_title():
    assert is_meta_commentary("Let me refactor the capture module")
    assert is_meta_commentary("  voy a revisar el hook  ")
    assert not is_meta_commentary("Reranker threshold 0.4 for hybrid mode")
    # Filler-opener titles are not narration — the candidate survives.
    assert not is_meta_commentary("Okay, fix for the flock race")


# ── confidence scoring: pure function ────────────────────────────────────────


def test_confidence_high_when_own_marker_present():
    conf = score_type_confidence("decision", "We decided to use MLX because latency dropped.")
    assert conf == 0.85


def test_confidence_very_high_with_multiple_own_markers():
    text = "We decided to use MLX. From now on this is the default embedder."
    assert score_type_confidence("decision", text) == 0.95


def test_confidence_mid_default_without_markers():
    # LLM-classified with zero corroborating markers → default mid.
    assert score_type_confidence("decision", "The sky is blue over the bay.") == 0.5


def test_confidence_low_when_markers_point_elsewhere():
    # Claimed decision, but the content carries a bug marker.
    assert score_type_confidence("decision", "The root cause was a stale cache.") == 0.35


def test_confidence_neutral_for_untyped_note():
    assert score_type_confidence("note", "Random observation about the weather.") == 0.6
    assert score_type_confidence("note", "The root cause was a stale cache.") == 0.4


def test_confidence_bounds():
    for type_ in ("decision", "preference", "bug", "fact", "note", "weird"):
        for text in ("", "decided to x. root cause y. i prefer z. turns out w."):
            assert 0.0 <= score_type_confidence(type_, text) <= 1.0


# ── intra-batch near-dup window: pure function ───────────────────────────────


class _VecStubMem:
    """Bare object exposing only what dedupe_batch touches."""

    class _E:
        def embed_query(self, text: str) -> list[float]:
            return [1.0, 0.0] if "SAMEFACT" in text else [0.0, 1.0]

    embedder = _E()


class _BrokenEmbedderMem:
    class _E:
        def embed_query(self, text: str) -> list[float]:
            raise RuntimeError("embedder cold")

    embedder = _E()


def test_dedupe_batch_collapses_retry_twins_keeps_longer():
    short = _cand("t", "SAMEFACT the fix was truncation")
    long = _cand("t", "SAMEFACT the fix was truncation to 1200 chars before rerank")
    distinct = _cand("other", "unrelated insight about BM25 tokenization")
    kept, dropped = dedupe_batch([short, long, distinct], _VecStubMem(), 0.85)
    assert dropped == 1
    assert kept == [long, distinct]  # better twin wins, order preserved


def test_dedupe_batch_confidence_beats_length():
    # Same embedding bucket; the typed candidate whose own marker matches
    # outranks the longer untyped one.
    weak = _cand("t", "SAMEFACT " + "filler words " * 10, type_="note")
    strong = _cand("t", "SAMEFACT we decided to truncate bodies", type_="decision")
    kept, dropped = dedupe_batch([weak, strong], _VecStubMem(), 0.85)
    assert dropped == 1
    assert kept == [strong]


def test_dedupe_batch_no_collapse_below_threshold():
    a = _cand("a", "SAMEFACT one thing")
    b = _cand("b", "completely different topic")
    kept, dropped = dedupe_batch([a, b], _VecStubMem(), 0.85)
    assert dropped == 0
    assert kept == [a, b]


def test_dedupe_batch_jaccard_fallback_when_embed_fails():
    twin1 = _cand("same title", "the fix was truncation before rerank")
    twin2 = _cand("same title", "the fix was truncation before rerank")
    other = _cand("other", "an entirely distinct BM25 diacritics note")
    kept, dropped = dedupe_batch([twin1, twin2, other], _BrokenEmbedderMem(), 0.85)
    assert dropped == 1
    assert kept == [twin1, other]


def test_dedupe_batch_single_candidate_passthrough():
    only = _cand("t", "body")
    kept, dropped = dedupe_batch([only], _BrokenEmbedderMem(), 0.85)
    assert kept == [only]
    assert dropped == 0


# ── _extract_and_save integration ────────────────────────────────────────────


def test_extract_and_save_drops_all_narration_candidate(mem_with_stub, monkeypatch):
    _wire(
        monkeypatch,
        [_cand("Plan de sesión", "Voy a revisar el config. Ahora voy a correr los tests.")],
    )
    out = capture_mod._extract_and_save(mem_with_stub, mem_with_stub.cfg, "u", "a")
    assert out["skipped_meta"] == 1
    assert out["saved"] == []
    assert out["candidates"] == 1  # extracted count is pre-hygiene


def test_extract_and_save_drops_narration_titled_candidate(mem_with_stub, monkeypatch):
    _wire(monkeypatch, [_cand("Let me check the reranker config", "Substantive body here.")])
    out = capture_mod._extract_and_save(mem_with_stub, mem_with_stub.cfg, "u", "a")
    assert out["skipped_meta"] == 1
    assert out["saved"] == []


def test_extract_and_save_strips_narration_but_saves_substance(mem_with_stub, monkeypatch):
    _wire(
        monkeypatch,
        [_cand("Root cause", "Let me check the config. The root cause was a stale cache.")],
    )
    saves: list[dict] = []
    orig_save = mem_with_stub.save
    monkeypatch.setattr(mem_with_stub, "save", lambda **kw: (saves.append(kw), orig_save(**kw))[1])
    out = capture_mod._extract_and_save(mem_with_stub, mem_with_stub.cfg, "u", "a")
    assert out["skipped_meta"] == 0
    assert len(out["saved"]) == 1
    assert saves[0]["content"] == "The root cause was a stale cache."


def test_extract_and_save_meta_filter_off_saves_narration_verbatim(mem_with_stub, monkeypatch):
    monkeypatch.setenv("MEMO_CAPTURE_META_FILTER", "0")
    body = "Voy a revisar el config. Ahora voy a correr los tests."
    _wire(monkeypatch, [_cand("Plan de sesión", body)])
    saves: list[dict] = []
    orig_save = mem_with_stub.save
    monkeypatch.setattr(mem_with_stub, "save", lambda **kw: (saves.append(kw), orig_save(**kw))[1])
    out = capture_mod._extract_and_save(mem_with_stub, mem_with_stub.cfg, "u", "a")
    assert out["skipped_meta"] == 0
    assert len(out["saved"]) == 1
    assert saves[0]["content"] == body  # today's behavior: untouched


def test_extract_and_save_collapses_batch_twins(mem_with_stub, monkeypatch):
    # Identical texts share a hash bucket in the stub embedder → cosine 1.0.
    twin = _cand("t", "el fix fue truncar el body antes del rerank")
    _wire(monkeypatch, [dict(twin), dict(twin)])
    out = capture_mod._extract_and_save(mem_with_stub, mem_with_stub.cfg, "u", "a")
    assert out["skipped_batch_dup"] == 1
    assert len(out["saved"]) == 1


def test_extract_and_save_batch_dedup_off_saves_both(mem_with_stub, monkeypatch):
    monkeypatch.setenv("MEMO_CAPTURE_BATCH_DEDUP", "0")
    twin = _cand("t", "el fix fue truncar el body antes del rerank")
    _wire(monkeypatch, [dict(twin), dict(twin)])
    out = capture_mod._extract_and_save(mem_with_stub, mem_with_stub.cfg, "u", "a")
    assert out["skipped_batch_dup"] == 0
    assert len(out["saved"]) == 2  # today's behavior


def test_extract_and_save_stamps_confidence_in_extra(mem_with_stub, monkeypatch):
    _wire(
        monkeypatch,
        [_cand("Stale cache in reranker", "The root cause was a stale cache.", type_="bug")],
    )
    out = capture_mod._extract_and_save(mem_with_stub, mem_with_stub.cfg, "u", "a")
    assert len(out["saved"]) == 1
    rec = mem_with_stub.get(out["saved"][0])
    assert rec is not None
    # bug marker corroborates the claimed type → 0.85
    assert rec.extra.get("capture_confidence") == 0.85


def test_extract_and_save_tags_uncertain_below_threshold(mem_with_stub, monkeypatch):
    monkeypatch.setenv("MEMO_CAPTURE_MIN_CONFIDENCE", "0.99")
    _wire(monkeypatch, [_cand("Weak claim", "Some untyped observation body.", type_="decision")])
    out = capture_mod._extract_and_save(mem_with_stub, mem_with_stub.cfg, "u", "a")
    assert out["uncertain"] == 1
    rec = mem_with_stub.get(out["saved"][0])
    assert rec is not None
    assert "_uncertain" in rec.tags
    assert rec.extra.get("capture_confidence") == 0.5


def test_extract_and_save_threshold_off_never_tags_uncertain(mem_with_stub, monkeypatch):
    # Default MEMO_CAPTURE_MIN_CONFIDENCE=0.0 → gating off, still saved untagged.
    _wire(monkeypatch, [_cand("Weak claim", "Some untyped observation body.", type_="decision")])
    out = capture_mod._extract_and_save(mem_with_stub, mem_with_stub.cfg, "u", "a")
    assert out["uncertain"] == 0
    rec = mem_with_stub.get(out["saved"][0])
    assert rec is not None
    assert "_uncertain" not in rec.tags


def test_extract_and_save_flags_off_matches_previous_behavior(mem_with_stub, monkeypatch):
    """META_FILTER=0 + BATCH_DEDUP=0 + MIN_CONFIDENCE unset → same saves as
    before the hygiene trio: narration untouched, twins both saved, no
    _uncertain tag, candidate tags/type/title/content passed through verbatim.
    (Only delta vs pre-trio: extra gains the capture_confidence stamp.)"""
    monkeypatch.setenv("MEMO_CAPTURE_META_FILTER", "0")
    monkeypatch.setenv("MEMO_CAPTURE_BATCH_DEDUP", "0")
    narration = _cand("Plan", "Voy a revisar el config y correr los tests.", tags=["plan"])
    twin = _cand("t", "el fix fue truncar el body", type_="fact", tags=["fix"])
    _wire(monkeypatch, [dict(narration), dict(twin), dict(twin)])
    saves: list[dict] = []
    orig_save = mem_with_stub.save
    monkeypatch.setattr(mem_with_stub, "save", lambda **kw: (saves.append(kw), orig_save(**kw))[1])
    out = capture_mod._extract_and_save(mem_with_stub, mem_with_stub.cfg, "u", "a")
    assert out["skipped_meta"] == 0
    assert out["skipped_batch_dup"] == 0
    assert out["uncertain"] == 0
    assert len(out["saved"]) == 3
    assert [s["content"] for s in saves] == [narration["body"], twin["body"], twin["body"]]
    assert [s["title"] for s in saves] == [narration["title"], twin["title"], twin["title"]]
    assert [s["type_"] for s in saves] == ["note", "fact", "fact"]
    assert [s["tags"] for s in saves] == [["plan"], ["fix"], ["fix"]]
    assert all("_uncertain" not in s["tags"] for s in saves)
