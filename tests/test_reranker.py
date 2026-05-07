"""Reranker tests — orchestration with stubs + real MLX smoke.

The orchestration tests monkeypatch `MLXReranker.score` so they don't
need to load the actual model, and exercise:

- ranking by descending score
- top_n truncation
- empty input
- body truncation before scoring
- score field overwrite on the returned records

The smoke test (`requires_mlx`) loads the real Qwen3-Reranker MLX
model and verifies the relative ordering on a known good/bad pair.
That is the only test that proves the wiring is end-to-end correct;
removing it would let "looks fine, totally broken" regressions through.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from memo.memory import MemoryRecord
from memo.reranker import MLXReranker

if TYPE_CHECKING:
    pass


def _rec(id_: str, title: str, body: str = "") -> MemoryRecord:
    return MemoryRecord(
        id=id_, path=f"x/{id_}.md", title=title, type="note",
        tags=[], created="2026-05-07", updated="2026-05-07",
        body=body, extra={}, score=None,
    )


def test_rerank_orders_by_descending_score(monkeypatch):
    rr = MLXReranker.__new__(MLXReranker)  # bypass __init__'s heavy fields
    rr._model = object()  # short-circuit _ensure_loaded
    rr._tokenizer = None
    rr._yes_id = 0
    rr._no_id = 1
    rr.task = "test"
    rr.max_seq_len = 4096

    scores_by_title = {"alpha": 0.9, "beta": 0.3, "gamma": 0.6}

    def _stub_score(self, query, doc):
        # Match by title prefix in the composed `title\n\nbody` string.
        for title, s in scores_by_title.items():
            if doc.startswith(title):
                return s
        return 0.0

    monkeypatch.setattr(MLXReranker, "score", _stub_score)
    monkeypatch.setattr(MLXReranker, "_ensure_loaded", lambda self: None)

    hits = [_rec("a", "alpha"), _rec("b", "beta"), _rec("g", "gamma")]
    out = rr.rerank("anything", hits)

    assert [h.id for h in out] == ["a", "g", "b"]
    assert [round(h.score, 2) for h in out] == [0.9, 0.6, 0.3]


def test_rerank_top_n_truncates_after_sort(monkeypatch):
    rr = MLXReranker.__new__(MLXReranker)
    rr._model = object()
    rr._tokenizer = None
    rr._yes_id = 0
    rr._no_id = 1
    rr.task = "t"
    rr.max_seq_len = 4096

    scores = {"a": 0.9, "b": 0.5, "c": 0.1}
    monkeypatch.setattr(
        MLXReranker, "score",
        lambda self, q, d: scores.get(d.split("\n")[0], 0.0),
    )
    monkeypatch.setattr(MLXReranker, "_ensure_loaded", lambda self: None)

    hits = [_rec("1", "a"), _rec("2", "b"), _rec("3", "c")]
    out = rr.rerank("q", hits, top_n=2)
    assert len(out) == 2
    assert [h.id for h in out] == ["1", "2"]


def test_rerank_empty_hits_returns_empty(monkeypatch):
    rr = MLXReranker.__new__(MLXReranker)
    monkeypatch.setattr(MLXReranker, "_ensure_loaded", lambda self: None)
    assert rr.rerank("q", [], top_n=5) == []


def test_rerank_truncates_body_before_scoring(monkeypatch):
    """Latency scales linearly with input length per pair. The body
    truncation is what keeps total latency in the 5-15s range on
    real corpora — without it, a single 50k-char informe blows the
    budget. This test pins the truncation length so future refactors
    don't accidentally remove it."""
    rr = MLXReranker.__new__(MLXReranker)
    rr._model = object()
    rr._tokenizer = None
    rr._yes_id = 0
    rr._no_id = 1
    rr.task = "t"
    rr.max_seq_len = 4096

    seen_docs: list[str] = []

    def _capture_score(self, query, doc):
        seen_docs.append(doc)
        return 0.5

    monkeypatch.setattr(MLXReranker, "score", _capture_score)
    monkeypatch.setattr(MLXReranker, "_ensure_loaded", lambda self: None)

    huge_body = "x" * 50_000
    hits = [_rec("1", "title", body=huge_body)]
    rr.rerank("q", hits, body_chars=1200)

    # Composed doc = "title\n\n" + body[:1200]; total length ≤ 1200 + 7.
    assert len(seen_docs) == 1
    assert len(seen_docs[0]) <= 1200 + len("title\n\n") + 5  # margin


def test_rerank_overwrites_score_on_returned_records(monkeypatch):
    """The returned records carry the rerank probability in `score`.
    Downstream callers (CLI display, recall-hook formatting) read
    `score` as the confidence — if we forgot to overwrite, they'd
    surface the stale RRF fusion score instead."""
    rr = MLXReranker.__new__(MLXReranker)
    rr._model = object()
    rr._tokenizer = None
    rr._yes_id = 0
    rr._no_id = 1
    rr.task = "t"
    rr.max_seq_len = 4096

    monkeypatch.setattr(MLXReranker, "score", lambda self, q, d: 0.42)
    monkeypatch.setattr(MLXReranker, "_ensure_loaded", lambda self: None)

    pre = replace(_rec("1", "x"), score=0.001)  # stale RRF score
    out = rr.rerank("q", [pre])
    assert out[0].score == pytest.approx(0.42)


# ── real MLX smoke ────────────────────────────────────────────────────────


@pytest.mark.requires_mlx
@pytest.mark.slow
def test_score_real_model_separates_relevant_from_irrelevant():
    """Loads the actual Qwen3-Reranker-0.6B-MLX-8Bit and verifies
    that a relevant doc scores >> an irrelevant one. This is the only
    test that catches end-to-end wiring breakage (token id mismatch,
    prompt-template drift, dtype regression)."""
    rr = MLXReranker()
    query = "favourite TV series"
    relevant = "List of best TV series: Breaking Bad, The Office, Dexter, True Detective."
    irrelevant = "Quarterly revenue report for Q3 2024 — supply chain logistics."

    p_rel = rr.score(query, relevant)
    p_irr = rr.score(query, irrelevant)

    assert 0.0 <= p_rel <= 1.0
    assert 0.0 <= p_irr <= 1.0
    # Looser margin than absolute thresholds so quantisation drift in
    # MLX-8bit weights doesn't fail the test on a model swap. The
    # ratio is what matters for ranking.
    assert p_rel > 0.5
    assert p_irr < 0.5
    assert p_rel - p_irr > 0.3
