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

import sys
from dataclasses import replace
from types import ModuleType
from typing import TYPE_CHECKING

import pytest

from memo.config import Config
from memo.memory import Memory, MemoryRecord
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

    def _stub_score_many(self, query, docs):
        # Match each composed `title\n\nbody` string by its title prefix.
        out = []
        for doc in docs:
            s = 0.0
            for title, val in scores_by_title.items():
                if doc.startswith(title):
                    s = val
                    break
            out.append(s)
        return out

    monkeypatch.setattr(MLXReranker, "score_many", _stub_score_many)
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
        MLXReranker, "score_many",
        lambda self, q, docs: [scores.get(d.split("\n")[0], 0.0) for d in docs],
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

    def _capture_score_many(self, query, docs):
        seen_docs.extend(docs)
        return [0.5 for _ in docs]

    monkeypatch.setattr(MLXReranker, "score_many", _capture_score_many)
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

    monkeypatch.setattr(MLXReranker, "score_many", lambda self, q, docs: [0.42 for _ in docs])
    monkeypatch.setattr(MLXReranker, "_ensure_loaded", lambda self: None)

    pre = replace(_rec("1", "x"), score=0.001)  # stale RRF score
    out = rr.rerank("q", [pre])
    assert out[0].score == pytest.approx(0.42)


def test_score_delegates_to_score_many(monkeypatch):
    """`score` is a thin wrapper over `score_many` (a 1-element batch) so
    single-pair callers and the batched path share one implementation."""
    rr = MLXReranker.__new__(MLXReranker)
    captured: dict[str, object] = {}

    def _stub_score_many(self, query, docs):
        captured["query"] = query
        captured["docs"] = list(docs)
        return [0.73 for _ in docs]

    monkeypatch.setattr(MLXReranker, "score_many", _stub_score_many)
    assert rr.score("q", "d") == pytest.approx(0.73)
    assert captured == {"query": "q", "docs": ["d"]}


def test_score_many_empty_returns_empty(monkeypatch):
    """No docs → no forward pass, empty list (before touching MLX)."""
    rr = MLXReranker.__new__(MLXReranker)
    rr._model = object()
    monkeypatch.setattr(MLXReranker, "_ensure_loaded", lambda self: None)
    assert rr.score_many("q", []) == []


def test_reranker_revision_downloads_pinned_snapshot(monkeypatch):
    calls: dict[str, str] = {}

    hf = ModuleType("huggingface_hub")

    def snapshot_download(*, repo_id: str, revision: str) -> str:
        calls["repo_id"] = repo_id
        calls["revision"] = revision
        return "/tmp/pinned-reranker"

    hf.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hf)

    mlx_lm = ModuleType("mlx_lm")

    class _Tokenizer:
        def convert_tokens_to_ids(self, token: str) -> int:
            return {"yes": 1, "no": 2}[token]

    def load(path: str) -> tuple[object, _Tokenizer]:
        calls["load_path"] = path
        return object(), _Tokenizer()

    mlx_lm.load = load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)

    rr = MLXReranker(
        model_path="vserifsaglam/Qwen3-Reranker-4B-4bit-MLX",
        revision="9655b27c01d2ff1c49f7e672a04b70d630161b46",
    )
    rr._ensure_loaded()

    assert calls == {
        "repo_id": "vserifsaglam/Qwen3-Reranker-4B-4bit-MLX",
        "revision": "9655b27c01d2ff1c49f7e672a04b70d630161b46",
        "load_path": "/tmp/pinned-reranker",
    }


def test_memory_ensure_reranker_rejects_missing_local_model_path(tmp_path, monkeypatch):
    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", lambda self, inputs: [[1.0, 0.0, 0.0, 0.0] for _ in inputs])
    cfg = Config(
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        embedder_dims=4,
        reranker_enabled=True,
        reranker_model=str(tmp_path / "missing-reranker"),
    )
    mem = Memory(cfg)
    try:
        with pytest.raises(FileNotFoundError) as excinfo:
            mem._ensure_reranker()
    finally:
        mem.close()

    message = str(excinfo.value)
    assert "reranker model path does not exist" in message
    assert str(tmp_path / "missing-reranker") in message


def test_rerank_hits_logs_score_failures(caplog):
    class _FailingReranker:
        def score(self, query: str, doc: str) -> float:
            raise RuntimeError("score boom")

    class _Memory:
        cfg = type("Cfg", (), {
            "reranker_enabled": True,
            "reranker_model": "stub-reranker",
            "reranker_revision": None,
        })()

        def _ensure_reranker(self):
            return _FailingReranker()

    hits = [{"id": "a", "title": "Alpha", "body": "body"}]
    with caplog.at_level("ERROR", logger="memo.memory.record"):
        out = Memory.rerank_hits(_Memory(), "query", hits)

    assert out == [{"id": "a", "title": "Alpha", "body": "body", "rerank_score": 0.0}]
    assert "reranker score failed" in caplog.text
    assert "stub-reranker" in caplog.text


# ── real MLX smoke ────────────────────────────────────────────────────────


@pytest.mark.requires_mlx
@pytest.mark.slow
def test_score_real_model_separates_relevant_from_irrelevant():
    """Loads the pinned Qwen3-Reranker-4B MLX model and verifies
    that a relevant doc scores >> an irrelevant one. This is the only
    test that catches end-to-end wiring breakage (token id mismatch,
    prompt-template drift, dtype regression)."""
    rr = MLXReranker(
        model_path="vserifsaglam/Qwen3-Reranker-4B-4bit-MLX",
        revision="9655b27c01d2ff1c49f7e672a04b70d630161b46",
    )
    query = "favourite TV series"
    relevant = "List of best TV series: Breaking Bad, The Office, Dexter, True Detective."
    irrelevant = "Quarterly revenue report for Q3 2024 — supply chain logistics."

    p_rel = rr.score(query, relevant)
    p_irr = rr.score(query, irrelevant)

    assert 0.0 <= p_rel <= 1.0
    assert 0.0 <= p_irr <= 1.0
    # The 4B MLX quantization is conservative in absolute probability,
    # so the separation is what matters for ranking.
    assert p_rel > 0.1
    assert p_irr < 0.05
    assert p_rel - p_irr > 0.1


@pytest.mark.requires_mlx
@pytest.mark.slow
def test_score_many_preserves_topk_set_vs_per_pair():
    """Batched `score_many` must return the same top-K *set* as per-pair `score`.

    `score_many` right-pads varying-length pairs into one forward and reads
    each row at its *last real token*. Under causal attention padding cannot
    leak into that position — a doc scored alone vs alone+trailing-pad is
    bit-identical (verified directly), and the head-slice projection is exact
    (linear). What is NOT bit-exact is the transformer *body*: the quantized
    GEMM is mildly batch-size dependent, so absolute P(yes) can drift ~0.03
    near the 0.5 boundary versus the B=1 path. That can swap two near-tied
    candidates *within* the result set, but it does not change which docs are
    retrieved — the top-K set is preserved, which is rerank's actual contract.
    This pins top-K set parity (what matters for retrieval) and bounds the
    drift, so a real gather/index regression — which scrambles scores
    wholesale — still trips."""
    rr = MLXReranker(
        model_path="vserifsaglam/Qwen3-Reranker-4B-4bit-MLX",
        revision="9655b27c01d2ff1c49f7e672a04b70d630161b46",
    )
    query = "favourite TV series"
    docs = [
        "List of best TV series: Breaking Bad, The Office, Dexter.",
        "Quarterly revenue report for Q3 2024 — supply chain logistics.",
        "My short notes on cooking pasta and a few Italian recipes I like.",
        "TV shows I love: True Detective, Better Call Saul, and The Wire.",
        "Grocery list: milk, eggs, bread, coffee, olive oil.",
        "Top streaming dramas this year ranked by critics and viewers.",
        "Sitcoms worth rewatching: Friends, Seinfeld, Parks and Rec.",
        "Tax filing deadlines and deductions for freelancers.",
        "Best HBO shows of the decade according to fans.",
        "Netflix originals I want to binge over the holidays.",
    ]
    per_pair = [rr.score(query, d) for d in docs]  # each is a B=1 batch
    batched = rr.score_many(query, docs)  # one padded multi-row forward

    assert len(batched) == len(docs)
    rank_pp = sorted(range(len(docs)), key=lambda i: per_pair[i], reverse=True)
    rank_bt = sorted(range(len(docs)), key=lambda i: batched[i], reverse=True)
    # Top-K retrieved set is identical (near-tied internal order may differ).
    for k in (3, 5):
        assert set(rank_bt[:k]) == set(rank_pp[:k])
    assert rank_bt[0] == rank_pp[0]  # the single best doc never moves
    # Bounded drift: a real gather/index bug would scramble scores far past this.
    assert max(abs(a - b) for a, b in zip(per_pair, batched, strict=True)) < 0.05
