"""The shed ladder in `Memory.search`: drop optional stages to stay inside the
wall-clock budget, and report every stage dropped.

The assertions are on the DEGRADATION DECISION, not on wall-clock racing: each
test picks a budget on one side of a documented `COST_*` constant and asserts
which stages ran, so the outcome is deterministic in CI. Nothing here sleeps to
provoke a rung.

Two failure modes this file exists to catch:
  1. a stage shed for budget reasons that does NOT report itself (a silent
     degradation is the bug the whole plan exists to prevent);
  2. a stage reported as budget-degraded that was never going to run anyway
     (its flag was off, or the mode does not use it) -- a false alarm is just
     as dishonest.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from memo.config import Config

# Budgets chosen against the COST_* constants in `memo.search_deadline`:
#   COST_RERANK_MS 4000 > COST_EMBED_MS 2000 > COST_EXPANSION_MS 1500
#     > COST_GRAPH_SIGNAL_MS 500
# HyDE is an add-on ON TOP of the embed it feeds, so it costs the SUM (3500).
_ONLY_RERANK_UNAFFORDABLE = 3900  # > 3500 (embed+hyde), < 4000 (rerank)
_ONLY_EXPANSION_UNAFFORDABLE = 3000  # > 2000 (embed), < 3500 (embed+hyde)
_EMBED_UNAFFORDABLE = 1000  # < 2000 (embed)
_GRAPH_UNAFFORDABLE = 400  # < 500 (graph signal)
_GENEROUS = 30000


def _stub_embed(self: Any, inputs: Any) -> list[list[float]]:
    out = []
    for s in inputs:
        h = sum(ord(c) for c in s) % 4
        v = [0.0] * 4
        v[h] = 1.0
        out.append(v)
    return out


@pytest.fixture
def mem(tmp_cfg: Config, monkeypatch: pytest.MonkeyPatch):
    """Seeded `Memory` with a deterministic 4-dim embedder.

    The store must be non-empty: `search()` short-circuits to `[]` (before any
    stage, therefore before any rung) when `store.count() == 0`, so an empty
    fixture would make every ladder assertion vacuously pass.
    """
    from memo.memory.facade import Memory

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _stub_embed)
    cfg = Config(
        data_dir=tmp_cfg.data_dir,
        vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir,
        embedder_dims=4,
        reranker_enabled=False,
    )
    m = Memory(cfg)
    m.save(content="ladder budget marker note one", title="Ladder Budget One", type_="note")
    m.save(content="ladder budget marker note two", title="Ladder Budget Two", type_="note")
    yield m
    m.close()


@pytest.fixture
def mem_reranking(tmp_cfg: Config, monkeypatch: pytest.MonkeyPatch):
    """Same, but with the cross-encoder enabled and stubbed.

    `reranker_enabled=True` alone would drag a real `MLXReranker` model load
    into the test (Apple-Silicon-only, seconds), so `_ensure_reranker` is
    replaced by a recorder. `built` is the evidence of whether rung one fired:
    a skipped rerank must never even construct the reranker.
    """
    from memo.memory.facade import Memory

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _stub_embed)
    cfg = Config(
        data_dir=tmp_cfg.data_dir,
        vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir,
        embedder_dims=4,
        reranker_enabled=True,
    )
    m = Memory(cfg)
    m.save(content="ladder budget marker note one", title="Ladder Budget One", type_="note")
    m.save(content="ladder budget marker note two", title="Ladder Budget Two", type_="note")

    built: list[str] = []

    class _FakeReranker:
        def rerank(self, query: str, hits: list[Any], **kwargs: Any) -> list[Any]:
            return hits

    def _ensure(self: Any) -> _FakeReranker:
        built.append("built")
        return _FakeReranker()

    monkeypatch.setattr(type(m), "_ensure_reranker", _ensure)
    m.rerank_built = built  # type: ignore[attr-defined]
    yield m
    m.close()


def _count_embed_queries(monkeypatch: pytest.MonkeyPatch, m: Any) -> list[str]:
    """Record every `embed_query` the search actually issues."""
    calls: list[str] = []
    real = m.embedder.embed_query

    def _spy(query: str) -> list[float]:
        calls.append(query)
        return real(query)

    monkeypatch.setattr(m.embedder, "embed_query", _spy)
    return calls


def _stub_hyde(monkeypatch: pytest.MonkeyPatch, m: Any) -> list[str]:
    """Replace the HyDE generator (a real LLM call) with a recorder."""
    calls: list[str] = []

    def _generate(self: Any, query: str) -> str:
        calls.append(query)
        return f"hypothetical answer for {query}"

    monkeypatch.setattr(type(m), "_generate_hyde_document", _generate)
    return calls


def _enable_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMO_GRAPH_SIGNAL_ENABLED", "1")
    monkeypatch.setenv("MEMO_GRAPH_PROJECTION_ENABLED", "1")


def _spy_projection(monkeypatch: pytest.MonkeyPatch, m: Any) -> list[int]:
    """Record whether the graph stage got as far as reading the projection —
    the first expensive thing it does after the enable checks."""
    calls: list[int] = []
    real = m.graph.projection.read_model

    def _spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(m.graph.projection, "read_model", _spy)
    return calls


# -- the healthy default ----------------------------------------------------


def test_a_generous_budget_sheds_nothing(mem) -> None:
    degraded: list[str] = []
    hits = mem.search(
        "ladder budget marker", mode="hybrid", _budget_ms=_GENEROUS, _degraded=degraded
    )
    assert degraded == []
    assert hits, "a generous budget must still return the seeded memories"


def test_the_default_budget_never_fires_on_a_healthy_search(mem) -> None:
    """No `_budget_ms` at all: the registered 30s default is far above any
    healthy search, so the ladder must stay silent."""
    degraded: list[str] = []
    mem.search("ladder budget marker", mode="hybrid", _degraded=degraded)
    assert degraded == []


def test_the_ladder_does_not_change_results_it_does_not_shed(mem) -> None:
    """Same query, with and without the new out-parameters: identical order."""
    plain = mem.search("ladder budget marker", mode="hybrid")
    with_ladder = mem.search(
        "ladder budget marker", mode="hybrid", _budget_ms=_GENEROUS, _degraded=[]
    )
    assert [r.id for r in plain] == [r.id for r in with_ladder]
    assert plain, "the fixture must return hits or this proves nothing"


# -- rung one: rerank -------------------------------------------------------


def test_rerank_alone_is_shed_when_only_the_reranker_does_not_fit(mem_reranking) -> None:
    degraded: list[str] = []
    mem_reranking.search(
        "ladder budget marker",
        mode="hybrid",
        _budget_ms=_ONLY_RERANK_UNAFFORDABLE,
        _degraded=degraded,
    )
    assert degraded == ["rerank_skipped"], (
        "3900ms affords the embed (2000) and HyDE-plus-embed (3500) but not the "
        "rerank (4000) — only rung one should fire"
    )
    assert mem_reranking.rerank_built == [], "the reranker was built despite being shed"


def test_rerank_is_not_reported_when_the_reranker_is_disabled(mem) -> None:
    """`mem`'s config has `reranker_enabled=False`: the stage was never going
    to run, so an exhausted budget must not claim to have shed it."""
    degraded: list[str] = []
    mem.search(
        "ladder budget marker", mode="hybrid", _budget_ms=_EMBED_UNAFFORDABLE, _degraded=degraded
    )
    assert "rerank_skipped" not in degraded


# -- rung two: query expansion (HyDE) ---------------------------------------


def test_expansion_is_shed_before_the_embed_it_feeds(mem, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMO_HYDE_ENABLED", "1")
    hyde = _stub_hyde(monkeypatch, mem)
    embeds = _count_embed_queries(monkeypatch, mem)

    degraded: list[str] = []
    mem.search(
        "ladder budget marker",
        mode="hybrid",
        _budget_ms=_ONLY_EXPANSION_UNAFFORDABLE,
        _degraded=degraded,
    )

    assert degraded == ["expansion_skipped"]
    assert hyde == [], "the HyDE LLM call ran despite an unaffordable expansion"
    assert embeds, "shedding the expansion must NOT shed the embed it feeds"


def test_a_slow_expansion_sheds_the_embed_it_was_about_to_feed(
    mem, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The embed's affordability must be re-checked AFTER HyDE, never snapshotted
    before it.

    `_generate_hyde_document` is an un-timeboxed `chat.chat()` — the
    `ChatBackend` protocol takes no deadline — so it can burn arbitrary
    wall-clock. A snapshot taken before it would be structurally guaranteed
    True on this branch (rung two only lets HyDE run when
    `afford(EXPANSION + EMBED)` held, which implies `afford(EMBED)`), making
    rung four unreachable exactly when the budget was actually spent: the
    300s-hang scenario, straight past the guard meant to stop it.

    Every other test here stubs HyDE as instantaneous, so no real time passes
    between a snapshot and its use — the one condition under which that bug is
    invisible. This test burns real wall-clock inside HyDE on purpose.

    The `COST_*` constants are scaled down to keep the test sub-second. The
    real 1500/2000 pair would force a >1.5s sleep (the gap between
    `EXPANSION + EMBED` and `EMBED` alone is exactly what must elapse), which
    belongs behind the `slow` marker and therefore outside the default gate.
    The structure under test — where the check is taken — is what the scaling
    preserves; the constants are documented estimates anyway.
    """
    monkeypatch.setenv("MEMO_HYDE_ENABLED", "1")
    monkeypatch.setattr("memo.memory.search_ops.COST_EXPANSION_MS", 200.0)
    monkeypatch.setattr("memo.memory.search_ops.COST_EMBED_MS", 500.0)

    slow_hyde: list[str] = []

    def _slow(self: Any, query: str) -> str:
        slow_hyde.append(query)
        time.sleep(0.6)
        return f"hypothetical answer for {query}"

    monkeypatch.setattr(type(mem), "_generate_hyde_document", _slow)
    embeds = _count_embed_queries(monkeypatch, mem)

    # 1000ms: affords HyDE-plus-embed (700) at entry, so rung two lets HyDE
    # run; the 600ms HyDE call then leaves ~400ms, which no longer affords the
    # embed (500).
    degraded: list[str] = []
    hits = mem.search("ladder budget marker", mode="hybrid", _budget_ms=1000, _degraded=degraded)

    assert slow_hyde, "HyDE must have run, or this proves nothing about the gap after it"
    assert degraded == ["embed_skipped_bm25_only"], (
        "the embed was not re-checked after HyDE spent the budget"
    )
    assert embeds == [], "the embedder ran on a budget HyDE had already spent"
    assert hits, "the shed must still fall back to BM25, not to nothing"


def test_expansion_is_not_reported_when_hyde_is_off(mem, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMO_HYDE_ENABLED", "0")
    degraded: list[str] = []
    mem.search(
        "ladder budget marker",
        mode="hybrid",
        _budget_ms=_ONLY_EXPANSION_UNAFFORDABLE,
        _degraded=degraded,
    )
    assert degraded == [], "a stage that was flag-off was reported as budget-degraded"


def test_expansion_is_shed_along_with_the_embed_when_neither_fits(
    mem, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MEMO_HYDE_ENABLED", "1")
    hyde = _stub_hyde(monkeypatch, mem)

    degraded: list[str] = []
    mem.search(
        "ladder budget marker", mode="hybrid", _budget_ms=_EMBED_UNAFFORDABLE, _degraded=degraded
    )

    assert degraded == ["expansion_skipped", "embed_skipped_bm25_only"]
    assert hyde == []


# -- rung three: graph signal -----------------------------------------------


def test_graph_signal_is_shed_when_it_does_not_fit(mem, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_graph(monkeypatch)
    projection = _spy_projection(monkeypatch, mem)

    degraded: list[str] = []
    trace: list[dict[str, Any]] = []
    # bm25 mode has no embed, no expansion and no rerank stage, so this budget
    # isolates rung three: the only thing 400ms cannot afford is the graph.
    mem.search(
        "ladder budget marker",
        mode="bm25",
        _budget_ms=_GRAPH_UNAFFORDABLE,
        _degraded=degraded,
        _trace=trace,
    )

    assert degraded == ["graph_signal_skipped"]
    assert projection == [], "the graph projection was read despite being shed"
    graph_traces = [t for t in trace if t["stage"] == "graph_signal"]
    assert graph_traces and graph_traces[0]["skipped"] == "budget"


def test_graph_signal_is_not_reported_when_the_flag_is_off(mem) -> None:
    degraded: list[str] = []
    mem.search(
        "ladder budget marker", mode="bm25", _budget_ms=_GRAPH_UNAFFORDABLE, _degraded=degraded
    )
    assert degraded == [], "a flag-off graph stage was reported as budget-degraded"


def test_graph_signal_runs_when_the_budget_affords_it(mem, monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_graph(monkeypatch)
    projection = _spy_projection(monkeypatch, mem)

    degraded: list[str] = []
    mem.search("ladder budget marker", mode="bm25", _budget_ms=_GENEROUS, _degraded=degraded)

    assert degraded == []
    assert projection == [1], "the graph stage did not run on a generous budget"


# -- rung four: the embed ---------------------------------------------------


def test_the_embed_is_shed_onto_the_bm25_only_path(mem, monkeypatch: pytest.MonkeyPatch) -> None:
    embeds = _count_embed_queries(monkeypatch, mem)

    degraded: list[str] = []
    hits = mem.search(
        "ladder budget marker", mode="hybrid", _budget_ms=_EMBED_UNAFFORDABLE, _degraded=degraded
    )

    assert degraded == ["embed_skipped_bm25_only"]
    assert embeds == [], "the embedder ran despite an unaffordable embed"
    assert hits, "shedding the embed must fall back to BM25, not to nothing"


def test_an_embedder_failure_is_reported_distinctly_from_a_budget_shed(
    mem, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The accidental BM25-only path (the embedder blew up) and the deliberate
    one (the budget shed it) must be DISTINGUISHABLE — and both must be
    reported. `degraded` is the only machine-readable signal that a "hybrid"
    result was BM25-only; leaving the crash path empty made an embedder outage
    look like an ordinary search with worse answers (the dominant failure mode
    on a machine whose recall daemon is down)."""

    def _boom(query: str) -> list[float]:
        raise RuntimeError("embedder unavailable")

    monkeypatch.setattr(mem.embedder, "embed_query", _boom)

    degraded: list[str] = []
    hits = mem.search(
        "ladder budget marker", mode="hybrid", _budget_ms=_GENEROUS, _degraded=degraded
    )

    assert degraded == ["embed_failed_bm25_only"]
    assert "embed_skipped_bm25_only" not in degraded, "a crash is not a budget shed"
    assert hits, "an embedder crash must still fall back to BM25, not to nothing"
    assert hits, "the accidental BM25-only fallback should still answer"


# -- the off switch ---------------------------------------------------------


def test_zero_budget_disables_the_whole_ladder(
    mem_reranking, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_budget_ms=0` means unlimited: every rung must be a total no-op, even
    with every guarded stage enabled."""
    monkeypatch.setenv("MEMO_HYDE_ENABLED", "1")
    _enable_graph(monkeypatch)
    hyde = _stub_hyde(monkeypatch, mem_reranking)
    embeds = _count_embed_queries(monkeypatch, mem_reranking)
    projection = _spy_projection(monkeypatch, mem_reranking)

    degraded: list[str] = []
    mem_reranking.search("ladder budget marker", mode="hybrid", _budget_ms=0, _degraded=degraded)

    assert degraded == []
    assert hyde, "HyDE was shed under an unlimited budget"
    assert embeds, "the embed was shed under an unlimited budget"
    assert projection == [1], "the graph stage was shed under an unlimited budget"
    assert mem_reranking.rerank_built == ["built"], "rerank was shed under an unlimited budget"


def test_search_without_the_out_parameter_is_unchanged(tmp_cfg) -> None:
    """The empty-store short-circuit still returns `[]` and never touches the
    ladder — the out-parameters are optional, not required."""
    from memo.memory.facade import Memory

    m = Memory(tmp_cfg)
    try:
        assert m.search("anything", mode="bm25") == []
    finally:
        m.close()
