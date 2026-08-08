"""Rerank is rung one of the shed ladder: the most expensive stage, and the
one that already knows how to fall back (`RerankBudgetExceeded` -> RRF
order). Under a spent deadline it must not start at all; under a tight-but-
live deadline it must cap the reranker's own budget instead of floating a
fixed 20.0s.

`_ensure_reranker` is stubbed in every test here on purpose: `_rerank`
itself has no `reranker_enabled` gate (that check lives one layer up, at
`search()`'s call site) -- it calls `_ensure_reranker()` unconditionally,
which would otherwise try to construct a real `MLXReranker` (model load,
MLX/Apple-Silicon only) even for an empty hit list. A test that lets that
happen either skips silently on non-Apple hardware or hangs on a real model
load -- neither proves anything about the budget math this task adds.
"""

from __future__ import annotations

import time
from typing import Any

from memo.memory.record import MemoryRecord
from memo.search_deadline import Deadline


def _rec(id_: str) -> MemoryRecord:
    return MemoryRecord(
        id=id_,
        path=f"x/{id_}.md",
        title=id_,
        type="note",
        tags=[],
        created="2026-08-06",
        updated="2026-08-06",
        body="body",
        extra={},
        score=1.0,
    )


class _FakeReranker:
    """Records every `rerank()` call's kwargs (notably `budget_s`); passes
    hits through unchanged so fusion has something deterministic to sort."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def rerank(self, query: str, hits: list[MemoryRecord], **kwargs: Any) -> list[MemoryRecord]:
        self.calls.append(kwargs)
        return hits


def _stub_reranker(monkeypatch: Any, mem: Any, fake: _FakeReranker) -> list[str]:
    """Replace `_ensure_reranker` with one that records whether it was ever
    invoked, without touching MLX. Returns the list it appends "built" to."""
    built: list[str] = []

    def _ensure(self: Any) -> _FakeReranker:
        built.append("built")
        return fake

    monkeypatch.setattr(type(mem), "_ensure_reranker", _ensure)
    return built


def test_rerank_is_skipped_when_the_deadline_cannot_afford_it(tmp_cfg, monkeypatch) -> None:
    from memo.memory.facade import Memory

    mem = Memory(tmp_cfg)
    try:
        fake = _FakeReranker()
        built = _stub_reranker(monkeypatch, mem, fake)

        spent = Deadline.start(5)
        time.sleep(0.02)
        degraded: list[str] = []
        hits = [_rec("a"), _rec("b")]

        out = mem._rerank("q", hits, top_n=1, deadline=spent, degraded=degraded)

        assert built == [], "the reranker was built despite an expired deadline"
        assert fake.calls == [], "the reranker was invoked despite an expired deadline"
        assert degraded == ["rerank_skipped"]
        assert out == hits[:1], "skip path must fall back to the RRF order, truncated to top_n"
    finally:
        mem.close()


def test_rerank_caps_its_budget_to_the_deadline_when_tighter_than_the_default(
    tmp_cfg, monkeypatch
) -> None:
    from memo.memory.facade import Memory

    mem = Memory(tmp_cfg)
    try:
        fake = _FakeReranker()
        _stub_reranker(monkeypatch, mem, fake)
        monkeypatch.delenv("MEMO_RERANK_BUDGET_S", raising=False)

        # 5s of budget clears the afford() gate (COST_RERANK_MS = 4000ms) but
        # is well under the fixed 20.0s default -- the reranker's own budget
        # must be capped down to (approximately) what's left, not the fixed
        # default.
        deadline = Deadline.start(5000)
        degraded: list[str] = []

        mem._rerank("q", [_rec("a")], top_n=1, deadline=deadline, degraded=degraded)

        assert len(fake.calls) == 1, "reranker should have run once"
        budget_s = fake.calls[0]["budget_s"]
        assert budget_s < 20.0, "budget_s was not capped down from the fixed default"
        assert 4.5 < budget_s <= 5.0, f"expected ~5s from the 5000ms deadline, got {budget_s}"
        assert degraded == []
    finally:
        mem.close()


def test_rerank_leaves_the_default_budget_alone_when_the_deadline_has_more_room(
    tmp_cfg, monkeypatch
) -> None:
    from memo.memory.facade import Memory

    mem = Memory(tmp_cfg)
    try:
        fake = _FakeReranker()
        _stub_reranker(monkeypatch, mem, fake)
        monkeypatch.delenv("MEMO_RERANK_BUDGET_S", raising=False)

        # 30s of deadline room is more generous than the 20.0s fixed default --
        # capping is a min(), so it must never inflate the budget past 20.0.
        deadline = Deadline.start(30000)
        degraded: list[str] = []

        out = mem._rerank("q", [], top_n=3, deadline=deadline, degraded=degraded)

        assert fake.calls == [{"top_n": None, "budget_s": 20.0}]
        assert degraded == []
        assert out == []
    finally:
        mem.close()


def test_rerank_without_a_deadline_behaves_exactly_as_before(tmp_cfg, monkeypatch) -> None:
    """No deadline passed -> identical to pre-Task-2 behavior: unconditional
    `_ensure_reranker()` call, fixed 20.0s budget, no degraded reporting."""
    from memo.memory.facade import Memory

    mem = Memory(tmp_cfg)
    try:
        fake = _FakeReranker()
        built = _stub_reranker(monkeypatch, mem, fake)
        monkeypatch.delenv("MEMO_RERANK_BUDGET_S", raising=False)

        out = mem._rerank("q", [], top_n=3)

        assert built == ["built"], "no-deadline callers must still build the reranker"
        assert fake.calls == [{"top_n": None, "budget_s": 20.0}]
        assert out == []
    finally:
        mem.close()
