from __future__ import annotations

from dataclasses import replace
from typing import Any

from memo.memory import Memory
from memo.memory.record import MemoryRecord
from memo.memory.search_scoring_ops import _SearchScoringMixin
from memo.tiers import VerificationState


def _rec(id_: str, score: float, **extra: Any) -> MemoryRecord:
    return MemoryRecord(
        id=id_,
        path=f"{id_}.md",
        title=id_,
        type="note",
        tags=[],
        created="2026-01-01T00:00:00+00:00",
        updated="2026-01-01T00:00:00+00:00",
        body="stored body",
        extra=dict(extra),
        score=score,
    )


class _Harness(_SearchScoringMixin):
    pass


def test_apply_quality_rerank_is_flag_gated(monkeypatch) -> None:
    hits = [_rec("old", 0.9, superseded_by="new"), _rec("new", 0.7)]
    monkeypatch.setenv("MEMO_QUALITY_RERANK", "0")
    assert [h.id for h in _Harness()._apply_quality_rerank(hits)] == ["old", "new"]

    monkeypatch.setenv("MEMO_QUALITY_RERANK", "1")
    out = _Harness()._apply_quality_rerank(hits)
    assert [h.id for h in out] == ["new", "old"]


def test_apply_quality_rerank_boosts_verified(monkeypatch) -> None:
    monkeypatch.setenv("MEMO_QUALITY_RERANK", "1")
    verified = replace(_rec("verified", 0.7), verification_state=VerificationState.VERIFIED)
    out = _Harness()._apply_quality_rerank([_rec("plain", 0.72), verified])
    assert out[0].id == "verified"


def test_search_with_trace_quality_rerank_requires_explicit_opt_in_and_flag(
    mem_with_stub: Memory,
    monkeypatch,
) -> None:
    mem_with_stub.save(content="alpha result one", title="Alpha One")
    mem_with_stub.save(content="alpha result two", title="Alpha Two")
    calls: list[list[str]] = []

    def _fake_quality(results: list[MemoryRecord]) -> list[MemoryRecord]:
        calls.append([r.id for r in results])
        return list(reversed(results))

    monkeypatch.setattr(mem_with_stub, "_apply_quality_rerank", _fake_quality)
    monkeypatch.setenv("MEMO_QUALITY_RERANK", "1")

    baseline = mem_with_stub.search_with_trace("alpha", limit=2, mode="bm25")
    assert calls == []
    assert "quality_rerank" not in [item["stage"] for item in baseline["trace"]]

    opted_in = mem_with_stub.search_with_trace("alpha", limit=2, mode="bm25", quality_rerank=True)
    assert calls == [[h.id for h in baseline["hits"]]]
    assert [h.id for h in opted_in["hits"]] == list(reversed([h.id for h in baseline["hits"]]))
    assert "quality_rerank" in [item["stage"] for item in opted_in["trace"]]

    calls.clear()
    monkeypatch.setenv("MEMO_QUALITY_RERANK", "0")
    flagged_off = mem_with_stub.search_with_trace(
        "alpha",
        limit=2,
        mode="bm25",
        quality_rerank=True,
    )
    assert calls == []
    assert "quality_rerank" not in [item["stage"] for item in flagged_off["trace"]]


def test_build_ask_context_opts_into_quality_rerank(mem_with_stub: Memory, monkeypatch) -> None:
    seen: dict[str, Any] = {}

    def _fake_search(query: str, **kwargs: Any) -> list[MemoryRecord]:
        seen["query"] = query
        seen.update(kwargs)
        return [_rec("answer", 0.9)]

    monkeypatch.setattr(mem_with_stub, "search", _fake_search)
    _, sources, _, _ = mem_with_stub._build_ask_context(
        "what about alpha?",
        k=1,
        type_=None,
        snippet_chars=200,
        include_repos=False,
    )

    assert sources
    assert seen["query"] == "what about alpha?"
    assert seen["quality_rerank"] is True
