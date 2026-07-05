"""Omissions tail: budget/filter-dropped qualifying hits leave a one-line trace."""
from __future__ import annotations

from types import SimpleNamespace


def _hit(i: int, body_len: int = 600):
    return SimpleNamespace(
        id=f"{i:08x}" + "0" * 24, title=f"Hit {i}", type="note", tags=[],
        created="2026-07-01T00:00:00", updated="2026-07-01T00:00:00",
        body="x" * body_len, score=0.8, extra={},
    )


def test_context_budget_drop_appends_tail(monkeypatch):
    from memo.recall_logic import render_recall_context

    monkeypatch.setenv("MEMO_RECALL_OMISSIONS_TAIL", "1")
    hits = [_hit(1), _hit(2), _hit(3), _hit(4)]
    out = render_recall_context(hits, [], turn=1, body_chars=500, token_budget=90)
    assert "more relevant" in out
    assert "/memo get" in out


def test_context_tail_off_by_default():
    from memo.recall_logic import render_recall_context

    hits = [_hit(1), _hit(2), _hit(3), _hit(4)]
    out = render_recall_context(hits, [], turn=1, body_chars=500, token_budget=80)
    assert "more relevant" not in out


def test_context_pre_dropped_omitted_param_counts(monkeypatch):
    from memo.recall_logic import render_recall_context

    monkeypatch.setenv("MEMO_RECALL_OMISSIONS_TAIL", "1")
    out = render_recall_context(
        [_hit(1, body_len=80)], [], turn=1, body_chars=200, token_budget=0,
        omitted=[_hit(8), _hit(9)],
    )
    assert "+2 more relevant" in out


def test_compact_budget_drop_appends_tail(monkeypatch):
    from memo.recall_logic import render_recall_compact

    monkeypatch.setenv("MEMO_RECALL_OMISSIONS_TAIL", "1")
    hits = [_hit(i, body_len=100) for i in range(1, 9)]
    out = render_recall_compact(hits, token_budget=40)
    assert "+  " not in out  # sanity: no stray formatting
    assert "more:" in out
