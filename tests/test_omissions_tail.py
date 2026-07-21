"""Omissions tail: budget/filter-dropped qualifying hits leave a one-line trace."""

from __future__ import annotations

from types import SimpleNamespace


def _hit(i: int, body_len: int = 600):
    return SimpleNamespace(
        id=f"{i:08x}" + "0" * 24,
        title=f"Hit {i}",
        type="note",
        tags=[],
        created="2026-07-01T00:00:00",
        updated="2026-07-01T00:00:00",
        body="x" * body_len,
        score=0.8,
        extra={},
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
        [_hit(1, body_len=80)],
        [],
        turn=1,
        body_chars=200,
        token_budget=0,
        omitted=[_hit(8), _hit(9)],
    )
    assert "+2 more relevant" in out


def test_single_hit_over_budget_flag_on_does_not_steal_body(monkeypatch):
    """Flag ON + one hit whose body is partially rendered + no further/omitted hits.

    The reserve must be 0 (no tail will appear), so the body preview must be
    identical to flag-OFF, and no tail line should be present.
    """
    from memo.recall_logic import render_recall_context

    # Use a tight budget that forces the partial-body path but still leaves
    # room for the body preview when the reserve is 0.
    hit = _hit(1, body_len=300)
    token_budget = 60  # ~240 chars — enough for title + partial body, no tail needed

    monkeypatch.setenv("MEMO_RECALL_OMISSIONS_TAIL", "0")
    out_off = render_recall_context([hit], [], turn=1, body_chars=300, token_budget=token_budget)

    monkeypatch.setenv("MEMO_RECALL_OMISSIONS_TAIL", "1")
    out_on = render_recall_context([hit], [], turn=1, body_chars=300, token_budget=token_budget)

    # Body preview must be identical — reserve must not steal chars when no tail appears.
    assert out_on == out_off, (
        f"flag ON altered output despite no tail appearing.\nOFF: {out_off!r}\nON:  {out_on!r}"
    )
    assert "more relevant" not in out_on


def test_compact_budget_drop_appends_tail(monkeypatch):
    from memo.recall_logic import render_recall_compact

    monkeypatch.setenv("MEMO_RECALL_OMISSIONS_TAIL", "1")
    hits = [_hit(i, body_len=100) for i in range(1, 9)]
    out = render_recall_compact(hits, token_budget=40)
    assert "+  " not in out  # sanity: no stray formatting
    assert "more:" in out


def test_compact_tail_counts_hits_not_lines_with_dossier(monkeypatch):
    """'+N more' must count omitted HITS. With MEMO_HIT_DOSSIER on, each hit
    renders 2 lines, and the old len(hit_lines)-based count halved N."""
    import re

    from memo.recall_logic import render_recall_compact

    monkeypatch.setenv("MEMO_RECALL_OMISSIONS_TAIL", "1")
    monkeypatch.setenv("MEMO_HIT_DOSSIER", "1")
    hits = [_hit(i, body_len=100) for i in range(1, 7)]  # 6 hits
    out = render_recall_compact(hits, token_budget=80)

    m = re.search(r"\+(\d+) more:", out)
    assert m, f"expected an omissions tail in:\n{out}"
    rendered = len(re.findall(r"^\[[0-9a-f]{8}\]", out, flags=re.M))
    assert 0 < rendered < len(hits)  # the budget actually cut mid-list
    assert int(m.group(1)) == len(hits) - rendered
