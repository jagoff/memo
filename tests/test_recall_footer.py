"""Tests for Task 5: scaffolding compression — footer collapses past turn 1."""
from memo import recall_logic as rl


def test_footer_full_on_first_turn(monkeypatch):
    monkeypatch.delenv("MEMO_RECALL_FOOTER", raising=False)
    monkeypatch.delenv("MEMO_RECALL_FOOTER_AFTER", raising=False)
    first = rl._render_footer(turn=1)
    later = rl._render_footer(turn=5)
    # later turns collapse to a shorter footer than turn 1 (fewer chars)
    assert len(later) < len(first)
    assert later.endswith("</memo-recall>")


def test_explicit_footer_flag_still_wins(monkeypatch):
    monkeypatch.setenv("MEMO_RECALL_FOOTER", "none")
    # user forced none → both turns honor it (turn arg cannot override an explicit flag)
    assert rl._render_footer(turn=1) == ""
    assert rl._render_footer(turn=9) == ""
