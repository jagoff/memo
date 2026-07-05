"""MEMO_RECALL_EPISTEMIC_LABELS: render-layer type/date/trust prefix per hit."""
from __future__ import annotations

from types import SimpleNamespace


def _hit(**kw):
    base = dict(
        id="a1b2c3d4" * 4, title="Título", type="decision", tags=[],
        created="2026-03-05T10:00:00", updated="2026-03-07T10:00:00",
        body="cuerpo suficientemente largo para renderizar " * 3, score=0.8, extra={},
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_epistemic_label_variants():
    from memo.recall_logic import epistemic_label

    assert epistemic_label(_hit()) == "decision · 2026-03"
    assert epistemic_label(_hit(type="synthesis")) == "~inferred · 2026-03"
    assert epistemic_label(_hit(tags=["_uncertain"])) == "?unverified"


def test_render_context_prefixes_when_flag_on(monkeypatch):
    from memo.recall_logic import render_recall_context

    monkeypatch.setenv("MEMO_RECALL_EPISTEMIC_LABELS", "1")
    out = render_recall_context([_hit()], [], turn=1, body_chars=200, token_budget=0)
    assert "⟨decision · 2026-03⟩" in out


def test_render_context_unchanged_when_flag_off():
    from memo.recall_logic import render_recall_context

    out = render_recall_context([_hit()], [], turn=1, body_chars=200, token_budget=0)
    assert "⟨" not in out


def test_render_compact_prefixes_when_flag_on(monkeypatch):
    from memo.recall_logic import render_recall_compact

    monkeypatch.setenv("MEMO_RECALL_EPISTEMIC_LABELS", "1")
    out = render_recall_compact([_hit(type="synthesis")], token_budget=0)
    assert "⟨~inferred · 2026-03⟩" in out
