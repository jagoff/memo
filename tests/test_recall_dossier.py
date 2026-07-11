"""hit-dossier: compact per-hit trust line in recall render (default off)."""

from __future__ import annotations

from types import SimpleNamespace

from memo import recall_logic


def _hit(**kw):
    base = dict(
        id="a1b2c3d4e5",
        title="Port is 8765",
        type="fact",
        tags=["project:memo"],
        updated="2026-07-10",
        score=0.83,
        extra={},
        verification_state=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_trust_dossier_renders_type_date_confband():
    line = recall_logic.trust_dossier(_hit(), None)
    assert "fact" in line
    assert "2026-07" in line
    assert "conf" in line.lower()


def test_trust_dossier_marks_disputed():
    line = recall_logic.trust_dossier(_hit(), ["ffff0000"])
    assert "⚔" in line
    assert "ffff0000" in line


def test_trust_dossier_no_dispute_marker_when_none():
    assert "⚔" not in recall_logic.trust_dossier(_hit(), None)


def _render_hit(**kw):
    """Fixture for the render-loop tests — includes `body`, required by both
    render_recall_context and render_recall_compact (unlike the brief's bare
    trust_dossier fixture above)."""
    base = dict(
        id="a1b2c3d4e5",
        title="Port is 8765",
        type="fact",
        tags=["project:memo"],
        updated="2026-07-10",
        score=0.83,
        extra={},
        verification_state=None,
        body="the port is 8765 for the dev server",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_render_context_unchanged_when_flag_off():
    """Flag-off: dossier line must not appear and render output must be
    byte-identical to the pre-dossier render (no _trust_ line, no ⚔)."""
    out = recall_logic.render_recall_context(
        [_render_hit()], [], turn=1, body_chars=200, token_budget=0
    )
    assert "_trust_" not in out
    assert "⚔" not in out


def test_render_compact_unchanged_when_flag_off():
    out = recall_logic.render_recall_compact([_render_hit()], token_budget=0)
    assert "_trust_" not in out
    assert "⚔" not in out


def test_render_context_shows_dossier_when_flag_on(monkeypatch):
    monkeypatch.setenv("MEMO_HIT_DOSSIER", "1")
    out = recall_logic.render_recall_context(
        [_render_hit()], [], turn=1, body_chars=200, token_budget=0
    )
    assert "_trust_" in out
    assert "fact" in out
    assert "conf" in out.lower()


def test_render_compact_shows_dossier_when_flag_on(monkeypatch):
    monkeypatch.setenv("MEMO_HIT_DOSSIER", "1")
    out = recall_logic.render_recall_compact([_render_hit()], token_budget=0)
    assert "_trust_" in out
