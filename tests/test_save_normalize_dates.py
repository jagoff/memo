"""MEMO_SAVE_NORMALIZE_DATES: relative dates annotated with ISO dates at save."""
from __future__ import annotations

import datetime as dt


def test_save_normalizes_relative_dates_when_flag_on(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_SAVE_NORMALIZE_DATES", "1")
    rec = mock_memory.save(content="decidimos ayer migrar el build a uv", title="Decisión build")
    expected = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    assert f"ayer ({expected})" in rec.body


def test_save_anchors_to_created_override_for_backdated_imports(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_SAVE_NORMALIZE_DATES", "1")
    rec = mock_memory.save(
        content="yesterday we shipped the fix",
        title="Backdated",
        created="2026-01-10T12:00:00",
    )
    assert "yesterday (2026-01-09)" in rec.body


def test_save_leaves_content_untouched_when_flag_off(mock_memory):
    rec = mock_memory.save(content="decidimos ayer migrar el build a uv", title="Sin flag")
    assert "ayer (" not in rec.body


def test_save_never_rewrites_reference_tier(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_SAVE_NORMALIZE_DATES", "1")
    rec = mock_memory.save(
        content="# Transcript\nayer hablamos del viaje y de un montón de cosas más largas",
        title="Chat §1/9",
        type_="reference",
    )
    assert "ayer (" not in rec.body
