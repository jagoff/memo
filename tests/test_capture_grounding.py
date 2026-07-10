"""grounding-judge wired into the capture path (default off)."""
from __future__ import annotations

from memo import capture_core


def _one_candidate(*_a, **_k):
    return [{"title": "Port is 8765", "type": "fact", "body": "The dashboard port is 8765.", "tags": [], "fact_edges": None}]


def _run(mock_memory, monkeypatch, score):
    # The stock candidate body is short — bypass the unrelated word-count
    # quality gate (default min 15) so it isn't dropped before reaching the
    # confidence/grounding gate under test.
    monkeypatch.setenv("MEMO_CAPTURE_MIN_WORDS", "0")
    monkeypatch.setattr(mock_memory, "_ensure_chat", lambda: object())
    monkeypatch.setattr(capture_core, "extract_insights", _one_candidate)
    monkeypatch.setattr(
        capture_core, "score_grounding", lambda *a, **k: score
    )
    return capture_core._extract_and_save(
        mock_memory, mock_memory.cfg,
        "user said the port changed", "assistant confirmed 8765",
    )


def test_low_grounding_candidate_is_quarantined(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_GROUNDING_JUDGE", "1")
    monkeypatch.setenv("MEMO_GROUNDING_WRITE_MIN", "0.5")
    out = _run(mock_memory, monkeypatch, score=0.1)
    saved_id = out["saved"][0]
    rec = mock_memory.get(saved_id)
    assert "_uncertain" in (rec.tags or [])
    assert rec.extra.get("grounding_score") == 0.1


def test_high_grounding_candidate_is_clean(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_GROUNDING_JUDGE", "1")
    monkeypatch.setenv("MEMO_GROUNDING_WRITE_MIN", "0.5")
    out = _run(mock_memory, monkeypatch, score=0.9)
    rec = mock_memory.get(out["saved"][0])
    assert "_uncertain" not in (rec.tags or [])
    assert rec.extra.get("grounding_score") == 0.9


def test_flag_off_never_calls_judge(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_GROUNDING_JUDGE", "0")
    monkeypatch.setenv("MEMO_CAPTURE_MIN_WORDS", "0")
    called = {"n": 0}
    monkeypatch.setattr(mock_memory, "_ensure_chat", lambda: object())
    monkeypatch.setattr(capture_core, "extract_insights", _one_candidate)

    def _boom(*a, **k):
        called["n"] += 1
        return 0.0

    monkeypatch.setattr(capture_core, "score_grounding", _boom)
    out = capture_core._extract_and_save(
        mock_memory, mock_memory.cfg, "u", "a",
    )
    rec = mock_memory.get(out["saved"][0])
    assert "_uncertain" not in (rec.tags or [])
    assert "grounding_score" not in (rec.extra or {})
    assert called["n"] == 0
