"""grounding-judge wired into the capture path (default off)."""

from __future__ import annotations

from memo import capture_core


def _one_candidate(*_a, **_k):
    return [
        {
            "title": "Port is 8765",
            "type": "fact",
            "body": "The dashboard port is 8765.",
            "tags": [],
            "fact_edges": None,
        }
    ]


def _run(mock_memory, monkeypatch, score):
    # The stock candidate body is short — bypass the unrelated word-count
    # quality gate (default min 15) so it isn't dropped before reaching the
    # confidence/grounding gate under test.
    monkeypatch.setenv("MEMO_CAPTURE_MIN_WORDS", "0")
    monkeypatch.setattr(mock_memory, "_ensure_chat", lambda: object())
    monkeypatch.setattr(capture_core, "extract_insights", _one_candidate)
    monkeypatch.setattr(capture_core, "score_grounding", lambda *a, **k: score)
    return capture_core._extract_and_save(
        mock_memory,
        mock_memory.cfg,
        "user said the port changed",
        "assistant confirmed 8765",
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
        mock_memory,
        mock_memory.cfg,
        "u",
        "a",
    )
    rec = mock_memory.get(out["saved"][0])
    assert "_uncertain" not in (rec.tags or [])
    assert "grounding_score" not in (rec.extra or {})
    assert called["n"] == 0


# ---------------------------------------------------------------------------
# Structured relative-date grounding (Task 5): ground_relative_dates emits a
# structured valid_at anchored to the Observation Date (capture/save time),
# never to today's clock.
# ---------------------------------------------------------------------------


def test_grounding_emits_structured_valid_at():
    from memo.memory.consolidate_ops import ground_relative_dates

    # observed on 2026-07-22; "yesterday" -> 2026-07-21
    text, valid_at = ground_relative_dates(
        "decided yesterday to use X", observed_at="2026-07-22T12:00:00"
    )
    assert valid_at is not None and valid_at.startswith("2026-07-21")
    # inline annotation behaviour preserved
    assert "2026-07-21" in text


def test_grounding_no_date_returns_none():
    from memo.memory.consolidate_ops import ground_relative_dates

    text, valid_at = ground_relative_dates(
        "a plain durable fact with no dates", observed_at="2026-07-22T12:00:00"
    )
    assert valid_at is None
    assert text == "a plain durable fact with no dates"


def test_grounding_ambiguous_two_days_returns_none():
    from memo.memory.consolidate_ops import ground_relative_dates

    # "yesterday" -> 07-21 and "3 days ago" -> 07-19 : two distinct anchors
    _text, valid_at = ground_relative_dates(
        "yesterday we started, but 3 days ago we planned",
        observed_at="2026-07-22T12:00:00",
    )
    assert valid_at is None


def test_grounding_accepts_datetime_observation():
    """A `datetime` Observation Date is reduced to its calendar day before
    resolving relative anchors."""
    import datetime as _dt

    from memo.memory.consolidate_ops import ground_relative_dates

    _text, valid_at = ground_relative_dates(
        "hoy arreglamos el bug", observed_at=_dt.datetime(2026, 7, 22, 15, 30, tzinfo=_dt.UTC)
    )
    assert valid_at == "2026-07-22"


def test_grounding_malformed_observation_never_raises():
    """An unparseable Observation Date returns `(text, None)` instead of raising."""
    from memo.memory.consolidate_ops import ground_relative_dates

    text, valid_at = ground_relative_dates("ayer fue el día", observed_at="not-a-date")
    assert text == "ayer fue el día" and valid_at is None


def test_save_grounding_sets_valid_at_from_observation(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_SAVE_NORMALIZE_DATES", "1")
    monkeypatch.setenv("MEMO_EMBEDDER_DIMS", "4")
    rec = mock_memory.save(
        content="decided yesterday to use X",
        type_="decision",
        title="X decision",
        created="2026-07-22T12:00:00",
    )
    assert rec.valid_at is not None
    assert rec.valid_at.startswith("2026-07-21")
    assert rec.valid_at != rec.created


def test_save_grounding_no_date_defaults_valid_at_to_created(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_SAVE_NORMALIZE_DATES", "1")
    monkeypatch.setenv("MEMO_EMBEDDER_DIMS", "4")
    rec = mock_memory.save(
        content="a plain durable fact",
        type_="fact",
        title="plain",
        created="2026-07-22T12:00:00",
    )
    assert rec.valid_at == rec.created


def test_save_explicit_valid_at_overrides_grounding(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_SAVE_NORMALIZE_DATES", "1")
    monkeypatch.setenv("MEMO_EMBEDDER_DIMS", "4")
    rec = mock_memory.save(
        content="decided yesterday to use X",
        type_="decision",
        title="X decision",
        created="2026-07-22T12:00:00",
        valid_at="2020-01-01T00:00:00",
    )
    assert rec.valid_at == "2020-01-01T00:00:00"
