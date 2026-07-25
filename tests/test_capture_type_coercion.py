"""Regression: an out-of-vocabulary type from the helper LLM must be coerced
to a valid one, not silently drop the mined insight at save time.

The extract prompt invites state-change insights ("switched from X to Y"); the
helper LLM naturally types them "state", which is NOT a valid memory type, so
``mem.save()`` raised and the insight was lost (``candidates:1, saved:0``).
Verified live during QA before the fix.
"""

from __future__ import annotations

from memo import capture_core


def _candidate(
    type_, body="Switched the reranker from the cross-encoder to a head-slice scorer; 12% better."
):
    def _f(*_a, **_k):
        return [
            {
                "title": "switched reranker scorer",
                "type": type_,
                "body": body,
                "tags": [],
                "fact_edges": None,
            }
        ]

    return _f


def _isolate(mock_memory, monkeypatch, type_):
    monkeypatch.setenv("MEMO_CAPTURE_MIN_WORDS", "0")
    monkeypatch.setenv("MEMO_GROUNDING_JUDGE", "0")
    monkeypatch.setenv("MEMO_CLAIM_SUPPORT", "0")
    monkeypatch.setattr(mock_memory, "_ensure_chat", lambda: object())
    monkeypatch.setattr(capture_core, "extract_insights", _candidate(type_))
    return capture_core._extract_and_save(mock_memory, mock_memory.cfg, "u", "a")


def test_unknown_type_state_is_coerced_not_dropped(mock_memory, monkeypatch):
    out = _isolate(mock_memory, monkeypatch, "state")
    assert out["candidates"] == 1
    assert out["save_failures"] == 0
    assert len(out["saved"]) == 1, "insight with hallucinated type must be saved, not dropped"
    assert mock_memory.get(out["saved"][0]).type == "note"


def test_mis_cased_valid_type_is_normalized(mock_memory, monkeypatch):
    out = _isolate(mock_memory, monkeypatch, "Decision")
    assert len(out["saved"]) == 1
    assert mock_memory.get(out["saved"][0]).type == "decision"


def test_valid_type_is_unchanged(mock_memory, monkeypatch):
    out = _isolate(mock_memory, monkeypatch, "bug")
    assert len(out["saved"]) == 1
    assert mock_memory.get(out["saved"][0]).type == "bug"
