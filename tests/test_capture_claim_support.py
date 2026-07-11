"""claim-support wired into the capture path (default off)."""
from __future__ import annotations

from memo import capture_core, claim_support


def _candidate(body):
    def _f(*_a, **_k):
        return [{"title": "outcome", "type": "note", "body": body, "tags": [], "fact_edges": None}]
    return _f


def test_unsupported_claim_downgrades_confidence(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_CLAIM_SUPPORT", "1")
    monkeypatch.setenv("MEMO_CLAIM_SUPPORT_CONFIDENCE", "0.5")
    monkeypatch.setenv("MEMO_CAPTURE_MIN_WORDS", "0")
    monkeypatch.setattr(mock_memory, "_ensure_chat", lambda: object())
    monkeypatch.setattr(capture_core, "extract_insights", _candidate("Fixed the timeout bug."))
    monkeypatch.setattr(claim_support, "_commit_exists", lambda s, r: False)

    out = capture_core._extract_and_save(mock_memory, mock_memory.cfg, "u", "a")
    sid = out["saved"][0]
    health = mock_memory.store.get_health_batch([sid])
    assert health.get(sid, {}).get("confidence") == 0.5


def test_supported_claim_keeps_confidence(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_CLAIM_SUPPORT", "1")
    monkeypatch.setenv("MEMO_CAPTURE_MIN_WORDS", "0")
    monkeypatch.setattr(mock_memory, "_ensure_chat", lambda: object())
    monkeypatch.setattr(
        capture_core, "extract_insights", _candidate("The dashboard runs on port 8765.")
    )
    out = capture_core._extract_and_save(mock_memory, mock_memory.cfg, "u", "a")
    sid = out["saved"][0]
    health = mock_memory.store.get_health_batch([sid])
    # no downgrade row written (neutral default) OR confidence stays 1.0
    assert health.get(sid, {}).get("confidence", 1.0) == 1.0


def test_flag_off_no_downgrade(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_CLAIM_SUPPORT", "0")
    monkeypatch.setenv("MEMO_CAPTURE_MIN_WORDS", "0")
    monkeypatch.setattr(mock_memory, "_ensure_chat", lambda: object())
    monkeypatch.setattr(capture_core, "extract_insights", _candidate("Fixed it."))
    out = capture_core._extract_and_save(mock_memory, mock_memory.cfg, "u", "a")
    sid = out["saved"][0]
    health = mock_memory.store.get_health_batch([sid])
    assert health.get(sid, {}).get("confidence", 1.0) == 1.0
