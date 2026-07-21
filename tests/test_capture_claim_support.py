"""claim-support wired into capture as a default-on quarantine boundary."""

from __future__ import annotations

from memo import capture_core, claim_support


def _candidate(body):
    def _f(*_a, **_k):
        return [{"title": "outcome", "type": "note", "body": body, "tags": [], "fact_edges": None}]

    return _f


def test_unsupported_claim_is_quarantined_by_default(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_CLAIM_SUPPORT_CONFIDENCE", "0.42")
    monkeypatch.setenv("MEMO_CAPTURE_MIN_WORDS", "0")
    monkeypatch.setattr(mock_memory, "_ensure_chat", lambda: object())
    monkeypatch.setattr(capture_core, "extract_insights", _candidate("Fixed the timeout bug."))
    monkeypatch.setattr(claim_support, "_commit_exists", lambda s, r: False)

    out = capture_core._extract_and_save(mock_memory, mock_memory.cfg, "u", "a")
    sid = out["saved"][0]
    rec = mock_memory.get(sid)
    health = mock_memory.store.get_health_batch([sid])
    assert rec is not None
    assert "_uncertain" in rec.tags
    assert rec.extra["claim_support_kind"] == "fixed"
    assert rec.extra["claim_support_reason"] == "fixed claim with no evidence ref"
    assert health.get(sid, {}).get("confidence") == 0.42


def test_supported_claim_keeps_confidence(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_CAPTURE_MIN_WORDS", "0")
    monkeypatch.setattr(mock_memory, "_ensure_chat", lambda: object())
    monkeypatch.setattr(
        capture_core, "extract_insights", _candidate("Fixed the dashboard bug; tests green.")
    )
    out = capture_core._extract_and_save(mock_memory, mock_memory.cfg, "u", "a")
    sid = out["saved"][0]
    rec = mock_memory.get(sid)
    health = mock_memory.store.get_health_batch([sid])
    assert rec is not None
    assert "_uncertain" not in rec.tags
    assert "claim_support_kind" not in rec.extra
    assert "claim_support_reason" not in rec.extra
    # no downgrade row written (neutral default) OR confidence stays 1.0
    assert health.get(sid, {}).get("confidence", 1.0) == 1.0


def test_flag_off_no_downgrade(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_CLAIM_SUPPORT", "0")
    monkeypatch.setenv("MEMO_CAPTURE_MIN_WORDS", "0")
    monkeypatch.setattr(mock_memory, "_ensure_chat", lambda: object())
    monkeypatch.setattr(capture_core, "extract_insights", _candidate("Fixed it."))
    out = capture_core._extract_and_save(mock_memory, mock_memory.cfg, "u", "a")
    sid = out["saved"][0]
    rec = mock_memory.get(sid)
    health = mock_memory.store.get_health_batch([sid])
    assert rec is not None
    assert "_uncertain" not in rec.tags
    assert "claim_support_kind" not in rec.extra
    assert "claim_support_reason" not in rec.extra
    assert health.get(sid, {}).get("confidence", 1.0) == 1.0


def test_non_outcome_memory_is_unchanged(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_CAPTURE_MIN_WORDS", "0")
    monkeypatch.setattr(mock_memory, "_ensure_chat", lambda: object())
    monkeypatch.setattr(
        capture_core,
        "extract_insights",
        _candidate("The dashboard listens on port 8765."),
    )

    out = capture_core._extract_and_save(mock_memory, mock_memory.cfg, "u", "a")

    rec = mock_memory.get(out["saved"][0])
    assert rec is not None
    assert "_uncertain" not in rec.tags
    assert "claim_support_kind" not in rec.extra
    assert "claim_support_reason" not in rec.extra
