from dataclasses import dataclass
from pathlib import Path

from memo import recall_logic


@dataclass
class _Hit:
    id: str
    title: str
    body: str | None
    score: float | None
    tags: list[str]
    type: str = "note"
    updated: str = "2026-07"


def test_gate_marks_low_confidence_hit(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MEMO_RECALL_CONFIDENCE_GATE", "1")
    # no calibration map -> identity; a low SCORE lands in the "low" band.
    hit = _Hit("a1b2c3d4", "T", "body body body", score=0.2, tags=[])
    out = recall_logic.render_recall_context(
        [hit], [], turn=1, body_chars=400, token_budget=0, state_dir=tmp_path
    )
    assert "⚠ unverified — consider checking" in out


def test_gate_silent_for_high_confidence_hit(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MEMO_RECALL_CONFIDENCE_GATE", "1")
    hit = _Hit("a1b2c3d4", "T", "body body body", score=0.9, tags=[])
    out = recall_logic.render_recall_context(
        [hit], [], turn=1, body_chars=400, token_budget=0, state_dir=tmp_path
    )
    assert "⚠ unverified — consider checking" not in out


def test_gate_off_by_default(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("MEMO_RECALL_CONFIDENCE_GATE", raising=False)
    hit = _Hit("a1b2c3d4", "T", "body body body", score=0.2, tags=[])
    out = recall_logic.render_recall_context(
        [hit], [], turn=1, body_chars=400, token_budget=0, state_dir=tmp_path
    )
    assert "⚠ unverified — consider checking" not in out


def test_gate_uses_calibration_map_to_demote(tmp_path: Path, monkeypatch):
    from memo import confidence_calibration as cc

    monkeypatch.setenv("MEMO_RECALL_CONFIDENCE_GATE", "1")
    # map demotes a "high" score-band down to "low" -> even a strong hit gates.
    cc.save_calibration(tmp_path, {"bins": {}, "map": {"high": "low"}})
    hit = _Hit("a1b2c3d4", "T", "body body body", score=0.95, tags=[])
    out = recall_logic.render_recall_context(
        [hit], [], turn=1, body_chars=400, token_budget=0, state_dir=tmp_path
    )
    assert "⚠ unverified — consider checking" in out
