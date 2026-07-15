import json
from pathlib import Path

import pytest

from memo import ask_gaps as ag


def _write_receipt(state_dir: Path, gaps):
    d = state_dir / "dream"
    d.mkdir(parents=True, exist_ok=True)
    import time

    (d / "last.json").write_text(
        json.dumps({"ts": time.time(), "anticipated": {"gaps": gaps}}), encoding="utf-8"
    )


def test_pick_gap_takes_top():
    gaps = [
        {"prompt": "how does sync lock work", "count": 5},
        {"prompt": "what is mmr", "count": 2},
    ]
    assert ag.pick_gap(gaps)["prompt"] == "how does sync lock work"
    assert ag.pick_gap([]) is None


def test_phrase_question_is_a_question_not_an_answer():
    q = ag.phrase_question({"prompt": "how does sync lock work", "count": 3})
    assert "how does sync lock work" in q
    assert "3" in q
    # never fabricates an answer — it re-asks
    assert "?" in q


def test_briefing_renders_one_gap_when_enabled(monkeypatch, tmp_cfg):
    monkeypatch.setenv("MEMO_ASK_GAPS_ENABLED", "1")
    _write_receipt(tmp_cfg.state_dir, [{"prompt": "how does sync lock work", "count": 4}])
    lines = ag.briefing_lines(tmp_cfg, session_id="s1")
    assert lines and "how does sync lock work" in lines[0]
    # same session, same gap -> deduped (not asked twice)
    assert ag.briefing_lines(tmp_cfg, session_id="s1") == []


def test_briefing_silent_when_disabled_but_shadow_logs(monkeypatch, tmp_cfg):
    monkeypatch.delenv("MEMO_ASK_GAPS_ENABLED", raising=False)
    _write_receipt(tmp_cfg.state_dir, [{"prompt": "what is mmr lambda", "count": 3}])
    assert ag.briefing_lines(tmp_cfg, session_id="s2") == []
    rows = ag.read_shadow(tmp_cfg.state_dir)
    assert len(rows) == 1 and rows[0]["rendered"] is False


def test_briefing_empty_without_receipt(tmp_cfg):
    assert ag.briefing_lines(tmp_cfg, session_id="s3") == []


def test_never_fabricates_missing_gaps(monkeypatch, tmp_cfg):
    monkeypatch.setenv("MEMO_ASK_GAPS_ENABLED", "1")
    _write_receipt(tmp_cfg.state_dir, [])  # no gaps at all
    assert ag.briefing_lines(tmp_cfg, session_id="s4") == []


def test_marker_rejects_session_id_traversal(tmp_path: Path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    outside = tmp_path / "outside.json"

    with pytest.raises(ValueError, match="session_id"):
        ag.note_asked(state_dir, "../../outside", "gap")

    assert not outside.exists()


def test_marker_rejects_symlinked_marker_directory(tmp_path: Path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (state_dir / ".ask_gaps_seen").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="unsafe"):
        ag.note_asked(state_dir, "safe-session", "gap")

    assert not (outside / "safe-session.json").exists()
