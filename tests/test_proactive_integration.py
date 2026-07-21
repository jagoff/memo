"""Task 12 — wire the proactive engine into briefing / statusline / Stop hook.

All wiring is behind `MEMO_PROACTIVE_ENABLED` (default off) and must degrade to
today's exact output when the flag is off or on any error.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from memo.proactive.engine import compute_routed
from memo.proactive.nudge import KIND_RELIABILITY, Nudge
from memo.proactive.store import ProactiveStore
from memo.proactive.surfaces import render_urgent_line


def test_urgent_emitted_once_then_cooled_down(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MEMO_PROACTIVE_ENABLED", "1")
    s = ProactiveStore(tmp_path / "p.db")
    s.put_candidates(
        [
            Nudge.make(
                KIND_RELIABILITY,
                subject_id="old1",
                urgency=0.95,
                value=0.9,
                title="stale",
                evidence=("new1",),
                action="memo review old1",
                created_at="2026-07-21T09:00:00Z",
            )
        ]
    )
    routed = compute_routed(s, now="2026-07-21T10:00:00Z", day="2026-07-21")
    assert routed.urgent is not None
    line = render_urgent_line(routed.urgent)
    s.mark_pushed("2026-07-21T10:00:00Z")
    assert "memo review old1" in line
    # within cooldown → no second push
    routed2 = compute_routed(s, now="2026-07-21T10:30:00Z", day="2026-07-21")
    assert routed2.urgent is None


# ── briefing wiring ─────────────────────────────────────────────────────────


def _seed_candidate(mem, *, now: str) -> None:
    store = ProactiveStore(mem.cfg.state_dir / "proactive.db")
    store.put_candidates(
        [
            Nudge.make(
                KIND_RELIABILITY,
                subject_id="old1",
                urgency=0.9,
                value=0.8,
                title="You may be relying on a superseded fact: use X not Y",
                evidence=("new1", "old1"),
                action="memo get old1",
                created_at=now,
            )
        ]
    )


def test_briefing_shows_proactive_section_when_enabled(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_PROACTIVE_ENABLED", "1")
    now = datetime.now(tz=UTC).isoformat()
    _seed_candidate(mock_memory, now=now)

    from memo.briefing import memo_native_briefing_lines

    joined = "\n".join(memo_native_briefing_lines(mock_memory, memory_of_day=False))

    assert "### Proactive" in joined
    assert "memo get old1" in joined


def test_briefing_omits_proactive_section_when_disabled(mock_memory, monkeypatch):
    monkeypatch.delenv("MEMO_PROACTIVE_ENABLED", raising=False)
    now = datetime.now(tz=UTC).isoformat()
    _seed_candidate(mock_memory, now=now)

    from memo.briefing import memo_native_briefing_lines

    joined = "\n".join(memo_native_briefing_lines(mock_memory, memory_of_day=False))

    assert "### Proactive" not in joined


def test_briefing_omits_proactive_section_when_no_candidates(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_PROACTIVE_ENABLED", "1")
    # No candidates seeded — the store exists (nightly dream refresh already ran)
    # but is empty. Must not print an empty/placeholder section.
    from memo.briefing import memo_native_briefing_lines

    joined = "\n".join(memo_native_briefing_lines(mock_memory, memory_of_day=False))

    assert "### Proactive" not in joined


# ── Stop hook urgent push ────────────────────────────────────────────────────


def test_capture_stop_prints_urgent_line_and_marks_pushed(tmp_path: Path, monkeypatch) -> None:
    from click.testing import CliRunner

    import memo.capture as capture_mod
    from memo.cli_capture import capture_stop
    from memo.config import Config

    state = tmp_path / "state"
    state.mkdir()
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("{}\n", encoding="utf-8")

    monkeypatch.setenv("MEMO_STATE_DIR", str(state))
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMO_NONINTERACTIVE", "1")
    monkeypatch.setenv("MEMO_PROACTIVE_ENABLED", "1")
    monkeypatch.setattr(
        capture_mod,
        "run_capture",
        lambda *a, **k: {"status": "no_trigger", "saved": [], "saved_titles": []},
    )

    cfg = Config.from_env()
    now = datetime.now(tz=UTC).isoformat()
    store = ProactiveStore(cfg.state_dir / "proactive.db")
    store.put_candidates(
        [
            Nudge.make(
                KIND_RELIABILITY,
                subject_id="old1",
                urgency=0.95,
                value=0.9,
                title="stale fact",
                evidence=("new1", "old1"),
                action="memo get old1",
                created_at=now,
            )
        ]
    )

    payload = json.dumps({"transcript_path": str(transcript), "session_id": "s1"})
    result = CliRunner().invoke(capture_stop, input=payload)

    assert result.exit_code == 0
    assert "memo get old1" in result.output
    assert store.pushes_today(datetime.now(tz=UTC).date().isoformat()) == 1


def test_capture_stop_proactive_disabled_prints_nothing_extra(tmp_path: Path, monkeypatch) -> None:
    """Flag off → today's exact Stop-hook output (no proactive section, no crash)."""
    from click.testing import CliRunner

    import memo.capture as capture_mod
    from memo.cli_capture import capture_stop
    from memo.config import Config

    state = tmp_path / "state"
    state.mkdir()
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("{}\n", encoding="utf-8")

    monkeypatch.setenv("MEMO_STATE_DIR", str(state))
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMO_NONINTERACTIVE", "1")
    monkeypatch.delenv("MEMO_PROACTIVE_ENABLED", raising=False)
    monkeypatch.setattr(
        capture_mod,
        "run_capture",
        lambda *a, **k: {"status": "no_trigger", "saved": [], "saved_titles": []},
    )

    cfg = Config.from_env()
    now = datetime.now(tz=UTC).isoformat()
    store = ProactiveStore(cfg.state_dir / "proactive.db")
    store.put_candidates(
        [
            Nudge.make(
                KIND_RELIABILITY,
                subject_id="old1",
                urgency=0.95,
                value=0.9,
                title="stale fact",
                evidence=("new1", "old1"),
                action="memo get old1",
                created_at=now,
            )
        ]
    )

    payload = json.dumps({"transcript_path": str(transcript), "session_id": "s1"})
    result = CliRunner().invoke(capture_stop, input=payload)

    assert result.exit_code == 0
    assert "memo get old1" not in result.output
    assert store.pushes_today(datetime.now(tz=UTC).date().isoformat()) == 0
