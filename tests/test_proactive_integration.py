"""Task 12 — wire the proactive engine into briefing / statusline / recall hook.

The urgent push rides the recall hook's `systemMessage` (the one synchronous
channel Claude Code renders to the user) via `engine.pull_urgent`; the Stop hook
no longer pushes (its async stdout is discarded). All wiring is behind
`MEMO_PROACTIVE_ENABLED` (default off) and must degrade to today's exact output
when the flag is off or on any error.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from memo.proactive.engine import compute_routed
from memo.proactive.nudge import KIND_RELIABILITY, Nudge
from memo.proactive.store import ProactiveStore
from memo.proactive.surfaces import render_urgent_line

pytestmark = pytest.mark.resource_hygiene


def test_urgent_emitted_once_then_cooled_down(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MEMO_PROACTIVE_ENABLED", "1")
    with ProactiveStore(tmp_path / "p.db") as store:
        store.put_candidates(
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
        routed = compute_routed(store, now="2026-07-21T10:00:00Z", day="2026-07-21")
        assert routed.urgent is not None
        line = render_urgent_line(routed.urgent)
        store.mark_pushed("2026-07-21T10:00:00Z")
        assert "memo review old1" in line
        # within cooldown → no second push
        routed2 = compute_routed(store, now="2026-07-21T10:30:00Z", day="2026-07-21")
        assert routed2.urgent is None


# ── briefing wiring ─────────────────────────────────────────────────────────


def _seed_candidate(mem, *, now: str) -> None:
    with ProactiveStore(mem.cfg.state_dir / "proactive.db") as store:
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


def test_compact_line_surfaces_digest_without_consuming_push(mock_memory, monkeypatch):
    """The `--compact` SessionStart capsule reaches the user via a one-liner —
    and, being a pull surface, must not consume the urgent push slot."""
    monkeypatch.setenv("MEMO_PROACTIVE_ENABLED", "1")
    now = datetime.now(tz=UTC).isoformat()
    _seed_candidate(mock_memory, now=now)

    from memo.briefing import proactive_compact_line

    line = proactive_compact_line(mock_memory)
    assert "memo get old1" in line
    assert "memo digest" in line

    with ProactiveStore(mock_memory.cfg.state_dir / "proactive.db") as store:
        assert store.pushes_today(datetime.now(tz=UTC).date().isoformat()) == 0


def test_compact_line_empty_when_disabled(mock_memory, monkeypatch):
    monkeypatch.delenv("MEMO_PROACTIVE_ENABLED", raising=False)
    _seed_candidate(mock_memory, now=datetime.now(tz=UTC).isoformat())

    from memo.briefing import proactive_compact_line

    assert proactive_compact_line(mock_memory) == ""


# ── recall-hook urgent push (pull_urgent owns the push slot) ─────────────────


def test_pull_urgent_marks_pushed_and_cools_down(tmp_path: Path, monkeypatch) -> None:
    """`pull_urgent` returns the due nudge and consumes exactly one push slot;
    a second call inside the cooldown returns None and does not re-push."""
    monkeypatch.setenv("MEMO_PROACTIVE_ENABLED", "1")

    from memo.proactive.engine import pull_urgent

    with ProactiveStore(tmp_path / "p.db") as store:
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
                    created_at="2026-07-21T09:00:00Z",
                )
            ]
        )

        urgent = pull_urgent(store, now="2026-07-21T10:00:00Z", day="2026-07-21")
        assert urgent is not None
        line = render_urgent_line(urgent)
        assert line.startswith("⚠️ memo:")
        assert "memo get old1" in line
        assert store.pushes_today("2026-07-21") == 1

        # within cooldown → nothing due, no extra push slot consumed
        assert pull_urgent(store, now="2026-07-21T10:30:00Z", day="2026-07-21") is None
        assert store.pushes_today("2026-07-21") == 1


def test_pull_urgent_noop_when_no_candidates(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MEMO_PROACTIVE_ENABLED", "1")

    from memo.proactive.engine import pull_urgent

    with ProactiveStore(tmp_path / "p.db") as store:
        assert pull_urgent(store, now="2026-07-21T10:00:00Z", day="2026-07-21") is None
        assert store.pushes_today("2026-07-21") == 0


def test_capture_stop_stays_silent_with_proactive_enabled(tmp_path: Path, monkeypatch) -> None:
    """The Stop hook no longer pushes proactive nudges (async stdout is
    discarded by Claude Code): capture-stop emits only `{}` and consumes no
    push slot, even with a due candidate and the flag on."""
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
    with ProactiveStore(cfg.state_dir / "proactive.db") as store:
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
    with ProactiveStore(cfg.state_dir / "proactive.db") as store:
        assert store.pushes_today(datetime.now(tz=UTC).date().isoformat()) == 0
