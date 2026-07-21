from pathlib import Path

from memo.cli_proactive import record_acted_if_matches
from memo.proactive.nudge import KIND_RELIABILITY, Nudge
from memo.proactive.store import ProactiveStore


def test_running_the_action_records_acted(tmp_path: Path):
    s = ProactiveStore(tmp_path / "p.db")
    s.put_candidates(
        [
            Nudge.make(
                KIND_RELIABILITY,
                subject_id="old1",
                urgency=0.9,
                value=0.8,
                title="t",
                evidence=("new1",),
                action="memo review old1",
                created_at="2026-07-21T10:00:00Z",
            )
        ]
    )
    record_acted_if_matches(
        s, command_line="memo review old1", now="2026-07-21T10:05:00Z", window_min=30
    )
    m = s.kind_multipliers(floor=0.2)
    assert m[KIND_RELIABILITY] >= 1.0  # acted counted


def test_mismatched_command_records_nothing(tmp_path: Path):
    s = ProactiveStore(tmp_path / "p.db")
    s.put_candidates(
        [
            Nudge.make(
                KIND_RELIABILITY,
                subject_id="old1",
                urgency=0.9,
                value=0.8,
                title="t",
                evidence=("new1",),
                action="memo review old1",
                created_at="2026-07-21T10:00:00Z",
            )
        ]
    )
    record_acted_if_matches(
        s, command_line="memo review other", now="2026-07-21T10:05:00Z", window_min=30
    )
    m = s.kind_multipliers(floor=0.2)
    assert KIND_RELIABILITY not in m


def test_outside_window_records_nothing(tmp_path: Path):
    s = ProactiveStore(tmp_path / "p.db")
    s.put_candidates(
        [
            Nudge.make(
                KIND_RELIABILITY,
                subject_id="old1",
                urgency=0.9,
                value=0.8,
                title="t",
                evidence=("new1",),
                action="memo review old1",
                created_at="2026-07-21T10:00:00Z",
            )
        ]
    )
    record_acted_if_matches(
        s, command_line="memo review old1", now="2026-07-21T11:00:00Z", window_min=30
    )
    m = s.kind_multipliers(floor=0.2)
    assert KIND_RELIABILITY not in m
