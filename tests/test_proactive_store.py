from pathlib import Path

import pytest

from memo.proactive.nudge import KIND_HEALTH, KIND_RELIABILITY, Nudge
from memo.proactive.store import ProactiveStore

pytestmark = pytest.mark.resource_hygiene


def _n(kind, sid, ev="m1"):
    return Nudge.make(
        kind,
        subject_id=sid,
        urgency=0.9,
        value=0.5,
        title=sid,
        evidence=(ev,),
        created_at="2026-07-21T00:00:00Z",
    )


def test_put_and_active_excludes_dismissed_and_snoozed(tmp_path: Path):
    with ProactiveStore(tmp_path / "p.db") as store:
        store.put_candidates([_n(KIND_RELIABILITY, "a"), _n(KIND_HEALTH, "b")])
        assert {n.id for n in store.active_candidates("2026-07-21T01:00:00Z")} == {
            _n(KIND_RELIABILITY, "a").id,
            _n(KIND_HEALTH, "b").id,
        }
        store.dismiss(_n(KIND_RELIABILITY, "a").id, "2026-07-21T01:00:00Z")
        store.snooze_kind(KIND_HEALTH, "2026-07-30T00:00:00Z")
        assert store.active_candidates("2026-07-21T02:00:00Z") == []


def test_put_candidates_tolerates_duplicate_ids(tmp_path: Path):
    """The write itself stays total when a batch repeats an id."""
    with ProactiveStore(tmp_path / "p.db") as store:
        store.put_candidates([_n(KIND_RELIABILITY, "a"), _n(KIND_RELIABILITY, "a")])
        assert len(store.active_candidates("2026-07-21T01:00:00Z")) == 1


def test_multipliers_demote_ignored_kind(tmp_path: Path):
    with ProactiveStore(tmp_path / "p.db") as store:
        for i in range(5):
            store.record_feedback(f"h{i}", KIND_HEALTH, "dismissed", "2026-07-21T00:00:00Z")
        store.record_feedback("r0", KIND_RELIABILITY, "acted", "2026-07-21T00:00:00Z")
        multipliers = store.kind_multipliers(floor=0.2)
        assert multipliers[KIND_HEALTH] == 0.2  # demoted to floor
        assert multipliers[KIND_RELIABILITY] >= 1.0  # acted → not demoted
