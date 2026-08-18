from pathlib import Path

import pytest

from memo.proactive.engine import push_gate, refresh_candidates
from memo.proactive.store import ProactiveStore

pytestmark = pytest.mark.resource_hygiene


class _FullFakeMem:
    """Fake mem exposing all five v1+v2 proactive accessors."""

    def superseded_pairs(self):
        return [("old1", "new1", "use X not Y")]

    def open_loops(self, limit):
        return [("m9", "finish int8 magnitude study")]

    def low_confidence_ids(self, *, threshold, limit):
        return ["low1", "low2"]

    def dead_memory_ids(self, *, limit):
        return ["dead1"]

    def recurring_pattern_pairs(self, *, limit):
        return [("m1", "recurring question"), ("m2", "another recurring question")]


def test_refresh_candidates_sums_all_five_detectors(tmp_path: Path):
    with ProactiveStore(tmp_path / "p.db") as store:
        n = refresh_candidates(_FullFakeMem(), store, now="2026-07-21T00:00:00Z")
        # 1 reliability + 1 continuity + 1 health (aggregate) + 1 roi (aggregate)
        # + 2 dejavu (one per pair) = 6
        assert n == 6
        candidates = store.active_candidates("2026-07-21T01:00:00Z")
        assert len(candidates) == 6
        for candidate in candidates:
            assert candidate.evidence, f"{candidate.kind} nudge has empty evidence"


def test_refresh_candidates_empty_sources_emit_no_v2_nudges(tmp_path: Path):
    class _EmptyMem:
        def superseded_pairs(self):
            return []

        def open_loops(self, limit):
            return []

        def low_confidence_ids(self, *, threshold, limit):
            return []

        def dead_memory_ids(self, *, limit):
            return []

        def recurring_pattern_pairs(self, *, limit):
            return []

    with ProactiveStore(tmp_path / "p2.db") as store:
        n = refresh_candidates(_EmptyMem(), store, now="2026-07-21T00:00:00Z")
        assert n == 0


def test_refresh_candidates_dedups_colliding_nudge_ids(tmp_path: Path):
    """Two patterns whose top hit is the same memory must not abort the refresh.

    `Nudge.id` is sha256(kind:subject_id) and it is the candidates primary key,
    so the duplicate used to raise `IntegrityError: UNIQUE constraint failed:
    proactive_candidates.id`, rolling back the DELETE+INSERT and leaving the
    nightly dream pass with stale candidates.
    """

    class _CollidingMem(_FullFakeMem):
        def recurring_pattern_pairs(self, *, limit):
            return [("m1", "same memory, first phrasing"), ("m1", "same memory, again")]

    with ProactiveStore(tmp_path / "dup.db") as store:
        n = refresh_candidates(_CollidingMem(), store, now="2026-07-21T00:00:00Z")
        # 1 reliability + 1 continuity + 1 health + 1 roi + 1 (deduped) dejavu
        assert n == 5
        candidates = store.active_candidates("2026-07-21T01:00:00Z")
        assert len(candidates) == 5
        assert len({c.id for c in candidates}) == 5


def test_push_gate_respects_cooldown_and_cap(tmp_path: Path):
    with ProactiveStore(tmp_path / "p.db") as store:
        assert (
            push_gate(
                store,
                now="2026-07-21T10:00:00Z",
                day="2026-07-21",
                cooldown_h=6,
                daily_cap=3,
            )
            is True
        )
        store.mark_pushed("2026-07-21T09:00:00Z")  # 1h ago < 6h cooldown
        assert (
            push_gate(
                store,
                now="2026-07-21T10:00:00Z",
                day="2026-07-21",
                cooldown_h=6,
                daily_cap=3,
            )
            is False
        )


def test_push_gate_daily_cap(tmp_path: Path):
    with ProactiveStore(tmp_path / "p.db") as store:
        for hour in (1, 2, 3):
            store.mark_pushed(f"2026-07-21T0{hour}:00:00Z")
        # 3 pushes today, all >6h before 'now' → cooldown ok but cap hit
        assert (
            push_gate(
                store,
                now="2026-07-21T23:00:00Z",
                day="2026-07-21",
                cooldown_h=6,
                daily_cap=3,
            )
            is False
        )
