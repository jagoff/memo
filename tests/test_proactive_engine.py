from pathlib import Path

from memo.proactive.engine import push_gate, refresh_candidates
from memo.proactive.store import ProactiveStore


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
    s = ProactiveStore(tmp_path / "p.db")
    n = refresh_candidates(_FullFakeMem(), s, now="2026-07-21T00:00:00Z")
    # 1 reliability + 1 continuity + 1 health (aggregate) + 1 roi (aggregate)
    # + 2 dejavu (one per pair) = 6
    assert n == 6
    candidates = s.active_candidates("2026-07-21T01:00:00Z")
    assert len(candidates) == 6
    for c in candidates:
        assert c.evidence, f"{c.kind} nudge has empty evidence"


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

    s = ProactiveStore(tmp_path / "p2.db")
    n = refresh_candidates(_EmptyMem(), s, now="2026-07-21T00:00:00Z")
    assert n == 0


def test_push_gate_respects_cooldown_and_cap(tmp_path: Path):
    s = ProactiveStore(tmp_path / "p.db")
    assert (
        push_gate(s, now="2026-07-21T10:00:00Z", day="2026-07-21", cooldown_h=6, daily_cap=3)
        is True
    )
    s.mark_pushed("2026-07-21T09:00:00Z")  # 1h ago < 6h cooldown
    assert (
        push_gate(s, now="2026-07-21T10:00:00Z", day="2026-07-21", cooldown_h=6, daily_cap=3)
        is False
    )


def test_push_gate_daily_cap(tmp_path: Path):
    s = ProactiveStore(tmp_path / "p.db")
    for h in (1, 2, 3):
        s.mark_pushed(f"2026-07-21T0{h}:00:00Z")
    # 3 pushes today, all >6h before 'now' → cooldown ok but cap hit
    assert (
        push_gate(s, now="2026-07-21T23:00:00Z", day="2026-07-21", cooldown_h=6, daily_cap=3)
        is False
    )
