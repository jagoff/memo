from memo.proactive.detectors.reliability import detect_reliability
from memo.proactive.nudge import KIND_RELIABILITY


class _FakeMem:
    def superseded_pairs(self):
        # (stale_id, superseding_id, title)
        return [("old1", "new1", "use X not Y")]


def test_reliability_nudges_cite_superseding_id():
    ns = detect_reliability(_FakeMem(), now="2026-07-21T00:00:00Z")
    assert len(ns) == 1
    n = ns[0]
    assert n.kind == KIND_RELIABILITY
    assert "new1" in n.evidence and n.urgency >= 0.7
    # The SUCCESSOR, not the stale side. `superseded_pairs` sources the stale
    # id from `memory_dir/inactive/`, and `memo get` does not read the archive:
    # on 2026-08-30 all seven ids the live digest offered as `memo get <id>`
    # answered "not found". This assertion used to pin `old1`, so the suite
    # certified the dead-end.
    assert n.action == "memo get new1"


def test_reliability_guarded_returns_empty_on_error():
    class Boom:
        def superseded_pairs(self):
            raise RuntimeError("boom")

    assert detect_reliability(Boom(), now="2026-07-21T00:00:00Z") == []
