from memo.proactive.arbiter import route
from memo.proactive.nudge import KIND_HEALTH, KIND_RELIABILITY, Nudge


def _n(kind, sid, urg, val):
    return Nudge.make(
        kind,
        subject_id=sid,
        urgency=urg,
        value=val,
        title=sid,
        evidence=("m1",),
        created_at="2026-07-21T00:00:00Z",
    )


def test_urgent_only_reliability_over_threshold_and_can_push():
    cands = [_n(KIND_RELIABILITY, "r", 0.95, 0.9), _n(KIND_HEALTH, "h", 0.99, 0.99)]
    mult = {KIND_RELIABILITY: 1.0, KIND_HEALTH: 1.0}
    r = route(cands, mult, digest_top=7, urgent_min=0.7, can_push=True, floor=0.2)
    assert r.urgent is not None and r.urgent.kind == KIND_RELIABILITY
    assert r.badge_count == 2


def test_no_push_when_cannot_push():
    cands = [_n(KIND_RELIABILITY, "r", 0.95, 0.9)]
    r = route(
        cands, {KIND_RELIABILITY: 1.0}, digest_top=7, urgent_min=0.7, can_push=False, floor=0.2
    )
    assert r.urgent is None


def test_floored_multiplier_keeps_reliability_visible():
    cands = [_n(KIND_RELIABILITY, "r", 0.95, 0.9)]
    r = route(
        cands, {KIND_RELIABILITY: 0.2}, digest_top=7, urgent_min=0.7, can_push=True, floor=0.2
    )
    # 0.2 floor: score = (0.6*0.95+0.4*0.9)*0.2 = 0.186 < 0.7 → not urgent, but still in digest
    assert r.urgent is None
    assert len(r.digest) == 1
