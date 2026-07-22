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
    r = route(cands, mult, digest_top=7, urgent_min=0.7, can_push=True)
    assert r.urgent is not None and r.urgent.kind == KIND_RELIABILITY
    assert len(r.digest) == 2


def test_no_push_when_cannot_push():
    cands = [_n(KIND_RELIABILITY, "r", 0.95, 0.9)]
    r = route(cands, {KIND_RELIABILITY: 1.0}, digest_top=7, urgent_min=0.7, can_push=False)
    assert r.urgent is None


def test_floored_multiplier_keeps_reliability_visible():
    """A reliability nudge whose kind multiplier is floored (repeated
    dismissals) must still break silence via urgent push — urgent eligibility
    uses the RAW urgency-based score (mult=1.0), not the demoted one. The
    adaptive multiplier only governs digest RANK, never the reliability
    safety alert (I2 review fix — the old behavior let the multiplier
    permanently self-mute the very alert it was demoting)."""
    reliability = _n(KIND_RELIABILITY, "r", 0.95, 0.9)
    health = _n(KIND_HEALTH, "h", 0.5, 0.5)
    r = route(
        [reliability, health],
        {KIND_RELIABILITY: 0.2, KIND_HEALTH: 1.0},
        digest_top=7,
        urgent_min=0.7,
        can_push=True,
    )
    # raw score = 0.6*0.95 + 0.4*0.9 = 0.93 >= 0.7 → urgent breaks silence
    assert r.urgent is not None
    assert r.urgent.kind == KIND_RELIABILITY
    # demoted score = 0.93*0.2 = 0.186 < health's 0.6*0.5+0.4*0.5 = 0.5 → digest
    # rank still demoted despite the urgent push
    assert [n.kind for n in r.digest] == [KIND_HEALTH, KIND_RELIABILITY]


def test_reliability_at_floor_multiplier_still_produces_urgent():
    cands = [_n(KIND_RELIABILITY, "r", 0.95, 0.9)]
    r = route(cands, {KIND_RELIABILITY: 0.2}, digest_top=7, urgent_min=0.7, can_push=True)
    assert r.urgent is not None
    assert r.urgent.kind == KIND_RELIABILITY
