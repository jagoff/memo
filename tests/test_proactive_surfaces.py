from memo.proactive.arbiter import Routed
from memo.proactive.nudge import KIND_CONTINUITY, KIND_RELIABILITY, Nudge
from memo.proactive.surfaces import render_digest, render_urgent_line


def _n(kind, sid, action=None):
    return Nudge.make(
        kind,
        subject_id=sid,
        urgency=0.9,
        value=0.5,
        title=f"t-{sid}",
        evidence=("m1",),
        action=action,
        created_at="2026-07-21T00:00:00Z",
    )


def test_urgent_line_has_action():
    line = render_urgent_line(_n(KIND_RELIABILITY, "old1", action="memo review old1"))
    assert "memo review old1" in line and line.startswith("⚠️ memo:")


def test_digest_empty_message():
    assert "nothing to surface" in render_digest(Routed([], None))


def test_digest_groups_kinds():
    routed = Routed([_n(KIND_RELIABILITY, "a"), _n(KIND_CONTINUITY, "b")], None)
    out = render_digest(routed)
    assert "Reliability" in out and "Continuity" in out
