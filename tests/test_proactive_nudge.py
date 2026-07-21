import pytest

from memo.proactive.nudge import KIND_RELIABILITY, Nudge


def test_make_hashes_id_and_requires_evidence():
    n = Nudge.make(
        KIND_RELIABILITY,
        subject_id="abc123",
        urgency=0.9,
        value=0.8,
        title="fact superseded",
        evidence=("abc123",),
        created_at="2026-07-21T00:00:00Z",
    )
    assert n.kind == KIND_RELIABILITY
    assert len(n.id) == 16
    # deterministic content address
    assert (
        n.id
        == Nudge.make(
            KIND_RELIABILITY,
            subject_id="abc123",
            urgency=0.1,
            value=0.1,
            title="other",
            evidence=("abc123",),
            created_at="x",
        ).id
    )


def test_make_rejects_empty_evidence():
    with pytest.raises(ValueError, match="evidence"):
        Nudge.make(
            KIND_RELIABILITY,
            subject_id="x",
            urgency=0.5,
            value=0.5,
            title="t",
            evidence=(),
            created_at="x",
        )
