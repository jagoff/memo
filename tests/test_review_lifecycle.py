from __future__ import annotations

from datetime import datetime

import pytest

from memo.memory.lifecycle_ops import review_interval_days
from memo.tiers import VerificationState


@pytest.mark.parametrize(
    ("type_", "tags", "expected"),
    [
        ("preference", [], 90),
        ("decision", ["config"], 90),
        ("decision", [], 180),
        ("note", ["policy"], 365),
        ("note", ["architecture"], 365),
        ("reference", ["config"], None),
        ("fact", [], None),
    ],
)
def test_review_policy_matrix(type_: str, tags: list[str], expected: int | None) -> None:
    assert review_interval_days(type_, tags) == expected


def test_save_schedules_only_policy_eligible_types(mock_memory) -> None:
    preference = mock_memory.save(
        content="prefer compact output", title="output preference", type_="preference"
    )
    fact = mock_memory.save(content="stable fact", title="plain fact", type_="fact")

    assert preference.review_after is not None
    assert fact.review_after is None
    delta = datetime.fromisoformat(preference.review_after) - datetime.fromisoformat(
        preference.updated
    )
    assert 89 <= delta.days <= 90


def test_mark_reviewed_persists_evidence_and_survives_reindex(mock_memory) -> None:
    record = mock_memory.save(
        content="reviewable config", title="config to review", type_="preference"
    )

    reviewed = mock_memory.mark_reviewed(
        record.id, evidence="confirmed against ticket 123", actor="tester"
    )
    evidence = mock_memory.store.review_evidence(record.id)
    mock_memory.reindex(force=True)
    reloaded = mock_memory.get(record.id)

    assert reviewed.verification_state == VerificationState.VERIFIED
    assert evidence[0]["evidence"] == "confirmed against ticket 123"
    assert evidence[0]["actor"] == "tester"
    assert reloaded is not None and reloaded.review_after == reviewed.review_after
    assert mock_memory.store.review_evidence(record.id) == evidence


def test_mark_reviewed_exact_retry_is_idempotent(mock_memory) -> None:
    record = mock_memory.save(
        content="reviewable retry config", title="retry review", type_="preference"
    )

    first = mock_memory.mark_reviewed(record.id, evidence="ticket 456", actor="tester")
    second = mock_memory.mark_reviewed(record.id, evidence="ticket 456", actor="tester")

    assert second.review_after == first.review_after
    assert len(mock_memory.store.review_evidence(record.id)) == 1


def test_judged_conflict_is_due_even_before_schedule(mock_memory) -> None:
    first = mock_memory.save(content="endpoint A", title="A", type_="fact")
    second = mock_memory.save(content="endpoint B", title="B", type_="fact")
    mock_memory.compare_memories(first.id, second.id, "conflicts_with", reason="different values")

    due = mock_memory.list_due_reviews()

    assert {row["id"] for row in due} == {first.id, second.id}
    assert all(row["open_conflict"] == 1 for row in due)


def test_invalidate_is_canonical_and_hides_current_recall(mock_memory) -> None:
    record = mock_memory.save(
        content="obsolete turquoise setting", title="obsolete setting", type_="fact"
    )

    invalidated = mock_memory.invalidate(record.id, reason="removed upstream")
    current = mock_memory.search("obsolete turquoise setting", limit=10)
    historical = mock_memory.search(
        "obsolete turquoise setting", limit=10, as_of=record.created
    )

    assert invalidated.invalid_at is not None
    assert record.id not in {row.id for row in current}
    assert record.id in {row.id for row in historical}
