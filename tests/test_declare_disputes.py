"""declare-disputes: a surfaced competing pair keeps BOTH sides (default off)."""

from __future__ import annotations

import pytest

from memo.contradict import PairRecord
from memo.memory.record import MemoryRecord


def _rec(rid: str, updated: str, score: float) -> MemoryRecord:
    return MemoryRecord(
        id=rid,
        path=f"{rid}.md",
        title=rid,
        type="fact",
        tags=[],
        created=updated,
        updated=updated,
        body="",
        score=score,
    )


def _competing_pair(older_id: str, newer_id: str, *, status: str = "competing") -> PairRecord:
    return PairRecord(
        pair_id=1,
        memory_id_a=older_id,
        memory_id_b=newer_id,
        relationship="contradiction",
        confidence=0.95,
        rationale="ports",
        status=status,
        detected_at="2026-01-01T00:00:00+00:00",
        resolved_at="2026-02-01T00:00:00+00:00",
        resolution_note="both",
    )


def test_competing_pair_both_kept_when_declared(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_CONTRADICT_PENALTY_ENABLED", "1")
    monkeypatch.setenv("MEMO_DECLARE_DISPUTES", "1")
    older = _rec("a" * 32, "2026-01-01T00:00:00+00:00", 1.0)
    newer = _rec("b" * 32, "2026-02-01T00:00:00+00:00", 0.9)
    pair = _competing_pair(older.id, newer.id)

    def _fake_pairs(ids, *, status="open"):
        # The pair is ALSO visible under status="open" here (unlike production,
        # where a resolved pair only ever has one status) so that, absent the
        # declare-skip, the legacy demote loop would still catch and penalize
        # it — proving declare-disputes is what keeps both sides at full score.
        return [pair] if status in ("competing", "open") else []

    monkeypatch.setattr(mock_memory.contradict_store, "pairs_for_ids", _fake_pairs)

    # score both hits; with declare-disputes on, neither should be penalized to 0.4x
    out = {r.id: r for r in mock_memory._apply_contradict_penalty([older, newer])}
    assert out[older.id].score == pytest.approx(1.0)
    assert out[newer.id].score == pytest.approx(0.9)


def test_flag_off_still_demotes(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_CONTRADICT_PENALTY_ENABLED", "1")
    monkeypatch.setenv("MEMO_DECLARE_DISPUTES", "0")
    older = _rec("a" * 32, "2026-01-01T00:00:00+00:00", 1.0)
    newer = _rec("b" * 32, "2026-02-01T00:00:00+00:00", 0.9)

    # legacy behavior preserved: an OPEN contradiction still demotes the older side
    # by MEMO_CONTRADICT_PENALTY (default 0.4), mirroring
    # test_evolution_penalty_demotes_older_when_both_present in test_contradict.py.
    open_pair = PairRecord(
        pair_id=2,
        memory_id_a=older.id,
        memory_id_b=newer.id,
        relationship="contradiction",
        confidence=0.9,
        rationale="ports",
        status="open",
        detected_at="2026-01-01T00:00:00+00:00",
        resolved_at=None,
        resolution_note=None,
    )

    def _fake_pairs_open(ids, *, status="open"):
        return [open_pair] if status == "open" else []

    monkeypatch.setattr(mock_memory.contradict_store, "pairs_for_ids", _fake_pairs_open)

    out = {r.id: r for r in mock_memory._apply_contradict_penalty([older, newer])}
    assert out[older.id].score == pytest.approx(1.0 * 0.4)
    assert out[newer.id].score == pytest.approx(0.9)
