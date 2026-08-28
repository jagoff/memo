from memo.proactive.detectors.roi import detect_roi
from memo.proactive.nudge import KIND_ROI


class _FakeMem:
    def dead_memory_ids(self, *, limit):
        return ["dead1", "dead2", "dead3"]


def test_roi_nudges_cite_dead_memory_ids():
    ns = detect_roi(_FakeMem(), now="2026-07-21T00:00:00Z")
    assert len(ns) == 1
    n = ns[0]
    assert n.kind == KIND_ROI
    assert n.evidence == ("dead1", "dead2", "dead3")
    assert n.action is None
    assert "3 memories never surfaced" in n.title


def test_roi_empty_source_emits_no_nudge():
    class _EmptyMem:
        def dead_memory_ids(self, *, limit):
            return []

    assert detect_roi(_EmptyMem(), now="2026-07-21T00:00:00Z") == []


def test_roi_guarded_returns_empty_on_error():
    class Boom:
        def dead_memory_ids(self, *, limit):
            raise RuntimeError("boom")

    assert detect_roi(Boom(), now="2026-07-21T00:00:00Z") == []


def _age(mem, memo_id: str, *, days: int) -> None:
    """Backdate `created` so the memory clears the staleness window."""
    mem.store._conn.execute(
        "UPDATE meta SET created = date('now', ?) WHERE id = ?",
        (f"-{days} day", memo_id),
    )
    mem.store._conn.commit()


def test_real_facade_dead_memory_ids_finds_never_accessed_durable_memory(mock_memory):
    rec = mock_memory.save(content="a fact nobody ever looked up again", type_="fact")
    _age(mock_memory, rec.id, days=400)

    ids = mock_memory.dead_memory_ids()

    assert rec.id in ids


def test_real_facade_dead_memory_ids_ignores_a_memory_too_young_to_prune(mock_memory):
    """Never-accessed is not the same as prunable.

    The nudge these ids feed says "candidates to prune", but the remediation
    it points at (`memo maintain`) only archives never-accessed memories older
    than `--stale-days`. Counting fresh ones tells the operator to delete work
    they saved this week: on the live corpus that was 1505 memories against
    which `memo maintain` would have archived zero.
    """
    rec = mock_memory.save(content="a fact saved moments ago", type_="fact")

    assert rec.id not in mock_memory.dead_memory_ids()
    assert mock_memory.dead_memory_count() == 0


def test_roi_title_reports_the_true_total_not_the_evidence_cap():
    """The title used to report `len(ids)`, which equals `limit` whenever the
    corpus has more dead memories than the evidence cap — so a corpus with
    1645 never-surfaced memories rendered as "10 memories never surfaced".
    Evidence stays capped; the count must be the real one."""

    class _TruncatedMem:
        def dead_memory_ids(self, *, limit):
            return [f"dead{i}" for i in range(limit)]

        def dead_memory_count(self):
            return 1645

    n = detect_roi(_TruncatedMem(), now="2026-07-21T00:00:00Z", limit=10)[0]

    assert "1645 memories never surfaced" in n.title
    assert len(n.evidence) == 10


def test_roi_falls_back_to_evidence_length_when_count_is_unavailable():
    """`mem` is duck-typed: a source without the newer count method must still
    emit its nudge rather than silently losing it to the guard."""

    class _NoCountMem:
        def dead_memory_ids(self, *, limit):
            return ["dead1", "dead2"]

    n = detect_roi(_NoCountMem(), now="2026-07-21T00:00:00Z")[0]

    assert "2 memories never surfaced" in n.title


def test_real_facade_dead_memory_count_matches_unlimited_ids(mock_memory):
    for i in range(3):
        rec = mock_memory.save(content=f"fact {i} nobody ever looked up", type_="fact")
        _age(mock_memory, rec.id, days=400)

    assert mock_memory.dead_memory_count() == len(mock_memory.dead_memory_ids(limit=999))
