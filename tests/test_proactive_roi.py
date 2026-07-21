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


def test_real_facade_dead_memory_ids_finds_never_accessed_durable_memory(mock_memory):
    rec = mock_memory.save(content="a fact nobody ever looked up again", type_="fact")

    ids = mock_memory.dead_memory_ids()

    assert rec.id in ids
