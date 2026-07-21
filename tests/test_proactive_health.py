from memo.proactive.detectors.health import detect_health
from memo.proactive.nudge import KIND_HEALTH


class _FakeMem:
    def low_confidence_ids(self, *, threshold, limit):
        return ["low1", "low2"]


def test_health_nudges_cite_low_confidence_ids():
    ns = detect_health(_FakeMem(), now="2026-07-21T00:00:00Z")
    assert len(ns) == 1
    n = ns[0]
    assert n.kind == KIND_HEALTH
    assert n.evidence == ("low1", "low2")
    assert n.action == "memo maintain"
    assert "2 low-confidence" in n.title


def test_health_empty_source_emits_no_nudge():
    class _EmptyMem:
        def low_confidence_ids(self, *, threshold, limit):
            return []

    assert detect_health(_EmptyMem(), now="2026-07-21T00:00:00Z") == []


def test_health_guarded_returns_empty_on_error():
    class Boom:
        def low_confidence_ids(self, *, threshold, limit):
            raise RuntimeError("boom")

    assert detect_health(Boom(), now="2026-07-21T00:00:00Z") == []


def test_real_facade_low_confidence_ids_reads_memory_health(mock_memory):
    rec = mock_memory.save(content="a shaky OCR'd note", type_="note")
    mock_memory.store.set_confidence_batch([(rec.id, 0.2)])

    ids = mock_memory.low_confidence_ids(threshold=0.4)

    assert rec.id in ids
