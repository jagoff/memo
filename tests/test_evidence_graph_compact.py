from __future__ import annotations

from memo.memory.evidence_graph_compact import compact_by_entity_overlap
from memo.memory.record import MemoryRecord


class _FakeGraph:
    def __init__(self, entities: dict[str, list[dict]], dfs: dict[str, float], total: int) -> None:
        self._entities = entities
        self._dfs = dfs
        self._total = total

    def memory_entities(self, memory_id: str) -> list[dict]:
        return self._entities.get(memory_id, [])

    def total_indexed_memories(self) -> int:
        return self._total

    def entity_doc_freqs(self, names) -> dict[str, float]:
        return {n: self._dfs[n] for n in names if n in self._dfs}


class _FakeMemory:
    def __init__(self, graph) -> None:
        self.graph = graph


def _hit(hid: str, score: float, title: str = "") -> MemoryRecord:
    return MemoryRecord(
        id=hid,
        path=f"{hid}.md",
        title=title or f"T{hid}",
        type="note",
        tags=[],
        created="2026-08-04T00:00:00",
        updated="2026-08-04T00:00:00",
        body="body",
        score=score,
    )


def test_ubiquitous_entity_overlap_does_not_collapse() -> None:
    entities = {
        "a": [{"name": "memo"}, {"name": "topic-x"}],
        "b": [{"name": "memo"}, {"name": "topic-y"}],
    }
    dfs = {"memo": 9.0, "topic-x": 1.0, "topic-y": 1.0}
    mem = _FakeMemory(_FakeGraph(entities, dfs, total=10))
    hits = [_hit("a", 1.0), _hit("b", 0.9)]

    out = compact_by_entity_overlap(hits, mem, min_idf_overlap=0.5)

    assert {h.id for h in out} == {"a", "b"}
    assert all(not h.extra.get("provenance", {}).get("related_ids") for h in out)


def test_rare_shared_entity_collapses() -> None:
    entities = {"c": [{"name": "topic-z"}], "d": [{"name": "topic-z"}]}
    dfs = {"topic-z": 1.0}
    mem = _FakeMemory(_FakeGraph(entities, dfs, total=10))
    hits = [_hit("c", 1.0), _hit("d", 0.9, title="Td")]

    out = compact_by_entity_overlap(hits, mem, min_idf_overlap=0.5)

    assert len(out) == 1
    assert out[0].id == "c"
    assert out[0].extra["provenance"]["related_ids"] == [("d", "Td")]


def test_no_entities_is_noop() -> None:
    mem = _FakeMemory(_FakeGraph({}, {}, total=10))
    hits = [_hit("e", 1.0), _hit("f", 0.9)]

    out = compact_by_entity_overlap(hits, mem, min_idf_overlap=0.5)

    assert {h.id for h in out} == {"e", "f"}
    assert all(not h.extra.get("provenance", {}).get("related_ids") for h in out)


def test_group_below_min_group_size_stays_uncollapsed() -> None:
    entities = {"c": [{"name": "topic-z"}], "d": [{"name": "topic-z"}], "g": []}
    dfs = {"topic-z": 1.0}
    mem = _FakeMemory(_FakeGraph(entities, dfs, total=10))
    hits = [_hit("c", 1.0), _hit("d", 0.9), _hit("g", 0.5)]

    out = compact_by_entity_overlap(hits, mem, min_idf_overlap=0.5, min_group_size=3)

    assert {h.id for h in out} == {"c", "d", "g"}
    assert all(not h.extra.get("provenance", {}).get("related_ids") for h in out)


def test_lookup_failure_returns_hits_unchanged() -> None:
    class _BoomGraph:
        def memory_entities(self, memory_id):
            raise RuntimeError("graph db locked")

    mem = _FakeMemory(_BoomGraph())
    hits = [_hit("a", 1.0), _hit("b", 0.9)]

    out = compact_by_entity_overlap(hits, mem, min_idf_overlap=0.5)

    assert out == hits
