from types import SimpleNamespace

from memo.context_compact import compact_hits_by_entity_overlap


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


def _hit(hid: str, score: float, title: str = "") -> SimpleNamespace:
    return SimpleNamespace(id=hid, title=title or f"T{hid}", score=score)


def test_ubiquitous_entity_overlap_does_not_collapse() -> None:
    entities = {
        "a": [{"name": "memo"}, {"name": "topic-x"}],
        "b": [{"name": "memo"}, {"name": "topic-y"}],
    }
    dfs = {"memo": 9.0, "topic-x": 1.0, "topic-y": 1.0}
    mem = _FakeMemory(_FakeGraph(entities, dfs, total=10))
    hits = [_hit("a", 1.0), _hit("b", 0.9)]

    out = compact_hits_by_entity_overlap(hits, mem, min_idf_overlap=0.5)

    assert [h.id for h in out] == ["a", "b"]


def test_rare_shared_entity_collapses() -> None:
    entities = {"c": [{"name": "topic-z"}], "d": [{"name": "topic-z"}]}
    dfs = {"topic-z": 1.0}
    mem = _FakeMemory(_FakeGraph(entities, dfs, total=10))
    hits = [_hit("c", 1.0), _hit("d", 0.9)]

    out = compact_hits_by_entity_overlap(hits, mem, min_idf_overlap=0.5)

    assert [h.id for h in out] == ["c"]


def test_no_entities_is_noop() -> None:
    mem = _FakeMemory(_FakeGraph({}, {}, total=10))
    hits = [_hit("e", 1.0), _hit("f", 0.9)]

    out = compact_hits_by_entity_overlap(hits, mem, min_idf_overlap=0.5)

    assert [h.id for h in out] == ["e", "f"]


def test_group_below_min_group_size_stays_uncollapsed() -> None:
    entities = {"c": [{"name": "topic-z"}], "d": [{"name": "topic-z"}], "g": []}
    dfs = {"topic-z": 1.0}
    mem = _FakeMemory(_FakeGraph(entities, dfs, total=10))
    hits = [_hit("c", 1.0), _hit("d", 0.9), _hit("g", 0.5)]

    out = compact_hits_by_entity_overlap(hits, mem, min_idf_overlap=0.5, min_group_size=3)

    assert [h.id for h in out] == ["c", "d", "g"]


def test_lookup_failure_returns_hits_unchanged() -> None:
    class _BoomGraph:
        def memory_entities(self, memory_id):
            raise RuntimeError("graph db locked")

    mem = _FakeMemory(_BoomGraph())
    hits = [_hit("a", 1.0), _hit("b", 0.9)]

    out = compact_hits_by_entity_overlap(hits, mem, min_idf_overlap=0.5)

    assert out == hits
