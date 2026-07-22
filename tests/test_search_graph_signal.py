from __future__ import annotations

from typing import Any


def _enable_graph(monkeypatch, *, reason: bool = True) -> None:
    monkeypatch.setenv("MEMO_GRAPH_PROJECTION_ENABLED", "1")
    monkeypatch.setenv("MEMO_GRAPH_SIGNAL_ENABLED", "1")
    monkeypatch.setenv("MEMO_GRAPH_REASON_ENABLED", "1" if reason else "0")
    monkeypatch.setenv("MEMO_GRAPH_MIN_ENTITY_IDF", "0")
    monkeypatch.setenv("MEMO_GRAPH_HUB_MAX_DOC_FREQ_RATIO", "1.0")
    monkeypatch.setenv("MEMO_HEALTH_SCORES_DISABLED", "1")


def _row(mem, memory_id: str, score: float) -> dict[str, Any]:
    row = mem.store.get(memory_id)
    assert row is not None
    return {**row, "score": score}


def test_search_attaches_curated_graph_reason_when_enabled(
    mem_with_stub,
    monkeypatch,
) -> None:
    _enable_graph(monkeypatch)
    mem = mem_with_stub
    rec_a = mem.save(
        content="MLX daemon anchor note with graph search marker",
        title="MLX Daemon",
        type_="note",
    )
    mem.save(
        content="Separate project note with graph search marker",
        title="Separate Hub",
        type_="note",
    )
    mem.graph.record_extraction(
        memory_id=rec_a.id,
        memory_date="2026-07-10",
        entities=[
            {"name": "MLX", "type": "technology"},
            {"name": "daemon", "type": "technology"},
        ],
        extracted_at="2026-07-10T00:00:00Z",
        extractor="explicit",
    )
    mem.rebuild_graph()

    trace: list[dict[str, object]] = []
    hits = mem.search("mlx", limit=5, mode="bm25", _trace=trace)

    assert any(t["stage"] == "graph_signal" for t in trace)
    reason_hits = [h for h in hits if (h.extra or {}).get("graph_reason")]
    assert reason_hits
    assert reason_hits[0].id == rec_a.id
    reason = reason_hits[0].extra["graph_reason"]
    assert reason["mode"] == "curated_proximity"
    assert reason["projection_version"]
    assert reason["edges"][0]["evidence_ids"] == [f"memory://{rec_a.id}"]


def test_search_graph_reason_includes_semantic_relations_when_enabled(
    mem_with_stub,
    monkeypatch,
) -> None:
    _enable_graph(monkeypatch)
    monkeypatch.setenv("MEMO_GRAPH_SEMANTIC_RELATIONS", "1")
    mem = mem_with_stub
    rec_a = mem.save(content="MLX daemon relation marker", title="MLX Relation Daemon")
    rec_b = mem.save(content="Separate relation marker", title="Separate Relation")
    mem.graph.record_extraction(
        memory_id=rec_a.id,
        memory_date="2026-07-10",
        entities=[
            {"name": "MLX", "type": "technology"},
            {"name": "daemon", "type": "technology"},
        ],
        extracted_at="2026-07-10T00:00:00Z",
        extractor="explicit",
    )
    mem.graph.upsert_semantic_relation(
        source_kind="memory",
        source_id=rec_a.id,
        target_kind="memory",
        target_id=rec_b.id,
        relation="extends",
        derived_from="test",
    )
    mem.rebuild_graph()

    hits = mem.search("mlx", limit=5, mode="bm25")
    reasons = [(h.extra or {}).get("graph_reason") for h in hits]

    assert any(reason and reason.get("relations") for reason in reasons)


def test_search_graph_signal_never_injects_outside_candidate_set(
    mem_with_stub,
    monkeypatch,
) -> None:
    _enable_graph(monkeypatch)
    # Deprecated compatibility switches must be accepted but inert.
    monkeypatch.setenv("MEMO_GRAPH_RETRIEVAL_ENABLED", "1")
    monkeypatch.setenv("MEMO_GRAPH_EXPANSION_ENABLED", "1")
    mem = mem_with_stub
    eligible = mem.save(content="eligible marker", title="Eligible")
    adjacent = mem.save(content="not returned by lexical retrieval", title="Adjacent")
    mem.graph.record_extraction(
        memory_id=eligible.id,
        memory_date="2026-07-10",
        entities=[
            {"name": "MLX", "type": "technology"},
            {"name": "daemon", "type": "technology"},
        ],
        extracted_at="2026-07-10T00:00:00Z",
        extractor="explicit",
    )
    mem.graph.record_extraction(
        memory_id=adjacent.id,
        memory_date="2026-07-10",
        entities=[{"name": "daemon", "type": "technology"}],
        extracted_at="2026-07-10T00:00:00Z",
        extractor="explicit",
    )
    mem.rebuild_graph()
    monkeypatch.setattr(
        mem.store,
        "search_bm25",
        lambda *_args, **_kwargs: [_row(mem, eligible.id, 0.9)],
    )

    hits = mem.search("mlx", mode="bm25", limit=5)

    assert [hit.id for hit in hits] == [eligible.id]
    assert adjacent.id not in {hit.id for hit in hits}


def test_graph_order_preserves_scores_used_by_recall_gates(
    mem_with_stub,
    monkeypatch,
) -> None:
    mem = mem_with_stub
    monkeypatch.setenv("MEMO_GRAPH_HUB_MAX_DOC_FREQ_RATIO", "1.0")
    first = mem.save(content="ordering marker generic", title="Base First")
    second = mem.save(content="ordering marker daemon", title="Graph First")
    mem.graph.record_extraction(
        memory_id=second.id,
        memory_date="2026-07-10",
        entities=[
            {"name": "MLX", "type": "technology"},
            {"name": "daemon", "type": "technology"},
        ],
        extracted_at="2026-07-10T00:00:00Z",
        extractor="explicit",
    )
    mem.rebuild_graph()

    def _ranked_rows(*_args, **_kwargs):
        return [_row(mem, first.id, 0.9), _row(mem, second.id, 0.8)]

    monkeypatch.setattr(mem.store, "search_bm25", _ranked_rows)
    monkeypatch.setenv("MEMO_HEALTH_SCORES_DISABLED", "1")
    before = mem.search("mlx", mode="bm25", limit=5)
    _enable_graph(monkeypatch, reason=False)
    after = mem.search("mlx", mode="bm25", limit=5)

    assert {hit.id: hit.score for hit in after} == {hit.id: hit.score for hit in before}
    assert [hit.id for hit in before] == [first.id, second.id]
    assert [hit.id for hit in after] == [second.id, first.id]


def test_missing_projection_is_exact_identity(mem_with_stub, monkeypatch) -> None:
    mem = mem_with_stub
    first = mem.save(content="missing projection marker", title="First")
    second = mem.save(content="missing projection marker", title="Second")

    def _ranked_rows(*_args, **_kwargs):
        return [_row(mem, first.id, 0.9), _row(mem, second.id, 0.8)]

    monkeypatch.setattr(mem.store, "search_bm25", _ranked_rows)
    monkeypatch.setenv("MEMO_HEALTH_SCORES_DISABLED", "1")
    base = mem.search("mlx", mode="bm25", limit=5)
    _enable_graph(monkeypatch)
    trace: list[dict[str, object]] = []
    enabled = mem.search("mlx", mode="bm25", limit=5, _trace=trace)

    assert [(hit.id, hit.score) for hit in enabled] == [(hit.id, hit.score) for hit in base]
    graph_stage = next(item for item in trace if item["stage"] == "graph_signal")
    assert graph_stage["skipped"] == "projection_missing"
