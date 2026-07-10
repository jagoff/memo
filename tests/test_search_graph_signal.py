from __future__ import annotations


def test_search_attaches_graph_reason_when_enabled(mem_with_stub, monkeypatch) -> None:
    monkeypatch.setenv("MEMO_GRAPH_SIGNAL_ENABLED", "1")
    monkeypatch.setenv("MEMO_GRAPH_REASON_ENABLED", "1")
    monkeypatch.setenv("MEMO_GRAPH_HUB_MAX_DOC_FREQ_RATIO", "1.0")
    monkeypatch.setattr("memo.embedder.MLXEmbedder.unload", lambda self: None)
    mem = mem_with_stub

    rec_a = mem.save(
        content="MLX daemon anchor note with graph search marker",
        title="MLX Daemon",
        type_="note",
    )
    rec_b = mem.save(
        content="MLX separate project note with graph search marker and unrelated body",
        title="MLX Hub",
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
        )
    mem.graph.rebuild_edges()
    mem.graph.record_extraction(
        memory_id=rec_b.id,
        memory_date="2026-07-10",
        entities=[
            {"name": "memo", "type": "project"},
        ],
        extracted_at="2026-07-10T00:00:00Z",
    )
    mem.graph.rebuild_edges()

    trace: list[dict[str, object]] = []
    hits = mem.search("mlx", limit=5, mode="bm25", _trace=trace)

    assert any(t["stage"] == "graph_signal" for t in trace)
    reason_hits = [h for h in hits if (h.extra or {}).get("graph_reason")]
    assert reason_hits
    assert reason_hits[0].id == rec_a.id
    assert reason_hits[0].extra["graph_reason"]["mode"] == "proximity"
