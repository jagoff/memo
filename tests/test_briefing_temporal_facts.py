from __future__ import annotations

from memo.briefing import temporal_fact_lines


def test_temporal_fact_lines_shows_recent_live_facts(mock_memory):
    rec = mock_memory.save(content="memo records graph facts", title="Graph facts")
    mock_memory.fact_edges.upsert_fact(
        subject="memo capture",
        predicate="records",
        object="graph facts",
        source_record_id=rec.id,
        valid_at="2026-07-10T00:00:00+00:00",
        confidence=0.75,
    )

    lines = temporal_fact_lines(mock_memory, limit=3)
    joined = "\n".join(lines)

    assert "### Temporal facts" in joined
    assert "**memo capture** records **graph facts**" in joined
    assert rec.id[:8] in joined
    assert "conf=0.75" in joined


def test_temporal_fact_lines_empty_store_returns_nothing(mock_memory):
    assert temporal_fact_lines(mock_memory) == []
