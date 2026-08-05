"""Regression: renaming a `fact` must not leave its assertion edge asserting
the old title.

Found running memo as an end user: after `memo rename`, the SessionStart
briefing's "Temporal facts" section still showed the pre-rename title. A
`fact` with no declared edges gets a coarse `memory asserts <title>` edge at
save time, and the update path never refreshed it — the save paths were the
only callers of the fact-edge upsert.

The old assertion is closed rather than deleted: memo's fact edges are
bi-temporal, so "this memory asserted X until now" stays queryable.
"""

from __future__ import annotations


def _assertion_edges(memory, record_id: str, *, include_inactive: bool = False) -> list[dict]:
    return [
        edge
        for edge in memory.fact_edges.query(
            source_record_id=record_id, include_inactive=include_inactive
        )
        if edge.get("predicate") == "asserts"
    ]


def test_rename_closes_the_stale_assertion_and_opens_the_new_one(mock_memory) -> None:
    record = mock_memory.save(
        content="Deploys run at 03:00 UTC.", title="deploy window", type_="fact"
    )
    before = _assertion_edges(mock_memory, record.id)
    assert [edge["object"] for edge in before] == ["deploy window"]
    assert before[0]["invalid_at"] is None

    mock_memory.update(record.id, title="deploy window (v2)")

    # The current view asserts only the new title — this is what the briefing
    # and the graph read.
    assert [edge["object"] for edge in _assertion_edges(mock_memory, record.id)] == [
        "deploy window (v2)"
    ]
    # The old assertion is closed, not erased: memo's edges are bi-temporal.
    history = {
        edge["object"]: edge
        for edge in _assertion_edges(mock_memory, record.id, include_inactive=True)
    }
    assert set(history) == {"deploy window", "deploy window (v2)"}
    assert history["deploy window"]["invalid_at"] is not None, "stale assertion stayed open"
    assert history["deploy window (v2)"]["invalid_at"] is None


def test_editing_only_the_body_leaves_the_assertion_alone(mock_memory) -> None:
    record = mock_memory.save(
        content="Deploys run at 03:00 UTC.", title="deploy window", type_="fact"
    )

    mock_memory.update(record.id, content="Deploys run at 05:00 UTC.")

    edges = _assertion_edges(mock_memory, record.id)
    assert [edge["object"] for edge in edges] == ["deploy window"]
    assert edges[0]["invalid_at"] is None


def test_renaming_a_non_fact_writes_no_assertion_edge(mock_memory) -> None:
    record = mock_memory.save(content="a note body", title="note title", type_="note")

    mock_memory.update(record.id, title="note title (v2)")

    assert _assertion_edges(mock_memory, record.id) == []
