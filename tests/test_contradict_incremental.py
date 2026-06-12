"""contradict scan incremental: push `since` to the DB so incremental runs
only fetch new/changed anchors, instead of fetching `max_memorias` rows and
filtering client-side (which silently drops fresh anchors past the limit).
"""

from __future__ import annotations


def _backdate(mock_memory, rec_id: str, ts: str) -> None:
    with mock_memory.store._conn:
        mock_memory.store._conn.execute(
            "UPDATE meta SET updated=? WHERE id=?", (ts, rec_id)
        )


def test_list_updated_since_filters_at_db_level(mock_memory):
    old = mock_memory.save(content="old anchor", title="Old")
    new = mock_memory.save(content="new anchor", title="New")
    _backdate(mock_memory, old.id, "2000-01-01T00:00:00Z")

    res = mock_memory.list(updated_since="2010-01-01T00:00:00Z")
    ids = {r.id for r in res}
    assert new.id in ids
    assert old.id not in ids


def test_list_updated_since_respects_limit_over_recency(mock_memory):
    # With many old rows and the limit small, a client-side `since` filter
    # would fetch the old rows first and drop the fresh one. DB-level filter
    # returns the fresh anchor within the limit.
    fresh = mock_memory.save(content="fresh", title="Fresh")
    for i in range(5):
        old = mock_memory.save(content=f"old {i}", title=f"Old {i}")
        _backdate(mock_memory, old.id, "2001-01-01T00:00:00Z")

    res = mock_memory.list(limit=2, updated_since="2010-01-01T00:00:00Z")
    assert [r.id for r in res] == [fresh.id]


def test_scan_corpus_pushes_since_to_db(mock_memory, monkeypatch):
    captured: dict = {}
    orig = mock_memory.list

    def spy(**kwargs):
        captured.update(kwargs)
        return orig(**kwargs)

    monkeypatch.setattr(mock_memory, "list", spy)
    mock_memory.contradict_scanner.scan_corpus(since="2020-01-01T00:00:00Z", max_memorias=10)
    assert captured.get("updated_since") == "2020-01-01T00:00:00Z"
