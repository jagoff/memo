"""_uncertain tag: recall-excluding via exclude_tags, still searchable on demand."""

from __future__ import annotations


def test_search_exclude_tags_drops_uncertain(mock_memory):
    q = mock_memory.save(
        content="dato incierto sobre llamas andinas " * 3, title="Incierto", tags=["_uncertain"]
    )
    ok = mock_memory.save(content="dato firme sobre llamas andinas " * 3, title="Firme")
    ids = {h.id for h in mock_memory.search("llamas", mode="vec", exclude_tags={"_uncertain"})}
    assert ok.id in ids
    assert q.id not in ids


def test_search_without_exclude_tags_still_finds_uncertain(mock_memory):
    q = mock_memory.save(
        content="dato incierto sobre llamas andinas " * 3, title="Incierto2", tags=["_uncertain"]
    )
    ids = {h.id for h in mock_memory.search("llamas", mode="vec")}
    assert q.id in ids


def test_bm25_leg_also_excluded(mock_memory):
    q = mock_memory.save(
        content="dato incierto sobre ornitorrincos australianos " * 3,
        title="IncBM",
        tags=["_uncertain"],
    )
    ids = {
        h.id for h in mock_memory.search("ornitorrincos", mode="bm25", exclude_tags={"_uncertain"})
    }
    assert q.id not in ids


def test_uncertain_exclusion_helper_follows_flag(monkeypatch):
    from memo.recall_logic import uncertain_exclusion

    assert uncertain_exclusion() == {"_uncertain"}  # default True (opt-out)
    monkeypatch.setenv("MEMO_RECALL_EXCLUDE_UNCERTAIN", "0")
    assert uncertain_exclusion() is None
