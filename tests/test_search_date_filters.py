"""date_from/date_to hard window on Memory.search / VecStore.search."""

from __future__ import annotations

import pytest


def _backdate(mem, id_: str, updated: str) -> None:
    mem.store._conn.execute("UPDATE meta SET updated = ? WHERE id = ?", (updated, id_))
    mem.store._conn.commit()


def test_vec_search_date_from_drops_old_rows(mock_memory):
    old = mock_memory.save(content="hecho viejo sobre zanahorias moradas " * 3, title="Viejo")
    new = mock_memory.save(content="hecho nuevo sobre zanahorias moradas " * 3, title="Nuevo")
    _backdate(mock_memory, old.id, "2026-01-01T00:00:00")
    ids = {h.id for h in mock_memory.search("zanahorias", mode="vec", date_from="2026-06-01")}
    assert new.id in ids
    assert old.id not in ids


def test_vec_search_date_to_is_end_of_day_inclusive(mock_memory):
    old = mock_memory.save(content="hecho viejo sobre zanahorias moradas " * 3, title="Viejo2")
    _backdate(mock_memory, old.id, "2026-01-15T18:30:00")
    ids = {h.id for h in mock_memory.search("zanahorias", mode="vec", date_to="2026-01-15")}
    assert old.id in ids  # bare date must include the whole day


def test_bm25_search_respects_date_window(mock_memory):
    old = mock_memory.save(
        content="reunión sobre flamencos rosados del proyecto " * 3, title="ViejoBM"
    )
    new = mock_memory.save(
        content="reunión sobre flamencos rosados del sprint " * 3, title="NuevoBM"
    )
    _backdate(mock_memory, old.id, "2026-01-01T00:00:00")
    ids = {h.id for h in mock_memory.search("flamencos", mode="bm25", date_from="2026-06-01")}
    assert new.id in ids
    assert old.id not in ids


@pytest.mark.parametrize("mode", ["bm25", "exact", "fuzzy"])
def test_bm25_date_filter_does_not_spend_limit_on_ineligible_hits(mock_memory, mode):
    old = [
        mock_memory.save(
            content="crowdneedle crowdneedle crowdneedle",
            title=f"Old {i}",
            auto_project=False,
        )
        for i in range(25)
    ]
    eligible = mock_memory.save(
        content="crowdneedle",
        title="Eligible",
        auto_project=False,
    )
    with mock_memory.store._conn:
        mock_memory.store._conn.executemany(
            "UPDATE meta SET updated = ? WHERE id = ?",
            [("2026-01-01T00:00:00+00:00", record.id) for record in old],
        )
        mock_memory.store._conn.execute(
            "UPDATE meta SET updated = ? WHERE id = ?",
            ("2026-07-01T00:00:00+00:00", eligible.id),
        )

    hits = mock_memory.search(
        "crowdneedle",
        mode=mode,
        date_from="2026-06-01",
        limit=1,
    )

    assert [hit.id for hit in hits] == [eligible.id]


@pytest.mark.parametrize("mode", ["bm25", "exact", "fuzzy"])
def test_bm25_tag_filter_does_not_spend_limit_on_ineligible_hits(mock_memory, mode):
    for i in range(25):
        mock_memory.save(
            content="tagcrowd tagcrowd tagcrowd",
            title=f"Blocked {i}",
            tags=["blocked"],
            auto_project=False,
        )
    eligible = mock_memory.save(
        content="tagcrowd",
        title="Eligible",
        auto_project=False,
    )

    hits = mock_memory.search(
        "tagcrowd",
        mode=mode,
        exclude_tags={"blocked"},
        limit=1,
    )

    assert [hit.id for hit in hits] == [eligible.id]


def test_lexical_filter_candidate_fetch_is_bounded(mock_memory, monkeypatch):
    requested: list[int] = []

    class _SpyTantivy:
        def search_bm25(self, query: str, limit: int):
            requested.append(limit)
            return []

    spy = _SpyTantivy()
    monkeypatch.setattr(mock_memory.store, "_get_tantivy", lambda: spy)

    mock_memory.store.search_bm25(
        "boundedtoken",
        limit=100,
        date_from="2026-01-01",
    )
    mock_memory.store.search_bm25("boundedtoken", limit=100)

    assert requested == [1_000, 100]


def test_lexical_explicit_empty_filters_preserve_unfiltered_ranking(mock_memory, monkeypatch):
    monkeypatch.setattr(mock_memory.store, "_get_tantivy", lambda: None)
    for i in range(5):
        mock_memory.save(
            content=f"stabletoken stabletoken detail-{i}",
            title=f"Stable {i}",
            auto_project=False,
        )

    plain = mock_memory.store.search_bm25("stabletoken", limit=10)
    explicit_empty = mock_memory.store.search_bm25(
        "stabletoken",
        limit=10,
        date_from=None,
        date_to=None,
        exclude_tags=set(),
    )

    assert [row["id"] for row in plain] == [row["id"] for row in explicit_empty]
