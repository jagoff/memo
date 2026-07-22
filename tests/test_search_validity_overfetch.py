"""finding-7: vec-search must widen the kNN pool when the validity gate can
drop rows, so filtered-out nearest neighbours don't cause an under-fill
(fewer than `limit` results). The widening is gated on the corpus actually
holding invalid rows (partial-index EXISTS), so an all-valid corpus pays
nothing on the hot recall path.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from memo.memory.record import _now_iso
from memo.store import VecStore


@pytest.fixture
def store(tmp_path: Path):
    s = VecStore(tmp_path / "vec.db", dims=4)
    yield s
    s.close()


def _emb(*xs: float) -> list[float]:
    norm = math.sqrt(sum(x * x for x in xs)) or 1.0
    return [x / norm for x in xs]


def _past() -> str:
    """Yesterday in memo's stored local-offset shape → lexicographically < now."""
    return (datetime.fromisoformat(_now_iso()) - timedelta(days=1)).isoformat(
        timespec="milliseconds"
    )


def _put(store: VecStore, id_: str, emb: list[float]) -> None:
    store.upsert(
        id_=id_,
        path=f"memory/{id_}.md",
        title=f"Memory {id_}",
        type_="fact",
        tags=[],
        created="2026-01-01T00:00:00-03:00",
        updated="2026-01-01T00:00:00-03:00",
        body_hash=f"hash-{id_}",
        embedding=emb,
    )


def test_index_has_invalid_reflects_corpus(store: VecStore) -> None:
    _put(store, "a", _emb(1, 0, 0, 0))
    assert store._index_has_invalid() is False
    store.update_validity(id_="a", valid_at=None, invalid_at=_past())
    assert store._index_has_invalid() is True


def test_search_overfetches_past_invalid_nearest_to_fill_limit(store: VecStore) -> None:
    # The two NEAREST neighbours of the query are invalid; the two valid ones
    # are farther. A plain vec.k = limit would fetch only the two invalid
    # nearest and return zero after the gate — the under-fill this fixes.
    _put(store, "near1", _emb(1.0, 0.0, 0, 0))  # closest, invalid
    _put(store, "near2", _emb(0.99, 0.14, 0, 0))  # 2nd closest, invalid
    _put(store, "far1", _emb(0.9, 0.44, 0, 0))  # valid
    _put(store, "far2", _emb(0.8, 0.6, 0, 0))  # valid
    for closed in ("near1", "near2"):
        store.update_validity(id_=closed, valid_at=None, invalid_at=_past())

    hits = store.search(_emb(1, 0, 0, 0), limit=2, include_invalid=False)

    ids = {h["id"] for h in hits}
    assert len(hits) == 2, f"under-fill: got {len(hits)} valid hits, expected 2"
    assert ids == {"far1", "far2"}
    assert "near1" not in ids and "near2" not in ids


def test_search_all_valid_returns_nearest_unchanged(store: VecStore) -> None:
    # Budget guard: an all-valid corpus must behave exactly as before —
    # nearest-first, no dropped rows.
    _put(store, "a", _emb(1.0, 0.0, 0, 0))
    _put(store, "b", _emb(0.9, 0.44, 0, 0))
    _put(store, "c", _emb(0.5, 0.87, 0, 0))

    hits = store.search(_emb(1, 0, 0, 0), limit=2, include_invalid=False)

    assert [h["id"] for h in hits] == ["a", "b"]


def test_search_include_invalid_returns_closed_rows(store: VecStore) -> None:
    _put(store, "near1", _emb(1.0, 0.0, 0, 0))
    _put(store, "far1", _emb(0.8, 0.6, 0, 0))
    store.update_validity(id_="near1", valid_at=None, invalid_at=_past())

    hits = store.search(_emb(1, 0, 0, 0), limit=2, include_invalid=True)

    assert {h["id"] for h in hits} == {"near1", "far1"}
