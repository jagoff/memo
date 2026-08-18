"""`update_meta` must never hand the tantivy index a NULL body.

The live corpus contains meta rows whose FTS `body` is NULL. `update_meta`
read that column straight into `add_document`, where tantivy raised
`normalize() argument 2 must be str, not None` — caught, logged, and followed
by `_mark_tantivy_unhealthy()`, so a single such row silently downgraded the
BM25 backend from tantivy to FTS5 for the rest of the process. Observed every
night in ~/Library/Logs/memo/nightly.err through 2026-08-13.
"""

from __future__ import annotations

from pathlib import Path

from memo.store import VecStore


class _StrictTantivy:
    """Stands in for the real index: rejects a non-str field the way tantivy does."""

    def __init__(self) -> None:
        self.added: list[tuple[str, str, str, str]] = []

    def delete_document(self, id_: str) -> None:
        pass

    def add_document(self, id_: str, title: str, tags: str, body: str) -> None:
        for field in (id_, title, tags, body):
            if not isinstance(field, str):
                raise TypeError("normalize() argument 2 must be str, not None")
        self.added.append((id_, title, tags, body))

    def commit(self) -> None:
        pass


def _store_with_null_fts_body(tmp_path: Path) -> tuple[VecStore, str]:
    store = VecStore(tmp_path / "vec.db", dims=4)
    id_ = "a" * 32
    store.upsert(
        id_=id_,
        path=str(tmp_path / "a.md"),
        title="original title",
        type_="note",
        tags=["one"],
        created="2026-01-01T00:00:00+00:00",
        updated="2026-01-01T00:00:00+00:00",
        body_hash="h0",
        embedding=[0.1, 0.2, 0.3, 0.4],
        body_text="some body",
    )
    # The shape the live index actually holds for ~6k rows.
    with store._tx() as cx:
        cx.execute("UPDATE fts SET body = NULL WHERE id = ?", (id_,))
    return store, id_


def test_update_meta_coerces_a_null_body_before_the_tantivy_write(tmp_path, monkeypatch):
    store, id_ = _store_with_null_fts_body(tmp_path)
    fake = _StrictTantivy()
    monkeypatch.setattr(store, "_get_tantivy", lambda: fake)

    updated = store.update_meta(
        id_=id_,
        title="new title",
        type_="note",
        tags=["one", "two"],
        updated="2026-02-01T00:00:00+00:00",
    )

    assert updated is True
    assert fake.added == [(id_, "new title", "one two", "")]
    store.close()
