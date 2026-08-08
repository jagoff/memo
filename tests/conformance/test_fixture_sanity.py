"""The fixture itself is load-bearing: every other conformance test trusts that
it really seeded a corpus of the requested size, on disk and in the index."""

from __future__ import annotations

import pytest

from memo.store import VecStore

from . import conftest as big_corpus_conftest
from .conftest import DIMS, seeded_id

pytestmark = pytest.mark.conformance

# Seeding may commit the Tantivy index a bounded number of times (the bulk
# rebuild plus the closing flush) -- never once per document.
_MAX_SEED_COMMITS = 2


def test_index_holds_every_seeded_memory(big_corpus, corpus_size) -> None:
    store = VecStore(big_corpus.db_path, dims=DIMS)
    try:
        rows = store.count()
    finally:
        store.close()
    assert rows == corpus_size


def test_markdown_files_match_the_index(big_corpus, corpus_size) -> None:
    on_disk = list(big_corpus.memory_dir.rglob("*.md"))
    assert len(on_disk) == corpus_size


def test_seeded_ids_are_addressable(big_corpus) -> None:
    store = VecStore(big_corpus.db_path, dims=DIMS)
    try:
        row = store.get(seeded_id(0))
    finally:
        store.close()
    assert row is not None
    assert row["title"].startswith("Conformance memory 0")


def test_seeding_does_not_commit_the_fts_index_per_document(tmp_path_factory, monkeypatch) -> None:
    """Guards the hang: seeding must not take a Tantivy writer lease per document.

    `VecStore.upsert` dual-writes to Tantivy with an exclusive writer lease and
    one `commit()` per call, measured at ~85ms/doc against this corpus -- ~17
    minutes of silent setup at corpus_size=10001, which is what
    "tests/conformance hangs" actually was. Tantivy is a darwin/arm64-only extra
    (pyproject `[tantivy]`) and CI installs dev/cpu/http on Linux, so CI never
    paid the cost and never caught it. This guard is therefore Mac-only by
    construction, exactly like the defect.

    Both halves matter: the commit budget catches a return to the per-document
    dual-write, and the non-empty-index assertion catches "fixing" it by leaving
    Tantivy switched off, which would silently drop the BM25 conformance
    assertions onto a different backend than this machine dispatches to.
    """
    pytest.importorskip("tantivy")
    from memo.store.tantivy_index import TantivyFTSIndex

    # Pin the backend so an ambient kill-switch cannot make this vacuous.
    monkeypatch.setenv("MEMO_TANTIVY_ENABLED", "1")
    monkeypatch.delenv("MEMO_FTS_BACKEND", raising=False)

    commits = 0
    real_commit = TantivyFTSIndex.commit

    def counting_commit(self: TantivyFTSIndex) -> None:
        nonlocal commits
        commits += 1
        real_commit(self)

    monkeypatch.setattr(TantivyFTSIndex, "commit", counting_commit)

    size = 50
    gen = big_corpus_conftest.big_corpus.__wrapped__(tmp_path_factory, corpus_size=size)
    cfg = next(gen)
    seed_commits = commits
    # Exhaust the generator rather than close() it: the fixture's `mp.undo()`
    # sits after the `yield`, so GeneratorExit would leak its env patches.
    with pytest.raises(StopIteration):
        next(gen)

    assert seed_commits <= _MAX_SEED_COMMITS, (
        f"seeding {size} memories committed the Tantivy index {seed_commits} times; "
        f"a per-document commit costs ~85ms and makes the 10001-memory fixture "
        f"take ~17 minutes"
    )

    index = TantivyFTSIndex.open_or_create(cfg.db_path.parent / "tantivy")
    try:
        hits = index.search_bm25("topic00", limit=5)
    finally:
        index.close()
    assert hits, "fixture left the Tantivy index empty -- BM25 conformance is not covered"
