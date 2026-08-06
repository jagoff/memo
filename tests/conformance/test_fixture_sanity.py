"""The fixture itself is load-bearing: every other conformance test trusts that
it really seeded a corpus of the requested size, on disk and in the index."""

from __future__ import annotations

import pytest

from memo.store import VecStore

from .conftest import DIMS, seeded_id

pytestmark = pytest.mark.conformance


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
