"""Unit tests for the tantivy FTS backend (src/memo/store/tantivy_index.py).

Tests that require tantivy are skipped when the package is not installed.
The fallback path (FTS5 when tantivy is absent) is tested via monkeypatching
so it runs in all environments.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from memo.store.tantivy_index import TantivyFTSIndex, _fold_diacritics, _tantivy_available

requires_tantivy = pytest.mark.skipif(not _tantivy_available(), reason="tantivy not installed")


# ---------------------------------------------------------------------------
# _fold_diacritics (pure Python — no skip needed)
# ---------------------------------------------------------------------------


def test_fold_diacritics_strips_accents() -> None:
    assert _fold_diacritics("decisión") == "decision"
    assert _fold_diacritics("ARQUITECTURA") == "arquitectura"
    assert _fold_diacritics("café") == "cafe"
    assert _fold_diacritics("Ñoño") == "nono"


def test_fold_diacritics_ascii_unchanged() -> None:
    assert _fold_diacritics("hello world") == "hello world"


# ---------------------------------------------------------------------------
# TantivyFTSIndex (skipped without tantivy)
# ---------------------------------------------------------------------------


@pytest.fixture()
def idx(tmp_path: Path) -> Iterator[TantivyFTSIndex]:
    index = TantivyFTSIndex.open_or_create(tmp_path / "tantivy")
    yield index
    index.close()


@requires_tantivy
def test_exists_false_before_creation(tmp_path: Path) -> None:
    assert not TantivyFTSIndex.exists(tmp_path / "nonexistent")


@requires_tantivy
def test_exists_true_after_creation(tmp_path: Path) -> None:
    index = TantivyFTSIndex.open_or_create(tmp_path / "idx")
    assert TantivyFTSIndex.exists(tmp_path / "idx")
    index.close()


@requires_tantivy
def test_close_waits_for_merging_threads_and_is_idempotent() -> None:
    """Closing permits immediate directory cleanup even after many commits."""
    with tempfile.TemporaryDirectory() as tmp:
        index = TantivyFTSIndex.open_or_create(Path(tmp) / "tantivy")
        for i in range(100):
            index.add_document(
                f"id{i}",
                f"document number {i}",
                "",
                f"content for merge segment {i}",
            )
            index.commit()
        index.close()
        index.close()


@requires_tantivy
def test_long_lived_reader_does_not_block_other_writer_and_refreshes(tmp_path: Path) -> None:
    """Separate MCP/CLI-style handles can alternate writes on one index."""
    index_dir = tmp_path / "tantivy"
    reader = TantivyFTSIndex.open_or_create(index_dir)
    writer = TantivyFTSIndex.open_or_create(index_dir)
    try:
        writer.add_document("from-cli", "cross process freshness", "", "first commit")
        writer.commit()
        assert reader.search_bm25("freshness", 10)[0]["id"] == "from-cli"

        reader.add_document("from-mcp", "second writer lease", "", "another commit")
        reader.commit()
        assert writer.search_bm25("lease", 10)[0]["id"] == "from-mcp"
    finally:
        reader.close()
        writer.close()


@requires_tantivy
def test_add_and_search_basic(idx: TantivyFTSIndex) -> None:
    idx.add_document("id1", "Python testing guide", "pytest coverage", "How to write good tests")
    idx.commit()
    results = idx.search_bm25("testing", 10)
    assert len(results) == 1
    assert results[0]["id"] == "id1"
    assert 0.0 < results[0]["score"] < 1.0


@requires_tantivy
def test_search_returns_empty_on_no_match(idx: TantivyFTSIndex) -> None:
    idx.add_document("id1", "Python testing", "", "guide")
    idx.commit()
    results = idx.search_bm25("javascript", 10)
    assert results == []


@requires_tantivy
def test_diacritic_folding_at_index_time(idx: TantivyFTSIndex) -> None:
    idx.add_document("id1", "Decisión sobre arquitectura", "", "")
    idx.commit()
    # Query without accent must match document indexed with accent
    results = idx.search_bm25("decision", 10)
    assert len(results) == 1
    assert results[0]["id"] == "id1"


@requires_tantivy
def test_diacritic_folding_at_query_time(idx: TantivyFTSIndex) -> None:
    idx.add_document("id1", "decision arquitectura", "", "")
    idx.commit()
    # Query with accent must match document indexed without accent
    results = idx.search_bm25("decisión", 10)
    assert len(results) == 1


@requires_tantivy
def test_field_boosts_title_ranks_higher(idx: TantivyFTSIndex) -> None:
    # "python" in title → should score higher than "python" in body only
    idx.add_document("title_hit", "Python language", "", "some content")
    idx.add_document("body_hit", "some language", "", "Python is great")
    idx.commit()
    results = idx.search_bm25("python", 10)
    assert len(results) == 2
    ids_in_order = [r["id"] for r in results]
    assert ids_in_order[0] == "title_hit"


@requires_tantivy
def test_delete_removes_from_results(idx: TantivyFTSIndex) -> None:
    idx.add_document("id1", "Python testing", "", "")
    idx.commit()
    assert len(idx.search_bm25("testing", 10)) == 1

    idx.delete_document("id1")
    idx.commit()
    assert idx.search_bm25("testing", 10) == []


@requires_tantivy
def test_upsert_pattern_delete_then_add(idx: TantivyFTSIndex) -> None:
    idx.add_document("id1", "Old title", "", "")
    idx.commit()
    idx.delete_document("id1")
    idx.add_document("id1", "New title with unique word xyzzy", "", "")
    idx.commit()
    assert idx.search_bm25("xyzzy", 10)[0]["id"] == "id1"
    assert idx.search_bm25("old", 10) == []


@requires_tantivy
def test_rebuild_clears_and_reindexes(idx: TantivyFTSIndex) -> None:
    idx.add_document("stale", "stale document", "", "")
    idx.commit()

    records = [
        {"id": "new1", "title": "fresh record one", "tags": "", "body": ""},
        {"id": "new2", "title": "fresh record two", "tags": "", "body": ""},
    ]
    idx.rebuild(records)

    assert idx.search_bm25("stale", 10) == []
    assert len(idx.search_bm25("fresh", 10)) == 2


@requires_tantivy
def test_rebuild_empty_records(idx: TantivyFTSIndex) -> None:
    idx.add_document("id1", "existing", "", "")
    idx.commit()
    idx.rebuild([])
    assert idx.search_bm25("existing", 10) == []


@requires_tantivy
def test_score_normalization_in_zero_one(idx: TantivyFTSIndex) -> None:
    idx.add_document("id1", "Python testing guide", "pytest", "unit tests")
    idx.commit()
    results = idx.search_bm25("testing", 10)
    assert results
    score = results[0]["score"]
    assert 0.0 < score < 1.0, f"score {score} not in (0,1)"


@requires_tantivy
def test_fuzzy_matches_typo(idx: TantivyFTSIndex) -> None:
    idx.add_document("id1", "arquitectura del sistema", "", "")
    idx.commit()
    # One char missing → should match with fuzzy
    results = idx.search_fuzzy("arquitectur", 10)
    assert len(results) == 1
    assert results[0]["id"] == "id1"


@requires_tantivy
def test_fuzzy_typo_in_body(idx: TantivyFTSIndex) -> None:
    idx.add_document("id1", "some title", "", "refactoring codebase structure")
    idx.commit()
    results = idx.search_fuzzy("refactorin", 10)  # missing 'g'
    assert any(r["id"] == "id1" for r in results)


@requires_tantivy
def test_empty_query_returns_empty(idx: TantivyFTSIndex) -> None:
    idx.add_document("id1", "test", "", "")
    idx.commit()
    assert idx.search_bm25("", 10) == []
    assert idx.search_bm25("   ", 10) == []
    assert idx.search_fuzzy("", 10) == []


@requires_tantivy
def test_thread_safety_concurrent_upserts(tmp_path: Path) -> None:
    """Concurrent add+commit calls must not corrupt the index."""
    idx = TantivyFTSIndex.open_or_create(tmp_path / "tantivy")
    errors: list[Exception] = []

    def worker(i: int) -> None:
        try:
            idx.add_document(f"id{i}", f"document number {i}", "", f"content {i}")
            idx.commit()
        except Exception as exc:
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(worker, i) for i in range(8)]
        for f in futures:
            f.result()

    assert not errors, f"thread errors: {errors}"
    results = idx.search_bm25("document", 20)
    assert len(results) == 8
    idx.close()


# ---------------------------------------------------------------------------
# Fallback: FTS5 used when tantivy absent (monkeypatched)
# ---------------------------------------------------------------------------


def test_fallback_to_fts5_when_tantivy_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When tantivy is patched out, VecStore.search_bm25 uses FTS5."""
    import memo.store.store as store_mod

    # Patch where _tantivy_available is used (imported directly into store.py).
    monkeypatch.setattr(store_mod, "_tantivy_available", lambda: False)

    from memo.store.store import VecStore

    db = tmp_path / "test.db"
    store = VecStore(db, dims=4)
    assert store._get_tantivy() is None

    store.upsert(
        id_="abc",
        path="test.md",
        title="Python testing",
        type_="note",
        tags=["python"],
        created="2024-01-01T00:00:00",
        updated="2024-01-01T00:00:00",
        body_hash="abc",
        embedding=[0.5, 0.5, 0.5, 0.5],
        body_text="unit test coverage",
    )

    results = store.search_bm25("testing", limit=5)
    assert len(results) == 1
    assert results[0]["id"] == "abc"
    store.close()
