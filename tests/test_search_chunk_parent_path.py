"""Chunk flood from the `parent_path` schema (MEMO_SEARCH_CHUNK_PARENT).

Found running memo as an end user: `memo search "Comandos disponibles CLI por
función"` returned 10/10 hits that were ten chunks of the *same* source file.
MEMO_SEARCH_CHUNK_PARENT did not help, because two chunk->parent schemas
coexist and only one was handled:

- `MEMO_CHUNK_INGEST` writes `extra.parent_id` AND materializes a parent
  record. `_map_chunks_to_parents` resolves it — that path works.
- `memo ingest --chunk` writes `extra.parent_path` + `chunk_seq` and never
  materializes a parent record. `self.get(parent_id)` had nothing to resolve,
  so every chunk survived and flooded the window. That is 92% of the chunked
  corpus, including the repro above.

There is no parent record to resolve to and none is fabricated: the collapse
keeps the best-ranked chunk of each source document — a real, citable row with
its own id — and drops its siblings. Freed slots refill from the wide
candidate pool (`_CHUNK_PARENT_POOL_FACTOR`), which the flag already widens.
"""

from __future__ import annotations

import pytest

QUERY = "comandos disponibles CLI por funcion"
SOURCE_DOC = "Vault/Referencia/Comandos.md"


def _ingest_chunk(mem, seq: int, *, parent_path: str = SOURCE_DOC):
    """A chunk exactly as `memo ingest --chunk` writes it: parent_path +
    chunk_seq in `extra`, type=reference, and NO parent_id."""
    return mem.save(
        # The body carries `parent_path` so two source documents never collide
        # on memo's content dedup — each chunk stays its own stored record.
        content=(
            f"## Comandos disponibles CLI por funcion — seccion {seq}\n\n"
            "Listado de comandos disponibles del CLI agrupados por funcion, "
            f"fragmento {seq} del documento de referencia {parent_path}."
        ),
        title=f"Comandos (§{seq}) — {parent_path}",
        type_="reference",
        tags=["chunk"],
        extra={
            "parent_path": parent_path,
            "chunk_seq": seq,
            "chunk_count": 6,
            "chunk_heading": f"seccion {seq}",
        },
    )


@pytest.fixture
def flooded_corpus(mock_memory, monkeypatch):
    """Six chunks of ONE source doc (parent_path only) plus unrelated
    documents that match the query at least as well."""
    monkeypatch.setenv("MEMO_SEARCH_CHUNK_PARENT", "1")
    chunk_ids = [_ingest_chunk(mock_memory, seq).id for seq in range(6)]
    others = [
        mock_memory.save(
            content=(
                "Comandos disponibles CLI por funcion: referencia independiente "
                f"numero {index} con el listado completo de comandos por funcion."
            ),
            title=f"Otra referencia de comandos {index}",
            type_="reference",
        ).id
        for index in range(4)
    ]
    return mock_memory, chunk_ids, others


def _parent_paths(hits) -> list[str]:
    return [(hit.extra or {}).get("parent_path") for hit in hits]


def test_one_source_document_no_longer_dominates_the_window(flooded_corpus) -> None:
    """The bug itself: pre-fix every slot was a chunk of SOURCE_DOC."""
    memory, _chunk_ids, _others = flooded_corpus

    hits = memory.search(QUERY, limit=5, mode="bm25", type_=None)

    from_source_doc = [path for path in _parent_paths(hits) if path == SOURCE_DOC]
    assert len(from_source_doc) <= 1, (
        f"{len(from_source_doc)}/{len(hits)} hits are chunks of the same source "
        f"document — the flood is still crowding out every other answer"
    )


def test_collapse_keeps_exactly_one_chunk_per_source_document(flooded_corpus) -> None:
    memory, _chunk_ids, _others = flooded_corpus

    hits = memory.search(QUERY, limit=10, mode="bm25", type_=None)

    seen = [path for path in _parent_paths(hits) if path]
    assert len(seen) == len(set(seen)), f"sibling chunks survived the collapse: {seen}"


def test_survivor_is_a_real_citable_record_not_a_synthesized_one(flooded_corpus) -> None:
    """No parent record exists for parent_path; the survivor must still be a
    row that actually exists and can be fetched back by its own id."""
    memory, chunk_ids, _others = flooded_corpus

    hits = memory.search(QUERY, limit=10, mode="bm25", type_=None)

    survivors = [hit for hit in hits if (hit.extra or {}).get("parent_path") == SOURCE_DOC]
    assert survivors, "the source document vanished entirely from the results"
    for hit in survivors:
        assert hit.id in chunk_ids, f"{hit.id} is not one of the ingested chunks"
        assert memory.get(hit.id) is not None, f"{hit.id} does not resolve to a stored record"


def test_freed_slots_refill_with_other_documents(flooded_corpus) -> None:
    """Collapsing must not silently shrink the result: the slots the dropped
    siblings occupied go to the next distinct documents."""
    memory, _chunk_ids, others = flooded_corpus

    hits = memory.search(QUERY, limit=5, mode="bm25", type_=None)

    ids = [hit.id for hit in hits]
    assert len(hits) > 1, "collapsing shrank the result instead of refilling it"
    assert any(other in ids for other in others), (
        "no independent document made it into the window after the collapse"
    )


def test_flag_off_leaves_the_parent_path_chunks_alone(mock_memory, monkeypatch) -> None:
    """The fallback stays behind MEMO_SEARCH_CHUNK_PARENT — no second flag."""
    monkeypatch.delenv("MEMO_SEARCH_CHUNK_PARENT", raising=False)
    for seq in range(6):
        _ingest_chunk(mock_memory, seq)

    hits = mock_memory.search(QUERY, limit=10, mode="bm25", type_=None)

    from_source_doc = [path for path in _parent_paths(hits) if path == SOURCE_DOC]
    assert len(from_source_doc) > 1, "collapsed with the flag off"


def test_explicit_reference_ask_keeps_every_chunk(flooded_corpus) -> None:
    """An explicit `type_="reference"` ask wants the fragments themselves."""
    memory, _chunk_ids, _others = flooded_corpus

    hits = memory.search(QUERY, limit=10, mode="bm25", type_="reference")

    from_source_doc = [path for path in _parent_paths(hits) if path == SOURCE_DOC]
    assert len(from_source_doc) > 1, "explicit tier ask lost its chunks"


def test_distinct_source_documents_are_not_collapsed_together(mock_memory, monkeypatch) -> None:
    """Collapsing is per source document, not a blanket chunk filter."""
    monkeypatch.setenv("MEMO_SEARCH_CHUNK_PARENT", "1")
    for seq in range(3):
        _ingest_chunk(mock_memory, seq, parent_path="Vault/A.md")
    for seq in range(3):
        _ingest_chunk(mock_memory, seq, parent_path="Vault/B.md")

    hits = mock_memory.search(QUERY, limit=10, mode="bm25", type_=None)

    assert {"Vault/A.md", "Vault/B.md"} <= set(_parent_paths(hits)), (
        "one of the two source documents was collapsed away entirely"
    )


def test_pool_widens_only_when_a_chunk_can_reach_it(mock_memory, monkeypatch) -> None:
    """Recall-hook budget (5s): the hook SQL-excludes the reference tier, so
    no chunk can reach its pool and a 4x fetch would be pure waste."""
    monkeypatch.setenv("MEMO_SEARCH_CHUNK_PARENT", "1")

    assert mock_memory._chunk_parent_pool_limit(10, None, None) > 10, (
        "the pool must widen when chunks can reach it, or the collapse shrinks the result"
    )
    assert mock_memory._chunk_parent_pool_limit(10, None, {"reference"}) == 10, (
        "widened a pool that excludes the reference tier — wasted hook budget"
    )
    assert mock_memory._chunk_parent_pool_limit(10, "reference", None) == 10, (
        "widened for an explicit reference ask, which never collapses"
    )


def test_pool_is_untouched_with_the_flag_off(mock_memory, monkeypatch) -> None:
    """Flag off (the default) must be a byte-for-byte no-op on retrieval."""
    monkeypatch.delenv("MEMO_SEARCH_CHUNK_PARENT", raising=False)

    assert mock_memory._chunk_parent_pool_limit(10, None, None) == 10


def test_parent_id_chunks_still_resolve_to_the_parent_record(mock_memory, monkeypatch) -> None:
    """Regression: the MEMO_CHUNK_INGEST schema keeps resolving to the real
    parent RECORD, unchanged — the parent_path fallback must not shadow it."""
    monkeypatch.setenv("MEMO_SEARCH_CHUNK_PARENT", "1")
    parent = mock_memory.save(
        content="Comandos disponibles CLI por funcion: documento padre completo.",
        title="Comandos padre",
        type_="reference",
    )
    for seq in range(5):
        mock_memory.save(
            content=(
                f"## Comandos disponibles CLI por funcion — parte {seq}\n\n"
                "Fragmento con parent_id que debe colapsar al registro padre."
            ),
            title=f"Comandos padre (§{seq})",
            type_="reference",
            # The MEMO_CHUNK_INGEST shape: both keys, parent_id authoritative.
            extra={"parent_id": parent.id, "parent_path": "Vault/Padre.md", "chunk_seq": seq},
        )

    hits = mock_memory.search(QUERY, limit=10, mode="bm25", type_=None)

    ids = [hit.id for hit in hits]
    assert parent.id in ids, "the parent record no longer stands in for its chunks"
    assert ids.count(parent.id) == 1, "the parent surfaced more than once"
    assert not [hit for hit in hits if (hit.extra or {}).get("parent_id")], (
        "a raw parent_id chunk survived the collapse"
    )
