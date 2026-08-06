"""Regression: collapsing chunks to their parent must not shrink the result.

Found running memo as an end user. "qué dominios cubre synapse como cerebro
neutral" returned eight hits that were eight chunks of the *same* note —
everything else was crowded out. Turning on MEMO_SEARCH_CHUNK_PARENT made it
worse in a different way: the collapse ran after the trim to `limit`, so the
eight chunks became **one** result instead of one plus the next seven distinct
documents.

Collapsing now runs on the wide pool, before rerank and before the trim, so
the caller gets `limit` distinct documents — and the reranker stops spending
its window on fragments of a single note.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def chunked_corpus(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_SEARCH_CHUNK_PARENT", "1")
    parent = mock_memory.save(
        content="Synapse cubre dominios de orquestación, memoria y federación.",
        title="Synapse dominios",
        type_="reference",
    )
    for index in range(6):
        mock_memory.save(
            content=(
                f"## Synapse dominios — fragmento {index}\n\n"
                "Cubre orquestación, memoria federada y ruteo entre agentes locales."
            ),
            title=f"Synapse dominios (§{index})",
            type_="reference",
            extra={"parent_id": parent.id},
        )
    others = [
        mock_memory.save(
            content=(
                f"## Synapse nota independiente {index}\n\n"
                "Documento separado que también describe dominios de Synapse."
            ),
            title=f"Synapse otra {index}",
            type_="reference",
        ).id
        for index in range(4)
    ]
    return mock_memory, parent.id, others


def test_collapsed_chunks_leave_room_for_other_documents(chunked_corpus) -> None:
    memory, parent_id, others = chunked_corpus

    hits = memory.search("Synapse dominios", limit=5, mode="hybrid", type_=None)

    ids = [hit.id for hit in hits]
    assert ids.count(parent_id) <= 1, "the parent was returned more than once"
    assert len(set(ids)) == len(ids), "duplicate documents in the result"
    assert len(hits) > 1, (
        "collapsing shrank the result instead of refilling it with the next distinct documents"
    )
    assert any(other in ids for other in others), "no other document made it into the window"


def test_the_parent_stands_in_for_its_chunks(chunked_corpus) -> None:
    memory, parent_id, _ = chunked_corpus

    hits = memory.search("Synapse dominios", limit=5, mode="hybrid", type_=None)

    returned = {hit.id for hit in hits}
    assert parent_id in returned, "the parent note did not stand in for its chunks"
    assert not [hit for hit in hits if (hit.extra or {}).get("parent_id")], (
        "a raw chunk survived the collapse"
    )
