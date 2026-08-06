"""Regression: `load_bodies=False` is an I/O optimisation, not a ranking change.

Found running memo as an end user. For "cómo me conecto a la VPN de avature":

    memo search  -> the answer at rank 2  (score 0.363)
    memo ask     -> "I couldn't find an answer in the saved memories"

`ask()` passes `load_bodies=False` to defer disk reads until after reranking,
but materialisation then set `body=""` on every candidate, so the
cross-encoder scored **titles only**. On the live corpus the expected document
fell from rank 2 to rank 21 — outside the top-k fed to the LLM — and memo
refused to answer a question it had the answer for. `ask`, `chat ask` and the
`memo_ask` MCP tool all take that path.

The hybrid path already had the right idea: pull the pool's text from the FTS
index in one batched query. It was just gated on `load_bodies`. That gate now
only controls whether the canonical `.md` is read from disk, which is what the
lazy contract is actually about.
"""

from __future__ import annotations

import pytest

VPN_BODY = (
    "Para conectarte a la VPN corporativa usá el cliente Tunnelblick con el "
    "perfil que está en el drive compartido."
)


@pytest.fixture
def corpus(mock_memory):
    """Bodies carry the discriminating text; titles deliberately do not."""
    vpn = mock_memory.save(content=VPN_BODY, title="Add Subdomain", type_="note")
    mock_memory.save(
        content="Notas generales de infraestructura del proveedor.",
        title="Overview",
        type_="note",
    )
    mock_memory.save(
        content="Registro de costos mensuales de la nube.", title="Aws costs", type_="note"
    )
    # Config is frozen; the reranker gate reads cfg.reranker_enabled.
    object.__setattr__(
        mock_memory, "cfg", mock_memory.cfg.model_copy(update={"reranker_enabled": True})
    )
    return mock_memory, vpn.id


@pytest.fixture
def text_reranker(corpus, monkeypatch):
    """A reranker that can only do its job if it is handed body text — which is
    exactly what a cross-encoder is."""
    memory, _ = corpus
    seen: list[list] = []

    def rerank_on_body(query: str, hits, *, top_n: int):
        seen.append(list(hits))
        ordered = sorted(hits, key=lambda hit: "vpn" not in (hit.body or "").lower())
        return ordered[:top_n]

    monkeypatch.setattr(memory, "_rerank", rerank_on_body)
    return seen


def test_the_reranker_sees_body_text_even_when_bodies_are_lazy(corpus, text_reranker) -> None:
    memory, _ = corpus

    memory.search("cómo me conecto a la VPN", limit=3, mode="hybrid", load_bodies=False)

    assert text_reranker, "the rerank stage did not run"
    assert any(hit.body.strip() for hit in text_reranker[0]), (
        "every candidate reached the reranker with an empty body — it can only "
        "score titles, which silently changes the ranking"
    )


def test_lazy_bodies_do_not_change_the_ranking(corpus, text_reranker) -> None:
    memory, vpn_id = corpus
    query = "cómo me conecto a la VPN"

    eager = memory.search(query, limit=3, mode="hybrid", load_bodies=True)
    lazy = memory.search(query, limit=3, mode="hybrid", load_bodies=False)

    assert [hit.id for hit in lazy] == [hit.id for hit in eager]
    assert lazy[0].id == vpn_id, "the body-matching memory must still rank first"


def test_lazy_callers_still_get_unloaded_bodies(corpus, text_reranker) -> None:
    """The contract the flag exists for: no canonical disk read per candidate,
    so `ask()` re-resolves the .md itself for the survivors."""
    memory, _ = corpus

    lazy = memory.search("cómo me conecto a la VPN", limit=3, mode="hybrid", load_bodies=False)

    assert lazy
    assert all(hit.body == "" for hit in lazy)
