"""Regression: `ask()` must not silently disable the reranker the install is
configured to use.

Found running memo as an end user. "cómo me conecto a la VPN de avature":

    memo search -> the procedure at rank 2
    memo ask    -> "I couldn't find an answer in the saved memories"

`_build_ask_context` defaulted to `disable_reranker=True`, documented as "RRF
is sufficient for synthesis and reranker adds ~150ms latency". On a real
11k-memory corpus RRF alone buried the answer below the top-k fed to the LLM,
so memo refused questions it had answers for — the failure mode
`memo eval chat` reports as expected_source_hit + forbid_refusal.

Retrieval quality for the answer path now follows the same configuration as
every other retrieval surface: if the install has a reranker, `ask` uses it.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def memory_with_reranker(mock_memory):
    object.__setattr__(
        mock_memory, "cfg", mock_memory.cfg.model_copy(update={"reranker_enabled": True})
    )
    mock_memory.save(content="the VPN profile lives in the shared drive", title="Net", type_="note")
    return mock_memory


def test_ask_reranks_when_the_install_has_a_reranker(memory_with_reranker, monkeypatch) -> None:
    calls: list[int] = []

    def counting_rerank(query, hits, *, top_n, deadline=None, degraded=None):
        calls.append(len(hits))
        return list(hits)[:top_n]

    monkeypatch.setattr(memory_with_reranker, "_rerank", counting_rerank)
    monkeypatch.setattr(memory_with_reranker, "_ensure_chat", lambda: None)

    memory_with_reranker.ask("how do I reach the VPN", k=3)

    assert calls, "ask() skipped the cross-encoder the install is configured to use"


def test_an_explicit_opt_out_is_still_honoured(memory_with_reranker, monkeypatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(
        memory_with_reranker,
        "_rerank",
        lambda query, hits, *, top_n: (calls.append(len(hits)), list(hits)[:top_n])[1],
    )

    memory_with_reranker._build_ask_context(
        "how do I reach the VPN",
        k=3,
        type_=None,
        snippet_chars=200,
        include_repos=False,
        disable_reranker=True,
        use_context_pack=False,
    )

    assert not calls, "an explicit disable_reranker=True must still skip the reranker"
