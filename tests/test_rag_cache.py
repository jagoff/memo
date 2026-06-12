"""RAG context cache: session-scoped, corpus-version-invalidated, TTL'd.

Caches the retrieval half of ask() (hits + formatted sources) so repeated
asks within a session don't re-run search/rerank. Keyed by session_id +
corpus_version; any corpus change or TTL expiry invalidates.
"""

from __future__ import annotations

from memo.rag_cache import RagContextCache


def test_put_then_get_hits_within_ttl():
    c = RagContextCache(ttl_s=300)
    c.put("k", "value", corpus_version="v1", now=1000.0)
    assert c.get("k", corpus_version="v1", now=1100.0) == "value"


def test_get_misses_after_ttl():
    c = RagContextCache(ttl_s=300)
    c.put("k", "value", corpus_version="v1", now=1000.0)
    assert c.get("k", corpus_version="v1", now=1301.0) is None


def test_get_misses_on_corpus_version_change():
    c = RagContextCache(ttl_s=300)
    c.put("k", "value", corpus_version="v1", now=1000.0)
    assert c.get("k", corpus_version="v2", now=1001.0) is None


def test_lru_eviction_respects_max_entries():
    c = RagContextCache(ttl_s=300, max_entries=2)
    c.put("a", 1, corpus_version="v", now=1.0)
    c.put("b", 2, corpus_version="v", now=2.0)
    c.put("c", 3, corpus_version="v", now=3.0)  # evicts oldest ("a")
    assert c.get("a", corpus_version="v", now=4.0) is None
    assert c.get("c", corpus_version="v", now=4.0) == 3


# ── behavioral: ask() retrieval caching ────────────────────────────────────


def _count_searches(mock_memory, monkeypatch):
    state = {"n": 0}
    orig = mock_memory.search

    def spy(*args, **kwargs):
        state["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(mock_memory, "search", spy)
    return state


def test_ask_caches_retrieval_per_session(mock_memory, monkeypatch):
    mock_memory.save(content="the zebra fact lives here", title="Zebra")
    state = _count_searches(mock_memory, monkeypatch)

    mock_memory.ask("zebra?", session_id="s1")
    first = state["n"]
    assert first >= 1
    mock_memory.ask("zebra?", session_id="s1")  # same session + corpus → cached
    assert state["n"] == first, "second identical ask should reuse cached retrieval"


def test_ask_without_session_id_does_not_cache(mock_memory, monkeypatch):
    mock_memory.save(content="the zebra fact lives here", title="Zebra")
    state = _count_searches(mock_memory, monkeypatch)

    mock_memory.ask("zebra?")
    first = state["n"]
    mock_memory.ask("zebra?")
    assert state["n"] > first, "no session_id → no caching"


def test_ask_cache_invalidated_on_corpus_change(mock_memory, monkeypatch):
    mock_memory.save(content="the zebra fact lives here", title="Zebra")
    state = _count_searches(mock_memory, monkeypatch)

    mock_memory.ask("zebra?", session_id="s1")
    first = state["n"]
    mock_memory.save(content="a new giraffe fact", title="Giraffe")  # corpus changed
    mock_memory.ask("zebra?", session_id="s1")
    assert state["n"] > first, "corpus change must invalidate the cached retrieval"
