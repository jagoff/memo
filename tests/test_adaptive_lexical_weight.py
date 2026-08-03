"""MEMO_SEARCH_ADAPTIVE_LEXICAL_WEIGHT: short queries tilt RRF fusion lexical."""

from __future__ import annotations


def _capture_weights(mock_memory, monkeypatch, query: str) -> list[float]:
    captured: dict = {}

    def fake_fuse(*lists, limit, k, weights):
        captured["weights"] = weights
        return []

    monkeypatch.setattr("memo.memory.search_ops._rrf_fuse", fake_fuse)
    monkeypatch.setattr(mock_memory.store, "count", lambda: 1)
    mock_memory.search(query, mode="hybrid")
    return captured["weights"]


def test_short_query_tilts_lexical_when_flag_on(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_SEARCH_ADAPTIVE_LEXICAL_WEIGHT", "1")
    assert _capture_weights(mock_memory, monkeypatch, "sqlite-vec") == [0.35, 0.65, 0.65]


def test_long_query_keeps_default_weights(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_SEARCH_ADAPTIVE_LEXICAL_WEIGHT", "1")
    w = _capture_weights(mock_memory, monkeypatch, "cómo configuramos el reranker de producción")
    assert w == [0.5, 0.5, 0.5]


def test_explicit_user_weights_always_win(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_SEARCH_ADAPTIVE_LEXICAL_WEIGHT", "1")
    monkeypatch.setenv("MEMO_SEARCH_VEC_WEIGHT", "0.7")
    monkeypatch.setenv("MEMO_SEARCH_BM25_WEIGHT", "0.3")
    w = _capture_weights(mock_memory, monkeypatch, "sqlite-vec")
    assert w == [0.7, 0.3, 0.3]


def test_flag_off_is_inert(mock_memory, monkeypatch):
    assert _capture_weights(mock_memory, monkeypatch, "sqlite-vec") == [0.5, 0.5, 0.5]
