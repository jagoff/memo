"""MEMO_ASK_MULTI_ROUND: one sufficiency-checked extra retrieval round in ask."""
from __future__ import annotations

from memo.memory.record import MemoryRecord


def _rec(id_: str, title: str, body: str) -> MemoryRecord:
    return MemoryRecord(
        id=id_ * 4, path=f"2026/07/{id_}.md", title=title, type="note", tags=[],
        created="2026-07-01T00:00:00", updated="2026-07-01T00:00:00", body=body,
    )


def test_multi_round_adds_round2_hits(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_ASK_MULTI_ROUND", "1")
    a = _rec("aaaa1111", "Alpha note", "alpha body text " * 10)
    b = _rec("bbbb2222", "Beta note", "beta body text " * 10)
    calls: list[str] = []

    def fake_search(query, **kwargs):
        calls.append(query)
        return [a] if len(calls) == 1 else [b]

    monkeypatch.setattr(mock_memory, "search", fake_search)
    monkeypatch.setattr(
        "memo.memory.record.chat_with_timeout",
        lambda *args, **kwargs: {
            "message": {"content": '{"sufficient": false, "queries": ["beta refined"]}'}
        },
    )
    _, sources, _, hits = mock_memory._build_ask_context(
        "what about alpha?", k=5, type_=None, snippet_chars=200, include_repos=False
    )
    assert calls == ["what about alpha?", "beta refined"]
    assert {h.id for h in hits} == {a.id, b.id}


def test_multi_round_sufficient_verdict_keeps_round1(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_ASK_MULTI_ROUND", "1")
    a = _rec("aaaa1111", "Alpha note", "alpha body text " * 10)
    monkeypatch.setattr(mock_memory, "search", lambda q, **kw: [a])
    monkeypatch.setattr(
        "memo.memory.record.chat_with_timeout",
        lambda *args, **kwargs: {"message": {"content": '{"sufficient": true}'}},
    )
    _, _, _, hits = mock_memory._build_ask_context(
        "what about alpha?", k=5, type_=None, snippet_chars=200, include_repos=False
    )
    assert [h.id for h in hits] == [a.id]


def test_multi_round_off_by_default_never_calls_llm(mock_memory, monkeypatch):
    a = _rec("aaaa1111", "Alpha note", "alpha body text " * 10)
    monkeypatch.setattr(mock_memory, "search", lambda q, **kw: [a])

    def _boom(*args, **kwargs):
        raise AssertionError("LLM must not be called with the flag off")

    monkeypatch.setattr("memo.memory.record.chat_with_timeout", _boom)
    _, _, _, hits = mock_memory._build_ask_context(
        "what about alpha?", k=5, type_=None, snippet_chars=200, include_repos=False
    )
    assert [h.id for h in hits] == [a.id]
