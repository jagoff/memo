"""ask-strict-threshold: abstain when sources don't entail the answer (default off)."""
from __future__ import annotations

from memo.memory import ask_ops


def _prep(mock_memory, monkeypatch, *, answer, entail):
    # Force a non-empty sources set + a drafted answer without a real LLM.
    monkeypatch.setattr(
        ask_ops._AskOpsMixin, "_build_ask_context",
        lambda self, q, **k: (q, [{"id": "a1b2c3d4", "title": "t", "type": "note", "score": 0.9, "snippet": "port 8765"}], "ctx", []),
    )
    monkeypatch.setattr(ask_ops._AskOpsMixin, "_verbatim_short_circuit", lambda self, q, h: None)
    monkeypatch.setattr(mock_memory, "_ensure_chat", lambda: _Chat(answer))
    # ask_ops imports score_grounding via `from ... import` (same pattern as
    # capture_core.py), so the monkeypatch target is ask_ops's own bound name,
    # not grounding_judge's module attribute.
    monkeypatch.setattr(ask_ops, "score_grounding", lambda *a, **k: entail)


class _Chat:
    def __init__(self, answer):
        self._a = answer

    def chat(self, **k):
        return {"message": {"content": self._a}}

    def chat_stream(self, **k):
        yield self._a


def test_ask_abstains_below_floor(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_GROUNDING_ASK_MIN", "0.85")
    monkeypatch.setenv("MEMO_ASK_FALLBACK_MSG", "I couldn't find that.")
    _prep(mock_memory, monkeypatch, answer="The port is definitely 9999.", entail=0.2)
    out = mock_memory.ask("what port?")
    assert out["answer"] == "I couldn't find that."


def test_ask_keeps_entailed_answer(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_GROUNDING_ASK_MIN", "0.85")
    _prep(mock_memory, monkeypatch, answer="The port is 8765.", entail=0.95)
    out = mock_memory.ask("what port?")
    assert "8765" in out["answer"]


def test_ask_off_skips_judge(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_GROUNDING_ASK_MIN", "0")
    called = {"n": 0}
    _prep(mock_memory, monkeypatch, answer="whatever", entail=0.0)
    monkeypatch.setattr(ask_ops, "score_grounding", lambda *a, **k: called.__setitem__("n", called["n"] + 1) or 0.0)
    out = mock_memory.ask("q?")
    assert out["answer"] == "whatever"
    assert called["n"] == 0


def test_ask_stream_abstains_below_floor(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_GROUNDING_ASK_MIN", "0.85")
    monkeypatch.setenv("MEMO_ASK_FALLBACK_MSG", "I couldn't find that.")
    _prep(mock_memory, monkeypatch, answer="The port is definitely 9999.", entail=0.2)

    events = list(mock_memory.ask_stream("what port?"))

    assert events[-1]["event"] == "done"
    assert events[-1]["answer"] == "I couldn't find that."


def test_ask_stream_off_skips_judge(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_GROUNDING_ASK_MIN", "0")
    called = {"n": 0}
    _prep(mock_memory, monkeypatch, answer="whatever", entail=0.0)
    monkeypatch.setattr(
        ask_ops,
        "score_grounding",
        lambda *a, **k: called.__setitem__("n", called["n"] + 1) or 0.0,
    )

    events = list(mock_memory.ask_stream("q?"))

    assert events[-1]["answer"] == "whatever"
    assert called["n"] == 0
