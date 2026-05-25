"""Streaming chat_ask: MLXChat.chat_stream → Memory.ask_stream → Memory.chat_ask_stream."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from memo.config import Config
from memo.memory import Memory


@pytest.fixture
def mem_with_stub(tmp_cfg: Config, monkeypatch: pytest.MonkeyPatch) -> Memory:
    """Memory with deterministic 4-dim embedder, mirroring test_memory.py.

    Keeps `mem._chat` as None so MLXChat() is lazily constructed and the
    class-level monkeypatch on `chat_stream` takes effect.
    """
    cfg = Config(
        data_dir=tmp_cfg.data_dir,
        vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir,
        embedder_dims=4,
    )

    def _stub_embed(self, inputs):
        out = []
        for s in inputs:
            h = sum(ord(c) for c in s) % 4
            v = [0.0] * 4
            v[h] = 1.0
            out.append(v)
        return out

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _stub_embed)
    return Memory(cfg)


def _stub_stream(deltas: list[str]):
    def _stream(self, model: str, messages: list[dict[str, str]],
                options: dict[str, Any] | None = None) -> Iterator[str]:
        yield from deltas
    return _stream


def test_chat_ask_stream_emits_context_tokens_done(
    mem_with_stub: Memory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec = mem_with_stub.save(content="alpha decision body", title="Alpha")

    monkeypatch.setattr(
        "memo.llm.MLXChat.chat_stream",
        _stub_stream(["Hola ", "mundo", "."]),
    )

    events = list(mem_with_stub.chat_ask_stream("what about alpha?", k=2))
    kinds = [e["event"] for e in events]
    assert kinds[0] == "context"
    assert kinds[-1] == "done"
    assert kinds.count("token") == 3

    context = events[0]
    assert context["schema"] == "memo.chat_ask.v2"
    assert context["sources"], "context event must carry sources"
    assert context["citations"][0]["source"] == "memo"

    token_text = "".join(e["delta"] for e in events if e["event"] == "token")
    assert token_text == "Hola mundo."

    done = events[-1]
    assert done["synthesis_status"] == "ok"
    assert done["answer"] == "Hola mundo."
    assert done["sources"], "done must reiterate the source set"
    assert done["history_turns_used"] == 0
    assert done["retrieval_trace"][0]["stage"] == "memo.chat_ask_stream"
    assert done["synthesis_source"].startswith("memo.ask_stream:")
    # Source id_short matches the saved record so citations remap cleanly.
    assert any(rec.id[:8] == s["id_short"] for s in done["sources"])


def test_chat_ask_stream_error_mid_stream_emits_partial_done(
    mem_with_stub: Memory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    mem_with_stub.save(content="alpha decision body", title="Alpha")

    def _broken_stream(self, model, messages, options=None):
        yield "partial "
        yield "answer"
        raise RuntimeError("mlx blew up mid-decode")

    monkeypatch.setattr("memo.llm.MLXChat.chat_stream", _broken_stream)

    events = list(mem_with_stub.chat_ask_stream("alpha?", k=2))
    kinds = [e["event"] for e in events]
    assert kinds[0] == "context"
    assert kinds[-1] == "done"
    assert "token" in kinds

    done = events[-1]
    assert done["synthesis_status"] == "error"
    assert "mlx blew up" in done["synthesis_error"]
    # Partial accumulator preserved so the UI can surface it.
    assert done["answer"] == "partial answer"


def test_chat_ask_stream_empty_question_short_circuits(
    mem_with_stub: Memory,
) -> None:
    events = list(mem_with_stub.chat_ask_stream("   ", k=2))
    assert len(events) == 1
    done = events[0]
    assert done["event"] == "done"
    assert done["synthesis_status"] == "unavailable"
    assert done["synthesis_error"] == "empty question"
    assert done["answer"] == ""
    assert done["sources"] == []


def test_chat_ask_stream_no_hits_short_circuits(
    mem_with_stub: Memory,
) -> None:
    # No save() calls — empty corpus, retrieval returns nothing.
    events = list(mem_with_stub.chat_ask_stream("anything", k=2))
    kinds = [e["event"] for e in events]
    assert kinds == ["done"]
    done = events[0]
    assert "no encuentro" in done["answer"].lower()
    assert done["sources"] == []
    assert done["synthesis_status"] == "unavailable"
