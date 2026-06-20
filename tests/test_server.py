"""MCP server — tool surface + body snippet behavior.

These tests build a `Memory` with the stub embedder (no MLX), instantiate
the FastMCP server with `build_server(memory=...)`, and inspect the tool
return values directly via the `Memory` instance — bypassing the JSON-RPC
transport. This is enough to lock in:

- `memo_search` truncates `body` to the configured `body_chars`.
- `memo_get` / `memo_update` / `memo_delete` return the structured
  `ambiguous` shape instead of raising when a prefix collides.
"""

from __future__ import annotations

import uuid

import pytest

from memo.config import Config
from memo.memory import Memory
from memo.server import build_server


@pytest.fixture
def mem(tmp_cfg: Config, monkeypatch) -> Memory:
    cfg = Config(
        data_dir=tmp_cfg.data_dir,
        vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir,
        embedder_dims=4,
    )
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0] for _ in inputs],
    )
    mem = Memory(cfg)
    yield mem
    mem.close()


def _tool(server, name):
    """Resolve a registered FastMCP tool to its plain Python callable.

    `FastMCP.get_tool` is async and yields a `FunctionTool` whose
    underlying fn lives on `.fn`. We `asyncio.run` that one-shot and
    return the bare callable.
    """
    import asyncio

    tool = asyncio.run(server.get_tool(name))
    if tool is None:
        raise RuntimeError(f"tool {name!r} not registered")
    return tool.fn


def test_search_truncates_body(mem: Memory):
    huge = "x" * 5_000
    mem.save(content=huge, title="Huge")
    server = build_server(memory=mem)
    search = _tool(server, "memo_search")
    out = search(query="huge", body_chars=200)
    assert out, "search returned nothing"
    body = out[0]["body"]
    # Truncated body has the ellipsis suffix; allow a little slack for the rstrip.
    assert len(body) <= 201
    assert body.endswith("…")
    assert out[0]["body_truncated"] is True


def test_search_full_body_when_chars_huge(mem: Memory):
    mem.save(content="corto", title="Short")
    server = build_server(memory=mem)
    search = _tool(server, "memo_search")
    out = search(query="corto", body_chars=10_000)
    assert out
    assert out[0]["body"] == "corto"
    assert "body_truncated" not in out[0]


def test_search_trace_returns_hits_and_pipeline(mem: Memory):
    mem.save(content="alpha body", title="Alpha")
    server = build_server(memory=mem)
    search_trace = _tool(server, "memo_search_trace")

    out = search_trace(query="alpha", limit=3)

    assert out["hits"]
    assert out["trace"][0]["stage"] == "candidate_generation"
    assert out["trace"][-1]["stage"] == "final"


def test_get_returns_ambiguous_shape(mem: Memory, monkeypatch):
    fixed = iter([
        uuid.UUID("aaaaaaaa1111000000000000000000ff"),
        uuid.UUID("aaaaaaaa2222000000000000000000ff"),
    ])
    monkeypatch.setattr("memo.memory.uuid.uuid4", lambda: next(fixed))
    mem.save(content="a", title="A")
    mem.save(content="b", title="B")

    server = build_server(memory=mem)
    get = _tool(server, "memo_get")
    out = get(id="aaaaaaaa")
    assert isinstance(out, dict)
    assert out.get("error") == "ambiguous"
    assert len(out["matches"]) == 2


def test_memory_chat_ask_returns_v2_envelope(mem: Memory, monkeypatch):
    rec = mem.save(content="alpha delivery result", title="Alpha")

    def _stub_chat(self, model, messages, options=None):
        assert "previous alpha question" in messages[-1]["content"]
        assert "packet_status" in messages[-1]["content"]
        return {"message": {"content": f"Alpha answer [{rec.id[:8]}]."}}

    monkeypatch.setattr("memo.llm.MLXChat.chat", _stub_chat)

    server = build_server(memory=mem)
    chat_ask = _tool(server, "memo_chat_ask")
    out = chat_ask(
        question="what did alpha decide?",
        k=2,
        history=[{"role": "user", "text": "previous alpha question"}],
        context={"packet_status": "ready"},
    )

    assert out["schema"] == "memo.chat_ask.v2"
    assert out["synthesis_status"] == "ok"
    assert out["answer"].startswith("Alpha answer")
    assert out["citations"][0]["source"] == "memo"
    assert out["history_turns_used"] == 1
    assert out["context_keys"] == ["packet_status"]


def test_memory_chat_ask_no_hits_returns_unavailable(mem: Memory):
    server = build_server(memory=mem)
    chat_ask = _tool(server, "memo_chat_ask")
    out = chat_ask(question="missing context")

    assert out["schema"] == "memo.chat_ask.v2"
    assert out["synthesis_status"] == "unavailable"
    assert "no encuentro" in out["answer"].lower()
    assert out["sources"] == []
    assert out["citations"] == []
