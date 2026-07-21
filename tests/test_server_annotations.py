"""MCP ToolAnnotations coverage — readOnly/destructive/idempotent hints."""

from __future__ import annotations

import asyncio

import pytest

from memo.config import Config
from memo.memory import Memory
from memo.server import build_server


@pytest.fixture
def mem(tmp_path, monkeypatch):
    data, vault, state = tmp_path / "d", tmp_path / "v", tmp_path / "s"
    for p in (data, vault, state):
        p.mkdir()
    monkeypatch.setenv("MEMO_EMBEDDER_DIMS", "4")  # pin dims to the stub's output
    cfg = Config(
        data_dir=data,
        vault_path=vault,
        state_dir=state,
        reranker_enabled=False,
        embedder_dims=4,
    )
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0] for _ in inputs],
    )
    m = Memory(cfg)
    yield m
    m.close()


def _hints(server, name):
    tool = asyncio.run(server.get_tool(name))
    assert tool is not None, f"{name} not registered"
    assert tool.annotations is not None, f"{name} has no annotations"
    return tool.annotations


def test_core_read_tools_marked_read_only(mem):
    server = build_server(memory=mem)
    for name in (
        "memo_get",
        "memo_list",
        "memo_unified_briefing",
        "memo_search",
        "memo_context",
        "memo_ask",
        "memo_chat_ask",
        "memo_lint",
    ):
        h = _hints(server, name)
        assert h.readOnlyHint is True, name
        assert h.destructiveHint is False, name


def test_core_destructive_tools_marked(mem):
    server = build_server(memory=mem)
    for name in ("memo_delete", "memo_update", "memo_forget", "memo_rename"):
        h = _hints(server, name)
        assert h.readOnlyHint is False, name
        assert h.destructiveHint is True, name


def test_save_reindex_offload_hints(mem):
    server = build_server(memory=mem)
    save = _hints(server, "memo_save")
    assert save.readOnlyHint is False and save.destructiveHint is False
    assert _hints(server, "memo_reindex").idempotentHint is True
    assert _hints(server, "memo_offload").idempotentHint is True  # content-addressed


def test_mock_server_fallback_keeps_zero_arg_decorator_working():
    """The 22 server_* test modules stub server.tool as a ZERO-ARG decorator;
    annotated_tool must fall back instead of crashing them."""
    from memo.server_annotations import READ_ONLY, annotated_tool

    tools: dict = {}

    class _Srv:
        def tool(self):  # zero-arg — rejects the annotations kwarg
            def wrap(fn):
                tools[fn.__name__] = fn
                return fn

            return wrap

    @annotated_tool(_Srv(), **READ_ONLY)
    def memo_fake() -> dict:
        return {}

    assert "memo_fake" in tools


def test_every_registered_tool_has_annotations(mem, monkeypatch):
    monkeypatch.delenv("MEMO_MCP_SLIM", raising=False)  # full surface
    server = build_server(memory=mem)
    # This fastmcp exposes the registered tools via list_tools() (a list of
    # Tool objects), not a name->tool dict; the contract is the same: every
    # registered tool must carry annotations.
    tools = asyncio.run(server.list_tools())
    missing = sorted(t.name for t in tools if t.annotations is None)
    assert missing == [], f"tools without annotations: {missing}"
