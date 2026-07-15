"""End-to-end: MCP synthesis tools sample the client's model when enabled."""

from __future__ import annotations

import asyncio
from typing import Any

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
    m = Memory(cfg)
    m.save(content="stub fact for retrieval", title="Stub fact", type_="fact")
    yield m
    m.close()


def _call(server, tool: str, args: dict[str, Any], handler=None) -> Any:
    from fastmcp import Client

    async def _run():
        kwargs = {"sampling_handler": handler} if handler is not None else {}
        async with Client(server, **kwargs) as c:
            res = await c.call_tool(tool, args)
            return res.data

    return asyncio.run(_run())


def test_memo_ask_samples_client_model(mem: Memory, monkeypatch):
    monkeypatch.setenv("MEMO_SAMPLING_SYNTH_ENABLED", "1")
    server = build_server(mem)

    async def handler(messages, params, context):
        return "CLIENT SYNTH ANSWER"

    out = _call(server, "memo_ask", {"question": "what is the stub fact?"}, handler=handler)
    assert out["synthesizer"].startswith("client:")
    assert "CLIENT SYNTH ANSWER" in str(out.get("answer", ""))


def test_memo_ask_falls_back_to_mlx_without_handler(mem: Memory, monkeypatch):
    monkeypatch.setenv("MEMO_SAMPLING_SYNTH_ENABLED", "1")
    monkeypatch.setattr("memo.platform_detect.mlx_available", lambda: True)
    monkeypatch.setattr("memo.llm.MLXChat.__init__", lambda self: None)
    monkeypatch.setattr(
        "memo.llm.MLXChat.chat",
        lambda self, model, messages, options=None: {"message": {"content": "MLX ANSWER"}},
    )
    server = build_server(mem)
    out = _call(server, "memo_ask", {"question": "what is the stub fact?"})
    assert out["synthesizer"].startswith("mlx:")


def test_memo_ask_flag_off_never_samples(mem: Memory, monkeypatch):
    monkeypatch.delenv("MEMO_SAMPLING_SYNTH_ENABLED", raising=False)
    monkeypatch.setattr("memo.platform_detect.mlx_available", lambda: True)
    monkeypatch.setattr("memo.llm.MLXChat.__init__", lambda self: None)
    monkeypatch.setattr(
        "memo.llm.MLXChat.chat",
        lambda self, model, messages, options=None: {"message": {"content": "MLX ANSWER"}},
    )
    server = build_server(mem)

    async def handler(messages, params, context):  # pragma: no cover - must not run
        raise AssertionError("sampled with flag off")

    out = _call(server, "memo_ask", {"question": "what is the stub fact?"}, handler=handler)
    assert out["synthesizer"].startswith("mlx:")


def test_memo_chat_ask_samples_client_model(mem: Memory, monkeypatch):
    monkeypatch.setenv("MEMO_SAMPLING_SYNTH_ENABLED", "1")
    server = build_server(mem)

    async def handler(messages, params, context):
        return "CLIENT CHAT ANSWER"

    out = _call(server, "memo_chat_ask", {"question": "what is the stub fact?"}, handler=handler)
    assert out["synthesizer"].startswith("client:")
