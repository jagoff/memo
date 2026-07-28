"""MCP prompts: pinnable briefing/recall for MCP-only clients (fail-open)."""

from __future__ import annotations

from memo import server_prompts


class _StubServer:
    def __init__(self):
        self.prompts: dict[str, object] = {}

    def prompt(self, *a, **k):
        def _decorator(fn):
            self.prompts[k.get("name") or fn.__name__] = fn
            return fn

        return _decorator


def _register(mock_memory):
    server = _StubServer()
    server_prompts.register(server, mock_memory)
    return server


def test_both_prompts_registered(mock_memory):
    server = _register(mock_memory)
    assert set(server.prompts) == {"briefing", "recall"}


def test_recall_formats_hits(mock_memory):
    rec = mock_memory.save(content="port is 8765", title="port fact", type_="fact")
    server = _register(mock_memory)
    out = server.prompts["recall"](topic="port")
    assert out.startswith("Context from memo (recall: port):")
    assert f"[{rec.id[:8]}]" in out and "port fact" in out


def test_recall_no_hits_degrades(mock_memory):
    server = _register(mock_memory)
    out = server.prompts["recall"](topic="zzz-nothing-here-zzz")
    assert "no matching memories" in out


def test_briefing_returns_context(mock_memory):
    server = _register(mock_memory)
    out = server.prompts["briefing"]()
    assert out.startswith("Context from memo (briefing):")


def test_prompts_fail_open(mock_memory, monkeypatch):
    server = _register(mock_memory)
    monkeypatch.setattr(
        type(mock_memory), "search", property(lambda self: (_ for _ in ()).throw(RuntimeError))
    )
    out = server.prompts["recall"](topic="x")
    assert out.startswith("memo unavailable:")


def test_recall_logs_consult(mock_memory, monkeypatch):
    calls = []
    monkeypatch.setattr(
        server_prompts, "log_consult", lambda *a, **k: calls.append(k)
    )
    server = _register(mock_memory)
    server.prompts["recall"](topic="port")
    assert calls and calls[0]["source"] == "mcp-prompt"
