"""Tests for memo MCP resources."""

from __future__ import annotations

from types import SimpleNamespace

from memo.server_resources import register


class _FakeServer:
    def __init__(self) -> None:
        self.resources = {}

    def resource(self, uri: str):
        def _decorator(fn):
            self.resources[uri] = fn
            return fn

        return _decorator


def test_memory_resource_body_chars_env_respects_zero(monkeypatch) -> None:
    monkeypatch.setenv("MEMO_RESOURCE_BODY_CHARS", "0")
    server = _FakeServer()
    memory = SimpleNamespace(
        list=lambda limit: [],
        get=lambda id: SimpleNamespace(
            id=id,
            title="Title",
            type="note",
            tags=[],
            created="2026-01-01T00:00:00",
            updated="2026-01-01T00:00:00",
            body="abcdef",
        ),
    )

    register(server, memory)
    rendered = server.resources["memo://memory/{id}"]("abc123")

    assert "---\n\n…" in rendered
    assert "abcdef" not in rendered
    assert "preview only" in rendered
