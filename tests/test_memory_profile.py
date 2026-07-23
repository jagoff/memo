from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

from memo.memory_profile import build_memory_profile
from memo.server_profile import register


class _Record:
    id = "a" * 32
    title = "Current decision"
    type = "decision"
    body = "Use the local runtime."
    updated = "2026-07-23T12:00:00+00:00"
    extra: ClassVar[dict[str, float]] = {"confidence": 0.92}
    verification_state: ClassVar[SimpleNamespace] = SimpleNamespace(value="verified")


class _Memory:
    def __init__(self, root: Path) -> None:
        self.cfg = SimpleNamespace(memory_dir=root / "memory")
        self.cfg.memory_dir.mkdir(parents=True)

    def list(self, **kwargs):
        return [_Record()]


def test_profile_is_bounded_and_evidence_aware(tmp_path, monkeypatch):
    monkeypatch.setattr("memo.briefing.profile_lines", lambda cfg, cwd=None: ["Stable preference"])
    payload = build_memory_profile(_Memory(tmp_path), scope="project", limit=1)

    assert payload["schema"] == "memo.profile.v1"
    assert payload["available"] is True
    assert payload["stable"][0]["source"] == "profile_document"
    assert payload["active"][0]["evidence_ids"] == ["a" * 32]
    assert payload["active"][0]["confidence"] == 0.92


def test_profile_mcp_tool_logs_and_returns_payload(tmp_path, monkeypatch):
    monkeypatch.setattr("memo.briefing.profile_lines", lambda cfg, cwd=None: [])
    memory = _Memory(tmp_path)
    tools = {}

    class _Server:
        def tool(self, **kwargs):
            def decorate(fn):
                tools[fn.__name__] = fn
                return fn

            return decorate

    register(_Server(), memory)
    payload = tools["memo_profile"](scope="current", limit=1, source="test")
    assert payload["schema"] == "memo.profile.v1"
    assert payload["active"][0]["id_short"] == "aaaaaaaa"
