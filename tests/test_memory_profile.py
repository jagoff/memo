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


def test_profile_helpers_and_empty_branches(tmp_path, monkeypatch):
    from memo.memory_profile import _confidence, _freshness

    assert _confidence(SimpleNamespace(extra={"confidence": "bad"})) == 0.7
    assert (
        _confidence(SimpleNamespace(extra={}, verification_state=SimpleNamespace(value="stale")))
        == 0.5
    )
    assert _freshness("") is None
    assert _freshness("not-a-date") == "not-a-date"
    monkeypatch.setattr("memo.briefing.profile_lines", lambda cfg, cwd=None: ["x" * 500])

    class _Empty(_Memory):
        def list(self, **kwargs):
            return []

    payload = build_memory_profile(_Empty(tmp_path), scope="agent", limit=0, budget_chars=256)
    assert payload["active"] == []
    assert payload["omissions"] == ["stable profile truncated to the requested budget"]


def test_profile_rejects_unknown_scope(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="scope"):
        build_memory_profile(_Memory(tmp_path), scope="invalid")


def test_profile_handles_domain_error(tmp_path, monkeypatch):
    from memo.errors import MemoError

    class _Broken(_Memory):
        def list(self, **kwargs):
            raise MemoError("unavailable")

    monkeypatch.setattr("memo.briefing.profile_lines", lambda cfg, cwd=None: [])
    payload = build_memory_profile(_Broken(tmp_path))
    assert payload["available"] is False
    assert payload["omissions"]


def test_profile_cli_json_and_text(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from memo.cli_profile import profile_group

    memory = _Memory(tmp_path)
    memory.close = lambda: None
    monkeypatch.setattr("memo.cli_common.get_memory", lambda cfg: memory)
    monkeypatch.setattr(
        "memo.memory_profile.build_memory_profile",
        lambda *a, **k: {
            "available": True,
            "stable": [],
            "active": [{"id_short": "aaaa", "type": "note", "title": "A"}],
        },
    )
    runner = CliRunner()
    env = {"MEMO_DATA_DIR": str(tmp_path), "MEMO_STATE_DIR": str(tmp_path / "state")}
    result = runner.invoke(profile_group, ["memory", "--json"], env=env)
    assert result.exit_code == 0, result.output
    assert '"available": true' in result.output
    result = runner.invoke(profile_group, ["memory"], env=env)
    assert result.exit_code == 0, result.output
    assert "memory profile" in result.output


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


def test_profile_mcp_tool_returns_structured_invalid_scope(tmp_path):
    memory = _Memory(tmp_path)
    tools = {}

    class _Server:
        def tool(self, **kwargs):
            def decorate(fn):
                tools[fn.__name__] = fn
                return fn

            return decorate

    register(_Server(), memory)
    payload = tools["memo_profile"](scope="not-a-scope")
    assert payload["schema"] == "memo.error.v1"
    assert payload["error"]["code"] == "invalid_scope"
