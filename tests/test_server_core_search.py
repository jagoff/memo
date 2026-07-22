"""Read-only notification behavior for core MCP search tools."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock


class _RecordingServer:
    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}
        self.annotations: dict[str, dict[str, Any]] = {}

    def tool(self, *, annotations: dict[str, Any]):
        def wrapper(fn):
            self.tools[fn.__name__] = fn
            self.annotations[fn.__name__] = annotations
            return fn

        return wrapper


def test_read_only_tools_peek_notification_without_consuming(tmp_cfg, monkeypatch) -> None:
    import memo.server_core_search as search_server
    from memo.memory import Memory

    monkeypatch.setenv("MEMO_CONTEXT_SURFACE", "1")
    monkeypatch.setattr("memo.briefing.memo_native_briefing_lines", lambda *a, **k: [])
    monkeypatch.setattr("memo.briefing.synapse_briefing_lines", lambda *a, **k: [])
    monkeypatch.setattr(
        "memo.context_surface.build_context_surface",
        lambda *a, **k: {"schema": "memo.context.v1", "hits": []},
    )
    monkeypatch.setattr("memo.context_surface.consult_hits_from_context", lambda out: [])
    monkeypatch.setattr(search_server, "log_consult", lambda *a, **k: None)

    async def _fake_run_synth(memory, ctx, fn):
        return {"answer": "ok", "citations": []}, "stub"

    monkeypatch.setattr(search_server, "run_synth", _fake_run_synth)

    memory = MagicMock(spec=Memory)
    memory.cfg = tmp_cfg
    memory.search.return_value = []
    server = _RecordingServer()
    search_server.register(server, memory)

    notification = "※ MEMO auto-saved"
    path = tmp_cfg.state_dir / "pending_idle_notification.txt"
    path.write_text(notification + "\n", encoding="utf-8")

    calls = {
        "memo_unified_briefing": lambda: server.tools["memo_unified_briefing"](),
        "memo_search": lambda: server.tools["memo_search"](query="needle"),
        "memo_context": lambda: server.tools["memo_context"](question="needle"),
        "memo_ask": lambda: asyncio.run(server.tools["memo_ask"](question="needle")),
        "memo_chat_ask": lambda: asyncio.run(server.tools["memo_chat_ask"](question="needle")),
    }
    for name, call in calls.items():
        assert server.annotations[name]["readOnlyHint"] is True
        assert call()["notification"] == notification
        assert path.is_file(), f"{name} consumed the pending notification"


def test_notification_carries_presence_for_mcp_agents(tmp_cfg, monkeypatch) -> None:
    """MCP-only agents (no statusline) see today's activity in the notification
    field of every tool response, alongside any pending idle notice."""
    import memo.server_core_search as search_server
    from memo import presence
    from memo.memory import Memory

    monkeypatch.setattr(search_server, "log_consult", lambda *a, **k: None)
    memory = MagicMock(spec=Memory)
    memory.cfg = tmp_cfg
    memory.search.return_value = []
    server = _RecordingServer()
    search_server.register(server, memory)

    presence.bump(tmp_cfg.state_dir, recalls=2, saves=1)
    (tmp_cfg.state_dir / "pending_idle_notification.txt").write_text(
        "※ MEMO auto-saved\n", encoding="utf-8"
    )

    notification = server.tools["memo_search"](query="needle")["notification"]
    assert "※ memo today" in notification
    assert "🧠 2 recalled" in notification and "💾 1 saved" in notification
    assert "※ MEMO auto-saved" in notification  # idle notice preserved


def test_presence_notify_flag_off_silences_only_that_channel(tmp_cfg, monkeypatch) -> None:
    import memo.server_core_search as search_server
    from memo import presence
    from memo.memory import Memory

    monkeypatch.setattr(search_server, "log_consult", lambda *a, **k: None)
    monkeypatch.setenv("MEMO_PRESENCE_NOTIFY", "0")
    memory = MagicMock(spec=Memory)
    memory.cfg = tmp_cfg
    memory.search.return_value = []
    server = _RecordingServer()
    search_server.register(server, memory)

    presence.bump(tmp_cfg.state_dir, recalls=2)
    assert "※ memo today" not in server.tools["memo_search"](query="needle")["notification"]


def test_search_hits_bump_recall_presence(tmp_cfg, monkeypatch) -> None:
    """An MCP search that surfaces memories increments the recall counter, so an
    agent that never runs the Claude recall-hook still shows honest activity."""
    import memo.server_core_search as search_server
    from memo import presence
    from memo.memory import Memory

    monkeypatch.setattr(search_server, "log_consult", lambda *a, **k: None)
    memory = MagicMock(spec=Memory)
    memory.cfg = tmp_cfg
    rec = MagicMock()
    rec.to_dict.return_value = {"id": "abc", "body": "hit"}
    memory.search.return_value = [rec, rec]
    server = _RecordingServer()
    search_server.register(server, memory)

    server.tools["memo_search"](query="needle")
    assert presence.read_today(tmp_cfg.state_dir)["recalls"] == 2


def test_search_with_file_announces_dropped_date_filters_and_explain(tmp_cfg, monkeypatch) -> None:
    """search_by_file takes no date filters and computes no explain trace —
    memo_search must say so in the response instead of silently dropping the
    parameters (an agent would present unfiltered hits as date-filtered)."""
    import memo.server_core_search as search_server
    from memo.memory import Memory

    monkeypatch.setattr(search_server, "log_consult", lambda *a, **k: None)
    memory = MagicMock(spec=Memory)
    memory.cfg = tmp_cfg
    memory.search.return_value = []
    memory.search_by_file.return_value = []
    server = _RecordingServer()
    search_server.register(server, memory)
    search = server.tools["memo_search"]

    # file + date filters → explicit note, file path still used
    out = search(query="deploy steps", file="runbook.md", date_from="2026-01-01")
    assert "date filters" in out["note"] and "not applied" in out["note"]
    memory.search_by_file.assert_called_once()
    memory.search.assert_not_called()

    # file + explain → note that explain fields stay empty
    out = search(query="deploy steps", file="runbook.md", explain=True)
    assert "explain" in out["note"]

    # file + when → the parsed range triggers the same note
    out = search(query="deploy steps", file="runbook.md", when="yesterday")
    assert "date filters" in out.get("note", "")

    # no file → no note key
    out = search(query="deploy steps", date_from="2026-01-01")
    assert "note" not in out
    memory.search.assert_called_once()
