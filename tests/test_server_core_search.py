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
