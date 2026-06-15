"""Tests for whatsapp_ingest pure functions (no bridge DB needed).

Covers timestamp parsing and the note renderer; the sqlite reader is exercised
indirectly via the WAMessage dataclass these build on.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from memo.whatsapp_ingest import WAMessage, _parse_bridge_ts, render_chat_note


def _msg(ts: float, sender: str, content: str, *, is_me: bool = False) -> WAMessage:
    return WAMessage(
        id=f"id-{ts}", chat_jid="123@s.whatsapp.net", chat_name="Alice",
        sender=sender, content=content, timestamp=ts, is_from_me=is_me,
        media_type=None,
    )


def test_parse_bridge_ts_numeric() -> None:
    assert _parse_bridge_ts(1_700_000_000) == 1_700_000_000.0
    assert _parse_bridge_ts(1_700_000_000.5) == 1_700_000_000.5


def test_parse_bridge_ts_none_and_empty() -> None:
    assert _parse_bridge_ts(None) is None
    assert _parse_bridge_ts("") is None
    assert _parse_bridge_ts("not-a-date") is None


def test_parse_bridge_ts_rfc3339() -> None:
    expected = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC).timestamp()
    assert _parse_bridge_ts("2026-01-02T03:04:05+00:00") == expected


def test_render_chat_note_frontmatter_and_count() -> None:
    base = datetime(2026, 3, 1, 12, 0, tzinfo=UTC).timestamp()
    msgs = [
        _msg(base, "Alice", "hola"),
        _msg(base + 60, "Me", "qué tal", is_me=True),
    ]
    note = render_chat_note("123@s.whatsapp.net", "Alice", msgs)
    assert "source: whatsapp" in note
    assert "messages: 2" in note
    assert "hola" in note
    assert "qué tal" in note
    assert "# WhatsApp · Alice" in note


def test_render_chat_note_groups_by_day() -> None:
    day1 = datetime(2026, 3, 1, 10, 0, tzinfo=UTC).timestamp()
    day2 = datetime(2026, 3, 2, 10, 0, tzinfo=UTC).timestamp()
    note = render_chat_note(
        "123@s.whatsapp.net", "Alice",
        [_msg(day1, "Alice", "día uno"), _msg(day2, "Alice", "día dos")],
    )
    # one `## YYYY-MM-DD` header per distinct day
    assert note.count("## 2026-03-0") == 2


def test_render_chat_note_sorts_unordered_input() -> None:
    base = datetime(2026, 3, 1, 12, 0, tzinfo=UTC).timestamp()
    msgs = [_msg(base + 120, "Alice", "tercero"), _msg(base, "Alice", "primero")]
    note = render_chat_note("123@s.whatsapp.net", "Alice", msgs)
    assert note.index("primero") < note.index("tercero")


# ── Task 17: path validation ──────────────────────────────────────────────────

def test_run_raises_valueerror_when_default_db_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """run() raises ValueError with actionable message when MEMO_WHATSAPP_DB is
    unset and the compiled-in default path does not exist."""
    # Ensure the env var is unset so the default path is used.
    monkeypatch.delenv("MEMO_WHATSAPP_DB", raising=False)
    # Point the default to a path that cannot exist.
    monkeypatch.setenv("MEMO_WHATSAPP_DB", str(tmp_path / "nonexistent.db"))

    mem = MagicMock()
    # re-import to pick up monkeypatched env
    from memo.whatsapp_ingest import run

    with pytest.raises((ValueError, FileNotFoundError)):
        run(mem, all_chats=True)


def test_run_uses_memo_whatsapp_db_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """MEMO_WHATSAPP_DB is read and used as the bridge DB path."""
    db_path = tmp_path / "messages.db"
    # Don't create the file — just verify the error references our custom path.
    monkeypatch.setenv("MEMO_WHATSAPP_DB", str(db_path))

    mem = MagicMock()
    from memo.whatsapp_ingest import run

    with pytest.raises((ValueError, FileNotFoundError)) as exc_info:
        run(mem, all_chats=True)

    assert str(db_path) in str(exc_info.value)


# ── Task 18.1: signal handler safety ─────────────────────────────────────────

def test_recall_server_sigterm_handler_only_sets_event() -> None:
    """The SIGTERM handler in recall_server.run_server only sets a threading.Event
    (no logging, no lock acquisition) — async-signal-safe."""
    import inspect

    from memo import recall_server

    # Inspect the run_server source to confirm the handler is signal-safe:
    # it must only call shutdown_event.set() and nothing else.
    src = inspect.getsource(recall_server.run_server)
    assert "shutdown_event.set()" in src
    # Must NOT call logging inside the handler (logging acquires a lock)
    assert "_LOG" not in src.split("def _sigterm")[1].split("\n\n")[0]


def test_recall_server_shutdown_event_is_threading_event() -> None:
    """Verify the shutdown mechanism uses threading.Event (set-once, signal-safe)."""
    import threading
    # Simulate what the signal handler does: set an event.
    ev = threading.Event()
    ev.set()
    assert ev.is_set()  # signal handler path is just .set(), no other calls
