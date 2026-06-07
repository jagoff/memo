"""Tests for whatsapp_ingest pure functions (no bridge DB needed).

Covers timestamp parsing and the note renderer; the sqlite reader is exercised
indirectly via the WAMessage dataclass these build on.
"""

from __future__ import annotations

from datetime import UTC, datetime

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
