"""chat_digest_lines — chat-insight captures + active-session + goals digest
for the SessionStart briefing (Task 7)."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from memo.briefing import chat_digest_lines


def _iso_local(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))


def _invalid_at_offset(delta: timedelta) -> str:
    """Match `memo.memory.record._now_iso()`'s shape: local-offset, ms precision."""
    return (datetime.now(tz=UTC) + delta).astimezone().isoformat(timespec="milliseconds")


def _write_captures(state_dir: Path, rows: list[dict[str, Any]]) -> None:
    d = state_dir / "chat" / "insights"
    d.mkdir(parents=True, exist_ok=True)
    with (d / "captures.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_session(state_dir: Path, session_id: str, turns: list[dict[str, Any]]) -> None:
    d = state_dir / "chat" / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    with (d / f"{session_id}.jsonl").open("w", encoding="utf-8") as fh:
        for turn in turns:
            fh.write(json.dumps(turn, ensure_ascii=False) + "\n")


class _FakeStore:
    def __init__(self, goals: list[dict[str, Any]]) -> None:
        self._goals = goals

    def list_by_tag(self, tag: str, limit: int = 500) -> list[dict[str, Any]]:
        assert tag == "goal"
        return self._goals[:limit]


class _FakeMem:
    def __init__(self, goals: list[dict[str, Any]] | None = None) -> None:
        self.store = _FakeStore(goals or [])


def test_empty_state_dir_returns_empty(tmp_path: Path) -> None:
    assert chat_digest_lines(_FakeMem(), tmp_path) == []


def test_recent_captures_and_sessions_render(tmp_path: Path) -> None:
    now = time.time()
    _write_captures(
        tmp_path,
        [
            {
                "memoria_id": "a1",
                "title": "First insight",
                "score": 90,
                "chat_session_id": "s1",
                "captured_at": _iso_local(now - 3600),
            },
            {
                "memoria_id": "a2",
                "title": "Latest insight",
                "score": 95,
                "chat_session_id": "s1",
                "captured_at": _iso_local(now - 60),
            },
        ],
    )
    _write_session(
        tmp_path,
        "s1",
        [
            {"role": "user", "text": "hola", "ts": now - 7200},
            {"role": "assistant", "text": "hi", "ts": now - 7100},
        ],
    )
    mem = _FakeMem(goals=[{"title": "Ship task 7"}, {"title": "Ship task 8"}])

    lines = chat_digest_lines(mem, tmp_path)
    joined = "\n".join(lines)

    assert "2 insights captured" in joined
    assert "Latest insight" in joined
    assert "First insight" not in joined  # only the most recent title is shown
    assert "1 chat session" in joined
    assert "Ship task 7" in joined
    assert "Ship task 8" in joined


def test_old_captures_and_sessions_are_excluded(tmp_path: Path) -> None:
    now = time.time()
    _write_captures(
        tmp_path,
        [
            {
                "memoria_id": "a1",
                "title": "Stale insight",
                "score": 90,
                "chat_session_id": "s1",
                "captured_at": _iso_local(now - 48 * 3600),
            }
        ],
    )
    _write_session(
        tmp_path,
        "s1",
        [{"role": "user", "text": "hola", "ts": now - 48 * 3600}],
    )

    assert chat_digest_lines(_FakeMem(), tmp_path) == []


def test_goals_are_capped_at_three(tmp_path: Path) -> None:
    goals = [{"title": f"Goal {i}"} for i in range(5)]
    mem = _FakeMem(goals=goals)

    joined = "\n".join(chat_digest_lines(mem, tmp_path))

    assert "Goal 0" in joined
    assert "Goal 1" in joined
    assert "Goal 2" in joined
    assert "Goal 3" not in joined
    assert "Goal 4" not in joined


def test_malformed_capture_lines_are_skipped(tmp_path: Path) -> None:
    d = tmp_path / "chat" / "insights"
    d.mkdir(parents=True, exist_ok=True)
    (d / "captures.jsonl").write_text("{not json\n", encoding="utf-8")

    assert chat_digest_lines(_FakeMem(), tmp_path) == []


def test_flag_off_returns_empty_even_with_data(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("MEMO_BRIEFING_CHAT_DIGEST", "0")
    mem = _FakeMem(goals=[{"title": "Ship task 7"}])
    _write_captures(
        tmp_path,
        [
            {
                "memoria_id": "a1",
                "title": "Recent insight",
                "score": 90,
                "chat_session_id": "s1",
                "captured_at": _iso_local(time.time() - 60),
            }
        ],
    )

    assert chat_digest_lines(mem, tmp_path) == []


# ── invalidated (superseded) goals must not surface ─────────────────────────
# `list_by_tag` only filters soft-deletes; a goal closed via
# `lifecycle.invalidate_in_place` (contradicted/superseded) keeps its row in
# `meta` with `invalid_at` stamped. The same tolerant pattern the store's own
# recall uses (`invalid_at IS NULL OR invalid_at > now`, see
# `store/bm25_queries.py::_validity_filter`) must apply here.


def test_invalidated_goal_in_the_past_is_excluded(tmp_path: Path) -> None:
    mem = _FakeMem(
        goals=[
            {"title": "Closed goal", "invalid_at": _invalid_at_offset(-timedelta(hours=1))},
        ]
    )

    assert chat_digest_lines(mem, tmp_path) == []


def test_goal_with_future_invalid_at_is_included(tmp_path: Path) -> None:
    mem = _FakeMem(
        goals=[
            {"title": "Still open", "invalid_at": _invalid_at_offset(timedelta(days=1))},
        ]
    )

    assert "Still open" in "\n".join(chat_digest_lines(mem, tmp_path))


def test_goal_with_null_invalid_at_is_included(tmp_path: Path) -> None:
    mem = _FakeMem(goals=[{"title": "Never closed", "invalid_at": None}])

    assert "Never closed" in "\n".join(chat_digest_lines(mem, tmp_path))


def test_goal_with_unparseable_invalid_at_is_included(tmp_path: Path) -> None:
    # Tolerant: a corrupt/unexpected invalid_at must not hide a goal.
    mem = _FakeMem(goals=[{"title": "Weird timestamp", "invalid_at": "not-a-date"}])

    assert "Weird timestamp" in "\n".join(chat_digest_lines(mem, tmp_path))


def test_invalidated_goal_excluded_among_mixed_goals(tmp_path: Path) -> None:
    mem = _FakeMem(
        goals=[
            {"title": "Open goal", "invalid_at": None},
            {"title": "Closed goal", "invalid_at": _invalid_at_offset(-timedelta(hours=1))},
        ]
    )

    joined = "\n".join(chat_digest_lines(mem, tmp_path))
    assert "Open goal" in joined
    assert "Closed goal" not in joined


def test_invalidate_in_place_hides_goal_from_real_store(mock_memory) -> None:
    """Integration: exercise the production `lifecycle.invalidate_in_place`
    path (not just a hand-built row) so the fake-store unit tests above are
    proven against the real `meta` schema/shape too."""
    rec = mock_memory.save(content="We decided to ship v1", title="Ship v1", tags=["goal"])
    mock_memory.store.update_validity(
        id_=rec.id, valid_at=None, invalid_at=_invalid_at_offset(-timedelta(hours=1))
    )

    assert chat_digest_lines(mock_memory, mock_memory.cfg.state_dir) == []


# ── inter-source isolation ───────────────────────────────────────────────────


def test_session_store_failure_does_not_hide_captures_or_goals(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import memo.chat.sessions as sessions_mod

    def _boom(self: Any, limit: int = 50) -> Any:
        raise RuntimeError("sessions store is on fire")

    monkeypatch.setattr(sessions_mod.SessionStore, "list_sessions", _boom)

    _write_captures(
        tmp_path,
        [
            {
                "memoria_id": "a1",
                "title": "Recent insight",
                "score": 90,
                "chat_session_id": "s1",
                "captured_at": _iso_local(time.time() - 60),
            }
        ],
    )
    mem = _FakeMem(goals=[{"title": "Ship task 7"}])

    joined = "\n".join(chat_digest_lines(mem, tmp_path))

    assert "Recent insight" in joined
    assert "Ship task 7" in joined


# ── compositor wiring ────────────────────────────────────────────────────────


def test_native_briefing_includes_chat_digest_by_default(mock_memory) -> None:
    from memo.briefing import memo_native_briefing_lines

    mock_memory.save(content="We decided to ship task 7", title="Ship task 7", tags=["goal"])
    _write_captures(
        mock_memory.cfg.state_dir,
        [
            {
                "memoria_id": "a1",
                "title": "Recent insight",
                "score": 90,
                "chat_session_id": "s1",
                "captured_at": _iso_local(time.time() - 60),
            }
        ],
    )

    joined = "\n".join(memo_native_briefing_lines(mock_memory, memory_of_day=False))

    assert "Ship task 7" in joined
    assert "Recent insight" in joined


def test_native_briefing_respects_chat_digest_opt_out(mock_memory, monkeypatch) -> None:
    from memo.briefing import memo_native_briefing_lines

    monkeypatch.setenv("MEMO_BRIEFING_CHAT_DIGEST", "0")
    mock_memory.save(content="We decided to ship task 7", title="Ship task 7", tags=["goal"])
    _write_captures(
        mock_memory.cfg.state_dir,
        [
            {
                "memoria_id": "a1",
                "title": "Recent insight",
                "score": 90,
                "chat_session_id": "s1",
                "captured_at": _iso_local(time.time() - 60),
            }
        ],
    )

    joined = "\n".join(memo_native_briefing_lines(mock_memory, memory_of_day=False))

    assert "Recent insight" not in joined
