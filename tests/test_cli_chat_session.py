from __future__ import annotations

import json
import multiprocessing
import os
import stat
from pathlib import Path

import pytest
from click import ClickException
from click.testing import CliRunner

from memo import cli_chat_session
from memo.cli_chat_session import _append_turn, _load, _start_session
from memo.errors import StorageError


def _append_worker(state_dir: str, index: int, barrier: object) -> None:
    os.environ["MEMO_STATE_DIR"] = state_dir
    barrier.wait()  # type: ignore[attr-defined]
    _append_turn(
        session_id="shared-session",
        question=f"question-{index}",
        answer=f"answer-{index}",
        client=f"terminal-{index % 2}",
        turn_id=f"turn-{index}",
        role="user",
    )


def test_terminal_processes_cannot_overwrite_each_others_chat_turns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_dir = tmp_path / "state"
    monkeypatch.setenv("MEMO_STATE_DIR", str(state_dir))
    _start_session(session_id="shared-session", client="memo-chat")
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(8)
    processes = [
        context.Process(target=_append_worker, args=(str(state_dir), index, barrier))
        for index in range(8)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)
        assert not process.is_alive()
        assert process.exitcode == 0

    stored = _load(state_dir / "chat_sessions.json")["sessions"]["shared-session"]
    assert {turn["turn_id"] for turn in stored["turns"]} == {f"turn-{index}" for index in range(8)}
    assert {turn["client"] for turn in stored["turns"]} == {
        "terminal-0",
        "terminal-1",
    }


def test_chat_turn_idempotency_conflict_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path))
    _start_session(session_id="shared-session", client="memo-chat")
    _append_turn(
        session_id="shared-session",
        question="original",
        answer="answer",
        client="terminal-a",
        turn_id="turn-1",
        role="user",
    )

    with pytest.raises(ClickException, match="different payload"):
        _append_turn(
            session_id="shared-session",
            question="changed",
            answer="answer",
            client="terminal-a",
            turn_id="turn-1",
            role="user",
        )


def test_chat_session_corruption_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "chat_sessions.json"
    path.write_text("not-json", encoding="utf-8")

    with pytest.raises(StorageError, match="invalid JSON"):
        _load(path)


def _write_sessions(path: Path, sessions: dict[str, dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"sessions": sessions}),
        encoding="utf-8",
    )


def test_list_cursor_drains_more_than_1000_sessions(
    monkeypatch,
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "chat_sessions.json"
    sessions = {
        f"session-{index:04d}": {
            "session_id": f"session-{index:04d}",
            "client": "codex",
            "created_at": float(index),
            "turns": [],
        }
        for index in range(1_005)
    }
    _write_sessions(state_path, sessions)
    monkeypatch.setattr(cli_chat_session, "_path", lambda: state_path)
    runner = CliRunner()

    first = runner.invoke(
        cli_chat_session.chat_session_group,
        ["list", "--limit", "1000", "--cursor", "", "--json"],
    )
    assert first.exit_code == 0, first.output
    first_payload = json.loads(first.output)
    assert len(first_payload["sessions"]) == 1_000
    assert first_payload["has_more"] is True
    assert first_payload["next_cursor"] == "session-0999"

    second = runner.invoke(
        cli_chat_session.chat_session_group,
        [
            "list",
            "--limit",
            "1000",
            "--cursor",
            first_payload["next_cursor"],
            "--json",
        ],
    )
    assert second.exit_code == 0, second.output
    second_payload = json.loads(second.output)
    assert [row["session_id"] for row in second_payload["sessions"]] == [
        "session-1000",
        "session-1001",
        "session-1002",
        "session-1003",
        "session-1004",
    ]
    assert second_payload["has_more"] is False
    assert second_payload["next_cursor"] is None


def test_legacy_list_shape_and_order_are_preserved(
    monkeypatch,
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "chat_sessions.json"
    _write_sessions(
        state_path,
        {
            "older": {
                "session_id": "older",
                "created_at": 10.0,
                "turns": [],
            },
            "newer": {
                "session_id": "newer",
                "created_at": 20.0,
                "turns": [],
            },
        },
    )
    monkeypatch.setattr(cli_chat_session, "_path", lambda: state_path)

    result = CliRunner().invoke(
        cli_chat_session.chat_session_group,
        ["list", "--limit", "2", "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert list(payload) == ["sessions"]
    assert [row["session_id"] for row in payload["sessions"]] == [
        "newer",
        "older",
    ]
    assert [row["updated_at"] for row in payload["sessions"]] == [20.0, 10.0]


def test_append_adds_updated_at_to_legacy_session(
    monkeypatch,
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "chat_sessions.json"
    _write_sessions(
        state_path,
        {
            "legacy": {
                "session_id": "legacy",
                "client": "codex",
                "created_at": 10.0,
                "turns": [],
            }
        },
    )
    monkeypatch.setattr(cli_chat_session, "_path", lambda: state_path)
    monkeypatch.setattr(cli_chat_session.time, "time", lambda: 42.0)

    result = CliRunner().invoke(
        cli_chat_session.chat_session_group,
        ["append", "legacy", "seguí", "--turn-id", "turn-1", "--json"],
    )

    assert result.exit_code == 0, result.output
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["sessions"]["legacy"]["created_at"] == 10.0
    assert saved["sessions"]["legacy"]["updated_at"] == 42.0


def test_corrupt_store_fails_closed_without_overwriting(
    monkeypatch,
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "chat_sessions.json"
    original = "{not valid json\n"
    state_path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(cli_chat_session, "_path", lambda: state_path)

    result = CliRunner().invoke(
        cli_chat_session.chat_session_group,
        ["start", "--session-id", "must-not-overwrite"],
    )

    assert result.exit_code == 1
    assert "chat session store is corrupt" in result.output
    assert state_path.read_text(encoding="utf-8") == original


def test_session_store_is_written_with_owner_only_permissions(
    monkeypatch,
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "chat_sessions.json"
    monkeypatch.setattr(cli_chat_session, "_path", lambda: state_path)

    result = CliRunner().invoke(
        cli_chat_session.chat_session_group,
        ["start", "--session-id", "private-session"],
    )

    assert result.exit_code == 0, result.output
    assert stat.S_IMODE(state_path.stat().st_mode) == 0o600
