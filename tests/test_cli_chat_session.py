from __future__ import annotations

import json
import stat
from pathlib import Path

from click.testing import CliRunner

from memo import cli_chat_session


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
