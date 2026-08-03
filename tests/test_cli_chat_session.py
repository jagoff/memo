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


def _write_http_session(root: Path, session_id: str, turns: list[dict[str, object]]) -> None:
    from memo.chat.sessions import SessionStore

    store = SessionStore(root)
    for turn in turns:
        store.append_turn(session_id, turn["role"], turn["text"])


def test_get_finds_session_created_via_chat_http_api(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """`memo chat-session get` must see sessions memo chat serve's HTTP API wrote.

    Regression test: those sessions live in a separate per-session JSONL store
    (memo.chat.sessions.SessionStore) that this CLI's flat chat_sessions.json
    store never wrote to and, before this fix, never read from either.
    """
    state_path = tmp_path / "chat_sessions.json"
    http_root = tmp_path / "chat" / "sessions"
    monkeypatch.setattr(cli_chat_session, "_path", lambda: state_path)
    monkeypatch.setattr(cli_chat_session, "_http_sessions_root", lambda: http_root)
    _write_http_session(
        http_root,
        "57d6b73fb1bd",
        [
            {"role": "user", "text": "hello test"},
            {"role": "assistant", "text": "hi there"},
        ],
    )

    result = CliRunner().invoke(
        cli_chat_session.chat_session_group,
        ["get", "57d6b73fb1bd", "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["session_id"] == "57d6b73fb1bd"
    assert payload["source"] == "http"
    assert [t["role"] for t in payload["turns"]] == ["user", "assistant"]


def test_get_finds_session_created_via_cli_start(
    monkeypatch,
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "chat_sessions.json"
    _write_sessions(
        state_path,
        {
            "cs-native": {
                "session_id": "cs-native",
                "client": "codex",
                "created_at": 5.0,
                "turns": [],
            }
        },
    )
    monkeypatch.setattr(cli_chat_session, "_path", lambda: state_path)

    result = CliRunner().invoke(
        cli_chat_session.chat_session_group,
        ["get", "cs-native", "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["session_id"] == "cs-native"
    assert payload["source"] == "cli"


def test_get_treats_id_invalid_for_the_http_store_as_not_found(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A session id the CLI's own (looser) regex accepts but the HTTP
    SessionStore's (stricter, no-dots) regex rejects must not raise a raw
    ValueError — it's just absent from that store.
    """
    state_path = tmp_path / "chat_sessions.json"
    http_root = tmp_path / "chat" / "sessions"
    monkeypatch.setattr(cli_chat_session, "_path", lambda: state_path)
    monkeypatch.setattr(cli_chat_session, "_http_sessions_root", lambda: http_root)

    result = CliRunner().invoke(
        cli_chat_session.chat_session_group,
        ["get", "sess.with.dots", "--json"],
    )

    assert result.exit_code == 1
    assert "session not found" in result.output


def test_get_still_fails_cleanly_when_session_is_nowhere(
    monkeypatch,
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "chat_sessions.json"
    http_root = tmp_path / "chat" / "sessions"
    monkeypatch.setattr(cli_chat_session, "_path", lambda: state_path)
    monkeypatch.setattr(cli_chat_session, "_http_sessions_root", lambda: http_root)

    result = CliRunner().invoke(
        cli_chat_session.chat_session_group,
        ["get", "nonexistent", "--json"],
    )

    assert result.exit_code == 1
    assert "session not found" in result.output


def test_list_merges_cli_and_http_sessions(
    monkeypatch,
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "chat_sessions.json"
    http_root = tmp_path / "chat" / "sessions"
    _write_sessions(
        state_path,
        {
            "cs-native": {
                "session_id": "cs-native",
                "client": "codex",
                "created_at": 5.0,
                "turns": [],
            }
        },
    )
    monkeypatch.setattr(cli_chat_session, "_path", lambda: state_path)
    monkeypatch.setattr(cli_chat_session, "_http_sessions_root", lambda: http_root)
    _write_http_session(
        http_root,
        "abc123httpsess",
        [{"role": "user", "text": "hi"}, {"role": "assistant", "text": "hey"}],
    )

    result = CliRunner().invoke(
        cli_chat_session.chat_session_group,
        ["list", "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    ids_by_source = {row["session_id"]: row["source"] for row in payload["sessions"]}
    assert ids_by_source == {"cs-native": "cli", "abc123httpsess": "http"}
