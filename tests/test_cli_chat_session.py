from __future__ import annotations

import multiprocessing
import os
from pathlib import Path

import pytest
from click import ClickException

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
