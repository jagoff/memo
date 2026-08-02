"""Real and fallback terminal input presentation."""

from __future__ import annotations

import json
import os
import pty
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import memo.terminal_presenter as presenter


def test_tiocsti_preserves_every_input_byte(monkeypatch) -> None:
    master_fd, slave_fd = pty.openpty()
    target = Path(os.ttyname(slave_fd))
    injected: list[bytes] = []
    monkeypatch.setattr(
        presenter.fcntl, "ioctl", lambda _fd, _request, value: injected.append(value)
    )
    try:
        transport = presenter.deliver_input(target, b"hello\r", terminal_app="")

        assert injected == [b"h", b"e", b"l", b"l", b"o", b"\r"]
        assert transport == "tiocsti"
    finally:
        os.close(slave_fd)
        os.close(master_fd)


def test_ghostty_fallback_targets_exact_tty_and_submits_atomically(monkeypatch) -> None:
    calls: list[tuple[str, tuple[str, ...], dict[str, str]]] = []

    def deny(_tty: Path, _payload: bytes) -> None:
        raise PermissionError("kernel denied TIOCSTI")

    def run(script: str, *args: str, **kwargs: str):
        calls.append((script, args, kwargs))
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(presenter, "_deliver_tiocsti", deny)
    monkeypatch.setattr(presenter, "_run_osascript", run)

    transport = presenter.deliver_input(
        Path("/dev/ttys003"),
        b"hello from memo\r",
        terminal_app="Ghostty",
    )

    assert transport == "ghostty-applescript"
    assert calls[0][1] == ("/dev/ttys003", "1")
    assert calls[0][2] == {"prompt_text": "hello from memo"}
    assert "candidateTty is targetTty" in calls[0][0]
    assert 'send key "enter" to candidateTerminal' in calls[0][0]
    assert "System Events" not in calls[0][0]
    assert len(calls) == 1


def test_terminal_app_fallback_keeps_payload_out_of_argv_and_clipboard(monkeypatch) -> None:
    calls: list[tuple[str, tuple[str, ...], dict[str, str]]] = []
    monkeypatch.setattr(
        presenter,
        "_deliver_tiocsti",
        lambda _tty, _payload: (_ for _ in ()).throw(PermissionError()),
    )

    def run(script: str, *args: str, **kwargs: str):
        calls.append((script, args, kwargs))
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(presenter, "_run_osascript", run)

    transport = presenter.deliver_input(
        Path("/dev/ttys003"),
        b"hello 'quoted'\r",
        terminal_app="Terminal",
    )

    assert transport == "terminal-applescript"
    assert calls[0][1] == ("/dev/ttys003", "1")
    assert calls[0][2] == {"prompt_text": "hello 'quoted'"}
    assert "tty of candidateTab is targetTty" in calls[0][0]
    assert "clipboard" not in calls[0][0].lower()
    assert "do script promptText in foundTab" in calls[0][0]


def test_iterm_fallback_writes_to_exact_session_without_newline(monkeypatch) -> None:
    calls: list[tuple[str, tuple[str, ...], dict[str, str]]] = []
    monkeypatch.setattr(
        presenter,
        "_deliver_tiocsti",
        lambda _tty, _payload: (_ for _ in ()).throw(PermissionError()),
    )

    def run(script: str, *args: str, **kwargs: str):
        calls.append((script, args, kwargs))
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(presenter, "_run_osascript", run)

    transport = presenter.deliver_input(
        Path("/dev/ttys007"),
        b"draft only",
        terminal_app="iTerm2",
    )

    assert transport == "iterm-applescript"
    assert calls[0][1] == ("/dev/ttys007", "0")
    assert calls[0][2] == {"prompt_text": "draft only"}
    assert "tty of candidateSession is targetTty" in calls[0][0]
    assert "newline false" in calls[0][0]


def test_osascript_receives_sensitive_text_over_stdin_not_argv(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def run(command: list[str], **kwargs: object):
        captured["command"] = command
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(presenter.subprocess, "run", run)
    secret = 'secret "prompt"\nwith another line'

    presenter._run_osascript(
        "on run argv\nset promptText to __MEMO_PROMPT_LITERAL__\nreturn promptText\nend run",
        "/dev/ttys003",
        prompt_text=secret,
    )

    assert all(secret not in part for part in captured["command"])
    assert json.dumps(secret, ensure_ascii=False) in str(captured["input"])
    assert "__MEMO_PROMPT_LITERAL__" not in str(captured["input"])


def test_osascript_timeout_becomes_safe_oserror(monkeypatch) -> None:
    def timeout(*_args: object, **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired(["osascript"], timeout=5)

    monkeypatch.setattr(presenter.subprocess, "run", timeout)

    with pytest.raises(OSError, match="timed out"):
        presenter._run_osascript('return "ok"')


def test_partial_tiocsti_never_replays_payload_through_fallback(monkeypatch) -> None:
    injected: list[bytes] = []
    fallbacks: list[bytes] = []

    def inject(_fd: int, _request: int, value: bytes) -> None:
        if injected:
            raise PermissionError("kernel denied remaining bytes")
        injected.append(value)

    monkeypatch.setattr(presenter.fcntl, "ioctl", inject)
    monkeypatch.setattr(
        presenter,
        "_deliver_ghostty",
        lambda _tty, payload: fallbacks.append(payload),
    )

    with pytest.raises(OSError, match="partial"):
        presenter.deliver_input(Path("/dev/null"), b"abc", terminal_app="Ghostty")

    assert injected == [b"a"]
    assert fallbacks == []


def test_missing_platform_fallback_has_actionable_error(monkeypatch) -> None:
    monkeypatch.setattr(
        presenter,
        "_deliver_tiocsti",
        lambda _tty, _payload: (_ for _ in ()).throw(PermissionError()),
    )

    with pytest.raises(OSError, match="no exact-session fallback"):
        presenter.deliver_input(Path("/dev/pts/7"), b"hello\r", terminal_app="")
