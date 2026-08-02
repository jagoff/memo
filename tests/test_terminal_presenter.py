"""Real and fallback terminal input presentation."""

from __future__ import annotations

import os
import pty
from pathlib import Path
from types import SimpleNamespace

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


def test_ghostty_fallback_targets_exact_tty_and_submits_separately(monkeypatch) -> None:
    calls: list[tuple[str, tuple[str, ...]]] = []

    def deny(_tty: Path, _payload: bytes) -> None:
        raise PermissionError("kernel denied TIOCSTI")

    def run(script: str, *args: str):
        calls.append((script, args))
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(presenter, "_deliver_tiocsti", deny)
    monkeypatch.setattr(presenter, "_run_osascript", run)

    transport = presenter.deliver_input(
        Path("/dev/ttys003"),
        b"hello from memo\r",
        terminal_app="Ghostty",
    )

    assert transport == "ghostty-applescript"
    assert calls[0][1] == ("/dev/ttys003", "hello from memo")
    assert "candidateTty is targetTty" in calls[0][0]
    assert len(calls) == 2
    assert "key code 36" in calls[1][0]


def test_terminal_app_fallback_passes_payload_as_argv(monkeypatch) -> None:
    calls: list[tuple[str, tuple[str, ...]]] = []
    monkeypatch.setattr(
        presenter,
        "_deliver_tiocsti",
        lambda _tty, _payload: (_ for _ in ()).throw(PermissionError()),
    )

    def run(script: str, *args: str):
        calls.append((script, args))
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(presenter, "_run_osascript", run)

    transport = presenter.deliver_input(
        Path("/dev/ttys003"),
        b"hello 'quoted'\r",
        terminal_app="Terminal",
    )

    assert transport == "terminal-applescript"
    assert calls[0][1] == ("/dev/ttys003", "1", "hello 'quoted'")
    assert "tty of candidateTab is targetTty" in calls[0][0]


def test_iterm_fallback_writes_to_exact_session_without_newline(monkeypatch) -> None:
    calls: list[tuple[str, tuple[str, ...]]] = []
    monkeypatch.setattr(
        presenter,
        "_deliver_tiocsti",
        lambda _tty, _payload: (_ for _ in ()).throw(PermissionError()),
    )

    def run(script: str, *args: str):
        calls.append((script, args))
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(presenter, "_run_osascript", run)

    transport = presenter.deliver_input(
        Path("/dev/ttys007"),
        b"draft only",
        terminal_app="iTerm2",
    )

    assert transport == "iterm-applescript"
    assert calls[0][1] == ("/dev/ttys007", "0", "draft only")
    assert "tty of candidateSession is targetTty" in calls[0][0]
    assert "newline false" in calls[0][0]
