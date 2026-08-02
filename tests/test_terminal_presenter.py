"""Real and fallback terminal input presentation."""

from __future__ import annotations

import os
import pty
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import memo.terminal_presenter as presenter
from memo.errors import TerminalDeliveryError


def test_low_level_tiocsti_preserves_every_input_byte(monkeypatch) -> None:
    master_fd, slave_fd = pty.openpty()
    target = Path(os.ttyname(slave_fd))
    injected: list[bytes] = []
    monkeypatch.setattr(
        presenter.fcntl, "ioctl", lambda _fd, _request, value: injected.append(value)
    )
    try:
        presenter._deliver_tiocsti(target, b"hello\r")

        assert injected == [b"h", b"e", b"l", b"l", b"o", b"\r"]
    finally:
        os.close(slave_fd)
        os.close(master_fd)


def test_ghostty_fallback_targets_and_submits_to_exact_tty_atomically(monkeypatch) -> None:
    calls: list[tuple[str, tuple[str, ...], str | None]] = []

    def deny(_tty: Path, _payload: bytes) -> None:
        raise PermissionError("kernel denied TIOCSTI")

    def run(script: str, *args: str, prompt_text: str | None = None):
        calls.append((script, args, prompt_text))
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
    assert calls[0][2] == "hello from memo"
    assert "candidateTty is targetTty" in calls[0][0]
    assert 'send key "enter" to candidateTerminal' in calls[0][0]
    assert "focus" not in calls[0][0]
    assert "activate" not in calls[0][0]
    assert "System Events" not in calls[0][0]
    assert len(calls) == 1


@pytest.mark.parametrize("payload", [b"draft only", b"submit\r"])
def test_terminal_app_always_fails_before_any_global_or_tiocsti_input(
    monkeypatch,
    payload: bytes,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        presenter,
        "_deliver_tiocsti",
        lambda _tty, _payload: calls.append("tiocsti"),
    )
    monkeypatch.setattr(
        presenter,
        "_run_osascript",
        lambda *_args, **_kwargs: calls.append("osascript"),
    )

    with pytest.raises(OSError, match="no safe exact-session") as raised:
        presenter.deliver_input(
            Path("/dev/ttys003"),
            payload,
            terminal_app="Terminal",
        )

    assert raised.value.errno is not None
    assert calls == []


def test_iterm_fallback_writes_to_exact_session_without_newline(monkeypatch) -> None:
    calls: list[tuple[str, tuple[str, ...], str | None]] = []
    monkeypatch.setattr(
        presenter,
        "_deliver_tiocsti",
        lambda _tty, _payload: (_ for _ in ()).throw(PermissionError()),
    )

    def run(script: str, *args: str, prompt_text: str | None = None):
        calls.append((script, args, prompt_text))
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(presenter, "_run_osascript", run)

    transport = presenter.deliver_input(
        Path("/dev/ttys007"),
        b"draft only",
        terminal_app="iTerm2",
    )

    assert transport == "iterm-applescript"
    assert calls[0][1] == ("/dev/ttys007", "0")
    assert calls[0][2] == "draft only"
    assert 'tell application id "com.googlecode.iterm2"' in calls[0][0]
    assert "tty of candidateSession is targetTty" in calls[0][0]
    assert "newline false" in calls[0][0]


def test_osascript_timeout_is_raised_without_body_bearing_context(monkeypatch) -> None:
    secret = "secret prompt must not escape"
    calls: list[tuple[list[str], str | None]] = []

    def timeout(command, **kwargs):
        calls.append((command, kwargs.get("input")))
        raise subprocess.TimeoutExpired(command, 5)

    monkeypatch.setattr(presenter.subprocess, "run", timeout)

    with pytest.raises(OSError, match="automation timed out") as raised:
        presenter._run_osascript(
            f"set promptText to {presenter._PROMPT_LITERAL_MARKER}",
            "/dev/ttys003",
            prompt_text=secret,
        )

    assert secret not in repr(calls[0][0])
    assert secret in str(calls[0][1])
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert secret not in repr(raised.value)


def test_osascript_rejects_invalid_payload_marker_without_starting(monkeypatch) -> None:
    started = False

    def run(*_args: object, **_kwargs: object) -> None:
        nonlocal started
        started = True

    monkeypatch.setattr(presenter.subprocess, "run", run)

    with pytest.raises(TerminalDeliveryError, match="payload marker is invalid"):
        presenter._run_osascript('return "ok"', prompt_text="secret")

    assert started is False


def test_osascript_start_failure_is_redacted_and_body_free(monkeypatch) -> None:
    def missing_binary(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError("private executable path")

    monkeypatch.setattr(presenter.subprocess, "run", missing_binary)

    with pytest.raises(TerminalDeliveryError, match="automation could not start") as raised:
        presenter._run_osascript('return "ok"')

    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert "private executable path" not in str(raised.value)


def test_non_utf8_payload_is_redacted_and_body_free() -> None:
    with pytest.raises(TerminalDeliveryError, match="payload is not UTF-8") as raised:
        presenter._split_payload(b"secret\xffbody")

    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert "secret" not in repr(raised.value)


def test_partial_tiocsti_failure_never_falls_back_or_redelivers(monkeypatch) -> None:
    master_fd, slave_fd = pty.openpty()
    target = Path(os.ttyname(slave_fd))
    injected: list[bytes] = []

    def partial(_fd, _request, value: bytes) -> None:
        injected.append(value)
        if len(injected) == 2:
            raise PermissionError("kernel stopped TIOCSTI")

    monkeypatch.setattr(presenter.fcntl, "ioctl", partial)
    try:
        with pytest.raises(OSError, match="partial input delivery"):
            presenter._deliver_tiocsti(target, b"hello\r")

        assert injected == [b"h", b"e"]
    finally:
        os.close(slave_fd)
        os.close(master_fd)


def test_exact_session_failure_never_falls_back_to_tiocsti(monkeypatch) -> None:
    calls: list[str] = []

    def fail_exact(_tty: Path, _payload: bytes) -> None:
        calls.append("ghostty")
        raise OSError("ambiguous exact-session failure")

    monkeypatch.setattr(presenter, "_deliver_ghostty", fail_exact)
    monkeypatch.setattr(
        presenter,
        "_deliver_tiocsti",
        lambda _tty, _payload: calls.append("tiocsti"),
    )

    with pytest.raises(OSError, match="terminal input delivery failed"):
        presenter.deliver_input(Path("/dev/ttys003"), b"body\r", terminal_app="Ghostty")

    assert calls == ["ghostty"]


def test_tmux_transport_resolves_exact_pane_and_keeps_body_out_of_argv(monkeypatch) -> None:
    calls: list[tuple[list[str], bytes | None]] = []

    def deny(_tty: Path, _payload: bytes) -> None:
        raise PermissionError("hardened kernel")

    def run(args, **kwargs):
        calls.append((args, kwargs.get("input")))
        stdout = b"%1\t/dev/pts/8\n%2\t/dev/pts/7\n" if "list-panes" in args else b""
        return SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")

    monkeypatch.setattr(presenter, "_deliver_tiocsti", deny)
    monkeypatch.setattr(presenter.subprocess, "run", run)

    transport = presenter.deliver_input(
        Path("/dev/pts/7"),
        "linux secrét\r".encode(),
        terminal_app="tmux",
    )

    assert transport == "tmux"
    assert calls[0][0] == ["tmux", "list-panes", "-a", "-F", "#{pane_id}\t#{pane_tty}"]
    assert calls[1][1] == "linux secrét".encode()
    assert all("linux secrét" not in repr(args) for args, _body in calls)
    paste_args = next(args for args, _body in calls if "paste-buffer" in args)
    enter_args = next(args for args, _body in calls if "send-keys" in args)
    assert paste_args[-2:] == ["-t", "%2"]
    assert enter_args[-2:] == ["%2", "Enter"]


def test_tmux_cleanup_failure_does_not_mask_primary_delivery_error(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    def run(*args: str, input_bytes: bytes | None = None):
        calls.append(args)
        if "list-panes" in args:
            return SimpleNamespace(returncode=0, stdout=b"%2\t/dev/pts/7\n", stderr=b"")
        if "paste-buffer" in args:
            return SimpleNamespace(returncode=1, stdout=b"", stderr=b"")
        if "delete-buffer" in args:
            raise OSError("cleanup failed")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(presenter, "_run_tmux", run)

    with pytest.raises(OSError, match="could not deliver terminal input"):
        presenter._deliver_tmux(Path("/dev/pts/7"), b"body")

    assert any("delete-buffer" in args for args in calls)


def test_linux_capability_requires_tmux_or_explicit_kernel_opt_in(
    tmp_path,
    monkeypatch,
) -> None:
    policy = tmp_path / "legacy_tiocsti"
    monkeypatch.setattr(presenter, "_LINUX_TIOCSTI_POLICY", policy)
    monkeypatch.setattr(presenter.termios, "TIOCSTI", 0x5412, raising=False)

    policy.write_text("0\n", encoding="ascii")
    assert not presenter.exact_tty_transport_supported(Path("/dev/pts/7"), "")
    assert presenter.exact_tty_transport_supported(Path("/dev/pts/7"), "tmux")

    policy.write_text("1\n", encoding="ascii")
    assert presenter.exact_tty_transport_supported(Path("/dev/pts/7"), "")
    assert not presenter.exact_tty_transport_supported(Path("/dev/ttys007"), "")
    assert not presenter.exact_tty_transport_supported(Path("/dev/ttys007"), "Terminal")
    assert presenter.exact_tty_transport_supported(Path("/dev/ttys007"), "Ghostty")
    assert presenter.exact_tty_transport_supported(Path("/dev/ttys007"), "iTerm2")


def test_hardened_linux_without_exact_transport_is_rejected_explicitly(monkeypatch) -> None:
    monkeypatch.setattr(
        presenter,
        "_deliver_tiocsti",
        lambda _tty, _payload: (_ for _ in ()).throw(PermissionError("denied")),
    )

    with pytest.raises(OSError, match="no safe exact-TTY Linux input transport") as raised:
        presenter.deliver_input(Path("/dev/pts/7"), b"do not deliver\r", terminal_app="")

    assert raised.value.errno is not None
