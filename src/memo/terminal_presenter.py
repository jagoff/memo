"""Low-level, exact-TTY input delivery for registered local terminals."""

from __future__ import annotations

import errno
import fcntl
import json
import os
import secrets
import subprocess
import termios
from contextlib import suppress
from pathlib import Path

from memo.errors import TerminalDeliveryError

_LINUX_TIOCSTI_POLICY = Path("/proc/sys/dev/tty/legacy_tiocsti")
_PROMPT_LITERAL_MARKER = "__MEMO_PROMPT_LITERAL__"
_ITERM_BUNDLE_ID = "com.googlecode.iterm2"


class _PartialTIOCSTIError(OSError):
    """TIOCSTI failed after input may already have reached the terminal."""


def _deliver_tiocsti(tty: Path, payload: bytes) -> None:
    request = getattr(termios, "TIOCSTI", None)
    if request is None:
        raise OSError(errno.ENOTSUP, "TIOCSTI is unavailable")
    flags = os.O_RDWR | getattr(os, "O_NOCTTY", 0)
    fd = os.open(tty, flags)
    delivered = 0
    try:
        for byte in payload:
            try:
                fcntl.ioctl(fd, request, bytes([byte]))
            except OSError as exc:
                if delivered:
                    raise _PartialTIOCSTIError(
                        errno.EIO,
                        "TIOCSTI failed after partial input delivery",
                    ) from exc
                raise
            delivered += 1
    finally:
        try:
            os.close(fd)
        except OSError as exc:
            if delivered:
                raise _PartialTIOCSTIError(
                    errno.EIO,
                    "TTY close failed after input delivery",
                ) from exc
            raise


def _run_osascript(
    script: str,
    *args: str,
    prompt_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    command = ["osascript", "-e", script, *args]
    script_input: str | None = None
    if prompt_text is not None:
        if script.count(_PROMPT_LITERAL_MARKER) != 1:
            raise TerminalDeliveryError("terminal automation payload marker is invalid")
        script_input = script.replace(
            _PROMPT_LITERAL_MARKER,
            json.dumps(prompt_text, ensure_ascii=False),
        )
        command = ["osascript", "-", *args]
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            input=script_input,
        )
    except subprocess.TimeoutExpired:
        # Raise outside the exception handler so the TimeoutExpired object (whose
        # command contains argv, including the prompt) is not retained as context.
        failure = TerminalDeliveryError(errno.ETIMEDOUT, "terminal automation timed out")
    except OSError:
        failure = TerminalDeliveryError(
            errno.ENOTSUP,
            "terminal automation could not start",
        )
    raise failure


def _run_tmux(
    *args: str,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ["tmux", *args],
            check=False,
            capture_output=True,
            input=input_bytes,
            timeout=5,
        )
    except subprocess.TimeoutExpired:
        failure = TerminalDeliveryError(errno.ETIMEDOUT, "tmux transport timed out")
    except OSError:
        failure = TerminalDeliveryError(
            errno.ENOTSUP,
            "tmux exact-TTY input transport is unavailable",
        )
    raise failure


def _split_payload(payload: bytes) -> tuple[str, bool]:
    submit = payload.endswith(b"\r")
    body = payload[:-1] if submit else payload
    try:
        return body.decode("utf-8"), submit
    except UnicodeDecodeError:  # defensive: bridge always supplies UTF-8
        pass
    raise TerminalDeliveryError("terminal payload is not UTF-8")


def exact_tty_transport_supported(tty: Path, terminal_app: str) -> bool:
    """Return whether registration can advertise a safe exact-TTY transport.

    macOS terminal integrations target an application-owned TTY object. Linux
    ``/dev/pts`` registrations require tmux exact-pane delivery or an explicit
    kernel opt-in to legacy TIOCSTI; an unknown kernel policy fails closed.
    """
    # Terminal.app exposes no input API bound to a tab/session object. Global
    # System Events keystrokes and clipboard paste are focus-racy, so this
    # presenter is unavailable even to injected/internal capability callers.
    if terminal_app == "Terminal":
        return False
    if not tty.is_relative_to(Path("/dev/pts")):
        return terminal_app in {"Ghostty", "iTerm", "iTerm2", "tmux"}
    if terminal_app == "tmux":
        return True
    if terminal_app:
        return False
    if getattr(termios, "TIOCSTI", None) is None:
        return False
    try:
        return _LINUX_TIOCSTI_POLICY.read_text(encoding="ascii").strip() == "1"
    except OSError:
        return False


_GHOSTTY_INPUT_SCRIPT = r"""
on run argv
  set targetTty to item 1 of argv
  set submitPrompt to ((item 2 of argv) is "1")
  set promptText to __MEMO_PROMPT_LITERAL__

  tell application "Ghostty"
    repeat with candidateWindow in windows
      repeat with candidateTab in tabs of candidateWindow
        repeat with candidateTerminal in terminals of candidateTab
          try
            set candidateTty to tty of candidateTerminal
            if candidateTty is targetTty then
              if promptText is not "" then input text promptText to candidateTerminal
              if submitPrompt then send key "enter" to candidateTerminal
              return "ok"
            end if
          end try
        end repeat
      end repeat
    end repeat
  end tell
  return "not found"
end run
"""


def _deliver_ghostty(tty: Path, payload: bytes) -> None:
    body, submit = _split_payload(payload)
    result = _run_osascript(
        _GHOSTTY_INPUT_SCRIPT,
        str(tty),
        "1" if submit else "0",
        prompt_text=body,
    )
    if result.returncode != 0 or result.stdout.strip() != "ok":
        raise TerminalDeliveryError(
            errno.EIO,
            "Ghostty could not deliver to the registered TTY",
        )


def _deliver_iterm(tty: Path, payload: bytes, *, app_name: str) -> None:
    body, submit = _split_payload(payload)
    script = f"""
on run argv
  set targetTty to item 1 of argv
  set submitPrompt to ((item 2 of argv) is "1")
  set promptText to __MEMO_PROMPT_LITERAL__

  tell application id "{_ITERM_BUNDLE_ID}"
    repeat with candidateWindow in windows
      repeat with candidateTab in tabs of candidateWindow
        repeat with candidateSession in sessions of candidateTab
          if tty of candidateSession is targetTty then
            if submitPrompt then
              tell candidateSession to write text promptText
            else
              tell candidateSession to write text promptText newline false
            end if
            return "ok"
          end if
        end repeat
      end repeat
    end repeat
  end tell
  return "not found"
end run
"""
    result = _run_osascript(
        script,
        str(tty),
        "1" if submit else "0",
        prompt_text=body,
    )
    if result.returncode != 0 or result.stdout.strip() != "ok":
        raise TerminalDeliveryError(f"{app_name} could not find the registered TTY")


def _tmux_target_for_tty(tty: Path) -> str:
    panes = _run_tmux("list-panes", "-a", "-F", "#{pane_id}\t#{pane_tty}")
    if panes.returncode != 0:
        raise TerminalDeliveryError(errno.ENOTSUP, "tmux pane discovery is unavailable")
    for line in panes.stdout.decode("utf-8", errors="replace").splitlines():
        pane_id, separator, pane_tty = line.partition("\t")
        if separator and pane_tty == str(tty):
            return pane_id
    raise TerminalDeliveryError(errno.ENOTSUP, "tmux could not find the registered TTY")


def _paste_tmux_body(target: str, body: bytes) -> None:
    buffer_name = f"memo-{os.getpid()}-{secrets.token_hex(8)}"
    loaded = False
    try:
        result = _run_tmux("load-buffer", "-b", buffer_name, "-", input_bytes=body)
        if result.returncode != 0:
            raise TerminalDeliveryError(errno.EIO, "tmux could not stage terminal input")
        loaded = True
        result = _run_tmux("paste-buffer", "-b", buffer_name, "-d", "-t", target)
        if result.returncode != 0:
            raise TerminalDeliveryError(errno.EIO, "tmux could not deliver terminal input")
        loaded = False
    finally:
        if loaded:
            # Preserve the primary staging/delivery error. A named buffer
            # cleanup failure must never make a retry look safe.
            with suppress(OSError):
                _run_tmux("delete-buffer", "-b", buffer_name)


def _deliver_tmux(tty: Path, payload: bytes) -> None:
    """Deliver via the one tmux pane whose reported TTY exactly matches *tty*."""
    target = _tmux_target_for_tty(tty)
    body, submit = _split_payload(payload)
    if body:
        _paste_tmux_body(target, body.encode("utf-8"))
    if submit:
        result = _run_tmux("send-keys", "-t", target, "Enter")
        if result.returncode != 0:
            raise TerminalDeliveryError(errno.EIO, "tmux could not submit terminal input")


def _deliver_exact_session(tty: Path, payload: bytes, terminal_app: str) -> str:
    if terminal_app == "tmux":
        _deliver_tmux(tty, payload)
        return "tmux"
    if terminal_app == "Ghostty":
        _deliver_ghostty(tty, payload)
        return "ghostty-applescript"
    _deliver_iterm(tty, payload, app_name=terminal_app)
    return "iterm-applescript"


def deliver_input(tty: Path, payload: bytes, *, terminal_app: str) -> str:
    """Deliver sanitized bytes to one validated TTY and return its transport."""
    if terminal_app == "Terminal":
        raise TerminalDeliveryError(
            errno.ENOTSUP,
            "Terminal.app has no safe exact-session input transport",
        )
    if terminal_app in {"tmux", "Ghostty", "iTerm", "iTerm2"}:
        try:
            return _deliver_exact_session(tty, payload, terminal_app)
        except TerminalDeliveryError:
            raise
        except OSError:
            # Do not fall back from an exact-session API to a weaker TTY-bound
            # transport: the exact API may already have delivered part of the body.
            pass
        raise TerminalDeliveryError("terminal input delivery failed")

    if tty.is_relative_to(Path("/dev/pts")) and exact_tty_transport_supported(tty, ""):
        try:
            _deliver_tiocsti(tty, payload)
            return "tiocsti"
        except _PartialTIOCSTIError:
            # Retrying could duplicate an already-injected prefix.
            raise
        except OSError:
            raise TerminalDeliveryError(
                errno.ENOTSUP,
                "no safe exact-TTY Linux input transport is available",
            ) from None

    if tty.is_relative_to(Path("/dev/pts")):
        raise TerminalDeliveryError(
            errno.ENOTSUP,
            "no safe exact-TTY Linux input transport is available",
        )
    raise TerminalDeliveryError(errno.ENOTSUP, "terminal input injection is unavailable")
