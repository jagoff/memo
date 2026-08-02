"""Low-level, exact-TTY input delivery for registered local terminals."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import termios
from pathlib import Path

from memo.errors import TerminalDeliveryError

_PROMPT_LITERAL_MARKER = "__MEMO_PROMPT_LITERAL__"


class _PartialTIOCSTIError(OSError):
    """TIOCSTI failed after at least one byte reached the target."""


def _deliver_tiocsti(tty: Path, payload: bytes) -> None:
    request = getattr(termios, "TIOCSTI", None)
    if request is None:
        raise OSError("TIOCSTI is unavailable")
    flags = os.O_RDWR | getattr(os, "O_NOCTTY", 0)
    fd = os.open(tty, flags)
    injected = 0
    try:
        for byte in payload:
            try:
                fcntl.ioctl(fd, request, bytes([byte]))
            except OSError as exc:
                if injected:
                    raise _PartialTIOCSTIError(f"TIOCSTI stopped after {injected} bytes") from exc
                raise
            injected += 1
    finally:
        os.close(fd)


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
    except subprocess.TimeoutExpired as exc:
        raise TerminalDeliveryError("terminal automation timed out") from exc
    except OSError as exc:
        raise TerminalDeliveryError("terminal automation could not start") from exc


def _split_payload(payload: bytes) -> tuple[str, bool]:
    submit = payload.endswith(b"\r")
    body = payload[:-1] if submit else payload
    try:
        return body.decode("utf-8"), submit
    except UnicodeDecodeError as exc:  # defensive: bridge always supplies UTF-8
        raise TerminalDeliveryError("terminal payload is not UTF-8") from exc


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

_TERMINAL_INPUT_SCRIPT = r"""
on run argv
  set targetTty to item 1 of argv
  set submitPrompt to ((item 2 of argv) is "1")
  set promptText to __MEMO_PROMPT_LITERAL__

  tell application "Terminal"
    set foundTab to missing value
    set foundWindow to missing value
    repeat with candidateWindow in windows
      repeat with candidateTab in tabs of candidateWindow
        if tty of candidateTab is targetTty then
          set foundTab to candidateTab
          set foundWindow to candidateWindow
          exit repeat
        end if
      end repeat
      if foundTab is not missing value then exit repeat
    end repeat
    if foundTab is missing value then return "not found"
    if submitPrompt then
      do script promptText in foundTab
      return "ok"
    end if
    set selected tab of foundWindow to foundTab
    set index of foundWindow to 1
    activate
  end tell

  delay 0.1
  if promptText is not "" then
    tell application "System Events"
      tell process "Terminal" to keystroke promptText
    end tell
  end if
  return "ok"
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
        raise TerminalDeliveryError("Ghostty could not find the registered TTY")


def _deliver_terminal(tty: Path, payload: bytes) -> None:
    body, submit = _split_payload(payload)
    result = _run_osascript(
        _TERMINAL_INPUT_SCRIPT,
        str(tty),
        "1" if submit else "0",
        prompt_text=body,
    )
    if result.returncode != 0 or result.stdout.strip() != "ok":
        raise TerminalDeliveryError("Terminal could not find the registered TTY")


def _deliver_iterm(tty: Path, payload: bytes, *, app_name: str) -> None:
    body, submit = _split_payload(payload)
    app_literal = json.dumps(app_name)
    script = f"""
on run argv
  set targetTty to item 1 of argv
  set submitPrompt to ((item 2 of argv) is "1")
  set promptText to __MEMO_PROMPT_LITERAL__

  tell application {app_literal}
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


def deliver_input(tty: Path, payload: bytes, *, terminal_app: str) -> str:
    """Deliver sanitized bytes to one validated TTY and return its transport."""
    try:
        _deliver_tiocsti(tty, payload)
        return "tiocsti"
    except _PartialTIOCSTIError as partial_error:
        raise TerminalDeliveryError(
            "terminal input delivery was partial; refusing fallback"
        ) from partial_error
    except OSError as direct_error:
        try:
            if terminal_app == "Ghostty":
                _deliver_ghostty(tty, payload)
                return "ghostty-applescript"
            if terminal_app == "Terminal":
                _deliver_terminal(tty, payload)
                return "terminal-applescript"
            if terminal_app in {"iTerm", "iTerm2"}:
                _deliver_iterm(tty, payload, app_name=terminal_app)
                return "iterm-applescript"
        except TerminalDeliveryError:
            raise
        except OSError as fallback_error:
            raise TerminalDeliveryError("terminal input delivery failed") from fallback_error
        raise TerminalDeliveryError(
            "terminal input injection is unavailable: no exact-session fallback"
        ) from direct_error
