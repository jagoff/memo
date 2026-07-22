"""`memo recap` — cross-client session recap, memo's answer to Claude Code's
native "recap" line.

## Why this exists

Claude Code prints a dim, periodic recap line straight
to the user's terminal — its own UI feature, not something memo controls.
Every OTHER memo client (codex, opencode, devin) has no equivalent, and
memo's own recall hook can't fill that gap for them: the hook's
``systemMessage`` field (``MEMO_RECALL_SYSTEM_MESSAGE``, see
``recall_logic.build_system_message``) is delivered through Claude Code's
``UserPromptSubmit`` hook-JSON contract, which only Claude Code parses —
``wire_recall_hook`` only ever writes into ``~/.claude/settings.json``.

Instead this rides the channel that's ALREADY cross-client: the pending
idle-capture notification file (``state_dir/pending_idle_notification.txt``),
already surfaced as the ``notification`` field on ``memo_search``,
``memo_ask``, ``memo_chat_ask``, ``memo_context`` and
``memo_unified_briefing`` (see ``server_core_search.py``,
``server_idle_capture.py``) — every client calling ANY of those MCP tools
picks it up, no Claude-Code-specific plumbing required. It already carries
memo's own ``※`` glyph (``※ MEMO auto-saved``); recap reuses that convention.

Color-fidelity caveat: the line is emitted with a real dim ANSI wrapper
(``\\x1b[2m...\\x1b[0m``), matching Claude Code's own muted styling. Whether a
given client's UI renders that escape sequence (vs. a GUI pane that strips or
escapes it) is outside memo's control per client — this module cannot
guarantee identical color rendering everywhere, only that the same bytes
Claude Code itself uses are the ones emitted.

## Content sourcing (cheap, precomputed)

No new LLM call. Reuses the session snapshot's existing summary chain
(``session._clean_snapshot_summary``: ``running_summary`` → ``summary`` →
``last_user_msg``) — the same fields ``refresh_summary`` (Stop-hook,
self-throttled) and ``checkpoint`` already maintain for `memo resume` /
`memo continuity`. Reading it is one JSON file read.

## Cadence

Self-throttled the same way ``capture-tick`` throttles on ``updated`` —
here keyed on ``turn_count`` vs. a stamped ``last_recap_turn``
(``session.stamp_recap_turn``), gated by ``MEMO_RECAP_EVERY_N`` (default 6
turns). Firing is a cheap comparison; never a blocking call in the hot path.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

_RECAP_PREFIX = "※ memo recap:"
_DIM_START = "\x1b[2m"
# Matches memo's own dim-ANSI convention (resume/_tui.py's
# _RESUME_TUI_DIM_OPEN/_CLOSE): SGR 2 (faint) / SGR 22 (normal intensity),
# not a full SGR 0 reset — so it doesn't clobber other active attributes.
_DIM_END = "\x1b[22m"


def format_recap_line(content: str, *, disable_hint: str = "MEMO_RECAP=0") -> str:
    """Format a single dim ANSI recap line, mirroring Claude Code's own
    ``※ memo recap: <summary>   (disable in /config)`` styling.

    Returns ``""`` for blank/whitespace-only content (nothing worth showing).
    """
    text = (content or "").strip().replace("\n", " ")
    if not text:
        return ""
    return f"{_DIM_START}{_RECAP_PREFIX} {text}   (disable: {disable_hint}){_DIM_END}"


def recap_content(snapshot: dict[str, Any] | None) -> str:
    """Pick the best one-line goal/progress summary from a session snapshot.

    Reuses ``session._clean_snapshot_summary`` — the same precedence memo
    already uses for `memo resume` / `memo continuity` (running_summary >
    summary > last_user_msg, skipping slash-command noise). Returns ``""``
    when nothing usable is available (its ``"—"`` sentinel).
    """
    if not snapshot:
        return ""
    from memo.session import _clean_snapshot_summary

    text = _clean_snapshot_summary(snapshot, 140)
    return "" if text == "—" else text


def due_for_recap(snapshot: dict[str, Any] | None, *, every_n: int) -> bool:
    """True if enough turns have passed since the last recap fired.

    Keyed on ``turn_count`` vs. the stamped ``last_recap_turn`` — mirrors the
    watermark idiom ``capture-tick``/``stamp_recall_turn`` use elsewhere.
    ``every_n <= 0`` disables recap entirely (returns False).
    """
    if every_n <= 0 or not snapshot:
        return False
    turn_count = int(snapshot.get("turn_count") or 0)
    last_recap_turn = int(snapshot.get("last_recap_turn") or 0)
    return turn_count >= every_n and (turn_count - last_recap_turn) >= every_n


def compose_system_message(presence_line: str, recap_line: str, urgent_line: str = "") -> str:
    """Join the 🧠 presence line, the ⚠️ proactive-urgent line, and the
    ``※ memo recap:`` line into one ``systemMessage`` string, newline-separated,
    without clobbering any. Order: presence, urgent, recap — the urgent nudge
    sits directly under presence so it reads first.

    Any side may be empty (feature flag off, no hits, no nudge due, no recap
    due) — the rest are returned unchanged. All empty returns ``""``.
    """
    lines = [line for line in (presence_line, urgent_line, recap_line) if line]
    return "\n".join(lines)


def maybe_write_recap(
    state_dir: Path,
    session_id: str,
    *,
    every_n: int | None = None,
) -> str | None:
    """Best-effort: if a recap is due for this session, format it and write it
    to the shared pending-notification file so the next MCP tool response (or
    the recall-hook's ``systemMessage`` on Claude Code) surfaces it.

    Returns the formatted line on success, ``None`` on no-op (not due, no
    session, no content, or disabled). Never raises — this rides the 5s
    recall-hook budget as an optional decoration, same contract as
    ``_write_capture_notification``.
    """
    try:
        from memo.flags import flag_bool, flag_int

        if not flag_bool("MEMO_RECAP"):
            return None
        n = every_n
        if n is None:
            n = flag_int("MEMO_RECAP_EVERY_N")
            n = 6 if n is None else n
        if n <= 0 or not session_id:
            return None

        from memo.session import get_session, stamp_recap_turn

        snapshot = get_session(state_dir, session_id)
        if not snapshot or not due_for_recap(snapshot, every_n=n):
            return None

        content = recap_content(snapshot)
        if not content:
            return None

        line = format_recap_line(content)
        if not line:
            return None

        state_dir = Path(state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "pending_idle_notification.txt").write_text(line + "\n", encoding="utf-8")

        turn_count = int(snapshot.get("turn_count") or 0)
        with contextlib.suppress(Exception):
            stamp_recap_turn(state_dir, session_id, turn_count)

        return line
    except Exception:
        return None
