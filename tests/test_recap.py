"""Tests for `memo recap` — cross-client session recap line.

Mirrors Claude Code's native `※ recap: ...` line, but sourced from memo's
own session snapshot and delivered via the pending-notification channel so
every memo client (not just Claude Code) can surface it. See
`src/memo/cli_recap.py` for the design rationale.
"""

from __future__ import annotations

from pathlib import Path

from memo.cli_recap import (
    compose_system_message,
    due_for_recap,
    format_recap_line,
    maybe_write_recap,
    recap_content,
)
from memo.session import checkpoint, get_session, update_summary

# --- format_recap_line -------------------------------------------------


def test_format_recap_line_has_prefix_and_dim_ansi() -> None:
    line = format_recap_line("fix the login bug")

    assert line.startswith("\x1b[2m")
    assert line.endswith("\x1b[22m")
    assert "※ recap: fix the login bug" in line


def test_format_recap_line_has_disable_hint_referencing_flag() -> None:
    line = format_recap_line("some progress")

    assert "MEMO_RECAP=0" in line


def test_format_recap_line_strips_newlines_from_content() -> None:
    line = format_recap_line("line one\nline two")

    assert "\n" not in line
    assert "\x1b[2m" in line


def test_format_recap_line_empty_content_returns_empty_string() -> None:
    assert format_recap_line("") == ""
    assert format_recap_line("   ") == ""


# --- recap_content -------------------------------------------------


def test_recap_content_prefers_running_summary() -> None:
    snapshot = {
        "running_summary": "Refactored the auth module; tests passing.",
        "summary": "fallback summary",
        "last_user_msg": "fallback last msg",
    }

    assert recap_content(snapshot) == "Refactored the auth module; tests passing."


def test_recap_content_falls_back_to_summary_then_last_user_msg() -> None:
    snapshot = {"running_summary": None, "summary": "add dark mode toggle"}
    assert recap_content(snapshot) == "add dark mode toggle"

    snapshot2 = {"last_user_msg": "investigate the flaky test"}
    assert recap_content(snapshot2) == "investigate the flaky test"


def test_recap_content_empty_snapshot_returns_empty_string() -> None:
    assert recap_content({}) == ""
    assert recap_content(None) == ""  # type: ignore[arg-type]


def test_recap_content_skips_command_noise() -> None:
    # A command-wrapper-only prompt is noise (session.is_command_noise);
    # recap should fall through to "—" (session._clean_snapshot_summary's
    # sentinel), which recap_content treats as empty.
    snapshot = {
        "running_summary": None,
        "summary": None,
        "last_user_msg": "<local-command-stdout>Enabled plan mode</local-command-stdout>",
    }
    assert recap_content(snapshot) == ""


# --- due_for_recap (cadence) -------------------------------------------------


def test_due_for_recap_true_on_first_qualifying_turn() -> None:
    snapshot = {"turn_count": 6, "last_recap_turn": 0}
    assert due_for_recap(snapshot, every_n=6) is True


def test_due_for_recap_false_before_interval_elapsed() -> None:
    snapshot = {"turn_count": 3, "last_recap_turn": 0}
    assert due_for_recap(snapshot, every_n=6) is False


def test_due_for_recap_true_after_interval_since_last_recap() -> None:
    snapshot = {"turn_count": 12, "last_recap_turn": 6}
    assert due_for_recap(snapshot, every_n=6) is True


def test_due_for_recap_false_immediately_after_firing() -> None:
    snapshot = {"turn_count": 7, "last_recap_turn": 6}
    assert due_for_recap(snapshot, every_n=6) is False


def test_due_for_recap_zero_or_negative_every_n_disables() -> None:
    snapshot = {"turn_count": 100, "last_recap_turn": 0}
    assert due_for_recap(snapshot, every_n=0) is False
    assert due_for_recap(snapshot, every_n=-1) is False


def test_due_for_recap_missing_fields_default_to_zero() -> None:
    assert due_for_recap({}, every_n=6) is False  # turn_count 0 < every_n


# --- maybe_write_recap (glue) -------------------------------------------------


def test_maybe_write_recap_writes_notification_when_due(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    sid = "recap-sess-0001"
    for _ in range(6):
        checkpoint(state_dir, session_id=sid, cwd=str(tmp_path))
    update_summary(state_dir, sid, "working on the recap feature")

    result = maybe_write_recap(state_dir, sid, every_n=6)

    assert result is not None
    assert "※ recap:" in result
    notif_path = state_dir / "pending_idle_notification.txt"
    assert notif_path.exists()
    assert "※ recap:" in notif_path.read_text(encoding="utf-8")


def test_maybe_write_recap_stamps_last_recap_turn(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    sid = "recap-sess-0002"
    for _ in range(6):
        checkpoint(state_dir, session_id=sid, cwd=str(tmp_path))
    update_summary(state_dir, sid, "working on X")

    maybe_write_recap(state_dir, sid, every_n=6)

    snap = get_session(state_dir, sid)
    assert snap is not None
    assert snap.get("last_recap_turn") == 6


def test_maybe_write_recap_noop_when_not_due(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    sid = "recap-sess-0003"
    checkpoint(state_dir, session_id=sid, cwd=str(tmp_path))
    update_summary(state_dir, sid, "just started")

    result = maybe_write_recap(state_dir, sid, every_n=6)

    assert result is None
    assert not (state_dir / "pending_idle_notification.txt").exists()


def test_maybe_write_recap_noop_when_no_session(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    result = maybe_write_recap(state_dir, "does-not-exist-0000", every_n=6)

    assert result is None


def test_maybe_write_recap_noop_when_no_content(tmp_path: Path) -> None:
    """Session exists and is due, but has no usable summary content at all
    (no transcript, no enrichment) — should not write an empty/junk recap
    line."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    sid = "recap-sess-0004"
    for _ in range(6):
        checkpoint(state_dir, session_id=sid, cwd=str(tmp_path))

    result = maybe_write_recap(state_dir, sid, every_n=6)

    assert result is None
    assert not (state_dir / "pending_idle_notification.txt").exists()


def test_maybe_write_recap_disabled_flag_is_noop(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MEMO_RECAP", "0")
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    sid = "recap-sess-0005"
    for _ in range(6):
        checkpoint(state_dir, session_id=sid, cwd=str(tmp_path))
    update_summary(state_dir, sid, "working on the recap feature")

    result = maybe_write_recap(state_dir, sid, every_n=6)

    assert result is None
    assert not (state_dir / "pending_idle_notification.txt").exists()


def test_maybe_write_recap_never_raises_on_corrupt_state(tmp_path: Path) -> None:
    """Best-effort: a corrupt/unwritable state dir must not raise (hook-safety)."""
    state_dir = tmp_path / "does-not-exist-and-is-not-created"

    result = maybe_write_recap(state_dir, "any-session-id-0000", every_n=6)

    assert result is None


# --- compose_system_message -------------------------------------------------


def test_compose_system_message_joins_both_lines_with_newline() -> None:
    combined = compose_system_message("🧠 memo · 1: some title", "※ recap: fix the bug")

    assert combined == "🧠 memo · 1: some title\n※ recap: fix the bug"


def test_compose_system_message_presence_only() -> None:
    assert compose_system_message("🧠 memo · 1: some title", "") == "🧠 memo · 1: some title"


def test_compose_system_message_recap_only() -> None:
    assert compose_system_message("", "※ recap: fix the bug") == "※ recap: fix the bug"


def test_compose_system_message_both_empty_returns_empty_string() -> None:
    assert compose_system_message("", "") == ""
