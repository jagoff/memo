"""Guards for the dream progress spinner's TTY gating.

Regression cover for the bug where `Console(force_terminal=False)` hard-disabled
`is_terminal` in Rich 15 (short-circuits before isatty), so the dream pipeline's
spinner/progress bar never rendered in a real terminal — the pipeline looked
frozen after the pre-dream inventory panel with no feedback.
"""

from __future__ import annotations

from rich.console import Console

import memo.cli_common as cli_common
from memo.dream_utils import _make_progress


def test_global_console_auto_detects_terminal() -> None:
    # Must stay None (auto-detect) — a forced value would short-circuit
    # is_terminal and kill colour + the spinner in real terminals.
    assert cli_common.console._force_terminal is None


def test_make_progress_enabled_on_a_real_terminal(monkeypatch) -> None:
    monkeypatch.delenv("MEMO_NONINTERACTIVE", raising=False)
    monkeypatch.setattr(cli_common, "console", Console(force_terminal=True))
    progress = _make_progress()
    assert progress.disable is False


def test_make_progress_disabled_off_a_terminal(monkeypatch) -> None:
    monkeypatch.delenv("MEMO_NONINTERACTIVE", raising=False)
    monkeypatch.setattr(cli_common, "console", Console(force_terminal=False))
    progress = _make_progress()
    assert progress.disable is True


def test_make_progress_disabled_when_noninteractive(monkeypatch) -> None:
    # An explicit non-interactive run (launchd) stays quiet even on a TTY.
    monkeypatch.setenv("MEMO_NONINTERACTIVE", "1")
    monkeypatch.setattr(cli_common, "console", Console(force_terminal=True))
    progress = _make_progress()
    assert progress.disable is True
