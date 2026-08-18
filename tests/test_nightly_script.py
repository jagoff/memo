"""Contract tests for the shipped `launchd/memo-nightly.sh`.

The script is a template users install verbatim, so its two safety properties
are worth pinning: it must be POSIX-sh clean, and it must not burn the day's
slot when a run dies halfway.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.resource_hygiene

SCRIPT = Path(__file__).resolve().parents[1] / "launchd" / "memo-nightly.sh"


def test_script_is_posix_sh_clean() -> None:
    sh = shutil.which("sh")
    assert sh is not None
    proc = subprocess.run([sh, "-n", str(SCRIPT)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_due_guard_stamp_is_written_after_the_passes_not_before() -> None:
    """A run killed mid-way must be retried by the next hourly tick.

    Stamping up front blocked every tick for MEMO_NIGHTLY_MIN_INTERVAL_H (20h),
    so the 2026-08-17 run that stopped after contradict-resolve skipped gc and
    consolidate for the whole day with no error anywhere.
    """
    body = SCRIPT.read_text(encoding="utf-8")
    stamp_write = body.index('date +%s > "$_stamp"')
    first_pass = body.index('log "start codegraph-sync"')
    assert stamp_write > first_pass, "stamp must be written after the passes run"


def test_uses_a_portable_lock_not_flock() -> None:
    """macOS ships no flock(1) — a flock-guarded lock silently no-ops there."""
    body = SCRIPT.read_text(encoding="utf-8")
    assert 'mkdir "$_lockdir"' in body
    # No executable flock call (the comment explaining why may mention it).
    assert not [
        ln for ln in body.splitlines() if "flock" in ln and not ln.lstrip().startswith("#")
    ]


def test_rotates_its_own_logs() -> None:
    body = SCRIPT.read_text(encoding="utf-8")
    assert "log-rotate" in body
