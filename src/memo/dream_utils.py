"""`memo dream` utilities — lock management, convergence guard, and helpers.

Extracted from `cli_dream.py` and `cli_dream_passes.py` for clarity. Keeps
the main dream dispatcher focused on orchestration. Includes:
  - Lock management (single-owner flock)
  - Convergence guard (check if corpus changed since last run)
  - Helper utilities (timestamps, paths, fingerprints, progress bar)
"""

from __future__ import annotations

import json
import logging as _logging
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

if TYPE_CHECKING:
    from memo.config import Config
    from memo.memory.facade import Memory

_log = _logging.getLogger(__name__)


def _iso_now() -> str:
    """Return current UTC time in ISO format (seconds precision)."""
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="seconds")


def _state_path(cfg: Config) -> Path:
    """Return the dream state directory."""
    return cfg.state_dir / "dream"


def _harvested_labels_path(cfg: Config) -> Path:
    """Return the path to harvested eval labels."""
    return cfg.state_dir / "eval" / "harvested_labels.json"


def _older_id(mem: Any, id_a: str, id_b: str) -> tuple[str, str]:
    """Return (older_id, newer_id) by comparing updated timestamps."""
    ra, rb = mem.get(id_a), mem.get(id_b)
    ua = getattr(ra, "updated", "") or ""
    ub = getattr(rb, "updated", "") or ""
    if ua and ub:
        return (id_a, id_b) if ua <= ub else (id_b, id_a)
    return id_a, id_b


def _corpus_fingerprint(mem: Memory) -> str | None:
    """Cheap change-signal: (row_count, max_updated_ts) from meta table.
    Any save/edit/delete moves at least one row."""
    try:
        row = mem.store._conn.execute(
            "SELECT COUNT(*), COALESCE(MAX(updated), '') FROM meta"
        ).fetchone()
        return f"{row[0]}:{row[1]}"
    except Exception:
        return None


def _make_progress() -> Progress:
    """Create a Rich progress bar, disabled in non-TTY environments."""
    from memo.cli_common import console
    from memo.flags import flag_bool

    # Non-interactive runs (launchd, piped output) skip the live-render
    # ANSI control stream to reduce output noise. Key the decision off the SAME
    # console the Progress renders to (stdout) — not sys.stderr — so a redirected
    # stderr can't silence a spinner on an interactive stdout, and vice versa.
    disable = flag_bool("MEMO_NONINTERACTIVE") or not console.is_terminal
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=24),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
        disable=disable,
    )


def acquire_dream_lock(cfg: Config) -> Any:
    """Acquire exclusive file lock on .dream.lock. Raises OSError if unavailable.
    Caller must call release_dream_lock() to clean up."""
    import fcntl as _fcntl

    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cfg.state_dir / ".dream.lock"
    fh = lock_path.open("w")
    try:
        _flags = _fcntl.fcntl(fh.fileno(), _fcntl.F_GETFD)
        _fcntl.fcntl(fh.fileno(), _fcntl.F_SETFD, _flags | _fcntl.FD_CLOEXEC)
        _fcntl.flock(fh.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB)
        return fh
    except OSError:
        fh.close()
        raise


def release_dream_lock(fh: Any) -> None:
    """Release the dream lock file handle."""
    if fh is not None:
        with suppress(Exception):
            fh.close()


def read_previous_fingerprint(cfg: Config) -> str | None:
    """Read the corpus fingerprint from the previous dream run."""
    try:
        last_json = _state_path(cfg) / "last.json"
        fp = json.loads(last_json.read_text(encoding="utf-8")).get("corpus_fp")
        return fp
    except Exception:
        return None


def check_convergence(
    force: bool,
    dry_run: bool,
    prev_fp: str | None,
    curr_fp: str | None,
    signal_gathered_count: int,
) -> bool:
    """Check if the corpus has converged since the last run.
    Returns True if converged (no work to redo), False otherwise."""
    return (
        not force
        and not dry_run
        and prev_fp is not None
        and curr_fp is not None
        and curr_fp == prev_fp
        and signal_gathered_count == 0
    )
