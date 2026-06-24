"""Auto-update on memo-mcp start.

When ``MEMO_AUTO_UPDATE`` is enabled, memo-mcp checks (throttled) whether a
newer **tagged** release exists in the git repo and, if so, spawns a detached
``memo upgrade --to-tag <tag>`` in the background. The running process keeps
the old code (you can't hot-swap a live interpreter) — the new version takes
effect on the NEXT memo-mcp start.

Design choices (2026-06-22):
- Trigger is a git **tag** (``vX.Y.Z``), not any commit, so an un-tagged push
  (work in progress / a broken commit) never propagates to the fleet.
- Default OFF (memo is public); enabled per-machine via the flag / install-mcp.
- Network + git failures are swallowed: auto-update must never break or delay a
  memo-mcp startup.
"""

from __future__ import annotations

import logging
import subprocess
import sys

from memo.config import Config
from memo.flags import flag_bool, flag_int, flag_str

_log = logging.getLogger(__name__)

DEFAULT_REPO = "https://github.com/jagoff/memo.git"
_CHECK_STAMP = "auto_update_check"


def _parse_semver(tag: str) -> tuple[int, int, int] | None:
    """``v1.2.3`` / ``1.2.3`` → ``(1, 2, 3)``. Anything non-numeric → None.

    Pre-release/build suffixes (``1.2.3-rc1``) are ignored on the patch field so
    they sort below the plain release, which is fine for an "is there a newer
    stable" check.
    """
    core = tag.strip().lstrip("vV").split("-", 1)[0].split("+", 1)[0]
    parts = core.split(".")
    if len(parts) != 3:
        return None
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        return None


def is_newer(remote: str, local: str) -> bool:
    """True iff ``remote`` is a strictly higher semver than ``local``."""
    r, lo = _parse_semver(remote), _parse_semver(local)
    if r is None or lo is None:
        return False
    return r > lo


def latest_remote_tag(repo_url: str, *, timeout: int = 10) -> str | None:
    """Highest ``vX.Y.Z`` tag in the remote repo, or None on any failure.

    Uses ``git ls-remote --tags --refs`` so no clone/fetch is needed and the
    probe stays cheap. Dereferenced (``--refs``) so peeled ``^{}`` lines are
    excluded.
    """
    try:
        cp = subprocess.run(
            ["git", "ls-remote", "--tags", "--refs", repo_url],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        _log.debug("auto-update: ls-remote failed: %s", exc)
        return None
    if cp.returncode != 0:
        _log.debug("auto-update: ls-remote rc=%d: %s", cp.returncode, cp.stderr.strip())
        return None

    best: tuple[int, int, int] | None = None
    best_tag: str | None = None
    for line in cp.stdout.splitlines():
        ref = line.rsplit("refs/tags/", 1)[-1].strip()
        if not ref or ref == line:
            continue
        ver = _parse_semver(ref)
        if ver is not None and (best is None or ver > best):
            best, best_tag = ver, ref
    return best_tag


def _should_check(cfg: Config, interval_s: int, now: float) -> bool:
    """Throttle: True if no check stamp or it's older than ``interval_s``."""
    stamp = cfg.state_dir / _CHECK_STAMP
    try:
        last = float(stamp.read_text().strip())
    except (OSError, ValueError):
        return True
    return (now - last) >= interval_s


def _record_check(cfg: Config, now: float) -> None:
    stamp = cfg.state_dir / _CHECK_STAMP
    try:
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(str(now))
    except OSError as exc:
        _log.debug("auto-update: could not write check stamp: %s", exc)


def maybe_auto_update(cfg: Config | None = None) -> bool:
    """Entry point called at memo-mcp startup. Gated, throttled, non-blocking.

    Returns True iff a background update was spawned (mainly for tests). Never
    raises — any failure is logged at debug and swallowed so a startup is never
    delayed or broken by the updater.
    """
    try:
        if not flag_bool("MEMO_AUTO_UPDATE"):
            return False
        cfg = cfg or Config.from_env()
        import time

        now = time.time()
        interval = flag_int("MEMO_AUTO_UPDATE_INTERVAL_S") or 21600
        if not _should_check(cfg, interval, now):
            return False
        # Record the check up front so a slow/looping spawn can't re-trigger.
        _record_check(cfg, now)

        from memo import __version__

        repo = flag_str("MEMO_AUTO_UPDATE_REPO") or DEFAULT_REPO
        tag = latest_remote_tag(repo)
        if not tag or not is_newer(tag, __version__):
            return False

        _log.info("auto-update: %s → %s (spawning background upgrade)", __version__, tag)
        log_file = cfg.state_dir / "auto_update.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a") as fh:
            subprocess.Popen(
                [sys.executable, "-m", "memo.cli", "upgrade", "--to-tag", tag],
                stdout=fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        return True
    except Exception as exc:  # never break startup
        _log.debug("auto-update: skipped (%s)", exc)
        return False
