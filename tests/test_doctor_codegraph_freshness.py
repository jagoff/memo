"""Codegraph staleness is "behind the code", not "old".

The check compared the index file's mtime against a flat 24-hour window, so a
repo nobody has touched in two days reported::

    ! codegraph: index older than 24h — run `codegraph sync`

and running the sync answered "Already up to date" — measured 2026-08-09 on a
1,202-file index. A permanent warning that the documented remedy cannot clear
teaches the operator to ignore the line, which costs the real signal: an index
that is genuinely behind the source.

Staleness is now relative to the newest tracked source file. Age alone is only
reported when that comparison is unavailable (no git, unreadable tree), where
the old heuristic is still the best guess on offer.
"""

from __future__ import annotations

import time
from pathlib import Path

from memo.cli_doctor import _newest_source_mtime, codegraph_staleness

DAY = 24 * 3600.0


def _git(root: Path, *args: str) -> None:
    import subprocess

    subprocess.run(["git", *args], cwd=root, capture_output=True, check=True)


def test_newest_source_mtime_reads_tracked_files(tmp_path: Path) -> None:
    """The comparison point: the newest file git knows about."""
    import os

    _git(tmp_path, "init", "-b", "main")
    old = tmp_path / "old.py"
    new = tmp_path / "new.py"
    old.write_text("a = 1", encoding="utf-8")
    new.write_text("b = 2", encoding="utf-8")
    _git(tmp_path, "add", "old.py", "new.py")
    stamp = time.time() - 3 * DAY
    os.utime(old, (stamp, stamp))
    os.utime(new, (stamp + DAY, stamp + DAY))

    assert _newest_source_mtime(tmp_path) == stamp + DAY


def test_newest_source_mtime_ignores_untracked_files(tmp_path: Path) -> None:
    """A build artifact is not a reason to call the index stale."""
    import os

    _git(tmp_path, "init", "-b", "main")
    tracked = tmp_path / "mod.py"
    tracked.write_text("a = 1", encoding="utf-8")
    _git(tmp_path, "add", "mod.py")
    stamp = time.time() - 2 * DAY
    os.utime(tracked, (stamp, stamp))
    (tmp_path / "artifact.bin").write_text("fresh", encoding="utf-8")

    assert _newest_source_mtime(tmp_path) == stamp


def test_newest_source_mtime_is_none_outside_a_repo(tmp_path: Path) -> None:
    """`git ls-files` fails; there is nothing to compare against."""
    assert _newest_source_mtime(tmp_path / "not-a-repo") is None


def test_index_newer_than_sources_is_fresh(tmp_path: Path) -> None:
    source = tmp_path / "mod.py"
    source.write_text("x = 1", encoding="utf-8")
    db = tmp_path / "codegraph.db"
    db.write_text("index", encoding="utf-8")

    assert codegraph_staleness(db, newest_source_mtime=source.stat().st_mtime) is None


def test_an_old_but_current_index_is_fresh(tmp_path: Path) -> None:
    """The regression: untouched repo, week-old index, nothing to do."""
    db = tmp_path / "codegraph.db"
    db.write_text("index", encoding="utf-8")
    week_ago = time.time() - 7 * DAY
    import os

    os.utime(db, (week_ago, week_ago))

    assert codegraph_staleness(db, newest_source_mtime=week_ago - DAY) is None


def test_index_behind_the_source_is_stale(tmp_path: Path) -> None:
    db = tmp_path / "codegraph.db"
    db.write_text("index", encoding="utf-8")
    import os

    old = time.time() - 2 * DAY
    os.utime(db, (old, old))

    message = codegraph_staleness(db, newest_source_mtime=time.time())

    assert message is not None
    assert "codegraph sync" in message
    assert "behind" in message


def test_falls_back_to_age_when_sources_are_unknown(tmp_path: Path) -> None:
    """No git / unreadable tree: age is the only signal left."""
    db = tmp_path / "codegraph.db"
    db.write_text("index", encoding="utf-8")
    import os

    old = time.time() - 2 * DAY
    os.utime(db, (old, old))

    message = codegraph_staleness(db, newest_source_mtime=None)

    assert message is not None
    assert "24h" in message


def test_recent_index_with_unknown_sources_is_fresh(tmp_path: Path) -> None:
    db = tmp_path / "codegraph.db"
    db.write_text("index", encoding="utf-8")

    assert codegraph_staleness(db, newest_source_mtime=None) is None
