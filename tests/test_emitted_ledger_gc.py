"""Tests for `memo ops gc-emitted-ledgers` (Task 9: prune stale ledgers)."""

from __future__ import annotations

import os
import time
from pathlib import Path

from click.testing import CliRunner

from memo import emitted_ledger as el
from memo.cli_ops import ops_group


def test_gc_removes_ledgers_older_than_48h(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path))
    el.append(
        tmp_path,
        "old",
        [el.Entry(id="a", h="deadbeef", n=1, ref="memo-r/a", t=1, src="mcp")],
    )
    el.append(
        tmp_path,
        "live",
        [el.Entry(id="b", h="deadbeef", n=1, ref="memo-r/b", t=1, src="mcp")],
    )
    stale = time.time() - 60 * 60 * 72
    os.utime(el.ledger_path(tmp_path, "old"), (stale, stale))

    result = CliRunner().invoke(ops_group, ["gc-emitted-ledgers"])
    assert result.exit_code == 0, result.output
    assert "1" in result.output
    assert not el.ledger_path(tmp_path, "old").exists()
    assert el.ledger_path(tmp_path, "live").exists()


def test_gc_emitted_ledgers_missing_dir_is_success(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path))
    result = CliRunner().invoke(ops_group, ["gc-emitted-ledgers"])
    assert result.exit_code == 0, result.output
    assert "0" in result.output


def test_gc_emitted_ledgers_max_age_hours_option(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path))
    el.append(
        tmp_path,
        "recent",
        [el.Entry(id="a", h="deadbeef", n=1, ref="memo-r/a", t=1, src="mcp")],
    )
    stale = time.time() - 60 * 60 * 2  # 2h old
    os.utime(el.ledger_path(tmp_path, "recent"), (stale, stale))

    result = CliRunner().invoke(ops_group, ["gc-emitted-ledgers", "--max-age-hours", "1"])
    assert result.exit_code == 0, result.output
    assert "1" in result.output
    assert not el.ledger_path(tmp_path, "recent").exists()
