"""Tests for signal-gather phase in dream run."""
from __future__ import annotations

from unittest.mock import patch

from memo.cli_dream import _run_signal_gather


def test_signal_gather_returns_summary_keys() -> None:
    fake_result = {
        "status": "ok",
        "files_processed": 3,
        "saved": ["id1", "id2"],
        "skipped_dup": 1,
        "candidates": 5,
    }
    with patch("memo.cli_dream.mine_transcripts", return_value=fake_result):
        result = _run_signal_gather(since_days=7, file_limit=20)
    assert result["files_processed"] == 3
    assert result["memorias_saved"] == 2
    assert result["skipped_dup"] == 1


def test_signal_gather_no_files_returns_zeros() -> None:
    with patch("memo.cli_dream.mine_transcripts", return_value={"status": "no_files"}):
        result = _run_signal_gather(since_days=7, file_limit=20)
    assert result["memorias_saved"] == 0
    assert result["files_processed"] == 0


def test_signal_gather_exception_returns_zeros() -> None:
    with patch("memo.cli_dream.mine_transcripts", side_effect=Exception("boom")):
        result = _run_signal_gather(since_days=7, file_limit=20)
    assert result["memorias_saved"] == 0
    assert "error" in result
