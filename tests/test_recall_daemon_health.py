"""`memo health --daemon` snapshot + the stats it surfaces.

Covers the recall-daemon observability seam added in the trinity hardening
pass: `_DaemonStats` tracks last-request timestamp + error counts, and
`memo health --daemon` surfaces them over the socket (no MLX, no daemon
process required for the stopped-path test).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from click.testing import CliRunner

from memo.recall_stats import _DaemonStats


def test_daemon_stats_tracks_last_request_ts_and_errors() -> None:
    stats = _DaemonStats(started_at=time.time(), model="m", dims=4)
    before = stats.snapshot()
    assert before["last_request_ts"] is None
    assert before["total_requests"] == 0
    assert before["total_errors"] == 0

    t0 = time.time()
    stats.record("recall", 12.0)
    stats.record("embed_query", 8.0, error=True)
    snap = stats.snapshot()

    assert snap["total_requests"] == 2
    assert snap["total_errors"] == 1
    assert snap["last_request_ts"] is not None
    assert snap["last_request_ts"] >= t0


def test_health_daemon_reports_stopped_when_no_daemon(tmp_path: Path) -> None:
    from memo.cli import cli

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["health", "--daemon", "--json"],
        env={
            "MEMO_DATA_DIR": str(tmp_path / "data"),
            "MEMO_STATE_DIR": str(tmp_path / "state"),
        },
    )
    assert result.exit_code == 0, result.output
    snap = json.loads(result.output)
    assert snap["running"] is False
