"""`memo health` — corpus/index/embedder health summary + watch mode.

A single read-only snapshot of operational state: corpus size, index
dims, embedder profile, health-score coverage, and warnings.

Also tests the continuous `--watch` mode and its `--json` output.
"""

from __future__ import annotations

import json
import time
from unittest.mock import patch

from click.testing import CliRunner

from memo.cli import cli
from memo.cli_health import (
    _check_daemon,
    _check_db,
    _check_fds,
    _check_sync,
    _collect_watch_signals,
    _format_watch_line,
)
from memo.health_report import build_health_report


def test_health_report_empty_corpus(mock_memory):
    report = build_health_report(mock_memory)
    assert report["corpus"]["memorias"] == 0
    assert report["index"]["expected_dims"] == mock_memory.cfg.embedder_dims
    # An empty corpus should surface at least one warning.
    assert report["warnings"], "empty corpus should warn"


def test_health_report_counts_memorias(mock_memory):
    mock_memory.save(content="one", title="One", tags=["t"])
    mock_memory.save(content="two", title="Two", tags=["t"])
    report = build_health_report(mock_memory)
    assert report["corpus"]["memorias"] == 2


def test_health_report_warns_when_health_scores_unpopulated(mock_memory):
    mock_memory.save(content="one", title="One", tags=["t"])
    report = build_health_report(mock_memory)
    assert report["health_table"]["tracked"] == 0
    joined = " ".join(report["warnings"]).lower()
    assert "health" in joined or "dream" in joined or "contradict" in joined


def test_health_report_no_embedder_probe_by_default(mock_memory):
    report = build_health_report(mock_memory)
    assert report["embedder"]["latency_ms"] is None


def test_server_health_summary_tool(mock_memory):
    import asyncio

    from memo.server import build_server

    mock_memory.save(content="one", title="One", tags=["t"])
    server = build_server(memory=mock_memory)
    tool = asyncio.run(server.get_tool("memo_health_summary")).fn
    out = tool()
    assert out["corpus"]["memorias"] == 1


def test_cli_health_json(monkeypatch, mock_memory):
    monkeypatch.setattr("memo.cli_health._get_memory", lambda cfg: mock_memory)
    monkeypatch.setattr("memo.cli_health.Config.from_env", staticmethod(lambda: mock_memory.cfg))
    result = CliRunner().invoke(cli, ["health", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "corpus" in data
    assert data["corpus"]["memorias"] == 0


# ---------------------------------------------------------------------------
# Watch-mode signal helpers
# ---------------------------------------------------------------------------


def test_check_daemon_stopped_when_no_socket_or_pid(tmp_cfg):
    """Daemon reports 'stopped' when nothing exists in state_dir."""
    result = _check_daemon(tmp_cfg)
    assert result == "stopped"


def test_check_db_missing_when_db_absent(tmp_cfg):
    """DB reports 'missing' when memvec.db does not exist."""
    result = _check_db(tmp_cfg)
    assert result == "missing"


def test_check_db_ok_with_real_db(tmp_cfg):
    """DB reports 'ok' after a Memory instance has initialised the store."""
    from memo.memory import Memory

    mem = Memory(tmp_cfg)
    mem.close()
    result = _check_db(tmp_cfg)
    assert result == "ok"


def test_check_sync_unknown_when_history_db_absent(tmp_cfg):
    """Sync reports 'unknown' when history.db does not exist."""
    result = _check_sync(tmp_cfg)
    assert result == "unknown"


def test_check_sync_ok_when_history_db_fresh(tmp_cfg):
    """Sync reports 'ok' when history.db was modified recently."""
    history_db = tmp_cfg.history_db
    history_db.parent.mkdir(parents=True, exist_ok=True)
    history_db.touch()
    result = _check_sync(tmp_cfg)
    assert result == "ok"


def test_check_sync_stale_when_history_db_old(tmp_cfg, monkeypatch):
    """Sync reports 'stale' when history.db mtime is beyond the threshold."""

    history_db = tmp_cfg.history_db
    history_db.parent.mkdir(parents=True, exist_ok=True)
    history_db.touch()
    # Make time.time() return a value 25 hours in the future
    original = time.time
    monkeypatch.setattr("memo.cli_health.time", type("T", (), {"time": staticmethod(lambda: original() + 25 * 3600), "sleep": time.sleep})())
    result = _check_sync(tmp_cfg)
    assert result == "stale"


def test_check_fds_returns_nonnegative():
    """FD count is a non-negative integer (or -1 on failure)."""
    count = _check_fds()
    # On any sane platform this should succeed and return > 0
    assert count >= 0 or count == -1


def test_collect_watch_signals_schema(tmp_cfg):
    """_collect_watch_signals returns a dict with all required keys."""
    sig = _collect_watch_signals(tmp_cfg)
    for key in ("timestamp", "status", "daemon", "sync", "db", "fds"):
        assert key in sig, f"missing key: {key}"
    assert sig["status"] in ("healthy", "degraded", "error")
    assert sig["daemon"] in ("running", "stopped", "unknown")
    assert sig["sync"] in ("ok", "stale", "unknown")
    assert sig["db"] in ("ok", "missing", "error")


def test_collect_watch_signals_healthy_with_db(tmp_cfg):
    """Status is 'healthy' when daemon is mocked running + db ok + sync ok."""
    from memo.memory import Memory

    # Initialise DB
    mem = Memory(tmp_cfg)
    mem.close()

    # Put a fresh history.db in place
    tmp_cfg.history_db.parent.mkdir(parents=True, exist_ok=True)
    tmp_cfg.history_db.touch()

    with patch("memo.cli_health._check_daemon", return_value="running"), \
         patch("memo.cli_health._check_fds", return_value=42):
        sig = _collect_watch_signals(tmp_cfg)

    assert sig["status"] == "healthy"
    assert sig["daemon"] == "running"
    assert sig["db"] == "ok"
    assert sig["fds"] == 42


def test_collect_watch_signals_degraded_when_daemon_stopped(tmp_cfg):
    """Status is 'degraded' when daemon is stopped (even if db/sync are ok)."""
    from memo.memory import Memory

    mem = Memory(tmp_cfg)
    mem.close()
    tmp_cfg.history_db.parent.mkdir(parents=True, exist_ok=True)
    tmp_cfg.history_db.touch()

    with patch("memo.cli_health._check_daemon", return_value="stopped"), \
         patch("memo.cli_health._check_fds", return_value=10):
        sig = _collect_watch_signals(tmp_cfg)

    assert sig["status"] == "degraded"


def test_format_watch_line_healthy():
    sig = {
        "timestamp": "2026-06-13 10:30:00",
        "status": "healthy",
        "daemon": "running",
        "sync": "ok",
        "db": "ok",
        "fds": 42,
    }
    line = _format_watch_line(sig)
    assert "2026-06-13 10:30:00" in line
    assert "healthy" in line
    assert "daemon=running" in line
    assert "sync=ok" in line
    assert "db=ok" in line
    assert "fds=42" in line


def test_format_watch_line_degraded():
    sig = {
        "timestamp": "2026-06-13 10:30:30",
        "status": "degraded",
        "daemon": "stopped",
        "sync": "ok",
        "db": "ok",
        "fds": 150,
    }
    line = _format_watch_line(sig)
    assert "degraded" in line
    assert "daemon=stopped" in line
    assert "fds=150" in line


# ---------------------------------------------------------------------------
# Watch mode CLI integration (single-iteration via mocked sleep)
# ---------------------------------------------------------------------------


def test_cli_health_watch_json_single_iteration(monkeypatch, tmp_cfg):
    """--watch --json emits valid JSON and exits cleanly on first sleep."""
    call_count = 0

    def _fake_sleep(n: float) -> None:
        nonlocal call_count
        call_count += 1
        raise KeyboardInterrupt  # stop after one iteration

    monkeypatch.setattr("memo.cli_health.Config.from_env", staticmethod(lambda: tmp_cfg))
    monkeypatch.setattr("memo.cli_health.time", type("T", (), {
        "time": staticmethod(time.time),
        "sleep": staticmethod(_fake_sleep),
    })())

    result = CliRunner().invoke(cli, ["health", "--watch", "--json", "--interval", "5"])
    # May exit 0 or 1 depending on how Click handles KeyboardInterrupt
    lines = [ln.strip() for ln in result.output.strip().splitlines() if ln.strip()]
    assert len(lines) >= 1, f"expected at least one JSON line, got: {result.output!r}"
    data = json.loads(lines[0])
    for key in ("timestamp", "status", "daemon", "sync", "db", "fds"):
        assert key in data, f"missing key {key!r} in watch JSON"


def test_cli_health_watch_plain_single_iteration(monkeypatch, tmp_cfg):
    """--watch (plain mode) emits a status line and exits cleanly."""
    def _fake_sleep(n: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("memo.cli_health.Config.from_env", staticmethod(lambda: tmp_cfg))
    monkeypatch.setattr("memo.cli_health.time", type("T", (), {
        "time": staticmethod(time.time),
        "sleep": staticmethod(_fake_sleep),
    })())

    result = CliRunner().invoke(cli, ["health", "--watch", "--interval", "30"])
    lines = [ln.strip() for ln in result.output.strip().splitlines() if ln.strip()]
    assert len(lines) >= 1, f"expected at least one status line, got: {result.output!r}"
    line = lines[0]
    assert "daemon=" in line
    assert "sync=" in line
    assert "db=" in line
