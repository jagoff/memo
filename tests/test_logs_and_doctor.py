"""Tests for the new logs / doctor / history-error surfaces."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import time
from contextlib import closing
from pathlib import Path

import pytest
from click.testing import CliRunner

from memo.cli import cli
from memo.dashboard import append_recall_log, recall_log_path
from memo.history import HistoryStore
from memo.repo_index import _git, _git_timeout, _tracked_files


def _run(env_overrides: dict[str, str], *args: str):
    runner = CliRunner()
    base_env = {
        "MEMO_NONINTERACTIVE": "1",
        **env_overrides,
    }
    return runner.invoke(cli, list(args), env=base_env, catch_exceptions=False)


# ---------- repo_index timeout ---------------------------------------------


def test_git_timeout_env_parsing(monkeypatch):
    monkeypatch.delenv("MEMO_REPO_GIT_TIMEOUT_S", raising=False)
    assert _git_timeout(7.0) == 7.0

    monkeypatch.setenv("MEMO_REPO_GIT_TIMEOUT_S", "12.5")
    assert _git_timeout(7.0) == 12.5

    monkeypatch.setenv("MEMO_REPO_GIT_TIMEOUT_S", "0")
    assert _git_timeout(7.0) == 0.0

    monkeypatch.setenv("MEMO_REPO_GIT_TIMEOUT_S", "abc")
    assert _git_timeout(7.0) == 7.0


def test_git_raises_runtimeerror_on_timeout(monkeypatch):
    def _fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout", 1))

    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setenv("MEMO_REPO_GIT_TIMEOUT_S", "1")
    with pytest.raises(RuntimeError, match="git timed out after 1s"):
        _git(["git", "status"])


def test_tracked_files_raises_runtimeerror_on_timeout(tmp_path, monkeypatch):
    def _fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout", 1))

    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setenv("MEMO_REPO_GIT_TIMEOUT_S", "1")
    with pytest.raises(RuntimeError, match="git ls-files timed out"):
        _tracked_files(tmp_path)


# ---------- memo logs command ----------------------------------------------


def test_logs_paths_flag(tmp_path):
    result = _run(
        {"MEMO_DATA_DIR": str(tmp_path / "data"), "MEMO_STATE_DIR": str(tmp_path / "state")},
        "logs",
        "--paths",
    )
    assert result.exit_code == 0, result.output
    assert "recall:" in result.output
    assert "daemon:" in result.output
    assert "watcher:" in result.output


def test_logs_recall_renders_bail_reason(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    append_recall_log(
        state_dir,
        prompt="",
        hits=[],
        via="bail",
        reason="empty stdin",
    )

    result = _run(
        {"MEMO_DATA_DIR": str(tmp_path / "data"), "MEMO_STATE_DIR": str(state_dir)},
        "logs",
        "--source",
        "recall",
        "--tail",
        "5",
    )
    assert result.exit_code == 0, result.output
    import re

    stripped = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert "bail" in stripped
    assert "empty" in stripped


def test_append_recall_log_persists_reason_and_error(tmp_path):
    state_dir = tmp_path / "state"
    append_recall_log(
        state_dir,
        prompt="hi",
        hits=[],
        via="daemon_error",
        error="ConnectionRefused",
    )
    log_path = recall_log_path(state_dir)
    line = log_path.read_text(encoding="utf-8").strip().splitlines()[-1]
    entry = json.loads(line)
    assert entry["via"] == "daemon_error"
    assert entry["error"] == "ConnectionRefused"


# ---------- history error counter ------------------------------------------


def test_history_error_count_initial(tmp_path):
    with closing(HistoryStore(tmp_path / "history.db")) as h:
        assert h.error_count == 0


def test_history_error_count_increments_on_failure(tmp_path):
    with closing(HistoryStore(tmp_path / "history.db")) as h:
        # Close the underlying connection so every subsequent execute() raises
        # sqlite3.ProgrammingError("Cannot operate on a closed database.")
        # That exercises the swallow-and-count path without monkeypatching.
        h._conn.close()

        h.log_save(ts="2026-05-27T00:00:00Z", record_id="abc", title="t", type_="memo")
        h.log_update(
            ts="2026-05-27T00:00:00Z",
            record_id="abc",
            title="t",
            type_="memo",
            delta={"title": ("a", "b")},
        )
        h.log_delete(ts="2026-05-27T00:00:00Z", record_id="abc", title="t", type_="memo")

        assert h.error_count == 3


# ---------- doctor daemon health -------------------------------------------


def test_recall_daemon_health_not_running(tmp_path, monkeypatch):
    from memo.cli import _recall_daemon_health
    from memo.config import Config

    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "state"))
    cfg = Config.from_env()
    cfg.ensure_dirs()

    health = _recall_daemon_health(cfg)
    assert health["running"] is False
    assert health.get("note") == "not started"


def test_doctor_command_prints_fts5_and_daemon(tmp_path):
    result = _run(
        {"MEMO_DATA_DIR": str(tmp_path / "data"), "MEMO_STATE_DIR": str(tmp_path / "state")},
        "doctor",
    )
    # We don't assert exit_code (depends on whether MLX is importable on host).
    out = result.output
    assert "FTS5" in out or "fts5" in out.lower()
    assert "recall-daemon" in out


# ---------- doctor codegraph check ------------------------------------------


def _seed_codegraph_db(db_path: Path) -> None:
    """Minimal synthetic codegraph index (WAL, one call edge) — never the real DB."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(
        """
        CREATE TABLE nodes (id TEXT PRIMARY KEY, kind TEXT, name TEXT);
        CREATE TABLE edges (source TEXT, target TEXT, kind TEXT);
        INSERT INTO nodes (id, kind, name) VALUES
            ('function:a', 'function', 'Alpha'),
            ('function:b', 'function', 'Beta');
        INSERT INTO edges (source, target, kind) VALUES
            ('function:a', 'function:b', 'calls');
        """
    )
    conn.commit()
    conn.close()


def _doctor_plain_output(tmp_path: Path) -> str:
    """Run doctor and return ANSI-stripped, whitespace-collapsed output.

    Rich wraps long lines at the console width, so substring asserts must not
    depend on where a hint happens to break.
    """
    result = _run(
        {"MEMO_DATA_DIR": str(tmp_path / "data"), "MEMO_STATE_DIR": str(tmp_path / "state")},
        "doctor",
    )
    stripped = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    return " ".join(stripped.split())


def test_doctor_codegraph_missing_db_with_cli_warns(tmp_path, monkeypatch):
    """CLI installed but no index anywhere → absence IS signal, keep the WARN."""
    from memo import cli_doctor, codegraph_loader

    monkeypatch.chdir(tmp_path)  # keep _resolve_db() discovery away from any real repo
    monkeypatch.setattr(codegraph_loader, "CODEGRAPH_DB", tmp_path / ".codegraph" / "codegraph.db")
    monkeypatch.setattr(cli_doctor, "_codegraph_cli_version", lambda *a, **k: (1, 5, 0))

    out = _doctor_plain_output(tmp_path)
    assert "codegraph: index missing" in out
    assert "npm i -g @colbymchenry/codegraph && codegraph init" in out
    assert "codegraph: CLI v1.5.0" in out


def test_doctor_codegraph_not_installed_is_dim_note_not_warn(tmp_path, monkeypatch):
    """Neither index nor CLI (pipx/uv-tool installs) → informative note, no WARN."""
    from memo import cli_doctor, codegraph_loader

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(codegraph_loader, "CODEGRAPH_DB", tmp_path / ".codegraph" / "codegraph.db")
    monkeypatch.setattr(cli_doctor, "_codegraph_cli_version", lambda *a, **k: None)

    out = _doctor_plain_output(tmp_path)
    assert "codegraph: not installed" in out
    assert "codegraph: index missing" not in out
    assert "! codegraph" not in out
    assert "codegraph: CLI not found" not in out  # folded into the single note


def test_doctor_codegraph_fresh_wal_db_ok(tmp_path, monkeypatch):
    from memo import cli_doctor, codegraph_loader

    db = tmp_path / ".codegraph" / "codegraph.db"
    _seed_codegraph_db(db)
    monkeypatch.chdir(tmp_path)  # _resolve_db() discovers the seeded project DB
    monkeypatch.setattr(codegraph_loader, "CODEGRAPH_DB", db)
    monkeypatch.setattr(cli_doctor, "_codegraph_cli_version", lambda *a, **k: (1, 5, 0))

    out = _doctor_plain_output(tmp_path)
    assert "codegraph: index ok (nodes=2 edges=1" in out
    assert "codegraph: CLI v1.5.0" in out


def test_doctor_codegraph_stale_mtime_warns(tmp_path, monkeypatch):
    from memo import cli_doctor, codegraph_loader

    db = tmp_path / ".codegraph" / "codegraph.db"
    _seed_codegraph_db(db)
    two_days_ago = time.time() - 48 * 3600
    os.utime(db, (two_days_ago, two_days_ago))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(codegraph_loader, "CODEGRAPH_DB", db)
    monkeypatch.setattr(cli_doctor, "_codegraph_cli_version", lambda *a, **k: None)

    out = _doctor_plain_output(tmp_path)
    assert "index older than 24h" in out
    assert "codegraph sync" in out


def test_doctor_codegraph_old_cli_version_warns(tmp_path, monkeypatch):
    from memo import codegraph_loader

    db = tmp_path / ".codegraph" / "codegraph.db"
    _seed_codegraph_db(db)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(codegraph_loader, "CODEGRAPH_DB", db)

    real_run = subprocess.run

    def _fake_run(cmd, *args, **kwargs):
        if cmd and cmd[0] == "codegraph":
            return subprocess.CompletedProcess(cmd, 0, stdout="1.4.2\n", stderr="")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", _fake_run)

    out = _doctor_plain_output(tmp_path)
    assert "codegraph: CLI v1.4.2 < v1.5.0" in out
    assert "@colbymchenry/codegraph@latest" in out
