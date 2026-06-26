"""Regression tests for cli_session.py correctness."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from memo.cli_session import session_group


def test_idle_maintenance_capture_mode_does_not_raise_name_error(tmp_path: Path, monkeypatch) -> None:
    """Regression: _hb('captured-notified', saved=n) used undefined `n`.

    When capture mode runs and saves titles, the heartbeat call crashed with
    NameError, swallowed by the bare except. This test runs the detached-worker
    path end-to-end with a stubbed capture result and asserts no exception is
    raised (exit code 0) and the heartbeat log contains the 'captured-notified'
    entry.
    """
    state = tmp_path / "state"
    state.mkdir()
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("{}\n", encoding="utf-8")

    monkeypatch.setenv("MEMO_STATE_DIR", str(state))
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMO_NONINTERACTIVE", "1")
    monkeypatch.setenv("MEMO_SESSION_DEBUG", "1")

    with patch("memo.capture.run_capture_incremental") as mock_cap:
        mock_cap.return_value = {
            "status": "ok",
            "saved": ["mem-id-1"],
            "saved_titles": ["Test insight"],
        }
        with patch("memo.cli_capture._write_capture_notification"):
            with patch("memo.session.get_session") as mock_get_session:
                # Mock session data to survive the idle-maintenance checks
                mock_get_session.return_value = {
                    "session_id": "test-sid-001",
                    "updated": "2026-06-18T10:00:00+00:00",
                    "last_user_msg": "test message",
                }
                with patch("memo.session.read_last_user_msg") as mock_read_msg:
                    # Both calls to read_last_user_msg return the same message
                    # so the idle window is not interrupted by a new prompt
                    mock_read_msg.return_value = "test message"
                    runner = CliRunner()
                    result = runner.invoke(
                        session_group,
                        [
                            "idle-maintenance",
                            "--mode", "capture",
                            "--delay-secs", "0",
                            "--detached-worker",
                        ],
                        input=json.dumps({"session_id": "test-sid-001", "transcript_path": str(transcript)}),
                        catch_exceptions=False,
                    )

    # Must not crash with NameError
    assert result.exit_code == 0, f"exit={result.exit_code} output={result.output}"
    # Heartbeat log must exist and contain 'captured-notified'
    log = state / "idle_capture.log"
    assert log.exists(), "heartbeat log was not written"
    entries = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    stages = [e["stage"] for e in entries]
    assert "captured-notified" in stages, f"stages={stages}"
