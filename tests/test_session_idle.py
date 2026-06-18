from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from memo.cli import cli


def _payload(session_id: str, transcript: Path) -> str:
    return json.dumps({"session_id": session_id, "transcript_path": str(transcript), "cwd": str(transcript.parent)})


def test_idle_maintenance_capture_runs_when_session_is_still_current(tmp_cfg, monkeypatch, tmp_path):
    from memo import cli_session as cli_session_mod

    monkeypatch.setattr(cli_session_mod.Config, "from_env", lambda: tmp_cfg)
    monkeypatch.setenv("MEMO_SESSION_DISABLE", "0")
    monkeypatch.setenv("MEMO_CAPTURE_DISABLE", "0")

    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("")
    calls = []

    def _fake_get_session(state_dir, session_id):
        return {
            "session_id": session_id,
            "updated": "2026-06-18T10:00:00+00:00",
        }

    def _fake_capture_incremental(path, sid, debug=False):
        calls.append((Path(path), sid, debug))
        return {"status": "ok"}

    monkeypatch.setattr("memo.session.get_session", _fake_get_session)
    monkeypatch.setattr("memo.capture.run_capture_incremental", _fake_capture_incremental)

    result = CliRunner().invoke(
        cli,
        ["session", "idle-maintenance", "--mode", "capture", "--delay-secs", "0"],
        input=_payload("sid-1", transcript),
        env={"MEMO_NONINTERACTIVE": "1", "MEMO_STATE_DIR": str(tmp_cfg.state_dir)},
    )

    assert result.exit_code == 0, result.output
    assert calls == [(transcript, "sid-1", False)]


def test_idle_maintenance_capture_skips_when_session_changed(tmp_cfg, monkeypatch, tmp_path):
    from memo import cli_session as cli_session_mod

    monkeypatch.setattr(cli_session_mod.Config, "from_env", lambda: tmp_cfg)
    monkeypatch.setenv("MEMO_SESSION_DISABLE", "0")
    monkeypatch.setenv("MEMO_CAPTURE_DISABLE", "0")

    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("")
    calls = []
    snapshots = [
        {"session_id": "sid-2", "updated": "2026-06-18T10:00:00+00:00"},
        {"session_id": "sid-2", "updated": "2026-06-18T10:00:05+00:00"},
    ]

    def _fake_get_session(state_dir, session_id):
        return snapshots.pop(0)

    def _fake_capture_incremental(path, sid, debug=False):
        calls.append((Path(path), sid, debug))
        return {"status": "ok"}

    monkeypatch.setattr("memo.session.get_session", _fake_get_session)
    monkeypatch.setattr("memo.capture.run_capture_incremental", _fake_capture_incremental)

    result = CliRunner().invoke(
        cli,
        ["session", "idle-maintenance", "--mode", "capture", "--delay-secs", "0"],
        input=_payload("sid-2", transcript),
        env={"MEMO_NONINTERACTIVE": "1", "MEMO_STATE_DIR": str(tmp_cfg.state_dir)},
    )

    assert result.exit_code == 0, result.output
    assert calls == []


def test_idle_maintenance_reflect_runs_when_session_is_still_current(tmp_cfg, monkeypatch, tmp_path):
    from memo import cli_session as cli_session_mod

    monkeypatch.setattr(cli_session_mod.Config, "from_env", lambda: tmp_cfg)
    monkeypatch.setenv("MEMO_SESSION_DISABLE", "0")
    monkeypatch.setenv("MEMO_CAPTURE_DISABLE", "0")

    calls = []

    def _fake_get_session(state_dir, session_id):
        return {
            "session_id": session_id,
            "updated": "2026-06-18T10:00:00+00:00",
        }

    class _FakeMemory:
        def __init__(self, cfg):
            self.cfg = cfg

    def _fake_reflect(session_id, mem, cfg, debug=False):
        calls.append((session_id, isinstance(mem, _FakeMemory), debug))
        return {"status": "ok"}

    monkeypatch.setattr("memo.session.get_session", _fake_get_session)
    monkeypatch.setattr("memo.memory.Memory", _FakeMemory)
    monkeypatch.setattr("memo.cli_transcripts._reflect_session", _fake_reflect)

    result = CliRunner().invoke(
        cli,
        ["session", "idle-maintenance", "--mode", "reflect", "--delay-secs", "0"],
        input=json.dumps({"session_id": "sid-3", "cwd": str(tmp_path)}),
        env={"MEMO_NONINTERACTIVE": "1", "MEMO_STATE_DIR": str(tmp_cfg.state_dir)},
    )

    assert result.exit_code == 0, result.output
    assert calls == [("sid-3", True, False)]
