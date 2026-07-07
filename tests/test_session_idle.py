from __future__ import annotations

import json
import subprocess
from pathlib import Path

from click.testing import CliRunner

from memo.cli import cli


def _payload(session_id: str, transcript: Path) -> str:
    return json.dumps(
        {
            "session_id": session_id,
            "transcript_path": str(transcript),
            "cwd": str(transcript.parent),
        }
    )


def test_idle_maintenance_capture_runs_when_session_is_still_current(
    tmp_cfg, monkeypatch, tmp_path
):
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
        [
            "session",
            "idle-maintenance",
            "--mode",
            "capture",
            "--delay-secs",
            "0",
            "--detached-worker",
        ],
        input=_payload("sid-1", transcript),
        env={"MEMO_NONINTERACTIVE": "1", "MEMO_STATE_DIR": str(tmp_cfg.state_dir)},
    )

    assert result.exit_code == 0, result.output
    assert calls == [(transcript, "sid-1", False)]


def test_idle_maintenance_capture_skips_when_new_prompt_arrives(tmp_cfg, monkeypatch, tmp_path):
    """A NEW user prompt during the window → self-cancel (a fresh worker handles
    the next quiet period). Keyed on the transcript's last user message."""
    from memo import cli_session as cli_session_mod

    monkeypatch.setattr(cli_session_mod.Config, "from_env", lambda: tmp_cfg)
    monkeypatch.setenv("MEMO_SESSION_DISABLE", "0")
    monkeypatch.setenv("MEMO_CAPTURE_DISABLE", "0")

    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("")
    calls = []
    # Successive reads see a new user prompt → the user kept going.
    prompts = ["prompt A", "prompt B"]

    monkeypatch.setattr("memo.session.get_session", lambda sd, sid: {"session_id": sid})
    monkeypatch.setattr("memo.session.read_last_user_msg", lambda p: prompts.pop(0))

    def _fake_capture_incremental(path, sid, debug=False):
        calls.append((Path(path), sid, debug))
        return {"status": "ok"}

    monkeypatch.setattr("memo.capture.run_capture_incremental", _fake_capture_incremental)

    result = CliRunner().invoke(
        cli,
        [
            "session",
            "idle-maintenance",
            "--mode",
            "capture",
            "--delay-secs",
            "0",
            "--detached-worker",
        ],
        input=_payload("sid-2", transcript),
        env={"MEMO_NONINTERACTIVE": "1", "MEMO_STATE_DIR": str(tmp_cfg.state_dir)},
    )

    assert result.exit_code == 0, result.output
    assert calls == []


def test_idle_maintenance_capture_runs_despite_updated_bump_same_prompt(
    tmp_cfg, monkeypatch, tmp_path
):
    """Regression: the same turn's Stop checkpoint bumps `updated` without a new
    prompt. The worker must NOT self-cancel on that — else the inactivity capture
    never fires. Keyed on the user prompt (stable), not `updated`."""
    from memo import cli_session as cli_session_mod

    monkeypatch.setattr(cli_session_mod.Config, "from_env", lambda: tmp_cfg)
    monkeypatch.setenv("MEMO_SESSION_DISABLE", "0")
    monkeypatch.setenv("MEMO_CAPTURE_DISABLE", "0")

    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("")
    calls = []
    # `updated` changed between reads (Stop checkpoint), but the prompt did not.
    snapshots = [
        {"session_id": "sid-x", "updated": "2026-06-18T10:00:00+00:00"},
        {"session_id": "sid-x", "updated": "2026-06-18T10:00:05+00:00"},
    ]
    monkeypatch.setattr("memo.session.get_session", lambda sd, sid: snapshots.pop(0))
    monkeypatch.setattr("memo.session.read_last_user_msg", lambda p: "decidimos usar X")

    def _fake_capture_incremental(path, sid, debug=False):
        calls.append((Path(path), sid, debug))
        return {"status": "ok"}

    monkeypatch.setattr("memo.capture.run_capture_incremental", _fake_capture_incremental)

    result = CliRunner().invoke(
        cli,
        [
            "session",
            "idle-maintenance",
            "--mode",
            "capture",
            "--delay-secs",
            "0",
            "--detached-worker",
        ],
        input=_payload("sid-x", transcript),
        env={"MEMO_NONINTERACTIVE": "1", "MEMO_STATE_DIR": str(tmp_cfg.state_dir)},
    )

    assert result.exit_code == 0, result.output
    assert calls == [(transcript, "sid-x", False)]


def test_idle_maintenance_reflect_runs_when_session_is_still_current(
    tmp_cfg, monkeypatch, tmp_path
):
    from memo import cli_session as cli_session_mod

    monkeypatch.setattr(cli_session_mod.Config, "from_env", lambda: tmp_cfg)
    monkeypatch.setenv("MEMO_SESSION_DISABLE", "0")
    monkeypatch.setenv("MEMO_CAPTURE_DISABLE", "0")

    calls = []
    events: list[str] = []

    def _fake_get_session(state_dir, session_id):
        return {
            "session_id": session_id,
            "updated": "2026-06-18T10:00:00+00:00",
        }

    class _FakeMemory:
        def __init__(self, cfg):
            self.cfg = cfg

        def close(self):
            events.append("closed")

    def _fake_reflect(session_id, mem, cfg, debug=False):
        calls.append((session_id, isinstance(mem, _FakeMemory), debug))
        return {"status": "ok"}

    monkeypatch.setattr("memo.session.get_session", _fake_get_session)
    monkeypatch.setattr("memo.memory.Memory", _FakeMemory)
    monkeypatch.setattr("memo.cli_transcripts._reflect_session", _fake_reflect)

    result = CliRunner().invoke(
        cli,
        [
            "session",
            "idle-maintenance",
            "--mode",
            "reflect",
            "--delay-secs",
            "0",
            "--detached-worker",
        ],
        input=json.dumps({"session_id": "sid-3", "cwd": str(tmp_path)}),
        env={"MEMO_NONINTERACTIVE": "1", "MEMO_STATE_DIR": str(tmp_cfg.state_dir)},
    )

    assert result.exit_code == 0, result.output
    assert calls == [("sid-3", True, False)]
    assert events == ["closed"]


def test_idle_maintenance_detaches_worker_and_returns(tmp_cfg, monkeypatch, tmp_path):
    """Without --detached-worker, the hook re-spawns a detached worker
    (start_new_session) and returns WITHOUT running capture inline — so it
    survives Claude Code reaping the inline async hook at turn end."""
    import subprocess

    from memo import cli_session as cli_session_mod

    monkeypatch.setattr(cli_session_mod.Config, "from_env", lambda: tmp_cfg)
    monkeypatch.setenv("MEMO_SESSION_DISABLE", "0")
    monkeypatch.setenv("MEMO_CAPTURE_DISABLE", "0")

    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("")
    spawned: dict = {}
    captured: list = []

    class _FakeProc:
        def __init__(self, *a, **k):
            spawned["args"] = a[0] if a else k.get("args")
            spawned["new_session"] = k.get("start_new_session")

            class _Stdin:
                def write(self, _b):
                    pass

                def close(self):
                    pass

            self.stdin = _Stdin()

    monkeypatch.setattr(subprocess, "Popen", _FakeProc)
    monkeypatch.setattr(
        "memo.capture.run_capture_incremental",
        lambda *a, **k: captured.append(a) or {"status": "ok"},
    )

    result = CliRunner().invoke(
        cli,
        ["session", "idle-maintenance", "--mode", "capture", "--delay-secs", "0"],
        input=_payload("sid-d", transcript),
        env={"MEMO_NONINTERACTIVE": "1", "MEMO_STATE_DIR": str(tmp_cfg.state_dir)},
    )

    assert result.exit_code == 0, result.output
    assert spawned["new_session"] is True
    assert "--detached-worker" in spawned["args"]
    assert "idle-maintenance" in spawned["args"]
    assert captured == []  # capture runs in the detached child, not inline


def test_idle_maintenance_env_zero_delay_is_preserved(tmp_cfg, monkeypatch, tmp_path):
    """MEMO_SESSION_IDLE_CAPTURE_SECS=0 must be forwarded as 0, not 10."""
    from memo import cli_session as cli_session_mod

    monkeypatch.setattr(cli_session_mod.Config, "from_env", lambda: tmp_cfg)
    monkeypatch.setenv("MEMO_SESSION_DISABLE", "0")
    monkeypatch.setenv("MEMO_CAPTURE_DISABLE", "0")

    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("")
    spawned: dict = {}

    class _FakeProc:
        def __init__(self, *a, **k):
            spawned["args"] = a[0] if a else k.get("args")

            class _Stdin:
                def write(self, _b):
                    pass

                def close(self):
                    pass

            self.stdin = _Stdin()

    monkeypatch.setattr(subprocess, "Popen", _FakeProc)
    monkeypatch.setattr(
        "memo.capture.run_capture_incremental",
        lambda *a, **k: {"status": "ok"},
    )

    result = CliRunner().invoke(
        cli,
        ["session", "idle-maintenance", "--mode", "capture"],
        input=_payload("sid-e", transcript),
        env={
            "MEMO_NONINTERACTIVE": "1",
            "MEMO_STATE_DIR": str(tmp_cfg.state_dir),
            "MEMO_SESSION_IDLE_CAPTURE_SECS": "0",
        },
    )

    assert result.exit_code == 0, result.output
    assert "--delay-secs" in spawned["args"]
    assert spawned["args"][spawned["args"].index("--delay-secs") + 1] == "0"


def test_idle_maintenance_env_zero_reflect_delay_is_preserved(tmp_cfg, monkeypatch, tmp_path):
    """MEMO_SESSION_IDLE_REFLECT_SECS=0 must be forwarded as 0, not 300."""
    from memo import cli_session as cli_session_mod

    monkeypatch.setattr(cli_session_mod.Config, "from_env", lambda: tmp_cfg)
    monkeypatch.setenv("MEMO_SESSION_DISABLE", "0")
    monkeypatch.setenv("MEMO_CAPTURE_DISABLE", "0")

    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("")
    spawned: dict = {}

    class _FakeProc:
        def __init__(self, *a, **k):
            spawned["args"] = a[0] if a else k.get("args")

            class _Stdin:
                def write(self, _b):
                    pass

                def close(self):
                    pass

            self.stdin = _Stdin()

    monkeypatch.setattr(subprocess, "Popen", _FakeProc)
    monkeypatch.setattr(
        "memo.cli_transcripts._reflect_session",
        lambda *a, **k: {"status": "ok"},
    )

    result = CliRunner().invoke(
        cli,
        ["session", "idle-maintenance", "--mode", "reflect"],
        input=_payload("sid-f", transcript),
        env={
            "MEMO_NONINTERACTIVE": "1",
            "MEMO_STATE_DIR": str(tmp_cfg.state_dir),
            "MEMO_SESSION_IDLE_REFLECT_SECS": "0",
        },
    )

    assert result.exit_code == 0, result.output
    assert "--delay-secs" in spawned["args"]
    assert spawned["args"][spawned["args"].index("--delay-secs") + 1] == "0"
