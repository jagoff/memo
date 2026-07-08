from __future__ import annotations

import json

from click.testing import CliRunner

from memo.cli import cli


def test_reflect_quiet_closes_memory(tmp_cfg, monkeypatch) -> None:
    from memo import cli_transcripts as cli_transcripts_mod

    monkeypatch.setattr(cli_transcripts_mod.Config, "from_env", lambda: tmp_cfg)
    monkeypatch.setenv("MEMO_CAPTURE_DISABLE", "0")

    events: list[str] = []

    class _FakeMemory:
        def __init__(self, cfg):
            self.cfg = cfg

        def close(self) -> None:
            events.append("closed")

    def _fake_reflect_session(session_id, mem, cfg, dry_run=False, debug=False):
        assert isinstance(mem, _FakeMemory)
        return {"status": "ok", "session_id": session_id, "saved": [], "dry_run": dry_run}

    monkeypatch.setattr(
        "memo.session.list_sessions",
        lambda state_dir, limit=2: [{"session_id": "sid-reflect-close"}],
    )
    monkeypatch.setattr("memo.memory.Memory", _FakeMemory)
    monkeypatch.setattr("memo.cli_transcripts._reflect_session", _fake_reflect_session)

    result = CliRunner().invoke(
        cli,
        ["reflect", "--last", "--quiet"],
        env={"MEMO_NONINTERACTIVE": "1", "MEMO_STATE_DIR": str(tmp_cfg.state_dir)},
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "ok"
    assert events == ["closed"]
