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


# ── _reflect_session stamping semantics ──────────────────────────────────


def _patch_reflect_deps(monkeypatch, llm_content: str) -> list[str]:
    """Stub session/transcript/LLM plumbing for _reflect_session.

    Returns the list that records mark_reflected calls (session ids).
    """
    import memo.memory.record as record_mod
    import memo.session as session_mod
    from memo import cli_transcripts as ct

    monkeypatch.setattr(
        session_mod, "get_session", lambda state_dir, sid: {"transcript_path": "/tmp/fake.jsonl"}
    )
    stamped: list[str] = []
    monkeypatch.setattr(session_mod, "mark_reflected", lambda state_dir, sid: stamped.append(sid))
    monkeypatch.setattr(
        ct,
        "_read_full_transcript",
        lambda path: [
            ("user", "q1"),
            ("assistant", "a1"),
            ("user", "q2"),
            ("assistant", "a2"),
            ("user", "q3"),
        ],
    )
    monkeypatch.setattr(
        record_mod, "chat_with_timeout", lambda *a, **k: {"message": {"content": llm_content}}
    )
    return stamped


class _ChatlessMemory:
    def _ensure_chat(self) -> None:
        return None


def test_reflect_parse_failure_does_not_stamp(tmp_cfg, monkeypatch) -> None:
    """LLM prose (unparseable JSON) → parse_error, session NOT stamped so it retries."""
    from memo.cli_transcripts import _reflect_session

    stamped = _patch_reflect_deps(monkeypatch, "Sure! Here are my thoughts about the session.")

    res = _reflect_session("sid-parse-fail", _ChatlessMemory(), tmp_cfg)

    assert res["status"] == "parse_error"
    assert stamped == []


def test_reflect_non_dict_json_does_not_stamp(tmp_cfg, monkeypatch) -> None:
    """Valid JSON that isn't an object → parse_error, session NOT stamped."""
    from memo.cli_transcripts import _reflect_session

    stamped = _patch_reflect_deps(monkeypatch, '["not", "an", "object"]')

    res = _reflect_session("sid-non-dict", _ChatlessMemory(), tmp_cfg)

    assert res["status"] == "parse_error"
    assert stamped == []


def test_reflect_parsed_empty_still_stamps(tmp_cfg, monkeypatch) -> None:
    """Parse OK but legitimately nothing to save → ok + stamped (no reprocess loop)."""
    from memo.cli_transcripts import _reflect_session

    stamped = _patch_reflect_deps(
        monkeypatch,
        '{"session_title": "", "summary": "", "decisions": [], '
        '"facts": [], "bugs": [], "followups": []}',
    )

    res = _reflect_session("sid-empty-ok", _ChatlessMemory(), tmp_cfg)

    assert res["status"] == "ok"
    assert res["saved"] == []
    assert stamped == ["sid-empty-ok"]
