"""Session snapshot checkpointing.

Covers the contract that the SessionStart picker depends on:

- New `session_id` → new file with `created == updated` and
  `turn_count == 1`.
- Repeat `session_id` → same file, `created` preserved,
  `turn_count` bumped, `updated` advanced.
- `list_sessions` returns most-recent-first.
- `prune_lru` deletes the oldest beyond `cap`.
- `get_session` resolves a unique prefix.
- `read_last_user_msg` parses the Claude Code transcript JSONL shape.
- `format_relative` renders sane labels for the picker.

Git introspection is monkey-patched to a fixed payload so the tests
don't depend on the working tree's actual state — and so the Linux
CI box (which doesn't have a git repo at the synthetic `cwd`) still
exercises the same code paths.
"""

from __future__ import annotations

import contextlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from memo import session as session_mod
from memo.cli_session import session_group
from memo.session import (
    checkpoint,
    find_transcript_path,
    format_relative,
    gather_git_state,
    get_session,
    is_command_noise,
    list_sessions,
    prune_lru,
    read_last_user_msg,
    recent_prompts,
    render_active_memory,
    update_summary,
)


@pytest.fixture
def fake_git(monkeypatch):
    """Replace git introspection with deterministic output."""

    def _fake(cwd):
        return {
            "branch": "master",
            "head_commit": "abc1234 fix(thing): something",
            "modified_files": ["src/memo/foo.py"],
        }

    monkeypatch.setattr(session_mod, "gather_git_state", _fake)


def test_find_transcript_path_recovers_by_session_id(tmp_path, monkeypatch):
    """Some hook events omit transcript_path (seen 2026-06-27 onward) — the
    recovery path globs ~/.claude/projects/*/<session_id>.jsonl."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    proj = tmp_path / ".claude" / "projects" / "-Users-fer-repos-memo"
    proj.mkdir(parents=True)
    transcript = proj / "abc-123.jsonl"
    transcript.write_text("{}\n", encoding="utf-8")

    found = find_transcript_path("abc-123")

    assert found == str(transcript)


def test_find_transcript_path_missing_session_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".claude" / "projects").mkdir(parents=True)

    assert find_transcript_path("no-such-session") is None


def test_find_transcript_path_empty_session_id_returns_none():
    assert find_transcript_path("") is None


def test_checkpoint_creates_new_session(tmp_cfg, fake_git):
    snap = checkpoint(
        tmp_cfg.state_dir,
        session_id="sid-aaaa-1111",
        cwd=str(tmp_cfg.state_dir),
    )
    assert snap["session_id"] == "sid-aaaa-1111"
    assert snap["turn_count"] == 1
    assert snap["created"] == snap["updated"]
    assert snap["branch"] == "master"

    p = tmp_cfg.state_dir / "sessions" / "sid-aaaa-1111.json"
    assert p.is_file()


def test_recent_prompts_returns_last_n_oldest_first(tmp_cfg, fake_git):
    """recent_prompts feeds the recall-hook short-prompt expansion: it returns
    the most recent N prompt_trail entries, oldest first."""
    sid = "sid-trail-1"
    for p in ("primer prompt largo", "segundo prompt largo", "tercer prompt largo"):
        checkpoint(tmp_cfg.state_dir, session_id=sid, cwd=str(tmp_cfg.state_dir), prompt=p)
    assert recent_prompts(tmp_cfg.state_dir, sid, 2) == [
        "segundo prompt largo",
        "tercer prompt largo",
    ]
    # n larger than the trail returns the whole trail.
    assert len(recent_prompts(tmp_cfg.state_dir, sid, 99)) == 3


def test_recent_prompts_missing_session_or_zero_n(tmp_cfg):
    """No session, empty id, or n<=0 → [] (never raises — recall must not break)."""
    assert recent_prompts(tmp_cfg.state_dir, "does-not-exist", 2) == []
    assert recent_prompts(tmp_cfg.state_dir, "", 2) == []
    assert recent_prompts(tmp_cfg.state_dir, "x", 0) == []


def test_checkpoint_idempotent_upsert(tmp_cfg, fake_git):
    """Repeat checkpoint for same sid bumps turn_count, preserves
    created, advances updated."""
    s1 = checkpoint(
        tmp_cfg.state_dir,
        session_id="sid-bbbb-2222",
        cwd=str(tmp_cfg.state_dir),
    )
    # Force a second-resolution change so updated definitely advances.
    import time as _t

    _t.sleep(1.05)
    s2 = checkpoint(
        tmp_cfg.state_dir,
        session_id="sid-bbbb-2222",
        cwd=str(tmp_cfg.state_dir),
    )
    assert s1["created"] == s2["created"]
    assert s2["updated"] >= s1["updated"]
    assert s2["turn_count"] == 2

    # Only one file on disk.
    files = list((tmp_cfg.state_dir / "sessions").glob("*.json"))
    assert len(files) == 1


def test_list_sessions_sorted_recency(tmp_cfg, monkeypatch):
    """`list_sessions` returns most-recent-first regardless of file
    creation order."""
    sessions_dir = tmp_cfg.state_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    older = {
        "session_id": "old",
        "project": "p",
        "created": "2026-01-01T00:00:00+00:00",
        "updated": "2026-01-01T00:00:00+00:00",
        "turn_count": 1,
    }
    newer = {
        "session_id": "new",
        "project": "p",
        "created": "2026-05-08T00:00:00+00:00",
        "updated": "2026-05-08T00:00:00+00:00",
        "turn_count": 1,
    }
    (sessions_dir / "old.json").write_text(json.dumps(older))
    (sessions_dir / "new.json").write_text(json.dumps(newer))

    rows = list_sessions(tmp_cfg.state_dir, limit=10)
    assert [r["session_id"] for r in rows] == ["new", "old"]


def test_list_sessions_project_filter(tmp_cfg):
    sessions_dir = tmp_cfg.state_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (sessions_dir / "a.json").write_text(
        json.dumps(
            {
                "session_id": "a",
                "project": "memo",
                "updated": "2026-05-01T00:00:00+00:00",
                "turn_count": 1,
            }
        )
    )
    (sessions_dir / "b.json").write_text(
        json.dumps(
            {
                "session_id": "b",
                "project": "other",
                "updated": "2026-05-02T00:00:00+00:00",
                "turn_count": 1,
            }
        )
    )
    rows = list_sessions(tmp_cfg.state_dir, project="memo")
    assert len(rows) == 1
    assert rows[0]["session_id"] == "a"


def test_prune_lru_keeps_newest(tmp_cfg):
    """With cap=2, only the 2 sessions with the latest `updated` survive."""
    sessions_dir = tmp_cfg.state_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    for i, ts in enumerate(
        [
            "2026-01-01T00:00:00+00:00",
            "2026-02-01T00:00:00+00:00",
            "2026-03-01T00:00:00+00:00",
            "2026-04-01T00:00:00+00:00",
        ]
    ):
        (sessions_dir / f"s{i}.json").write_text(
            json.dumps(
                {
                    "session_id": f"s{i}",
                    "updated": ts,
                    "turn_count": 1,
                }
            )
        )
    n = prune_lru(tmp_cfg.state_dir, cap=2)
    assert n == 2
    surviving = sorted(p.name for p in sessions_dir.glob("*.json"))
    assert surviving == ["s2.json", "s3.json"]


def test_get_session_by_full_id(tmp_cfg, fake_git):
    snap = checkpoint(
        tmp_cfg.state_dir,
        session_id="full-id-12345",
        cwd=str(tmp_cfg.state_dir),
    )
    got = get_session(tmp_cfg.state_dir, "full-id-12345")
    assert got is not None
    assert got["session_id"] == snap["session_id"]


def test_get_session_by_prefix(tmp_cfg, fake_git):
    checkpoint(
        tmp_cfg.state_dir,
        session_id="prefix-test-9999",
        cwd=str(tmp_cfg.state_dir),
    )
    got = get_session(tmp_cfg.state_dir, "prefix-")
    assert got is not None
    assert got["session_id"] == "prefix-test-9999"


def test_get_session_short_prefix_rejected(tmp_cfg):
    """Prefixes <4 chars don't match — same convention as
    `Memory.resolve_id`."""
    assert get_session(tmp_cfg.state_dir, "abc") is None
    assert get_session(tmp_cfg.state_dir, "") is None


def test_read_last_user_msg(tmp_path):
    """Parse a synthetic Claude Code transcript and pull the last user msg."""
    t = tmp_path / "transcript.jsonl"
    t.write_text(
        "\n".join(
            [
                json.dumps({"type": "user", "message": {"content": "first prompt"}}),
                json.dumps(
                    {"type": "assistant", "message": {"content": [{"type": "text", "text": "ack"}]}}
                ),
                json.dumps(
                    {
                        "type": "user",
                        "message": {
                            "content": [{"type": "text", "text": "second prompt with detail"}]
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [
                                {"type": "tool_use", "name": "Bash"},
                                {"type": "text", "text": "doing it"},
                            ]
                        },
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    msg = read_last_user_msg(t)
    assert msg == "second prompt with detail"


def test_read_last_user_msg_missing_file(tmp_path):
    assert read_last_user_msg(tmp_path / "nope.jsonl") is None


def test_read_last_user_msg_skips_slash_command(tmp_path):
    """The latest user turn is slash-command plumbing → fall back to the
    prior real prompt, never the wrapper text."""
    t = tmp_path / "transcript.jsonl"
    t.write_text(
        "\n".join(
            [
                json.dumps({"type": "user", "message": {"content": "real prompt here"}}),
                json.dumps(
                    {"type": "assistant", "message": {"content": [{"type": "text", "text": "ok"}]}}
                ),
                json.dumps(
                    {
                        "type": "user",
                        "message": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": "<command-name>/clear</command-name>\n<command-message>clear</command-message>",
                                }
                            ]
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "user",
                        "message": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": "<local-command-stdout>Enabled plan mode</local-command-stdout>",
                                }
                            ]
                        },
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    assert read_last_user_msg(t) == "real prompt here"


def test_read_last_user_msg_strips_inline_wrapper(tmp_path):
    """A user turn mixing a wrapper with real text yields just the real text."""
    t = tmp_path / "transcript.jsonl"
    t.write_text(
        json.dumps(
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "text",
                            "text": "<command-args>--all</command-args>actual question",
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    assert read_last_user_msg(t) == "actual question"


def test_read_last_user_msg_skips_meta(tmp_path):
    """isMeta / isCompactSummary records are not real prompts."""
    t = tmp_path / "transcript.jsonl"
    t.write_text(
        "\n".join(
            [
                json.dumps({"type": "user", "message": {"content": "kept prompt"}}),
                json.dumps(
                    {"type": "user", "isMeta": True, "message": {"content": "<system reminder>"}}
                ),
                json.dumps(
                    {
                        "type": "user",
                        "isCompactSummary": True,
                        "message": {"content": "summary blob"},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    assert read_last_user_msg(t) == "kept prompt"


def test_is_command_noise_detects_truncated_wrapper():
    """Catches values truncated mid-tag (old stored summaries) that the tag-pair
    stripper alone cannot repair, while passing real prompts through."""
    assert is_command_noise(None)
    assert is_command_noise("")
    assert is_command_noise("<local-command-stdout>Enabled plan mode</local-command-")
    assert is_command_noise("<command-name>/clear</command-name>")
    assert is_command_noise("<task-notification>\n<task-id>blqnkhf83</task-id>")
    assert is_command_noise("<system-reminder>do the thing</system-reminder>")
    assert not is_command_noise("mergea a master")


def test_checkpoint_heals_persisted_noise_summary(tmp_cfg, fake_git, tmp_path):
    """A previously stored command-noise summary is recomputed, not preserved."""
    t = tmp_path / "transcript.jsonl"
    t.write_text(
        json.dumps({"type": "user", "message": {"content": "clean recovered prompt"}}),
        encoding="utf-8",
    )
    # First checkpoint plants junk directly in the snapshot, mimicking pre-fix data.
    snap = checkpoint(tmp_cfg.state_dir, session_id="heal-me", cwd=str(tmp_path))
    path = session_mod._session_path(tmp_cfg.state_dir, "heal-me")
    data = json.loads(path.read_text())
    data["summary"] = "<local-command-stdout>Enabled plan mode</local-command-"
    path.write_text(json.dumps(data))
    # Next checkpoint with a real transcript must overwrite the junk.
    snap = checkpoint(
        tmp_cfg.state_dir, session_id="heal-me", cwd=str(tmp_path), transcript_path=str(t)
    )
    assert snap["summary"] == "clean recovered prompt"


def test_render_active_memory_includes_session_context():
    md = "\n".join(
        render_active_memory(
            {
                "project": "memo",
                "branch": "master",
                "turn_count": 4,
                "running_summary": "Se dejó listo el nuevo bloque de memoria activa.",
                "modified_files": ["src/memo/session.py", "src/memo/cli_session.py"],
                "last_assistant_tail": "Quedó aplicado y verificado.",
                "prompt_trail": ["primer loop", "segundo loop", "tercer loop"],
            }
        )
    )
    assert "Active memory" in md
    assert "Se dejó listo" in md
    assert "memo" in md and "master" in md
    assert "session.py" in md and "cli_session.py" in md
    assert "Last reply" in md and "verificado" in md
    assert "Open loops (session)" in md and "tercer loop" in md


def test_format_relative_buckets():
    now = datetime(2026, 5, 8, 12, 0, 0, tzinfo=UTC)
    assert format_relative((now - timedelta(seconds=30)).isoformat(), now=now) == "<1m ago"
    assert format_relative((now - timedelta(minutes=5)).isoformat(), now=now) == "5m ago"
    assert format_relative((now - timedelta(hours=3)).isoformat(), now=now) == "3h ago"
    assert format_relative((now - timedelta(days=2)).isoformat(), now=now) == "2d ago"
    assert format_relative(None) == "—"
    assert format_relative("garbage") == "—"


def test_update_summary_patches_existing(tmp_cfg, fake_git):
    checkpoint(
        tmp_cfg.state_dir,
        session_id="patch-me-1234",
        cwd=str(tmp_cfg.state_dir),
    )
    assert update_summary(tmp_cfg.state_dir, "patch-me-1234", "fresh LLM-derived label")
    got = get_session(tmp_cfg.state_dir, "patch-me-1234")
    assert got["summary"] == "fresh LLM-derived label"


def test_update_summary_unknown_session(tmp_cfg):
    assert not update_summary(tmp_cfg.state_dir, "does-not-exist-1234", "x")


def test_checkpoint_requires_session_id(tmp_cfg):
    with pytest.raises(ValueError):
        checkpoint(tmp_cfg.state_dir, session_id="", cwd=str(tmp_cfg.state_dir))


def test_gather_git_state_outside_repo(tmp_path):
    """Outside a git repo, all fields are nullable but the function
    must not raise."""
    state = gather_git_state(tmp_path)
    assert state["branch"] is None
    assert state["head_commit"] is None
    assert state["modified_files"] == []


def test_list_sessions_cwd_filter(tmp_cfg):
    """`cwd` filter narrows to sessions for one absolute path; `project`
    is a coarser match by basename."""
    sessions_dir = tmp_cfg.state_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (sessions_dir / "a.json").write_text(
        json.dumps(
            {
                "session_id": "a",
                "cwd": "/tmp/proj-a",
                "project": "proj-a",
                "updated": "2026-05-08T00:00:00+00:00",
                "turn_count": 1,
            }
        )
    )
    (sessions_dir / "a2.json").write_text(
        json.dumps(
            {
                "session_id": "a2",
                "cwd": "/tmp/proj-a",
                "project": "proj-a",
                "updated": "2026-05-08T01:00:00+00:00",
                "turn_count": 1,
            }
        )
    )
    (sessions_dir / "b.json").write_text(
        json.dumps(
            {
                "session_id": "b",
                "cwd": "/tmp/proj-b",
                "project": "proj-b",
                "updated": "2026-05-08T02:00:00+00:00",
                "turn_count": 1,
            }
        )
    )
    rows = list_sessions(tmp_cfg.state_dir, cwd="/tmp/proj-a")
    assert [r["session_id"] for r in rows] == ["a2", "a"]


def test_list_sessions_cwd_no_match(tmp_cfg):
    sessions_dir = tmp_cfg.state_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (sessions_dir / "a.json").write_text(
        json.dumps(
            {
                "session_id": "a",
                "cwd": "/tmp/proj-a",
                "updated": "2026-05-08T00:00:00+00:00",
                "turn_count": 1,
            }
        )
    )
    rows = list_sessions(tmp_cfg.state_dir, cwd="/tmp/nope")
    assert rows == []


def test_list_sessions_cwd_resolves_symlinks(tmp_path, tmp_cfg):
    """Both the stored cwd and the filter cwd are run through
    `Path.resolve()` so macOS-isms (`/tmp` ↔ `/private/tmp`) and
    relative paths compare equal."""
    sessions_dir = tmp_cfg.state_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    target = tmp_path / "real-proj"
    target.mkdir()
    link = tmp_path / "link-proj"
    link.symlink_to(target)
    (sessions_dir / "a.json").write_text(
        json.dumps(
            {
                "session_id": "a",
                "cwd": str(target),
                "updated": "2026-05-08T00:00:00+00:00",
                "turn_count": 1,
            }
        )
    )
    rows = list_sessions(tmp_cfg.state_dir, cwd=str(link))
    assert len(rows) == 1
    assert rows[0]["session_id"] == "a"


def test_list_sessions_cwd_and_project_compose(tmp_cfg):
    """Both filters AND-applied: project narrows to the basename, cwd
    further narrows to the absolute path within that basename."""
    sessions_dir = tmp_cfg.state_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (sessions_dir / "a.json").write_text(
        json.dumps(
            {
                "session_id": "a",
                "cwd": "/work/memo",
                "project": "memo",
                "updated": "2026-05-08T00:00:00+00:00",
                "turn_count": 1,
            }
        )
    )
    (sessions_dir / "b.json").write_text(
        json.dumps(
            {
                "session_id": "b",
                "cwd": "/sandbox/memo",
                "project": "memo",
                "updated": "2026-05-08T01:00:00+00:00",
                "turn_count": 1,
            }
        )
    )
    rows = list_sessions(
        tmp_cfg.state_dir,
        project="memo",
        cwd="/sandbox/memo",
    )
    assert len(rows) == 1
    assert rows[0]["session_id"] == "b"


def test_gather_git_state_porcelain_first_line_intact(tmp_path, monkeypatch):
    """Regression: `git status --porcelain` starts each line with a
    2-char status code that may begin with a space (` M file`). An
    overly aggressive `.strip()` on the whole stdout eats the first
    line's leading space and shifts `line[3:]` by one, dropping the
    first character of the first reported file. Hit during smoke
    testing — the first modified file came back as `ooks/hooks.json`
    instead of `hooks/hooks.json`."""
    import subprocess as _sp

    from memo import session as _sess

    fake_status = " M hooks/hooks.json\n M src/memo/cli.py\n?? docs/\n"

    def fake_run(args, cwd=None, capture_output=None, text=None, timeout=None, check=None):
        kind = args[1] if len(args) > 1 else None

        # Mimic CompletedProcess shape with the fields _git reads.
        class _Result:
            returncode = 0
            stdout = ""

        if kind == "status":
            _Result.stdout = fake_status
        elif kind == "rev-parse":
            _Result.stdout = "master\n"
        elif kind == "log":
            _Result.stdout = "abc1234 something\n"
        return _Result()

    monkeypatch.setattr(_sp, "run", fake_run)
    state = _sess.gather_git_state(tmp_path)
    assert state["modified_files"] == [
        "hooks/hooks.json",
        "src/memo/cli.py",
        "docs/",
    ]


def test_idle_maintenance_capture_mode_does_not_raise_name_error(tmp_path, monkeypatch) -> None:
    """Regression: _hb('captured-notified', saved=n) used undefined `n`.

    When capture mode runs and saves titles with successful status, the heartbeat
    call should not raise NameError. The bug was that `n` was undefined.
    This test verifies successful capture doesn't crash (exit code 0).
    """
    from memo.cli_session import session_group
    from memo.session import checkpoint

    state = tmp_path / "state"
    state.mkdir()
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("{}\n", encoding="utf-8")

    monkeypatch.setenv("MEMO_STATE_DIR", str(state))
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMO_NONINTERACTIVE", "1")
    monkeypatch.setenv("MEMO_SESSION_DEBUG", "1")

    # Pre-create the session snapshot so idle-maintenance doesn't self-cancel
    sid = "test-sid-001"
    checkpoint(state, session_id=sid, cwd=str(tmp_path), transcript_path=str(transcript))

    with (
        patch("memo.capture.run_capture_incremental") as mock_cap,
        patch("memo.cli_capture._write_capture_notification"),
    ):
        mock_cap.return_value = {
            "status": "ok",
            "saved": ["mem-id-1"],
            "saved_titles": ["Test insight"],
        }
        runner = CliRunner()
        result = runner.invoke(
            session_group,
            [
                "idle-maintenance",
                "--mode",
                "capture",
                "--delay-secs",
                "0",
                "--detached-worker",
            ],
            input=json.dumps(
                {
                    "session_id": sid,
                    "transcript_path": str(transcript),
                }
            ),
            catch_exceptions=False,
        )

    # The exit code should be 0 — if there was a NameError, it would be caught
    # by the bare except and we'd still exit 0, but we verify it ran without crashing.
    assert result.exit_code == 0, f"exit={result.exit_code} output={result.output}"
    log = state / "idle_capture.log"
    # When successful capture completes with titles, _hb("captured-notified", saved=len(_titles))
    # is called. Check that the log exists and contains this stage.
    assert log.exists(), "heartbeat log was not written"
    entries = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    stages = [e["stage"] for e in entries]
    # If the capture ran to completion, "captured-notified" stage should be present
    assert "captured-notified" in stages, f"stages={stages}"


def test_idle_maintenance_reflect_mode_respects_maintain_disable(
    tmp_path: Path, monkeypatch
) -> None:
    """Regression: MEMO_MAINTAIN_DISABLE=1 must skip reflect mode to prevent OOM.

    Without this gate, every idle session spawns a full LLM load after 300s.
    On a 16GB Mac with multiple sessions this causes OOM kernel panics.
    """
    state = tmp_path / "state"
    state.mkdir()
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("{}\n", encoding="utf-8")

    monkeypatch.setenv("MEMO_STATE_DIR", str(state))
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMO_NONINTERACTIVE", "1")
    monkeypatch.setenv("MEMO_MAINTAIN_DISABLE", "1")

    # Pre-create the session snapshot so idle-maintenance doesn't self-cancel
    sid = "test-sid-reflect-002"
    checkpoint(state, session_id=sid, cwd=str(tmp_path), transcript_path=str(transcript))

    reflect_called = []

    with patch("memo.cli_transcripts._reflect_session") as mock_reflect:
        runner = CliRunner()
        result = runner.invoke(
            session_group,
            [
                "idle-maintenance",
                "--mode",
                "reflect",
                "--delay-secs",
                "0",
                "--detached-worker",
            ],
            input=json.dumps({"session_id": sid, "transcript_path": str(transcript)}),
        )
        reflect_called.append(mock_reflect.called)

    assert result.exit_code == 0
    assert not reflect_called[0], "reflect must not be called when MEMO_MAINTAIN_DISABLE=1"
    out = json.loads(result.output.strip()) if result.output.strip() else {}
    assert out.get("status") == "skipped_maintain_disabled", f"output={result.output!r}"


def test_reflect_flock_prevents_concurrent_reflect(tmp_path: Path, monkeypatch) -> None:
    """Regression: idle-maintenance reflect must skip when reflect.lock is held.

    Without flock in idle-maintenance, N sessions all pass the reflected_at check
    before any stamps it → N concurrent LLM loads (OOM risk on 16GB Macs).
    """
    import fcntl
    import threading

    state = tmp_path / "state"
    state.mkdir(parents=True)
    (tmp_path / "data").mkdir(parents=True)

    monkeypatch.setenv("MEMO_STATE_DIR", str(state))
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMO_NONINTERACTIVE", "1")

    # Create a real session so the worker reaches the flock check (not exits early on "not found").
    sid = "test-sid-flock-003"
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("{}\n", encoding="utf-8")
    checkpoint(state, session_id=sid, cwd=str(tmp_path), transcript_path=str(transcript))

    # Pre-hold the reflect.lock to simulate another session already running reflect.
    lock_path = state / "reflect.lock"
    lock_path.touch()
    # Held open intentionally (not a `with`): the lock must persist across the
    # concurrent-reflect simulation below; unlocked and closed at the test's end.
    lock_fd = open(lock_path, "w")  # noqa: SIM115
    fcntl.flock(lock_fd, fcntl.LOCK_EX)

    results: list[str | None] = []

    def try_reflect() -> None:
        runner = CliRunner()
        r = runner.invoke(
            session_group,
            ["idle-maintenance", "--mode", "reflect", "--delay-secs", "0", "--detached-worker"],
            input=json.dumps({"session_id": sid, "transcript_path": str(transcript)}),
        )
        out: dict[str, object] = {}
        with contextlib.suppress(Exception):
            out = json.loads(r.output.strip())
        results.append(out.get("status"))  # type: ignore[arg-type]

    t = threading.Thread(target=try_reflect)
    t.start()
    t.join(timeout=5)
    fcntl.flock(lock_fd, fcntl.LOCK_UN)
    lock_fd.close()

    assert not t.is_alive(), "idle-maintenance reflect hung instead of skipping under held lock"
    assert results, "idle-maintenance produced no output"
    assert results[0] == "skipped_concurrent", f"expected skipped_concurrent, got {results[0]!r}"


def test_autosave_persists_last_hook_payload_without_transcript(tmp_path, monkeypatch) -> None:
    """Regression: 2026-06-27 onward some hook events omit transcript_path,
    and autosave used to gate `last_hook_payload.json` persistence on having
    one — starving async hooks (checkpoint/idle-maintenance) of even the
    session_id fallback. Persistence must key on session_id alone."""
    from memo import session as session_mod
    from memo.cli_session import session_group

    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setenv("MEMO_STATE_DIR", str(state))
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMO_NONINTERACTIVE", "1")
    monkeypatch.setattr(session_mod, "find_transcript_path", lambda sid: None)

    sid = "test-sid-no-transcript"
    result = CliRunner().invoke(
        session_group,
        ["autosave"],
        input=json.dumps({"session_id": sid}),
    )

    assert result.exit_code == 0
    payload_file = state / "last_hook_payload.json"
    assert payload_file.exists(), "last_hook_payload.json must persist even without transcript_path"
    saved = json.loads(payload_file.read_text(encoding="utf-8"))
    assert saved["session_id"] == sid
    assert saved["transcript_path"] is None


def test_autosave_recovers_transcript_path_from_session_id(tmp_path, monkeypatch) -> None:
    """When the payload omits transcript_path, autosave should recover it via
    session_id before giving up — restoring the size-threshold check instead
    of silently no-oping every turn."""
    from memo import session as session_mod
    from memo.cli_session import session_group

    state = tmp_path / "state"
    state.mkdir()
    transcript = tmp_path / "recovered.jsonl"
    transcript.write_text("{}\n", encoding="utf-8")

    monkeypatch.setenv("MEMO_STATE_DIR", str(state))
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMO_NONINTERACTIVE", "1")
    monkeypatch.setattr(session_mod, "find_transcript_path", lambda sid: str(transcript))

    sid = "test-sid-recovered"
    result = CliRunner().invoke(
        session_group,
        ["autosave"],
        input=json.dumps({"session_id": sid}),
    )

    assert result.exit_code == 0
    saved = json.loads((state / "last_hook_payload.json").read_text(encoding="utf-8"))
    assert saved["transcript_path"] == str(transcript)


def test_checkpoint_cli_recovers_transcript_via_session_id(tmp_path, monkeypatch, fake_git) -> None:
    """Regression: session_checkpoint (Stop hook) must not persist a skeleton
    snapshot (session_id + nothing else) when transcript_path is missing from
    the payload but recoverable by session_id."""
    from memo.cli_session import session_group

    state = tmp_path / "state"
    state.mkdir()
    transcript = tmp_path / "recovered.jsonl"
    transcript.write_text(
        json.dumps({"type": "user", "message": {"role": "user", "content": "hola"}}) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("MEMO_STATE_DIR", str(state))
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMO_NONINTERACTIVE", "1")
    monkeypatch.setattr("memo.session.find_transcript_path", lambda sid: str(transcript))

    sid = "test-sid-checkpoint-recovered"
    result = CliRunner().invoke(
        session_group,
        ["checkpoint", "--cwd", str(tmp_path), "--json"],
        input=json.dumps({"session_id": sid}),
    )

    assert result.exit_code == 0
    snap = json.loads(result.output)
    assert snap["transcript_path"] == str(transcript)
    assert snap["last_user_msg"] == "hola"
