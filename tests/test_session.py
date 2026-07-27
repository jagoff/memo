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
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

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
    refresh_summary,
    render_active_memory,
    stamp_recall_turn,
    stamp_recap_turn,
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


def test_session_paths_reject_traversal(tmp_cfg, tmp_path, fake_git):
    outside = tmp_path / "outside.json"
    outside.write_text('{"secret": true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="session_id"):
        checkpoint(
            tmp_cfg.state_dir,
            session_id="../../outside",
            cwd=str(tmp_cfg.state_dir),
        )

    assert json.loads(outside.read_text(encoding="utf-8")) == {"secret": True}
    assert get_session(tmp_cfg.state_dir, "../../outside") is None


def test_checkpoint_rejects_symlinked_sessions_directory(tmp_path, fake_git):
    state_dir = tmp_path / "state"
    outside = tmp_path / "outside"
    state_dir.mkdir()
    outside.mkdir()
    (state_dir / "sessions").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="unsafe sessions directory"):
        checkpoint(state_dir, session_id="safe-session", cwd=str(tmp_path))

    assert not (outside / "safe-session.json").exists()


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


def test_session_recent_env_zero_limit_disables_panel(tmp_cfg, fake_git, monkeypatch, tmp_path):
    """MEMO_SESSION_RECENT_LIMIT=0 must mean show zero rows, not fall back to 12."""
    checkpoint(
        tmp_cfg.state_dir,
        session_id="sid-recent-0",
        cwd=str(tmp_path),
        transcript_path=str(tmp_path / "t.jsonl"),
    )
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        session_group,
        ["recent"],
        env={
            "MEMO_STATE_DIR": str(tmp_cfg.state_dir),
            "MEMO_DATA_DIR": str(tmp_cfg.data_dir),
            "MEMO_SESSION_RECENT_LIMIT": "0",
            "MEMO_NONINTERACTIVE": "1",
        },
    )

    assert result.exit_code == 0, result.output
    assert result.output.strip() == "{}"


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


def test_concurrent_checkpoints_preserve_turn_count_and_prompts(tmp_cfg, monkeypatch):
    """Same-session checkpoint RMW operations serialize across threads."""
    barrier = threading.Barrier(2)

    def concurrent_git_state(cwd):
        barrier.wait(timeout=3)
        return {
            "branch": "master",
            "head_commit": "abc123 concurrent",
            "modified_files": [],
        }

    monkeypatch.setattr(session_mod, "gather_git_state", concurrent_git_state)

    def write(prompt):
        return checkpoint(
            tmp_cfg.state_dir,
            session_id="concurrent-session",
            cwd=str(tmp_cfg.state_dir),
            prompt=prompt,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(write, prompt) for prompt in ("first prompt", "second prompt")]
        for future in futures:
            future.result(timeout=3)

    snapshot = get_session(tmp_cfg.state_dir, "concurrent-session")
    assert snapshot is not None
    assert snapshot["turn_count"] == 2
    assert set(snapshot["prompt_trail"]) == {"first prompt", "second prompt"}


def test_checkpoint_merges_stamp_written_during_git_probe(tmp_cfg, fake_git, monkeypatch):
    """A checkpoint must load latest state after its expensive probes."""
    session_id = "checkpoint-stamp-merge"
    checkpoint(tmp_cfg.state_dir, session_id=session_id, cwd=str(tmp_cfg.state_dir))
    entered = threading.Event()
    release = threading.Event()

    def slow_git_state(cwd):
        entered.set()
        assert release.wait(timeout=3)
        return {"branch": "master", "head_commit": "abc123 merge", "modified_files": []}

    monkeypatch.setattr(session_mod, "gather_git_state", slow_git_state)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            checkpoint,
            tmp_cfg.state_dir,
            session_id=session_id,
            cwd=str(tmp_cfg.state_dir),
            prompt="checkpoint prompt",
        )
        assert entered.wait(timeout=3)
        stamp_recall_turn(tmp_cfg.state_dir, session_id, 7)
        release.set()
        future.result(timeout=3)

    snapshot = get_session(tmp_cfg.state_dir, session_id)
    assert snapshot is not None
    assert snapshot["turn_count"] == 2
    assert snapshot["last_recall_turn"] == 7
    assert snapshot["prompt_trail"] == ["checkpoint prompt"]


def test_refresh_summary_reacquires_and_merges_latest_snapshot(
    tmp_cfg, fake_git, tmp_path, monkeypatch
):
    """LLM summary writeback never clobbers checkpoint/stamp fields."""
    session_id = "summary-writeback-merge"
    transcript = tmp_path / f"{session_id}.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps({"type": "user", "message": {"content": "design the change"}}),
                json.dumps(
                    {"type": "assistant", "message": {"content": "implemented the decision"}}
                ),
            ]
        ),
        encoding="utf-8",
    )
    for _ in range(3):
        checkpoint(
            tmp_cfg.state_dir,
            session_id=session_id,
            cwd=str(tmp_cfg.state_dir),
            transcript_path=str(transcript),
        )

    entered = threading.Event()
    release = threading.Event()

    class SlowChat:
        def chat(self, *args, **kwargs):
            entered.set()
            assert release.wait(timeout=3)
            return {"message": {"content": "Durable generated summary."}}

    monkeypatch.setattr("memo.llm.MLXChat", SlowChat)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(refresh_summary, tmp_cfg.state_dir, session_id)
        assert entered.wait(timeout=3)
        checkpoint(
            tmp_cfg.state_dir,
            session_id=session_id,
            cwd=str(tmp_cfg.state_dir),
            prompt="new prompt while summarizing",
        )
        stamp_recap_turn(tmp_cfg.state_dir, session_id, 4)
        release.set()
        assert future.result(timeout=3) is True

    snapshot = get_session(tmp_cfg.state_dir, session_id)
    assert snapshot is not None
    assert snapshot["turn_count"] == 4
    assert snapshot["prompt_trail"] == ["new prompt while summarizing"]
    assert snapshot["last_recap_turn"] == 4
    assert snapshot["running_summary"] == "Durable generated summary."
    assert snapshot["summary_turn"] == 3


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


def test_list_sessions_sorts_offset_timestamps_by_instant(tmp_cfg):
    """Recency sorting must compare UTC instants, not ISO timestamp text."""
    sessions_dir = tmp_cfg.state_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    lexically_later_but_older = {
        "session_id": "old-offset",
        "project": "p",
        "updated": "2026-01-01T02:00:00+03:00",
        "turn_count": 1,
    }
    actually_newer = {
        "session_id": "new-utc",
        "project": "p",
        "updated": "2026-01-01T00:30:00+00:00",
        "turn_count": 1,
    }
    (sessions_dir / "old-offset.json").write_text(json.dumps(lexically_later_but_older))
    (sessions_dir / "new-utc.json").write_text(json.dumps(actually_newer))

    rows = list_sessions(tmp_cfg.state_dir, limit=10)

    assert [r["session_id"] for r in rows] == ["new-utc", "old-offset"]


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


def test_prune_lru_compares_offset_timestamps_by_instant(tmp_cfg):
    """LRU pruning must keep the newest UTC instant when offsets differ."""
    sessions_dir = tmp_cfg.state_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    rows = {
        "old-offset": "2026-01-01T02:00:00+03:00",
        "new-utc": "2026-01-01T00:30:00+00:00",
    }
    for session_id, updated in rows.items():
        (sessions_dir / f"{session_id}.json").write_text(
            json.dumps({"session_id": session_id, "updated": updated, "turn_count": 1})
        )

    n = prune_lru(tmp_cfg.state_dir, cap=1)

    assert n == 1
    assert sorted(p.name for p in sessions_dir.glob("*.json")) == ["new-utc.json"]


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


def test_session_low_level_contracts_are_exact(tmp_path, monkeypatch) -> None:
    assert session_mod._instant_sort_key(None) == (0, 0.0, "")
    assert session_mod._instant_sort_key(" invalid ") == (0, 0.0, "invalid")
    assert session_mod._instant_sort_key("2026-01-01T00:00:00Z") == (
        1,
        datetime(2026, 1, 1, tzinfo=UTC).timestamp(),
        "2026-01-01T00:00:00Z",
    )
    assert session_mod._instant_sort_key("2026-01-01T00:00:00z") == (
        0,
        0.0,
        "2026-01-01T00:00:00z",
    )
    assert session_mod._instant_sort_key("2025-12-31T21:00:00-03:00")[:2] == (
        1,
        datetime(2026, 1, 1, tzinfo=UTC).timestamp(),
    )

    with pytest.raises(
        ValueError,
        match=r"^session_id must be 1-128 ASCII letters, digits, underscores, or hyphens$",
    ):
        session_mod.validate_session_id("bad/id")

    state_dir = tmp_path / "nested" / "state"
    assert session_mod.sessions_dir(state_dir) == state_dir / "sessions"
    assert (state_dir / "sessions").is_dir()

    now = session_mod._now_iso()
    parsed = datetime.fromisoformat(now)
    assert parsed.tzinfo == UTC
    assert parsed.microsecond == 0


def test_sessions_dir_rechecks_symlink_after_creation(tmp_path, monkeypatch) -> None:
    state_dir = tmp_path / "state"
    target = state_dir / "sessions"
    original_is_symlink = Path.is_symlink
    target_checks = 0

    def race_is_symlink(path: Path) -> bool:
        nonlocal target_checks
        if path == target:
            target_checks += 1
            return target_checks == 2
        return original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", race_is_symlink)

    with pytest.raises(ValueError, match=r"^unsafe sessions directory: "):
        session_mod.sessions_dir(state_dir)
    assert target_checks == 2


def test_session_json_io_uses_utf8_and_stable_unicode_format(tmp_path, monkeypatch) -> None:
    encodings: list[str | None] = []
    writes: list[tuple[Path, str]] = []
    original_read_text = Path.read_text

    def recording_read_text(path: Path, *args, **kwargs) -> str:
        encodings.append(kwargs.get("encoding"))
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", recording_read_text)
    monkeypatch.setattr(
        session_mod,
        "atomic_write_text",
        lambda path, text: writes.append((path, text)),
    )
    payload = {"message": "café", "nested": {"ok": True}}

    written = session_mod._write(tmp_path, "sid-io", payload)

    assert written == tmp_path / "sessions" / "sid-io.json"
    assert writes == [
        (
            written,
            '{\n  "message": "café",\n  "nested": {\n    "ok": true\n  }\n}',
        )
    ]

    original_read_text(written, encoding="utf-8") if written.exists() else None
    written.write_text(json.dumps(payload), encoding="utf-8")
    assert session_mod._load(tmp_path, "sid-io") == payload
    assert encodings == ["utf-8"]


def test_checkpoint_persists_complete_merge_contract(tmp_path, monkeypatch) -> None:
    state_dir = tmp_path / "state"
    cwd = tmp_path / "project"
    cwd.mkdir()
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("{}\n", encoding="utf-8")
    session_id = "sid-complete"
    existing = {
        "session_id": session_id,
        "cwd": "/old",
        "project": "old",
        "branch": "old",
        "head_commit": "old",
        "modified_files": ["old.py"],
        "transcript_path": "/old/transcript",
        "last_user_msg": "old user",
        "last_assistant_tail": "old assistant",
        "prompt_trail": ["one", "two", "three", "four", "five"],
        "running_summary": "Running summary",
        "summary_turn": 3,
        "summary": "Human summary",
        "created": "2025-01-01T00:00:00+00:00",
        "updated": "2025-01-02T00:00:00+00:00",
        "turn_count": 4,
        "last_recall_turn": 7,
        "last_recap_turn": 6,
        "custom": "preserved",
    }
    session_mod._write(state_dir, session_id, existing)
    pruner = MagicMock()

    def gather(path: Path) -> dict:
        assert path == cwd.resolve()
        return {
            "branch": "master",
            "head_commit": "abc123 exact",
            "modified_files": ["a.py", "b.py"],
        }

    monkeypatch.setattr(session_mod, "gather_git_state", gather)
    monkeypatch.setattr(session_mod, "read_last_user_msg", lambda path: "new user")
    monkeypatch.setattr(session_mod, "read_last_assistant_tail", lambda path: "new assistant")
    monkeypatch.setattr(session_mod, "_now_iso", lambda: "2026-07-27T12:34:56+00:00")
    monkeypatch.setattr(session_mod, "prune_lru", pruner)

    snapshot = checkpoint(
        state_dir,
        session_id=session_id,
        cwd=str(cwd),
        transcript_path=str(transcript),
        prompt="  final prompt  ",
        lru_cap=17,
    )

    assert snapshot == {
        "session_id": session_id,
        "cwd": str(cwd.resolve()),
        "project": "project",
        "branch": "master",
        "head_commit": "abc123 exact",
        "modified_files": ["a.py", "b.py"],
        "transcript_path": str(transcript),
        "last_user_msg": "new user",
        "last_assistant_tail": "new assistant",
        "prompt_trail": ["two", "three", "four", "five", "final prompt"],
        "running_summary": "Running summary",
        "summary_turn": 3,
        "summary": "Human summary",
        "created": "2025-01-01T00:00:00+00:00",
        "updated": "2026-07-27T12:34:56+00:00",
        "turn_count": 5,
        "last_recall_turn": 7,
        "last_recap_turn": 6,
        "custom": "preserved",
    }
    pruner.assert_called_once_with(state_dir, cap=17)
    assert session_mod._load(state_dir, session_id) == snapshot


def test_checkpoint_empty_session_has_no_synthetic_summary(tmp_path, monkeypatch) -> None:
    cwd = tmp_path / "project"
    cwd.mkdir()
    monkeypatch.setattr(
        session_mod,
        "gather_git_state",
        lambda _cwd: {"branch": None, "head_commit": None, "modified_files": []},
    )
    monkeypatch.setattr(session_mod, "_now_iso", lambda: "2026-01-01T00:00:00+00:00")
    monkeypatch.setattr(session_mod, "prune_lru", MagicMock())

    with pytest.raises(ValueError, match=r"^session_id required$"):
        checkpoint(tmp_path, session_id="", cwd=str(cwd))

    snapshot = checkpoint(tmp_path, session_id="sid-empty", cwd=str(cwd), lru_cap=0)
    assert snapshot["summary"] is None
    assert snapshot["last_user_msg"] is None
    assert snapshot["last_assistant_tail"] is None
    session_mod.prune_lru.assert_called_once_with(tmp_path, cap=0)


def test_turn_stamps_preserve_exact_fields(tmp_path, monkeypatch) -> None:
    session_mod._write(tmp_path, "sid-stamp", {"preserved": True})
    monkeypatch.setattr(session_mod, "_now_iso", lambda: "2026-07-27T00:00:00+00:00")

    stamp_recall_turn(tmp_path, "sid-stamp", 4)
    assert session_mod._load(tmp_path, "sid-stamp") == {
        "preserved": True,
        "session_id": "sid-stamp",
        "last_recall_turn": 4,
        "updated": "2026-07-27T00:00:00+00:00",
    }

    session_mod.stamp_recap_turn(tmp_path, "sid-stamp", 5)
    assert session_mod._load(tmp_path, "sid-stamp") == {
        "preserved": True,
        "session_id": "sid-stamp",
        "last_recall_turn": 4,
        "last_recap_turn": 5,
        "updated": "2026-07-27T00:00:00+00:00",
    }


def test_recent_prompts_filters_types_and_obeys_zero_boundary(tmp_path) -> None:
    session_mod._write(
        tmp_path,
        "sid-prompts",
        {"prompt_trail": [" one ", 7, "", "two", None, " three "]},
    )

    assert recent_prompts(tmp_path, "sid-prompts", 0) == []
    assert recent_prompts(tmp_path, "sid-prompts", 1) == [" three "]
    assert recent_prompts(tmp_path, "sid-prompts", 2) == ["two", " three "]
    assert recent_prompts(tmp_path, "", 2) == []


def test_session_listing_and_lookup_exact_boundaries_and_utf8(tmp_path, monkeypatch) -> None:
    directory = session_mod.sessions_dir(tmp_path)
    (directory / "a-mismatch.json").write_text(
        json.dumps({"session_id": "a-mismatch", "project": "other", "updated": "invalid"}),
        encoding="utf-8",
    )
    (directory / "abcd-session.json").write_text(
        json.dumps(
            {
                "session_id": "abcd-session",
                "project": "memo",
                "cwd": str(tmp_path),
                "updated": "2026-01-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    (directory / "missing-cwd.json").write_text(
        json.dumps(
            {
                "session_id": "missing-cwd",
                "project": "memo",
                "updated": "2027-01-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    encodings: list[str | None] = []
    original_read_text = Path.read_text

    def recording_read_text(path: Path, *args, **kwargs) -> str:
        encodings.append(kwargs.get("encoding"))
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", recording_read_text)

    rows = list_sessions(tmp_path, project="memo", cwd=str(tmp_path))
    assert [row["session_id"] for row in rows] == ["abcd-session"]
    assert get_session(tmp_path, "abc") is None
    assert get_session(tmp_path, "abcd")["session_id"] == "abcd-session"
    assert get_session(tmp_path, "abcd-session")["session_id"] == "abcd-session"
    assert encodings and set(encodings) == {"utf-8"}


def test_list_sessions_default_limit_is_ten(tmp_path) -> None:
    for index in range(11):
        session_mod._write(
            tmp_path,
            f"sid-{index:02d}",
            {
                "session_id": f"sid-{index:02d}",
                "updated": f"2026-01-{index + 1:02d}T00:00:00+00:00",
            },
        )

    rows = list_sessions(tmp_path)
    assert len(rows) == 10
    assert rows[0]["session_id"] == "sid-10"
    assert rows[-1]["session_id"] == "sid-01"


def test_prune_lru_zero_equal_invalid_and_ordering_contract(tmp_path) -> None:
    session_mod._write(tmp_path, "sid-old", {"updated": "2025-01-01T00:00:00+00:00"})
    session_mod._write(tmp_path, "sid-new", {"updated": "2026-01-01T00:00:00+00:00"})
    bad = session_mod.sessions_dir(tmp_path) / "sid-bad.json"
    bad.write_text("{bad json", encoding="utf-8")

    with pytest.raises(ValueError, match=r"^cap must be non-negative$"):
        prune_lru(tmp_path, cap=-1)
    assert prune_lru(tmp_path, cap=3) == 0
    assert prune_lru(tmp_path, cap=2) == 1
    assert not bad.exists()
    assert prune_lru(tmp_path, cap=0) == 2
    assert list(session_mod.sessions_dir(tmp_path).glob("*.json")) == []


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (59, "<1m ago"),
        (60, "1m ago"),
        (3599, "59m ago"),
        (3600, "1h ago"),
        (86399, "23h ago"),
        (86400, "1d ago"),
    ],
)
def test_format_relative_exact_boundaries(seconds: int, expected: str) -> None:
    now = datetime(2026, 1, 2, tzinfo=UTC)
    assert format_relative((now - timedelta(seconds=seconds)).replace(tzinfo=None).isoformat(), now) == expected


def test_autosave_uses_floor_kibibytes_and_exact_threshold(tmp_path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_bytes(b"x" * 1536)

    assert session_mod.check_autosave(
        tmp_path,
        session_id="sid-auto",
        transcript_path=str(transcript),
        threshold_kb=2,
    ) == (False, 1)
    assert session_mod.check_autosave(
        tmp_path,
        session_id="sid-auto",
        transcript_path=str(transcript),
        threshold_kb=1,
    ) == (True, 1)
    assert session_mod.check_autosave(
        tmp_path,
        session_id="sid-auto",
        transcript_path=None,
    ) == (False, 0)


def test_recent_summary_exchanges_exact_roles_labels_truncation_and_utf8(
    tmp_path,
    monkeypatch,
) -> None:
    transcript = tmp_path / "transcript.jsonl"
    long_text = "x" * 301
    transcript.write_text(
        "\n".join(
            [
                json.dumps({"type": "user", "message": {"content": "hello"}}),
                json.dumps({"role": "assistant", "content": "reply"}),
                json.dumps({"type": "tool", "message": {"content": "ignore"}}),
                "{invalid",
                json.dumps({"type": "assistant", "message": {"content": long_text}}),
            ]
        ),
        encoding="utf-8",
    )
    encodings: list[str | None] = []
    original_read_text = Path.read_text

    def recording_read_text(path: Path, *args, **kwargs) -> str:
        encodings.append(kwargs.get("encoding"))
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", recording_read_text)

    assert session_mod._recent_summary_exchanges(transcript) == [
        "[User] hello",
        "[Assistant] reply",
        f"[Assistant] {'x' * 300}",
    ]
    assert encodings == ["utf-8"]


def test_refresh_summary_forwards_exact_prompt_model_options_and_caps_output(
    tmp_path,
    monkeypatch,
) -> None:
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        "\n".join(
            json.dumps({"type": "user", "message": {"content": f"exchange-{index}"}})
            for index in range(11)
        ),
        encoding="utf-8",
    )
    session_mod._write(
        tmp_path,
        "sid-summary",
        {
            "session_id": "sid-summary",
            "transcript_path": str(transcript),
            "turn_count": 5,
            "summary_turn": 1,
        },
    )
    calls: list[tuple[str, list[dict[str, str]], dict[str, float | int]]] = []

    class FakeChat:
        def chat(self, model, messages, *, options):
            calls.append((model, messages, options))
            return {"message": {"content": f"  {'s' * 450}  "}}

    monkeypatch.setattr("memo.llm.MLXChat", FakeChat)

    assert refresh_summary(
        tmp_path,
        "sid-summary",
        helper_model="exact/model",
        min_new_turns=3,
    )

    expected_recent = "\n\n".join(f"[User] exchange-{index}" for index in range(1, 11))
    expected_prompt = (
        "Based on this work session, write ONE brief PARAGRAPH (2-3 sentences) "
        "in English that summarizes: (1) what was being worked on, (2) what decisions or "
        "progress were made, (3) what was left pending or was the next step.\n\n"
        f"Session:\n{expected_recent}\n\n"
        "Summary (2-3 sentences, no bullets or headings):"
    )
    assert calls == [
        (
            "exact/model",
            [{"role": "user", "content": expected_prompt}],
            {"temperature": 0.0, "num_predict": 150},
        )
    ]
    saved = session_mod._load(tmp_path, "sid-summary")
    assert saved["running_summary"] == "s" * 400
    assert saved["summary_turn"] == 5


def test_refresh_summary_does_not_invent_empty_model_output(tmp_path, monkeypatch) -> None:
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        json.dumps({"type": "user", "message": {"content": "work"}}),
        encoding="utf-8",
    )
    initial = {
        "session_id": "sid-empty-summary",
        "transcript_path": str(transcript),
        "turn_count": 3,
        "summary_turn": 0,
    }
    session_mod._write(tmp_path, "sid-empty-summary", initial)

    class EmptyChat:
        def chat(self, *_args, **_kwargs):
            return {"message": {}}

    monkeypatch.setattr("memo.llm.MLXChat", EmptyChat)

    assert not refresh_summary(tmp_path, "sid-empty-summary", min_new_turns=3)
    assert session_mod._load(tmp_path, "sid-empty-summary") == initial


def test_clean_summary_and_active_memory_render_exact_contract() -> None:
    assert render_active_memory({"project": "", "branch": "", "turn_count": 0})[3] == (
        "- **Context**: `—` · `—` · 0 turns"
    )

    assert session_mod._clean_snapshot_summary(
        {
            "running_summary": "<command-message>noise</command-message>",
            "summary": "line one\nline two",
            "last_user_msg": "fallback",
        },
        13,
    ) == "line one line"
    assert session_mod._clean_snapshot_summary(
        {"summary": "<command-message>noise</command-message>", "last_user_msg": "fallback"},
        20,
    ) == "fallback"

    snapshot = {
        "running_summary": "r" * 141,
        "project": "memo",
        "branch": "",
        "turn_count": 0,
        "modified_files": [" a.py ", 7, "", "b.py", "c.py", "d.py", "e.py"],
        "last_assistant_tail": "tail\n" + "z" * 170,
        "prompt_trail": ["one", 8, "", "two", "three", "four", "p" * 121],
    }
    assert render_active_memory(snapshot) == [
        "### Active memory",
        "",
        f"- **In progress**: {'r' * 140}",
        "- **Context**: `memo` · `—` · 0 turns",
        "- **Files touched**: `a.py`, `b.py`, `c.py`, `d.py`",
        f"- **Last reply**: tail {'z' * 155}",
        "- **Open loops (session)**:",
        f"  1. {'p' * 120}",
        "  2. four",
        "  3. three",
        "  4. two",
        "  5. one",
    ]


def test_checkpoint_without_transcript_preserves_existing_message_fields(
    tmp_path,
    monkeypatch,
) -> None:
    state = tmp_path / "state"
    cwd = tmp_path / "project"
    cwd.mkdir()
    session_mod._write(
        state,
        "sid-preserve",
        {
            "transcript_path": "/existing/transcript.jsonl",
            "last_user_msg": "existing user",
            "last_assistant_tail": "existing assistant",
        },
    )
    monkeypatch.setattr(
        session_mod,
        "gather_git_state",
        lambda _cwd: {"branch": None, "head_commit": None, "modified_files": []},
    )
    monkeypatch.setattr(session_mod, "prune_lru", MagicMock())
    monkeypatch.setattr(session_mod, "flag_int", lambda name: 12)

    snapshot = checkpoint(state, session_id="sid-preserve", cwd=str(cwd))

    assert snapshot["transcript_path"] == "/existing/transcript.jsonl"
    assert snapshot["last_user_msg"] == "existing user"
    assert snapshot["last_assistant_tail"] == "existing assistant"
    session_mod.prune_lru.assert_called_once_with(state, cap=12)


def test_next_turn_covers_empty_missing_and_existing_sessions(tmp_path) -> None:
    assert session_mod.next_turn(tmp_path, "") == 1
    assert session_mod.next_turn(tmp_path, "sid-missing") == 1
    session_mod._write(tmp_path, "sid-next", {"turn_count": 2})
    assert session_mod.next_turn(tmp_path, "sid-next") == 3
    session_mod._write(tmp_path, "sid-zero", {"turn_count": 0})
    assert session_mod.next_turn(tmp_path, "sid-zero") == 1


def test_list_sessions_keeps_failed_cwd_resolution_as_exact_filter(
    tmp_path,
    monkeypatch,
) -> None:
    session_mod._write(
        tmp_path,
        "sid-other",
        {
            "session_id": "sid-other",
            "cwd": "/other",
            "updated": "2026-01-01T00:00:00+00:00",
        },
    )
    original_resolve = Path.resolve

    def selective_resolve(path: Path, *args, **kwargs) -> Path:
        if str(path) == "/broken":
            raise OSError("unresolvable")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", selective_resolve)

    assert list_sessions(tmp_path, cwd="/broken") == []


def test_list_sessions_missing_cwd_never_matches_synthetic_path(tmp_path) -> None:
    session_mod._write(
        tmp_path,
        "sid-missing-cwd",
        {
            "session_id": "sid-missing-cwd",
            "updated": "2026-01-01T00:00:00+00:00",
        },
    )

    assert list_sessions(tmp_path, cwd=str(Path.cwd() / "XXXX")) == []


def test_get_session_prefix_skips_symlink_and_corrupt_candidates(tmp_path) -> None:
    directory = session_mod.sessions_dir(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text('{"session_id": "outside"}', encoding="utf-8")
    (directory / "pref-00.json").symlink_to(outside)
    (directory / "pref-01.json").write_text("{invalid", encoding="utf-8")
    (directory / "pref-02.json").write_text(
        '{"session_id": "pref-02"}',
        encoding="utf-8",
    )

    assert get_session(tmp_path, "pref") == {"session_id": "pref-02"}


def test_prune_lru_sorts_by_timestamp_not_filename_and_reads_utf8(
    tmp_path,
    monkeypatch,
) -> None:
    session_mod._write(
        tmp_path,
        "a-new",
        {"updated": "2026-01-01T00:00:00+00:00"},
    )
    session_mod._write(
        tmp_path,
        "z-old",
        {"updated": "2025-01-01T00:00:00+00:00"},
    )
    encodings: list[str | None] = []
    original_read_text = Path.read_text

    def recording_read_text(path: Path, *args, **kwargs) -> str:
        encodings.append(kwargs.get("encoding"))
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", recording_read_text)

    assert prune_lru(tmp_path, cap=1) == 1
    assert (session_mod.sessions_dir(tmp_path) / "a-new.json").is_file()
    assert not (session_mod.sessions_dir(tmp_path) / "z-old.json").exists()
    assert encodings == ["utf-8", "utf-8"]


def test_refresh_summary_returns_false_if_snapshot_disappears_before_commit(
    tmp_path,
    monkeypatch,
) -> None:
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        json.dumps({"type": "user", "message": {"content": "work"}}),
        encoding="utf-8",
    )
    initial = {
        "session_id": "sid-vanish",
        "transcript_path": str(transcript),
        "turn_count": 3,
        "summary_turn": 0,
    }
    loads = iter([initial, None])
    monkeypatch.setattr(session_mod, "_load", lambda *_args: next(loads))

    class Chat:
        def chat(self, *_args, **_kwargs):
            return {"message": {"content": "summary"}}

    monkeypatch.setattr("memo.llm.MLXChat", Chat)

    assert not refresh_summary(tmp_path, "sid-vanish", min_new_turns=3)


def test_recap_stamp_empty_and_failure_paths_are_observable(
    tmp_path,
    monkeypatch,
    caplog,
) -> None:
    writer = MagicMock()
    monkeypatch.setattr(session_mod, "_write", writer)
    session_mod.stamp_recap_turn(tmp_path, "", 1)
    writer.assert_not_called()

    monkeypatch.setattr(session_mod, "_load", lambda *_args: {})
    session_mod.stamp_recap_turn(tmp_path, "sid-recap", 8)
    payload = writer.call_args.args[2]
    assert payload == {"session_id": "sid-recap", "last_recap_turn": 8}

    writer.side_effect = OSError("disk full")
    caplog.set_level("DEBUG")
    session_mod.stamp_recap_turn(tmp_path, "sid-recap", 9)
    assert "session: failed to checkpoint recap turn: disk full" in caplog.messages


def test_autosave_missing_file_and_cooldown_boundaries(tmp_path, monkeypatch) -> None:
    assert session_mod.check_autosave(
        tmp_path,
        session_id="sid-cooldown",
        transcript_path=str(tmp_path / "missing.jsonl"),
        threshold_kb=1,
    ) == (False, 0)

    transcript = tmp_path / "transcript.jsonl"
    transcript.write_bytes(b"x" * 1024)
    fixed_now = datetime(2026, 7, 27, 12, 0, 0, tzinfo=UTC)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            assert tz == UTC
            return fixed_now

    monkeypatch.setattr(session_mod, "datetime", FixedDateTime)
    session_mod._write(
        tmp_path,
        "sid-cooldown",
        {
            "last_autosave_at": (fixed_now - timedelta(seconds=299)).isoformat(),
        },
    )
    assert session_mod.check_autosave(
        tmp_path,
        session_id="sid-cooldown",
        transcript_path=str(transcript),
        threshold_kb=1,
        cooldown_secs=300,
    ) == (False, 1)

    session_mod._write(
        tmp_path,
        "sid-cooldown",
        {
            "last_autosave_at": (fixed_now - timedelta(seconds=300)).isoformat(),
        },
    )
    assert session_mod.check_autosave(
        tmp_path,
        session_id="sid-cooldown",
        transcript_path=str(transcript),
        threshold_kb=1,
        cooldown_secs=300,
    ) == (True, 1)


def test_mark_autosaved_updates_existing_only(tmp_path, monkeypatch) -> None:
    session_mod._write(tmp_path, "sid-mark", {"preserved": True})
    monkeypatch.setattr(session_mod, "_now_iso", lambda: "2026-07-27T12:00:00+00:00")

    session_mod.mark_autosaved(tmp_path, "sid-mark")
    assert session_mod._load(tmp_path, "sid-mark") == {
        "preserved": True,
        "last_autosave_at": "2026-07-27T12:00:00+00:00",
    }

    session_mod.mark_autosaved(tmp_path, "sid-missing")
    assert session_mod._load(tmp_path, "sid-missing") is None


def test_clean_snapshot_summary_empty_fallback_is_exact() -> None:
    assert session_mod._clean_snapshot_summary({}, 80) == "—"


def test_prune_lru_corrupt_snapshot_precedes_invalid_and_pre_epoch(
    tmp_path,
) -> None:
    directory = session_mod.sessions_dir(tmp_path)
    corrupt = directory / "corrupt.json"
    corrupt.write_text("{invalid", encoding="utf-8")
    (directory / "invalid-time.json").write_text(
        '{"updated": "AAA"}',
        encoding="utf-8",
    )
    (directory / "pre-epoch.json").write_text(
        '{"updated": "1960-01-01T00:00:00+00:00"}',
        encoding="utf-8",
    )

    assert prune_lru(tmp_path, cap=2) == 1
    assert not corrupt.exists()
    assert (directory / "invalid-time.json").exists()
    assert (directory / "pre-epoch.json").exists()


def test_autosave_invalid_timestamp_has_stable_diagnostic(
    tmp_path,
    caplog,
) -> None:
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_bytes(b"x" * 1024)
    session_mod._write(
        tmp_path,
        "sid-invalid-time",
        {"last_autosave_at": "not-an-instant"},
    )
    caplog.set_level("DEBUG")

    assert session_mod.check_autosave(
        tmp_path,
        session_id="sid-invalid-time",
        transcript_path=str(transcript),
        threshold_kb=1,
    ) == (True, 1)
    assert "session: unparseable transcript line, skipping" in caplog.messages


def test_refresh_summary_llm_failure_has_stable_diagnostic(
    tmp_path,
    monkeypatch,
    caplog,
) -> None:
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        json.dumps({"type": "user", "message": {"content": "work"}}),
        encoding="utf-8",
    )
    session_mod._write(
        tmp_path,
        "sid-llm-failure",
        {
            "session_id": "sid-llm-failure",
            "transcript_path": str(transcript),
            "turn_count": 3,
            "summary_turn": 0,
        },
    )

    class FailingChat:
        def chat(self, *_args, **_kwargs):
            raise RuntimeError("model unavailable")

    monkeypatch.setattr("memo.llm.MLXChat", FailingChat)
    caplog.set_level("DEBUG")

    assert not refresh_summary(tmp_path, "sid-llm-failure", min_new_turns=3)
    assert "session: reflect summary LLM call failed: model unavailable" in caplog.messages
