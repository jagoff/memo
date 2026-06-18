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

import json
from datetime import UTC, datetime, timedelta

import pytest

from memo import session as session_mod
from memo.session import (
    checkpoint,
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
    assert "Memoria activa" in md
    assert "Se dejó listo" in md
    assert "memo" in md and "master" in md
    assert "session.py" in md and "cli_session.py" in md
    assert "Última respuesta" in md and "verificado" in md
    assert "Loops abiertos (sesión)" in md and "tercer loop" in md


def test_format_relative_buckets():
    now = datetime(2026, 5, 8, 12, 0, 0, tzinfo=UTC)
    assert format_relative((now - timedelta(seconds=30)).isoformat(), now=now) == "hace <1m"
    assert format_relative((now - timedelta(minutes=5)).isoformat(), now=now) == "hace 5m"
    assert format_relative((now - timedelta(hours=3)).isoformat(), now=now) == "hace 3h"
    assert format_relative((now - timedelta(days=2)).isoformat(), now=now) == "hace 2d"
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
