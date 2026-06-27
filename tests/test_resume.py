"""Tests for memo's cross-agent resume engine (src/memo/resume/).

Ported from synapse's test_resume.py, scoped to the native-agent discovery +
merge + format + execute surface memo absorbed (Forma B). The memflow /
checkpoint / interactive-TUI cases stay in synapse. Adds memo-specific coverage
for MemoSnapshotProvider (memo's own session snapshots, kept first-class) and
the `memo resume --agent` CLI dispatch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from memo.resume import (
    ClaudeNativeProvider,
    CodexNativeProvider,
    GeminiNativeProvider,
    MemoSnapshotProvider,
    ResumeAgent,
    ResumeCandidate,
    discover_resume_candidates,
    execute_resume_candidate,
    format_resume_candidates,
)
from memo.resume._utils import _format_relative_time


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


# ── native providers ─────────────────────────────────────────────────────────


def test_codex_provider_discovers_native_sessions_for_cwd(tmp_path: Path) -> None:
    cwd = tmp_path / "repo"
    other = tmp_path / "other"
    cwd.mkdir()
    other.mkdir()
    codex_home = tmp_path / "codex"
    _write_jsonl(
        codex_home / "sessions" / "2026" / "05" / "23" / "session-a.jsonl",
        [
            {
                "type": "session_meta",
                "payload": {
                    "id": "codex-a",
                    "timestamp": "2026-05-23T10:00:00Z",
                    "cwd": str(cwd),
                    "originator": "codex-tui",
                },
            },
            {
                "type": "event_msg",
                "payload": {"type": "user_message", "message": "resume the work"},
            },
        ],
    )
    _write_jsonl(
        codex_home / "sessions" / "2026" / "05" / "23" / "session-b.jsonl",
        [
            {
                "type": "session_meta",
                "payload": {
                    "id": "codex-b",
                    "timestamp": "2026-05-23T10:05:00Z",
                    "cwd": str(other),
                },
            }
        ],
    )

    rows = CodexNativeProvider(codex_home=codex_home).discover(
        agent="all", cwd=str(cwd), include_all_cwd=False, limit=10
    )

    assert [row.session_id for row in rows] == ["codex-a"]
    assert rows[0].uri == "codex://session/codex-a"
    assert rows[0].resume_command == ["codex", "resume", "codex-a"]
    assert rows[0].resume_mode == "native_resume"


def test_codex_provider_extracts_goal_from_bootstrap_prompts(tmp_path: Path) -> None:
    cwd = tmp_path / "repo"
    cwd.mkdir()
    codex_home = tmp_path / "codex"
    _write_jsonl(
        codex_home / "sessions" / "2026" / "05" / "23" / "session-subagent.jsonl",
        [
            {
                "type": "session_meta",
                "payload": {
                    "id": "codex-subagent",
                    "timestamp": "2026-05-23T10:00:00Z",
                    "cwd": str(cwd),
                },
            },
            {
                "type": "event_msg",
                "payload": {
                    "type": "user_message",
                    "message": "# AGENTS.md instructions for /tmp/repo <INSTRUCTIONS>noise",
                },
            },
            {
                "type": "event_msg",
                "payload": {
                    "type": "user_message",
                    "message": (
                        f"Repo: {cwd}. Read AGENTS.md and CLAUDE.md first. "
                        "You are Developer 1. Ownership: runtime files. "
                        "Goal: finish and harden the runtime loop: bounded loop mode, "
                        "signed history snapshots, retention pruning, and tests. "
                        "Preserve invariants."
                    ),
                },
            },
        ],
    )

    rows = CodexNativeProvider(codex_home=codex_home).discover(
        agent="codex", cwd=str(cwd), include_all_cwd=False, limit=10
    )

    assert rows[0].summary.startswith("finish and harden the runtime loop")
    assert not rows[0].summary.startswith("Repo:")


def test_claude_provider_discovers_native_sessions_for_cwd(tmp_path: Path) -> None:
    cwd = tmp_path / "repo"
    cwd.mkdir()
    claude_home = tmp_path / "claude"
    _write_jsonl(
        claude_home / "projects" / "-tmp-repo" / "claude-a.jsonl",
        [
            {"type": "last-prompt", "sessionId": "claude-a"},
            {"type": "attachment", "sessionId": "claude-a", "cwd": str(cwd), "version": "2.0.0"},
            {
                "type": "user",
                "message": {"content": [{"type": "text", "text": "continue the Claude task"}]},
            },
        ],
    )

    rows = ClaudeNativeProvider(claude_home=claude_home).discover(
        agent="claude", cwd=str(cwd), include_all_cwd=False, limit=10
    )

    assert len(rows) == 1
    assert rows[0].uri == "claude://session/claude-a"
    assert rows[0].resume_command == ["claude", "--resume", "claude-a"]
    assert rows[0].summary == "continue the Claude task"


def test_claude_provider_excludes_subagent_transcripts(tmp_path: Path) -> None:
    cwd = tmp_path / "repo"
    cwd.mkdir()
    claude_home = tmp_path / "claude"
    project = claude_home / "projects" / "-tmp-repo"
    _write_jsonl(
        project / "parent.jsonl",
        [
            {"type": "attachment", "sessionId": "parent", "cwd": str(cwd)},
            {"type": "user", "message": {"content": [{"type": "text", "text": "real session"}]}},
        ],
    )
    _write_jsonl(
        project / "subagents" / "agent-deadbeef.jsonl",
        [
            {"type": "attachment", "sessionId": "parent", "cwd": str(cwd)},
            {"type": "user", "message": {"content": [{"type": "text", "text": "subagent prompt"}]}},
        ],
    )
    _write_jsonl(
        project / "agent-cafef00d.jsonl",
        [
            {"type": "attachment", "sessionId": "parent", "cwd": str(cwd)},
            {
                "type": "user",
                "message": {"content": [{"type": "text", "text": "another subagent"}]},
            },
        ],
    )

    rows = ClaudeNativeProvider(claude_home=claude_home).discover(
        agent="claude", cwd=str(cwd), include_all_cwd=False, limit=10
    )

    assert [row.session_id for row in rows] == ["parent"]
    assert rows[0].summary == "real session"


def test_gemini_provider_discovers_native_sessions_for_cwd(tmp_path: Path) -> None:
    cwd = tmp_path / "repo"
    other = tmp_path / "other"
    cwd.mkdir()
    other.mkdir()
    gemini_home = tmp_path / "gemini"

    def _write_gemini(
        project: str, project_root: Path, session: str, rows: list[dict[str, object]]
    ) -> None:
        proj_dir = gemini_home / "tmp" / project
        (proj_dir / "chats").mkdir(parents=True, exist_ok=True)
        (proj_dir / ".project_root").write_text(str(project_root), encoding="utf-8")
        _write_jsonl(proj_dir / "chats" / f"session-{session}.jsonl", rows)

    _write_gemini(
        "repo",
        cwd,
        "2026-06-09T18-00-aaaa1111",
        [
            {
                "sessionId": "gemini-a",
                "startTime": "2026-06-09T18:00:00.000Z",
                "lastUpdated": "2026-06-09T18:05:00.000Z",
                "kind": "main",
            },
            {"type": "user", "content": [{"text": "<session_context>\nThis is the Gemini CLI."}]},
            {"type": "gemini", "content": [{"text": "ack"}]},
            {"type": "user", "content": [{"text": "wire up the resume picker"}]},
        ],
    )
    _write_gemini(
        "other",
        other,
        "2026-06-09T18-10-bbbb2222",
        [
            {
                "sessionId": "gemini-b",
                "startTime": "2026-06-09T18:10:00.000Z",
                "lastUpdated": "2026-06-09T18:11:00.000Z",
                "kind": "main",
            },
            {"type": "user", "content": [{"text": "different repo work"}]},
        ],
    )

    rows = GeminiNativeProvider(gemini_home=gemini_home).discover(
        agent="all", cwd=str(cwd), include_all_cwd=False, limit=10
    )

    assert [row.session_id for row in rows] == ["gemini-a"]
    assert rows[0].uri == "gemini://session/gemini-a"
    assert rows[0].resume_command[:2] == ["gemini", "--session-file"]
    assert rows[0].cwd == str(cwd.resolve())
    assert rows[0].summary == "wire up the resume picker"


# ── memo's own snapshots — kept first-class (memo's individuality) ────────────


def test_memo_snapshot_provider_reads_own_sessions(tmp_path: Path) -> None:
    cwd = tmp_path / "repo"
    cwd.mkdir()
    state_dir = tmp_path / "state"
    (state_dir / "sessions").mkdir(parents=True)
    (state_dir / "sessions" / "memo-sess-1.json").write_text(
        json.dumps(
            {
                "session_id": "memo-sess-1",
                "cwd": str(cwd),
                "project": "repo",
                "summary": "memo snapshot work",
                "updated": "2026-05-23T10:00:00+00:00",
                "created": "2026-05-23T09:00:00+00:00",
                "turn_count": 3,
            }
        ),
        encoding="utf-8",
    )

    rows = MemoSnapshotProvider(state_dir=state_dir).discover(
        agent="all", cwd=str(cwd), include_all_cwd=True, limit=10
    )

    assert len(rows) == 1
    assert rows[0].agent == "claude"
    assert rows[0].provider == "memo"
    assert rows[0].uri == "memo://session/memo-sess-1"
    assert rows[0].resume_mode == "native_resume"
    assert rows[0].resume_command == ["claude", "--resume", "memo-sess-1"]
    assert rows[0].summary == "memo snapshot work"


# ── orchestration: merge / format / relative time / execute ───────────────────


class _StaticProvider:
    def __init__(self, name: str, rows: list[ResumeCandidate]) -> None:
        self.name = name
        self._rows = rows

    def discover(
        self, *, agent: ResumeAgent, cwd: str, include_all_cwd: bool, limit: int
    ) -> list[ResumeCandidate]:
        return self._rows


def test_discovery_merges_duplicate_sessions_by_agent_and_session_id() -> None:
    native = ResumeCandidate(
        agent="claude",
        provider="claude-native",
        uri="claude://session/same",
        session_id="same",
        title="native",
        updated_at="2026-05-23T10:00:00Z",
        resume_mode="native_resume",
        resume_command=["claude", "--resume", "same"],
        provenance=["claude://session/same"],
    )
    memo = ResumeCandidate(
        agent="claude",
        provider="memo",
        uri="memo://session/same",
        session_id="same",
        title="memo",
        updated_at="2026-05-23T10:05:00Z",
        summary="richer memo summary",
        resume_mode="native_resume",
        resume_command=["claude", "--resume", "same"],
        provenance=["memo://session/same"],
    )

    report = discover_resume_candidates(
        providers=[_StaticProvider("claude-native", [native]), _StaticProvider("memo", [memo])],
        include_all_cwd=True,
        limit=10,
    )

    assert report.schema == "memo.resume_candidates.v1"
    assert len(report.candidates) == 1
    # Native wins the merge base; memo's richer summary still merges in.
    assert report.candidates[0].provider == "claude-native"
    assert report.candidates[0].summary == "richer memo summary"
    assert report.candidates[0].provenance == ["claude://session/same", "memo://session/same"]


def test_format_resume_candidates_uses_terminal_width(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    title = "finish and harden the runtime loop with signed history snapshots"
    candidate = ResumeCandidate(
        agent="codex",
        provider="codex-native",
        uri="codex://session/abc",
        session_id="abc",
        title=title,
        summary=title,
        updated_at="2026-05-23T10:00:00Z",
    )
    report = discover_resume_candidates(
        providers=[_StaticProvider("codex-native", [candidate])], include_all_cwd=True, limit=10
    )

    monkeypatch.setattr(
        "memo.resume.shutil.get_terminal_size", lambda fallback: os.terminal_size((160, 24))
    )
    wide = format_resume_candidates(report)
    monkeypatch.setattr(
        "memo.resume.shutil.get_terminal_size", lambda fallback: os.terminal_size((58, 24))
    )
    narrow = format_resume_candidates(report)

    assert title in wide
    assert "..." not in wide.splitlines()[1]
    assert "..." in narrow.splitlines()[1]


def test_format_relative_time_buckets_minutes_hours_days() -> None:
    from datetime import UTC, datetime, timedelta

    now = datetime(2026, 5, 25, 18, 0, 0, tzinfo=UTC)
    cases = [
        (now - timedelta(seconds=15), "now"),
        (now - timedelta(minutes=5), "5m ago"),
        (now - timedelta(hours=4), "4h ago"),
        (now - timedelta(days=2), "2d ago"),
        (now - timedelta(days=45), "1mo ago"),
        (now - timedelta(days=400), "1y ago"),
    ]
    for ts, expected in cases:
        assert _format_relative_time(ts.isoformat().replace("+00:00", "Z"), now=now) == expected


def test_format_relative_time_handles_invalid_input() -> None:
    assert _format_relative_time("") == ""
    assert _format_relative_time("not-a-date") == ""


def test_execute_native_resume_replaces_process(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = ResumeCandidate(
        agent="codex",
        provider="codex-native",
        uri="codex://session/abc",
        session_id="abc",
        title="codex",
        updated_at="2026-05-23T10:00:00Z",
        resume_mode="native_resume",
        resume_command=["codex", "resume", "abc"],
    )
    captured: dict[str, Any] = {}

    class ExecCalled(RuntimeError):
        pass

    def fake_which(name: str) -> str:
        assert name == "codex"
        return "/bin/codex"

    def fake_execvp(file: str, args: list[str]) -> None:
        captured["file"] = file
        captured["args"] = args
        raise ExecCalled

    monkeypatch.setattr("memo.resume.shutil.which", fake_which)
    monkeypatch.setattr("memo.resume.os.execvp", fake_execvp)

    with pytest.raises(ExecCalled):
        execute_resume_candidate(candidate)

    assert captured == {"file": "/bin/codex", "args": ["/bin/codex", "resume", "abc"]}


# ── CLI dispatch: default (memo) vs federated (--agent all) ───────────────────


def _agent_home_env(tmp_path: Path) -> dict[str, str]:
    """Point every agent-home env at an empty tmp dir so the federated picker
    stays hermetic (never reads the developer's real ~/.claude / ~/.codex / …)."""
    empty = tmp_path / "empty_homes"
    empty.mkdir(exist_ok=True)
    return {
        "CLAUDE_HOME": str(empty / "claude"),
        "CODEX_HOME": str(empty / "codex"),
        "DEVIN_HOME": str(empty / "devin"),
        "GEMINI_HOME": str(empty / "gemini"),
        "OPENCODE_DATA": str(empty / "opencode"),
    }


def _seed_memo_snapshot(state_dir: Path, cwd: Path) -> None:
    (state_dir / "sessions").mkdir(parents=True, exist_ok=True)
    (state_dir / "sessions" / "cli-sess.json").write_text(
        json.dumps(
            {
                "session_id": "cli-sess",
                "cwd": str(cwd),
                "project": cwd.name,
                "summary": "cli session work",
                "updated": "2026-05-23T10:00:00+00:00",
                "turn_count": 2,
            }
        ),
        encoding="utf-8",
    )


def test_resume_cli_default_lists_memo_snapshots(tmp_path: Path) -> None:
    """Default `memo resume --json` stays memo-only (list shape) — preserves the
    contract synapse's MemoResumeProvider depends on."""
    from memo.cli import cli

    state_dir = tmp_path / "state"
    data_dir = tmp_path / "data"
    cwd = tmp_path / "repo"
    cwd.mkdir()
    _seed_memo_snapshot(state_dir, cwd)

    result = CliRunner().invoke(
        cli,
        ["resume", "--json"],
        env={
            "MEMO_STATE_DIR": str(state_dir),
            "MEMO_DATA_DIR": str(data_dir),
            "MEMO_NONINTERACTIVE": "1",
        },
    )

    assert result.exit_code == 0, result.output
    loaded = json.loads(result.output)
    assert isinstance(loaded, list)
    assert any(row.get("session_id") == "cli-sess" for row in loaded)


def test_resume_cli_bare_default_federates(tmp_path: Path) -> None:
    """Bare `memo resume` (no --agent, no --json) federates across agents — it
    surfaces a codex session, which the memo-only mode would never show."""
    from memo.cli import cli

    state_dir = tmp_path / "state"
    data_dir = tmp_path / "data"
    homes = _agent_home_env(tmp_path)
    _write_jsonl(
        Path(homes["CODEX_HOME"]) / "sessions" / "2026" / "05" / "23" / "session-z.jsonl",
        [
            {
                "type": "session_meta",
                "payload": {
                    "id": "codex-z",
                    "timestamp": "2026-05-23T10:00:00Z",
                    "cwd": str(tmp_path),
                },
            },
            {
                "type": "event_msg",
                "payload": {"type": "user_message", "message": "federated codex work"},
            },
        ],
    )

    env = {
        "MEMO_STATE_DIR": str(state_dir),
        "MEMO_DATA_DIR": str(data_dir),
        "MEMO_NONINTERACTIVE": "1",
        **homes,
    }
    result = CliRunner().invoke(cli, ["resume", "--all-cwd"], env=env)

    assert result.exit_code == 0, result.output
    assert "codex" in result.output


def test_resume_cli_agent_all_federates(tmp_path: Path) -> None:
    """`memo resume --agent all --json` emits the federated report (parity with
    synapse resume), with memo's own snapshot merged in."""
    from memo.cli import cli

    state_dir = tmp_path / "state"
    data_dir = tmp_path / "data"
    cwd = tmp_path / "repo"
    cwd.mkdir()
    _seed_memo_snapshot(state_dir, cwd)

    env = {
        "MEMO_STATE_DIR": str(state_dir),
        "MEMO_DATA_DIR": str(data_dir),
        "MEMO_NONINTERACTIVE": "1",
        **_agent_home_env(tmp_path),
    }
    result = CliRunner().invoke(cli, ["resume", "--agent", "all", "--all-cwd", "--json"], env=env)

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["schema"] == "memo.resume_candidates.v1"
    assert any(
        c["session_id"] == "cli-sess" and c["provider"] == "memo" for c in report["candidates"]
    )


# ── interactive picker (TUI) logic — pure, no TTY ─────────────────────────────


def _tui_cands() -> list[ResumeCandidate]:
    return [
        ResumeCandidate(
            agent="claude",
            provider="claude-native",
            uri="claude://s/a",
            session_id="aaaa",
            title="alpha repo work",
            updated_at="2026-05-23T10:00:00Z",
            cwd="/repo",
        ),
        ResumeCandidate(
            agent="codex",
            provider="codex-native",
            uri="codex://s/b",
            session_id="bbbb",
            title="beta task",
            updated_at="2026-05-23T09:00:00Z",
            cwd="/repo",
        ),
    ]


@pytest.mark.parametrize(
    ("seq", "expected"),
    [
        ("\x1b[A", "up"),
        ("\x1b[B", "down"),
        ("\x1bOA", "up"),
        ("q", "quit"),
        ("\n", "enter"),
        ("\x1b", "quit"),
    ],
)
def test_resume_key_from_sequence(seq: str, expected: str) -> None:
    from memo.resume._tui import _resume_key_from_sequence

    assert _resume_key_from_sequence(seq) == expected


@pytest.mark.parametrize(
    ("seq", "expected"),
    [
        ("\x1b[A", "up"),
        ("\x1b[B", "down"),
        ("\t", "tab"),
        ("\x7f", "backspace"),
        ("\x03", "ctrl_c"),
        ("a", "a"),
    ],
)
def test_rich_key_from_sequence(seq: str, expected: str) -> None:
    from memo.resume._tui import _rich_key_from_sequence

    assert _rich_key_from_sequence(seq) == expected


def test_resume_tui_dispatch_navigation_and_select() -> None:
    from memo.resume._tui import _resume_tui_dispatch, _resume_tui_visible, _ResumeTuiState

    state = _ResumeTuiState(
        candidates=_tui_cands(), current_cwd="/repo", filter_mode="all", index=1
    )
    visible = _resume_tui_visible(state)
    assert _resume_tui_dispatch("down", state, visible) == ""
    assert state.index == 2
    assert _resume_tui_dispatch("enter", state, visible) == "select"
    # index 0 is the "[ Start a new session ]" row → Enter quits (start fresh).
    state.index = 0
    assert _resume_tui_dispatch("enter", state, visible) == "quit"
    assert _resume_tui_dispatch("ctrl_c", state, visible) == "quit"


def test_resume_tui_typing_filters_visible_list() -> None:
    from memo.resume._tui import _resume_tui_dispatch, _resume_tui_visible, _ResumeTuiState

    state = _ResumeTuiState(
        candidates=_tui_cands(), current_cwd="/repo", filter_mode="all", index=2
    )
    _resume_tui_dispatch("b", state, _resume_tui_visible(state))
    assert state.query == "b"
    assert state.index == 1  # typing resets selection to the first match
    assert [c.session_id for c in _resume_tui_visible(state)] == ["bbbb"]


def test_filter_resume_candidates_by_cwd(tmp_path: Path) -> None:
    from memo.resume._tui import _filter_resume_candidates

    a, b = _tui_cands()
    elsewhere = ResumeCandidate(
        agent="gemini",
        provider="gemini-native",
        uri="g://s/c",
        session_id="cccc",
        title="other repo",
        updated_at="2026-05-23T08:00:00Z",
        cwd="/elsewhere",
    )
    out = _filter_resume_candidates(
        [a, b, elsewhere], query="", filter_mode="cwd", current_cwd="/repo"
    )
    assert {c.session_id for c in out} == {"aaaa", "bbbb"}


# ── improvements: mtime cap, summary merge, project filter, <id> --json ───────


def test_mtime_capped_returns_newest_first_capped(tmp_path: Path) -> None:
    import os

    from memo.resume._utils import _mtime_capped

    paths = []
    for i in range(5):
        p = tmp_path / f"f{i}.jsonl"
        p.write_text("x", encoding="utf-8")
        os.utime(p, (1000 + i * 100, 1000 + i * 100))  # f4 newest, f0 oldest
        paths.append(p)
    capped = _mtime_capped(paths, cap=3)
    assert [p.name for p in capped] == ["f4.jsonl", "f3.jsonl", "f2.jsonl"]


def test_merge_prefers_memo_running_summary() -> None:
    """Native wins identity, but memo's richer LLM running_summary wins the
    summary/title fields (regression: the raw last-user-text used to clobber it)."""
    native = ResumeCandidate(
        agent="claude",
        provider="claude-native",
        uri="claude://s/x",
        session_id="x",
        title="fix the bug",
        summary="fix the bug",
        updated_at="2026-05-23T10:00:00Z",
    )
    memo = ResumeCandidate(
        agent="claude",
        provider="memo",
        uri="memo://s/x",
        session_id="x",
        title="Refactored the resume picker and added mtime gating",
        summary="Refactored the resume picker and added mtime gating",
        updated_at="2026-05-23T10:05:00Z",
    )
    report = discover_resume_candidates(
        providers=[_StaticProvider("claude-native", [native]), _StaticProvider("memo", [memo])],
        include_all_cwd=True,
        limit=10,
    )
    assert len(report.candidates) == 1
    merged = report.candidates[0]
    assert merged.provider == "claude-native"  # native keeps identity
    assert merged.summary == "Refactored the resume picker and added mtime gating"
    assert merged.title == "Refactored the resume picker and added mtime gating"


def test_resume_cli_federated_session_id_json_returns_single(tmp_path: Path) -> None:
    from memo.cli import cli

    state_dir = tmp_path / "state"
    cwd = tmp_path / "repo"
    cwd.mkdir()
    _seed_memo_snapshot(state_dir, cwd)
    env = {
        "MEMO_STATE_DIR": str(state_dir),
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_NONINTERACTIVE": "1",
        **_agent_home_env(tmp_path),
    }
    result = CliRunner().invoke(
        cli, ["resume", "cli-sess", "--agent", "all", "--all-cwd", "--json"], env=env
    )
    assert result.exit_code == 0, result.output
    obj = json.loads(result.output)
    # A single candidate dict, not a {schema, candidates:[…]} report.
    assert obj["session_id"] == "cli-sess"
    assert obj["provider"] == "memo"


def test_resume_cli_federated_project_filter(tmp_path: Path) -> None:
    from memo.cli import cli

    state_dir = tmp_path / "state"
    cwd = tmp_path / "repo"
    cwd.mkdir()
    _seed_memo_snapshot(state_dir, cwd)  # project basename == "repo"
    homes = _agent_home_env(tmp_path)
    _write_jsonl(
        Path(homes["CODEX_HOME"]) / "sessions" / "session-z.jsonl",
        [
            {
                "type": "session_meta",
                "payload": {
                    "id": "codex-z",
                    "timestamp": "2026-05-23T10:00:00Z",
                    "cwd": str(tmp_path),
                },
            },
            {"type": "event_msg", "payload": {"type": "user_message", "message": "elsewhere work"}},
        ],
    )
    env = {
        "MEMO_STATE_DIR": str(state_dir),
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_NONINTERACTIVE": "1",
        **homes,
    }
    result = CliRunner().invoke(
        cli, ["resume", "--agent", "all", "--all-cwd", "--project", "repo", "--json"], env=env
    )
    assert result.exit_code == 0, result.output
    sids = {c["session_id"] for c in json.loads(result.output)["candidates"]}
    assert "cli-sess" in sids  # project repo kept
    assert "codex-z" not in sids  # different project dropped
