from __future__ import annotations

import contextlib
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from ._parsers import (
    _claude_candidate,
    _codex_candidate,
    _devin_candidate,
    _gemini_candidate,
    _is_claude_subagent_transcript,
)
from ._types import ResumeAgent, ResumeCandidate, ResumeProvider
from ._utils import (
    _agent_matches,
    _clip,
    _mtime_capped,
    _parse_instant,
    _prefer_status,
    _resolve_cwd,
    _same_cwd,
    _sort_key,
)


class DevinNativeProvider:
    name = "devin-native"

    def __init__(self, *, devin_home: Path | str | None = None) -> None:
        self.devin_home = (
            Path(devin_home).expanduser()
            if devin_home is not None
            else Path(
                os.environ.get("DEVIN_HOME") or Path.home() / ".local/share/devin/cli"
            ).expanduser()
        )

    def discover(
        self,
        *,
        agent: ResumeAgent,
        cwd: str,
        include_all_cwd: bool,
        limit: int,
    ) -> list[ResumeCandidate]:
        if not _agent_matches("devin", agent):
            return []
        root = self.devin_home / "transcripts"
        if not root.is_dir():
            return []
        rows: list[ResumeCandidate] = []
        for path in _mtime_capped(root.glob("*.json")):
            candidate = _devin_candidate(path)
            if candidate is None:
                continue
            if not include_all_cwd and not _same_cwd(candidate.cwd, cwd):
                continue
            rows.append(candidate)
        rows.sort(key=lambda item: _sort_key(item.updated_at), reverse=True)
        return rows[: max(1, int(limit))]


class CodexNativeProvider:
    name = "codex-native"

    def __init__(self, *, codex_home: Path | str | None = None) -> None:
        self.codex_home = (
            Path(codex_home).expanduser()
            if codex_home is not None
            else Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex").expanduser()
        )

    def discover(
        self,
        *,
        agent: ResumeAgent,
        cwd: str,
        include_all_cwd: bool,
        limit: int,
    ) -> list[ResumeCandidate]:
        if not _agent_matches("codex", agent):
            return []
        root = self.codex_home / "sessions"
        if not root.is_dir():
            return []
        rows: list[ResumeCandidate] = []
        for path in _mtime_capped(root.rglob("*.jsonl")):
            candidate = _codex_candidate(path)
            if candidate is None:
                continue
            if not include_all_cwd and not _same_cwd(candidate.cwd, cwd):
                continue
            rows.append(candidate)
        rows.sort(key=lambda item: _sort_key(item.updated_at), reverse=True)
        return rows[: max(1, int(limit))]


class ClaudeNativeProvider:
    name = "claude-native"

    def __init__(self, *, claude_home: Path | str | None = None) -> None:
        self.claude_home = (
            Path(claude_home).expanduser()
            if claude_home is not None
            else Path(os.environ.get("CLAUDE_HOME") or Path.home() / ".claude").expanduser()
        )

    def discover(
        self,
        *,
        agent: ResumeAgent,
        cwd: str,
        include_all_cwd: bool,
        limit: int,
    ) -> list[ResumeCandidate]:
        if not _agent_matches("claude", agent):
            return []
        root = self.claude_home / "projects"
        if not root.is_dir():
            return []
        rows: list[ResumeCandidate] = []
        # Drop subagent sidechains before the mtime cap so they don't displace
        # real resumable sessions from the parse budget.
        transcripts = (p for p in root.rglob("*.jsonl") if not _is_claude_subagent_transcript(p))
        for path in _mtime_capped(transcripts):
            candidate = _claude_candidate(path)
            if candidate is None:
                continue
            if not include_all_cwd and not _same_cwd(candidate.cwd, cwd):
                continue
            rows.append(candidate)
        rows.sort(key=lambda item: _sort_key(item.updated_at), reverse=True)
        return rows[: max(1, int(limit))]


class GeminiNativeProvider:
    name = "gemini-native"

    def __init__(self, *, gemini_home: Path | str | None = None) -> None:
        self.gemini_home = (
            Path(gemini_home).expanduser()
            if gemini_home is not None
            else Path(os.environ.get("GEMINI_HOME") or Path.home() / ".gemini").expanduser()
        )

    def discover(
        self,
        *,
        agent: ResumeAgent,
        cwd: str,
        include_all_cwd: bool,
        limit: int,
    ) -> list[ResumeCandidate]:
        if not _agent_matches("gemini", agent):
            return []
        root = self.gemini_home / "tmp"
        if not root.is_dir():
            return []
        rows: list[ResumeCandidate] = []
        # Gemini stores chats per project: ~/.gemini/tmp/<project>/chats/session-*.jsonl
        # The project's real cwd lives in the sibling `.project_root` file.
        for path in _mtime_capped(root.glob("*/chats/session-*.jsonl")):
            candidate = _gemini_candidate(path)
            if candidate is None:
                continue
            if not include_all_cwd and not _same_cwd(candidate.cwd, cwd):
                continue
            rows.append(candidate)
        rows.sort(key=lambda item: _sort_key(item.updated_at), reverse=True)
        return rows[: max(1, int(limit))]


class OpencodeNativeProvider:
    """Reads sessions from opencode's SQLite database at
    ``~/.local/share/opencode/opencode.db``."""

    name = "opencode-native"

    def __init__(self, *, db_path: Path | str | None = None) -> None:
        self.db_path = (
            Path(db_path).expanduser()
            if db_path is not None
            else Path(
                os.environ.get("OPENCODE_DATA") or Path.home() / ".local/share/opencode"
            ).expanduser()
            / "opencode.db"
        )

    def discover(
        self,
        *,
        agent: ResumeAgent,
        cwd: str,
        include_all_cwd: bool,
        limit: int,
    ) -> list[ResumeCandidate]:
        if not _agent_matches("opencode", agent):
            return []
        if not self.db_path.is_file():
            return []
        try:
            import sqlite3

            con = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            with contextlib.closing(con):
                con.row_factory = sqlite3.Row
                rows = con.execute(
                    "SELECT id, directory, title, time_updated FROM session"
                    " ORDER BY time_updated DESC LIMIT 200"
                ).fetchall()
        except Exception:
            return []
        results: list[ResumeCandidate] = []
        for row in rows:
            session_cwd = str(row["directory"] or "")
            if not include_all_cwd and not _same_cwd(session_cwd, cwd):
                continue
            ts_raw = row["time_updated"]
            if ts_raw:
                try:
                    updated_at = datetime.fromtimestamp(int(ts_raw) / 1000, tz=UTC).isoformat()
                except (ValueError, TypeError, OSError):
                    updated_at = str(ts_raw)
            else:
                updated_at = ""
            sid = str(row["id"])
            results.append(
                ResumeCandidate(
                    agent="opencode",
                    provider=self.name,
                    uri=f"opencode://session/{sid}",
                    session_id=sid,
                    title=str(row["title"] or ""),
                    cwd=session_cwd,
                    updated_at=updated_at,
                    resume_command=["opencode", "--session", sid],
                )
            )
        results.sort(key=lambda r: _sort_key(r.updated_at), reverse=True)
        return results[: max(1, int(limit))]


def _memo_snapshot_candidate(snap: dict[str, object]) -> ResumeCandidate | None:
    """Map one memo session snapshot dict (state_dir/sessions/*.json) to a candidate."""
    session_id = str(snap.get("session_id") or "").strip()
    if not session_id:
        return None
    summary = str(
        snap.get("running_summary") or snap.get("summary") or snap.get("last_user_msg") or ""
    )
    prompt_trail_raw = snap.get("prompt_trail")
    prompt_trail = (
        [str(p).strip() for p in prompt_trail_raw if isinstance(p, str) and p.strip()]
        if isinstance(prompt_trail_raw, list)
        else []
    )
    # memo only snapshots Claude Code today, but read an explicit `agent` if a
    # future writer stores one rather than hardcoding the resume command.
    agent = str(snap.get("agent") or "claude")
    uri = f"memo://session/{session_id}"
    return ResumeCandidate(
        agent=agent,
        provider="memo",
        uri=uri,
        session_id=session_id,
        title=_clip(summary or f"memo session {session_id[:8]}", 240),
        updated_at=str(snap.get("updated") or snap.get("created") or ""),
        cwd=_resolve_cwd(str(snap.get("cwd") or "")),
        summary=_clip(summary, 1000),
        resume_mode="native_resume" if agent == "claude" else "context_resume",
        resume_command=["claude", "--resume", session_id] if agent == "claude" else [],
        provenance=[uri],
        metadata={
            "project": str(snap.get("project") or ""),
            "turn_count": snap.get("turn_count"),
            "path": str(snap.get("transcript_path") or ""),
            "prompt_trail": prompt_trail,
        },
    )


class MemoSnapshotProvider:
    """memo's own session snapshots — the native, individual memo surface.

    Reads ``state_dir/sessions/*.json`` (written by memo's Stop /
    UserPromptSubmit hooks via ``memo session checkpoint``) directly, with no
    subprocess hop. These are Claude Code sessions, so they resume natively via
    ``claude --resume``. This is memo keeping its individuality: the federated
    picker spans other agents' native stores, but memo's own recall-grounded
    snapshots (``running_summary`` / ``prompt_trail``) stay first-class here.
    """

    name = "memo"

    def __init__(self, *, state_dir: Path | str | None = None) -> None:
        self.state_dir = Path(state_dir).expanduser() if state_dir is not None else None

    def _resolved_state_dir(self) -> Path:
        if self.state_dir is not None:
            return self.state_dir
        from memo.config import Config

        return Config.from_env().state_dir

    def discover(
        self,
        *,
        agent: ResumeAgent,
        cwd: str,
        include_all_cwd: bool,
        limit: int,
    ) -> list[ResumeCandidate]:
        if not _agent_matches("claude", agent):
            return []
        from memo.session import list_sessions

        snapshots = list_sessions(
            self._resolved_state_dir(),
            limit=max(1, int(limit)) * 4,
            cwd=cwd if not include_all_cwd else None,
        )
        rows: list[ResumeCandidate] = []
        for snap in snapshots:
            candidate = _memo_snapshot_candidate(snap)
            if candidate is None:
                continue
            if not include_all_cwd and not _same_cwd(candidate.cwd, cwd):
                continue
            rows.append(candidate)
        rows.sort(key=lambda item: _sort_key(item.updated_at), reverse=True)
        return rows[: max(1, int(limit))]


def default_resume_providers() -> list[ResumeProvider]:
    return [
        DevinNativeProvider(),
        CodexNativeProvider(),
        ClaudeNativeProvider(),
        GeminiNativeProvider(),
        OpencodeNativeProvider(),
        MemoSnapshotProvider(),
    ]


def _merge_candidates(candidates: Sequence[ResumeCandidate]) -> list[ResumeCandidate]:
    merged: dict[tuple[str, str], ResumeCandidate] = {}
    for candidate in candidates:
        key = (candidate.agent, candidate.session_id)
        existing = merged.get(key)
        if existing is None:
            merged[key] = candidate
            continue
        merged[key] = _merge_candidate(existing, candidate)
    return list(merged.values())


def _merge_candidate(existing: ResumeCandidate, incoming: ResumeCandidate) -> ResumeCandidate:
    base, other = (existing, incoming)
    if _provider_rank(incoming.provider) < _provider_rank(existing.provider):
        base, other = incoming, existing
    provenance = list(dict.fromkeys([*base.provenance, *other.provenance]))
    metadata = {**other.metadata, **base.metadata, f"{other.provider}_uri": other.uri}
    # Native wins identity (real cwd / transcript path), but memo's snapshot has
    # the LLM `running_summary` — strictly richer than the raw last-user-text the
    # native parser extracts. Prefer the memo side's summary/title when present.
    memo_side = next((c for c in (base, other) if c.provider == "memo" and c.summary), None)
    summary = memo_side.summary if memo_side else (base.summary or other.summary)
    title = (memo_side.title if memo_side and memo_side.title else base.title) or other.title
    base_updated = _parse_instant(base.updated_at)
    other_updated = _parse_instant(other.updated_at)
    if base_updated is None:
        updated_at = other.updated_at
    elif other_updated is None:
        updated_at = base.updated_at
    else:
        updated_at = base.updated_at if base_updated >= other_updated else other.updated_at
    return ResumeCandidate(
        agent=base.agent,
        provider=base.provider,
        uri=base.uri,
        session_id=base.session_id,
        title=title,
        updated_at=updated_at,
        cwd=base.cwd or other.cwd,
        summary=summary,
        status=_prefer_status(base.status, other.status),
        resume_mode=base.resume_mode if base.resume_mode == "native_resume" else other.resume_mode,
        resume_command=base.resume_command or other.resume_command,
        provenance=provenance,
        metadata=metadata,
    )


def _provider_rank(provider: str) -> int:
    # Native-transcript providers win the merge base (they carry the real cwd +
    # resumable session id); memo's own snapshot merges its richer summary in.
    return {
        "codex-native": 0,
        "claude-native": 0,
        "devin-native": 0,
        "gemini-native": 0,
        "opencode-native": 0,
        "memo": 1,
    }.get(provider, 9)
