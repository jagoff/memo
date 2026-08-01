"""Conversation crystallization — synthesize a chat session into a durable memo memory.

Ported from the archived synapse `crystallize.py` (prompt kept verbatim), adapted to
call `Memory` directly instead of federating through separate memflow/memo backends.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from memo.chat.config import ChatConfig
from memo.chat.sessions import SessionStore

_log = logging.getLogger(__name__)

_CRYSTAL_PROMPT = """\
Analyze this conversation transcript and extract a concise session crystal.
Return ONLY valid JSON with these exact fields:
{{
  "title": "short descriptive title (max 80 chars, include date)",
  "situation": "1-2 sentences: what was being worked on",
  "decisions": ["decision 1", "decision 2"],
  "learnings": ["learning 1", "learning 2"],
  "goal_progress": ["goal: what advanced"],
  "body": "2-4 paragraph prose summary of the session",
  "tags": ["tag1", "tag2", "tag3"]
}}

Transcript (most recent {n_turns} turns):
{transcript}

Today: {today}
"""

_DEDUP_WINDOW_S = 1800


def _content_of(out: Any) -> str:
    if isinstance(out, dict):
        message = out.get("message")
        if isinstance(message, dict):
            return str(message.get("content", ""))
        return str(out.get("content", "") or out.get("response", ""))
    return str(out)


def _format_transcript(turns: list[dict[str, Any]], max_chars: int = 4000) -> str:
    lines = [f"{str(t.get('role', '?')).upper()}: {str(t.get('text', ''))[:500]}" for t in turns]
    raw = "\n".join(lines)
    return raw[-max_chars:] if len(raw) > max_chars else raw


def _heuristic_crystal(session_id: str, turns: list[dict[str, Any]]) -> dict[str, Any]:
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    return {
        "title": f"Session {session_id[:12]} — {today}",
        "situation": "",
        "decisions": [],
        "learnings": [],
        "goal_progress": [],
        "body": f"Session with {len(turns)} turns. LLM synthesis unavailable.",
        "tags": ["session"],
    }


def _synthesize_crystal(
    memory: Any, session_id: str, turns: list[dict[str, Any]]
) -> dict[str, Any]:
    try:
        chat = memory._ensure_chat()
        transcript = _format_transcript(turns)
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        prompt = _CRYSTAL_PROMPT.format(n_turns=len(turns), transcript=transcript, today=today)
        out = chat.chat(
            memory.cfg.llm_model,
            [{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "max_tokens": 600},
        )
        content = _content_of(out)
        payload = content[content.index("{") : content.rindex("}") + 1]
        crystal = json.loads(payload)
        if isinstance(crystal, dict) and crystal.get("title"):
            return crystal
    except Exception as exc:
        _log.debug("crystallize: LLM synthesis failed, using heuristic fallback: %s", exc)
    return _heuristic_crystal(session_id, turns)


def _build_text(crystal: dict[str, Any]) -> str:
    body_parts = [str(crystal.get("body") or "")]
    if crystal.get("decisions"):
        body_parts.append("\n**Decisions:**\n" + "\n".join(f"- {d}" for d in crystal["decisions"]))
    if crystal.get("learnings"):
        body_parts.append(
            "\n**Learnings:**\n" + "\n".join(f"- {item}" for item in crystal["learnings"])
        )
    if crystal.get("goal_progress"):
        body_parts.append(
            "\n**Goal progress:**\n" + "\n".join(f"- {g}" for g in crystal["goal_progress"])
        )
    full_text = f"{crystal.get('title', '')}\n\n" + "\n".join(body_parts)
    return full_text[:3000]


def _dedup_path(memory: Any) -> Path:
    return Path(memory.cfg.state_dir) / "chat" / "crystallize_last.json"


def _load_dedup_state(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_dedup_state(path: Path, state: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state), encoding="utf-8")


def crystallize_session(
    memory: Any,
    session_id: str | None = None,
    *,
    n_turns: int = 30,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Synthesize a chat session into a durable `decision` memory (a "crystal").

    With no `session_id`, crystallizes the most recently active session. Guarded
    by a 30-minute dedup window per session id (writes/checks
    `cfg.state_dir/chat/crystallize_last.json`) so an auto-trigger calling this
    repeatedly for the same session doesn't spam duplicate memories.
    """
    result: dict[str, Any] = {
        "ok": False,
        "dry_run": dry_run,
        "session_id": session_id,
        "crystal": None,
        "memory_id": None,
        "skipped": False,
        "error": None,
    }

    chat_cfg = ChatConfig.load(memory.cfg.state_dir)
    store = SessionStore(chat_cfg.sessions_dir)

    if not session_id:
        sessions = store.list_sessions(limit=1)
        if not sessions:
            result["error"] = "No sessions found."
            return result
        session_id = str(sessions[0]["session_id"])
    result["session_id"] = session_id

    dedup_path = _dedup_path(memory)
    dedup_state = _load_dedup_state(dedup_path)
    last_ts = dedup_state.get(session_id)
    if last_ts is not None and (time.time() - float(last_ts)) < _DEDUP_WINDOW_S:
        result["skipped"] = True
        return result

    try:
        turns = store.get(session_id)[-n_turns:]
    except ValueError as exc:
        result["error"] = str(exc)
        return result
    if not turns:
        result["error"] = f"No turns found for session {session_id}."
        return result

    crystal = _synthesize_crystal(memory, session_id, turns)
    result["crystal"] = crystal

    if dry_run:
        result["ok"] = True
        return result

    text = _build_text(crystal)
    tags = [*(crystal.get("tags") or [])[:8], "session-crystal"]
    record = memory.save(
        content=text,
        title=str(crystal.get("title") or "")[:120] or None,
        type_="decision",
        tags=tags,
    )

    result["memory_id"] = record.id
    result["ok"] = True

    dedup_state[session_id] = time.time()
    _write_dedup_state(dedup_path, dedup_state)

    return result


__all__ = ["crystallize_session"]
