"""Measured per-session token accounting from the Claude Code transcript.

Reads real `usage` (output_tokens) per assistant turn, joins with memo's
injection cost (context_cost.log) and grounding (grounding.log) by session_id,
and persists a durable per-session ledger. Runs in the Stop hook only — never
in the 5s recall hook. Pure stdlib + memo.dashboard_logs + memo.flags; no MLX,
no memo.memory.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

LEDGER_SCHEMA = "memo.token_meter.sessions.v1"


@dataclass(frozen=True)
class TurnUsage:
    index: int
    answer_tok: int
    tool_tok: int
    n_tool_steps: int


@dataclass(frozen=True)
class SessionUsage:
    session_id: str
    n_turns: int
    answer_tok: int
    tool_tok: int
    output_tok: int


def _is_human_prompt(row: dict) -> bool:
    """A real human prompt boundary (not a tool_result carrier)."""
    if row.get("type") != "user":
        return False
    content = (row.get("message") or {}).get("content")
    if isinstance(content, str):
        return True
    if isinstance(content, list):
        has_text = any(isinstance(b, dict) and b.get("type") == "text" for b in content)
        has_tool_result = any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
        )
        return has_text and not has_tool_result
    return False


def _assistant_out(row: dict) -> tuple[str, int, bool] | None:
    """(message_id, output_tokens, has_tool_use) for a main-thread assistant row."""
    if row.get("type") != "assistant" or row.get("isSidechain"):
        return None
    msg = row.get("message") or {}
    usage = msg.get("usage") or {}
    mid = str(msg.get("id") or "")
    out = int(usage.get("output_tokens") or 0)
    content = msg.get("content")
    has_tool = isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") == "tool_use" for b in content
    )
    return mid, out, has_tool


def iter_prompt_turns(rows: list[dict]) -> list[TurnUsage]:
    """Split the transcript into prompt-turns; measure answer vs tool output.

    A turn spans one human prompt to the next. Assistant messages are deduped
    by message id (a message can span several JSONL rows sharing one usage).
    """
    turns: list[TurnUsage] = []
    idx = -1
    seen_ids: set[str] = set()
    # per-turn accumulator of (out, has_tool) in order, one per unique message id
    steps: list[tuple[int, bool]] = []

    def _flush() -> None:
        if idx < 0:
            return
        if not steps:
            turns.append(TurnUsage(idx, 0, 0, 0))
            return
        answer_tok = steps[-1][0]
        tool_tok = sum(o for o, _ in steps[:-1])
        n_tool_steps = sum(1 for _, t in steps if t)
        turns.append(TurnUsage(idx, answer_tok, tool_tok, n_tool_steps))

    for row in rows:
        if _is_human_prompt(row):
            _flush()
            idx += 1
            seen_ids = set()
            steps = []
            continue
        parsed = _assistant_out(row)
        if parsed is None or idx < 0:
            continue
        mid, out, has_tool = parsed
        if mid and mid in seen_ids:
            continue
        if mid:
            seen_ids.add(mid)
        steps.append((out, has_tool))
    _flush()
    return turns


def session_usage(transcript_path: Path) -> SessionUsage | None:
    """Parse a transcript file into a per-session usage aggregate."""
    p = Path(transcript_path).expanduser()
    if not p.is_file():
        return None
    rows: list[dict] = []
    sid = ""
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(o)
            sid = sid or str(o.get("sessionId") or "")
    except OSError:
        return None
    turns = iter_prompt_turns(rows)
    answer = sum(t.answer_tok for t in turns)
    tool = sum(t.tool_tok for t in turns)
    return SessionUsage(sid, len(turns), answer, tool, answer + tool)
