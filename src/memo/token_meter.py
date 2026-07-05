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


from memo.dashboard_logs import read_context_cost_log, read_grounding_log  # noqa: E402
from memo.dashboard_metrics import GROUNDED_SCORE  # noqa: E402

_CHARS_PER_TOKEN = 4


def ledger_path(state_dir: Path) -> Path:
    return state_dir / "token_meter.json"


def _lock_path(state_dir: Path) -> Path:
    return state_dir / "token_meter.json.lock"


def _read_ledger(state_dir: Path) -> dict:
    path = ledger_path(state_dir)
    if not path.is_file():
        return {"schema": LEDGER_SCHEMA, "sessions": {}}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema": LEDGER_SCHEMA, "sessions": {}}
    if not isinstance(doc, dict) or not isinstance(doc.get("sessions"), dict):
        return {"schema": LEDGER_SCHEMA, "sessions": {}}
    return doc


def _write_ledger(state_dir: Path, ledger: dict) -> None:
    path = ledger_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(ledger, ensure_ascii=False, indent=2))
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _injected_chars_for(state_dir: Path, session_id: str) -> int:
    return sum(
        int(e.get("chars") or 0)
        for e in read_context_cost_log(state_dir)
        if e.get("kind") == "recall" and e.get("session_id") == session_id
    )


def _grounded_for(state_dir: Path, session_id: str) -> int:
    seen: set[tuple] = set()
    for r in read_grounding_log(state_dir):
        if r.get("session_id") != session_id:
            continue
        score = r.get("used_score")
        if isinstance(score, (int, float)) and float(score) >= GROUNDED_SCORE:
            seen.add((r.get("turn"), r.get("recall_id")))
    return len(seen)


def roll(state_dir: Path, session_id: str, transcript_path: str | Path | None) -> dict:
    """Fold this session's measured usage into the durable per-session ledger."""
    if not session_id or not transcript_path:
        return _read_ledger(state_dir)
    su = session_usage(Path(transcript_path))
    if su is None:
        return _read_ledger(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    with _lock_path(state_dir).open("a+", encoding="utf-8") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        ledger = _read_ledger(state_dir)
        from datetime import UTC, datetime

        ledger.setdefault("sessions", {})[session_id] = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "n_turns": su.n_turns,
            "answer_tok": su.answer_tok,
            "tool_tok": su.tool_tok,
            "injected_chars": _injected_chars_for(state_dir, session_id),
            "grounded": _grounded_for(state_dir, session_id),
        }
        _write_ledger(state_dir, ledger)
    return ledger


def summarize(state_dir: Path) -> dict:
    ledger = _read_ledger(state_dir)
    rows = list(ledger.get("sessions", {}).values())
    answer = sum(int(r.get("answer_tok", 0)) for r in rows)
    tool = sum(int(r.get("tool_tok", 0)) for r in rows)
    injected_chars = sum(int(r.get("injected_chars", 0)) for r in rows)
    grounded = sum(int(r.get("grounded", 0)) for r in rows)

    def _rate(subset: list[dict]) -> float | None:
        turns = sum(int(r.get("n_turns", 0)) for r in subset)
        tk = sum(int(r.get("tool_tok", 0)) for r in subset)
        return round(tk / turns, 2) if turns else None

    grounded_ss = [r for r in rows if int(r.get("grounded", 0)) > 0]
    ungrounded_ss = [
        r for r in rows if int(r.get("grounded", 0)) == 0 and int(r.get("injected_chars", 0)) > 0
    ]
    g_rate = _rate(grounded_ss)
    u_rate = _rate(ungrounded_ss)
    delta = round(u_rate - g_rate, 2) if (g_rate is not None and u_rate is not None) else None
    return {
        "schema": LEDGER_SCHEMA,
        "sessions": len(rows),
        "answer_tok": answer,
        "tool_tok": tool,
        "injected_tokens": (injected_chars + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN,
        "grounded": grounded,
        "proxy": {
            "grounded_tool_tok_per_turn": g_rate,
            "ungrounded_tool_tok_per_turn": u_rate,
            "delta": delta,
        },
        "ledger_path": str(ledger_path(state_dir)),
    }
