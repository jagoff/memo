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
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

LEDGER_SCHEMA = "memo.token_meter.sessions.v2"


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
    input_tok: int = 0
    cache_read_tok: int = 0
    cache_creation_tok: int = 0
    models: dict[str, int] | None = None


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


def _transcript_input_side(rows: list[dict]) -> tuple[int, int, int, dict[str, int]]:
    """Aggregate the prompt-side `usage` fields the transcript exposes.

    Claude Code stamps each assistant message with the API usage of its call:
    ``input_tokens`` (this call's full prompt footprint), and the two cache
    splits ``cache_read_input_tokens`` / ``cache_creation_input_tokens``.
    ``input_tokens`` is cumulative-per-call, so the session footprint is the
    MAX; the cache splits are per-call volumes, so they SUM (that is also
    what the provider bills against). Returns ``(input_max, cache_read_sum,
    cache_creation_sum, models)`` where ``models`` tallies output tokens per
    distinct model name — ccusage-style per-model accounting off the same
    transcript (Stop hook only; pure stdlib).

    Some Claude Code builds stamp a degenerate ``input_tokens`` (a fixed 1–2
    per call, while the cache splits stay sane). The cache splits are real
    billed volumes either way, so they are always summed; ``input_max`` is
    reported as 0 (unknown) when every call's footprint looks degenerate —
    honest zero beats a wrong number.
    """
    input_max = 0
    cache_read = 0
    cache_creation = 0
    models: dict[str, int] = {}
    seen_ids: set[str] = set()
    for row in rows:
        if row.get("type") != "assistant" or row.get("isSidechain"):
            continue
        msg = row.get("message") or {}
        usage = msg.get("usage") or {}
        model = str(msg.get("model") or "")
        if model == "<synthetic>":
            # Internal generations (titles, summaries) are not user-facing
            # spend; counting them inflates the answer/tool narrative.
            continue
        mid = str(msg.get("id") or "")
        if mid and mid in seen_ids:
            continue  # streaming rows repeat the same message's usage
        if mid:
            seen_ids.add(mid)
        input_max = max(input_max, int(usage.get("input_tokens") or 0))
        cache_read += int(usage.get("cache_read_input_tokens") or 0)
        cache_creation += int(usage.get("cache_creation_input_tokens") or 0)
        out = int(usage.get("output_tokens") or 0)
        if model:
            models[model] = models.get(model, 0) + out
    if input_max < 100:
        input_max = 0
    return input_max, cache_read, cache_creation, models


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
    input_max, cache_read, cache_creation, models = _transcript_input_side(rows)
    return SessionUsage(
        sid,
        len(turns),
        answer,
        tool,
        answer + tool,
        input_tok=input_max,
        cache_read_tok=cache_read,
        cache_creation_tok=cache_creation,
        models=models or None,
    )


from memo.dashboard_logs import (  # noqa: E402
    read_context_cost_log,
    read_grounding_log,
    read_recall_log,
)
from memo.dashboard_metrics import GROUNDED_SCORE  # noqa: E402

_CHARS_PER_TOKEN = 4

# ---------------------------------------------------------------------------
# Precision bands — learned score-band suppression (Task 9 / Lever 3)
# ---------------------------------------------------------------------------


def _band_key(score: float) -> str:
    """0.05-wide bucket key for a recall top-score. E.g. 0.63 → '0.60', 0.65 → '0.65'.

    Uses a small epsilon before floor to absorb FP rounding artifacts like
    ``0.55 / 0.05 == 10.999…`` which would otherwise mis-bucket an exact boundary.
    """
    idx = math.floor(score / 0.05 + 1e-9)
    return f"{round(idx * 0.05, 2):.2f}"


def suppress_score(top_score: float, bands: dict) -> bool:
    """Return True if the score's band is marked suppress=True in *bands*."""
    band = bands.get(_band_key(top_score))
    if band is None:
        return False
    return bool(band.get("suppress"))


def _precision_bands_path(state_dir: Path) -> Path:
    return state_dir / "precision_bands.json"


def load_precision_bands(state_dir: Path) -> dict:
    """Load the cached precision-bands dict (cheap read for the 5 s recall hook)."""
    p = _precision_bands_path(state_dir)
    if not p.is_file():
        return {}
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def learn_precision_bands(state_dir: Path, *, min_samples: int = 20) -> dict:
    """Bucket historical recall top-scores; flag bands with zero grounding.

    Runs in the Stop hook (reads capped JSONL logs).  Writes the result to
    ``state_dir/precision_bands.json`` so the recall hook can read it cheaply.
    Returns the bands dict.

    A band is marked ``suppress=True`` when it has ≥ *min_samples* recall
    events and zero of them were ever grounded (used_score ≥ GROUNDED_SCORE).
    """
    # Build the set of (session_id, turn) pairs that had a grounding event.
    grounded_turns: set[tuple] = set()
    for r in read_grounding_log(state_dir, limit=4000):
        sid = r.get("session_id")
        turn = r.get("turn")
        raw = r.get("used_score")
        if (
            sid
            and turn is not None
            and isinstance(raw, (int, float))
            and float(raw) >= GROUNDED_SCORE
        ):
            grounded_turns.add((sid, turn))

    # Bucket recall log entries by score band.
    # read_recall_log default limit=10 is for UI; use 200 (the log cap) here.
    band_total: dict[str, int] = {}
    band_grounded_count: dict[str, int] = {}

    for entry in read_recall_log(state_dir, limit=200):
        hits = entry.get("hits")
        if not hits:
            continue  # bail entries have hits=[]
        top_score = hits[0].get("score")
        if top_score is None or not isinstance(top_score, (int, float)):
            continue
        sid = entry.get("session_id")
        turn = entry.get("turn")
        if sid is None or turn is None:
            continue
        key = _band_key(float(top_score))
        band_total[key] = band_total.get(key, 0) + 1
        if (sid, turn) in grounded_turns:
            band_grounded_count[key] = band_grounded_count.get(key, 0) + 1

    # Build the result dict.
    bands: dict[str, dict] = {}
    for key in sorted(set(band_total) | set(band_grounded_count)):
        total = band_total.get(key, 0)
        grounded = band_grounded_count.get(key, 0)
        bands[key] = {
            "total": total,
            "grounded": grounded,
            "suppress": total >= min_samples and grounded == 0,
        }

    # Persist to cache (atomic replace).
    path = _precision_bands_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(bands, ensure_ascii=False, indent=2))
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise

    return bands


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

        # `_read_ledger` returns the on-disk doc verbatim, including a
        # `schema` field that may predate this code (e.g. a v1 ledger written
        # before the per-model/prompt-side fields below existed). The row
        # this call is about to write is always CURRENT-schema-shaped, so the
        # file as a whole is now at least that capable -- stamp the header
        # accordingly rather than leaving it to claim a stale version while
        # its rows have already moved on.
        ledger["schema"] = LEDGER_SCHEMA
        ledger.setdefault("sessions", {})[session_id] = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "n_turns": su.n_turns,
            "answer_tok": su.answer_tok,
            "tool_tok": su.tool_tok,
            "input_tok": su.input_tok,
            "cache_read_tok": su.cache_read_tok,
            "cache_creation_tok": su.cache_creation_tok,
            "models": su.models or {},
            "injected_chars": _injected_chars_for(state_dir, session_id),
            "grounded": _grounded_for(state_dir, session_id),
        }
        _write_ledger(state_dir, ledger)
        with contextlib.suppress(Exception):
            learn_precision_bands(state_dir)
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

    # `delta` is only the tool-loop half of the trade. Injecting a recall block
    # costs tokens on every turn memo fires, whether or not the answer used it,
    # so a savings claim that omits it is not a savings claim. Net is what the
    # user actually keeps.
    injected_tokens = (injected_chars + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN
    measured_turns = sum(int(r.get("n_turns", 0)) for r in rows)
    injected_per_turn = round(injected_tokens / measured_turns, 2) if measured_turns else None
    net_per_turn = (
        round(delta - injected_per_turn, 2)
        if (delta is not None and injected_per_turn is not None)
        else None
    )
    grounded_turns = sum(int(r.get("n_turns", 0)) for r in grounded_ss)
    ungrounded_turns = sum(int(r.get("n_turns", 0)) for r in ungrounded_ss)
    models: dict[str, int] = {}
    for r in rows:
        for model, out in (r.get("models") or {}).items():
            models[str(model)] = models.get(str(model), 0) + int(out)
    return {
        "schema": LEDGER_SCHEMA,
        "sessions": len(rows),
        "answer_tok": answer,
        "tool_tok": tool,
        "input_tok": sum(int(r.get("input_tok", 0)) for r in rows),
        "cache_read_tok": sum(int(r.get("cache_read_tok", 0)) for r in rows),
        "cache_creation_tok": sum(int(r.get("cache_creation_tok", 0)) for r in rows),
        "models": dict(sorted(models.items(), key=lambda kv: kv[1], reverse=True)),
        "injected_tokens": injected_tokens,
        "grounded": grounded,
        "proxy": {
            "grounded_tool_tok_per_turn": g_rate,
            "ungrounded_tool_tok_per_turn": u_rate,
            "delta": delta,
            "injected_tok_per_turn": injected_per_turn,
            "net_tok_per_turn": net_per_turn,
            "grounded_turns": grounded_turns,
            "ungrounded_turns": ungrounded_turns,
        },
        "ledger_path": str(ledger_path(state_dir)),
    }
