"""Terminal dashboard — `memo tui`.

Live, read-only view over the memo corpus and runtime state. Uses Rich
Live (already a dep) so there's no Textual / curses overhead. Refresh
every `--refresh` seconds (default 1.0). Press Ctrl+C to exit.

Panels
------
- **corpus** — total memorias, breakdown by type, distinct project tags
- **runtime** — MLX model warm/cold flags, vault path + size, watcher
  status (launchd job)
- **recent saves** — last N entries from `history.db` (op=save)
- **recent recalls** — last N entries from the recall log JSONL
  (`~/.local/share/memo/recall.log`), written by `memo recall-hook`
- **top tags** — most frequent tags across the corpus
- **activity** — sparklines of saves/day and recalls/day

The dashboard is intentionally allocation-cheap on the refresh path —
the heavy work (corpus scan, launchctl probe) is sampled at a slower
cadence than the UI tick so a 1 s refresh doesn't thrash the disk.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import subprocess
import threading
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Unicode block characters for sparklines (8 levels, low→high).
_SPARK = "▁▂▃▄▅▆▇█"
_WATCH_LABEL = "com.fer.memo.watch"


# -------------------- recall log (write side) --------------------

def recall_log_path(state_dir: Path) -> Path:
    return state_dir / "recall.log"


def _bail_breakdown(bail_rows: list[dict[str, Any]]) -> dict[str, int]:
    """Classify bail rows by cause so 'bailed=40%' isn't opaque.

    Intentional bails (slash_command, short_prompt) are expected noise.
    Lossy bails (min_sim) indicate the score floor is too aggressive.
    """
    counts: dict[str, int] = {"slash_command": 0, "short_prompt": 0, "min_sim": 0, "other": 0}
    for row in bail_rows:
        reason = row.get("reason", "")
        if "slash command" in reason:
            counts["slash_command"] += 1
        elif "too short" in reason:
            counts["short_prompt"] += 1
        elif "min_sim" in reason or "no hits above" in reason:
            counts["min_sim"] += 1
        else:
            counts["other"] += 1
    return counts


def recall_health(state_dir: Path, *, limit: int = 200) -> dict[str, Any]:
    """Summarise the recall ring buffer — is memo actually consulted and
    returning confident hits, or a write-only store nobody reads?

    Computed purely from the existing recall.log (no hot-path cost):
      - sampled:          events considered (newest `limit`)
      - fired / bailed:   recall ran vs. skipped (short/slash/empty prompt)
      - bail_breakdown:   bailed counts by cause (slash_command, short_prompt,
                          min_sim — lossy, other). min_sim > 0 means hits exist
                          but all fell below the floor; the others are intentional.
      - hit_rate:         share of *fired* recalls that surfaced ≥1 memoria
      - strong_hit_rate:  share of *fired* recalls with a top score > 0.85
                          (high-confidence match — the honest relevance number)
      - grounded_rate:    PRIMARY usefulness metric. Share of surfaced memorias
                          that the Stop-hook grounding detector found in the answer.
                          90%+ is excellent. Requires correlatable session_id+turn rows.
      - referenced_rate:  lower bound on "used" — requires an explicit MCP fetch or
                          click event after recall, which almost never happens in normal
                          flow. Expect 0; grounded_rate is the real signal.
      - median_top_score: median best-hit score on hit events (confidence)
      - p50_latency_ms:   median end-to-end recall latency

    Double-fire pairs (same prompt logged subprocess+daemon) are collapsed
    first so totals and rates aren't inflated.
    """
    rows = dedup_double_fire(read_recall_log(state_dir, limit=limit))
    fired = [r for r in rows if r.get("via") in ("daemon", "subprocess")]
    bail_rows = [r for r in rows if r.get("via") == "bail"]
    bailed = len(bail_rows)
    with_hits = [r for r in fired if r.get("hits")]

    top_scores: list[float] = []
    strong = 0
    for r in with_hits:
        hits = r.get("hits") or []
        score = hits[0].get("score") if hits else None
        if isinstance(score, (int, float)):
            top_scores.append(float(score))
            if float(score) > STRONG_SCORE:
                strong += 1
    top_scores.sort()
    lats = sorted(
        int(r["latency_ms"]) for r in fired
        if isinstance(r.get("latency_ms"), (int, float))
    )

    def _median(xs: list[Any]) -> Any:
        return xs[len(xs) // 2] if xs else None

    ref = referenced_rate(state_dir, rows)
    grounded = grounded_rate(state_dir, rows)

    return {
        "sampled": len(rows),
        "fired": len(fired),
        "bailed": bailed,
        "bail_breakdown": _bail_breakdown(bail_rows),
        "hit_rate": round(len(with_hits) / len(fired), 3) if fired else None,
        "strong_hit_rate": round(strong / len(fired), 3) if fired else None,
        # grounded_rate is the primary usefulness signal. referenced_rate is a
        # lower bound (almost always 0 — requires explicit MCP fetch events).
        "grounded_rate": grounded["grounded_rate"],
        "grounded": grounded["grounded"],
        "grounded_surfaced": grounded["surfaced"],
        "referenced_rate": ref["referenced_rate"],
        "referenced": ref["referenced"],
        "surfaced": ref["surfaced"],
        "median_top_score": round(_median(top_scores), 3) if top_scores else None,
        "p50_latency_ms": _median(lats),
    }


# Consumers the source-of-truth contract expects to read memo. A consumer
# that never appears in the consult log is a silent gap, not "no data" —
# surface it explicitly so "nobody reads memo" can't hide behind an empty
# table. (See CLAUDE.md "Source of truth — role & contract".)
EXPECTED_CONSUMERS = ("claude-code", "synapse", "memflow", "codex", "devin", "windsurf")


def consumer_label(row: dict[str, Any]) -> str:
    """Classify a recall-log row by the consumer that issued the consult.

    The recall-hook (Claude Code) tags rows with `via` in daemon/subprocess;
    MCP tools tag `via=mcp:<tool>` and an optional `source` naming the caller
    (synapse/memflow/agent). We collapse to a stable consumer name so the
    usefulness report answers "who actually reads memo".
    """
    # Explicit client tag (set by the recall-hook / installers) wins — it names
    # the actual front-end (claude-code/codex/devin/windsurf) rather than the
    # transport. Fall back to source (synapse/memflow) then via.
    client = (row.get("client") or "").strip().lower()
    if client:
        return client
    source = (row.get("source") or "").strip().lower()
    if source:
        return source
    via = (row.get("via") or "").strip().lower()
    if via in ("daemon", "subprocess", "daemon_error"):
        return "claude-code"
    if via.startswith("mcp:"):
        return "mcp:unknown"
    if via == "bail":
        return "claude-code"  # bails are recall-hook skips
    return "unknown"


def consult_breakdown(state_dir: Path, *, limit: int = 500) -> dict[str, Any]:
    """Per-consumer usefulness summary over the consult ring buffer.

    Answers the question `memo usefulness` exists for: who consults memo,
    how often, with what hit rate, and — critically — which expected
    consumers are silent. Pure read over the existing recall.log; no
    hot-path cost.
    """
    rows = dedup_double_fire(read_recall_log(state_dir, limit=limit))
    # Grounded keys (session_id, turn, recall_id) from the grounding ledger, so
    # each consumer gets an outcome-based "did calling memo help" number, not
    # just a hit rate.
    grounded_keys: set[tuple[str, int, str]] = set()
    for g in read_grounding_log(state_dir):
        sid = g.get("session_id")
        turn = g.get("turn")
        rid = g.get("recall_id")
        score = g.get("used_score")
        if (
            sid and isinstance(turn, int) and rid
            and isinstance(score, (int, float)) and float(score) >= GROUNDED_SCORE
        ):
            grounded_keys.add((sid, turn, rid))

    by: dict[str, dict[str, Any]] = {}
    for r in rows:
        name = consumer_label(r)
        agg = by.setdefault(
            name,
            {
                "consults": 0, "fired": 0, "with_hits": 0, "strong": 0,
                "top_scores": [], "last_seen": None,
                "surfaced": set(), "grounded": set(),
            },
        )
        agg["consults"] += 1
        is_bail = (r.get("via") or "") == "bail"
        if not is_bail:
            agg["fired"] += 1
            hits = r.get("hits") or []
            if hits:
                agg["with_hits"] += 1
                score = hits[0].get("score")
                if isinstance(score, (int, float)):
                    agg["top_scores"].append(float(score))
                    if float(score) > STRONG_SCORE:
                        agg["strong"] += 1
            # Per-consumer grounded join (only correlatable rows).
            sid = r.get("session_id")
            turn = r.get("turn")
            if sid and isinstance(turn, int):
                for h in hits:
                    hid = h.get("id")
                    if not hid:
                        continue
                    key = (sid, turn, hid)
                    agg["surfaced"].add(key)
                    if key in grounded_keys:
                        agg["grounded"].add(key)
        ts = r.get("ts")
        if ts and (agg["last_seen"] is None or ts > agg["last_seen"]):
            agg["last_seen"] = ts

    consumers: list[dict[str, Any]] = []
    for name, agg in by.items():
        scores = sorted(agg["top_scores"])
        fired = agg["fired"]
        n_surfaced = len(agg["surfaced"])
        consumers.append({
            "consumer": name,
            "consults": agg["consults"],
            "fired": fired,
            "bailed": agg["consults"] - fired,
            "hit_rate": round(agg["with_hits"] / fired, 3) if fired else None,
            "strong_hit_rate": round(agg["strong"] / fired, 3) if fired else None,
            "grounded_rate": round(len(agg["grounded"]) / n_surfaced, 3) if n_surfaced else None,
            "grounded_surfaced": n_surfaced,
            "median_top_score": round(scores[len(scores) // 2], 3) if scores else None,
            "last_seen": agg["last_seen"],
        })
    consumers.sort(key=lambda c: c["consults"], reverse=True)
    silent = [c for c in EXPECTED_CONSUMERS if c not in by]
    return {"sampled": len(rows), "consumers": consumers, "silent": silent}


def append_recall_log(
    state_dir: Path,
    *,
    prompt: str,
    hits: list[dict[str, Any]],
    cap: int = 200,
    mode: str | None = None,
    latency_ms: int | None = None,
    via: str | None = None,
    source: str | None = None,
    reason: str | None = None,
    error: str | None = None,
    session_id: str | None = None,
    turn: int | None = None,
    client: str | None = None,
) -> None:
    """Append a recall event to the JSONL ring buffer. Rotates by
    keeping only the most recent `cap` lines after writing. Errors are
    swallowed — the recall hook must never fail because of telemetry.

    `session_id` + `turn` are the correlation keys that let the Stop-hook
    grounding detector match this recall to the answer of the same exchange
    (see `grounding.py`). `hits[].snippet` caches the recall-block text so
    grounding needs no store read. All three are optional and additive:
    pre-correlation rows stay valid (counted in access stats, excluded from
    utility denominators).
    """
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        entry: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "prompt": prompt[:200],
            "hits": [
                {
                    "id": h.get("id", "")[:8],
                    "score": h.get("score"),
                    "title": h.get("title", "")[:80],
                    **({"snippet": h["snippet"][:240]} if h.get("snippet") else {}),
                }
                for h in hits[:5]
            ],
        }
        if mode is not None:
            entry["mode"] = mode
        if latency_ms is not None:
            entry["latency_ms"] = latency_ms
        if via is not None:
            entry["via"] = via
        if source is not None:
            entry["source"] = source
        if reason is not None:
            entry["reason"] = reason[:200]
        if error is not None:
            entry["error"] = error[:200]
        if session_id is not None:
            entry["session_id"] = session_id
        if turn is not None:
            entry["turn"] = turn
        if client is not None:
            entry["client"] = client
        path = recall_log_path(state_dir)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        # Rotate when the file gets too big (cheap line count). Skip
        # rotation 99% of the time — only fire when over cap*1.5.
        if path.stat().st_size > 1024 * 200:  # ~200 KB
            lines = path.read_text(encoding="utf-8").splitlines()[-cap:]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as exc:
        _log.debug("dashboard: log trim failed for %s: %s", path, exc)


def read_recall_log(state_dir: Path, *, limit: int = 10) -> list[dict[str, Any]]:
    path = recall_log_path(state_dir)
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        _log.debug("dashboard: log read failed for %s: %s", path, exc)
        return []
    out: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    out.reverse()  # newest first
    return out


# Confidence floor above which a hit counts as a STRONG match (high-cosine,
# genuinely on-topic) rather than a weak grasp. median_top_score in the wild
# sits ~0.84, so ~half of "hits" are weak — strong_hit_rate is the honest
# "when memo fires, is it actually relevant" number.
STRONG_SCORE = 0.85


def _parse_ts(value: Any) -> Any:
    from datetime import datetime
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _row_quality(row: dict[str, Any]) -> tuple[int, float]:
    hits = row.get("hits") or []
    top = hits[0].get("score") if hits else None
    return (len(hits), float(top) if isinstance(top, (int, float)) else 0.0)


def dedup_double_fire(rows: list[dict[str, Any]], *, window_s: float = 15.0) -> list[dict[str, Any]]:
    """Collapse double-fire pairs into one logical consult.

    The same prompt logged twice seconds apart (subprocess fallback + the
    daemon that finished late) is ONE consult, not two — counting both inflates
    consult totals and skews hit_rate. Merge rows with an identical non-empty
    prompt within `window_s`, keeping the higher-quality row (more hits, then
    higher top score). Empty-prompt bails (distinct slash-command skips) are
    never merged. Pre-`feat/recall-productive` logs contain these pairs; the
    fix stops new ones, this keeps historical reports honest.
    """
    kept: list[dict[str, Any]] = []
    anchor: dict[str, tuple[int, Any]] = {}  # prompt -> (kept index, ts)
    for r in rows:
        prompt = (r.get("prompt") or "").strip()
        if not prompt:
            kept.append(r)
            continue
        ts = _parse_ts(r.get("ts"))
        prev = anchor.get(prompt)
        if (
            prev is not None
            and ts is not None
            and prev[1] is not None
            and abs((ts - prev[1]).total_seconds()) <= window_s
        ):
            idx = prev[0]
            if _row_quality(r) > _row_quality(kept[idx]):
                kept[idx] = r
            anchor[prompt] = (idx, ts)
            continue
        kept.append(r)
        anchor[prompt] = (len(kept) - 1, ts)
    return kept


# -------------------- usage log (closed-loop "used" signal) --------------------
# recall.log proves a memoria was SHOWN. This proves one was acted on: every
# `memory_get` / `memo get` / contextual click appends {ts, id} here. Cross-
# referenced against recall.log, it yields referenced_rate — a LOWER BOUND on
# usefulness (explicit fetch-through only; the model usually consumes the
# injected recall text inline without a fetch, so true "used" is higher).

def usage_log_path(state_dir: Path) -> Path:
    return state_dir / "usage.log"


def append_usage_log(state_dir: Path, memoria_id: str, *, cap: int = 500) -> None:
    """Record that a memoria was fetched/clicked. Best-effort, never raises."""
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "id": (memoria_id or "")[:8],
        }
        path = usage_log_path(state_dir)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if path.stat().st_size > 1024 * 100:  # ~100 KB
            lines = path.read_text(encoding="utf-8").splitlines()[-cap:]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as exc:
        _log.debug("dashboard: log trim failed for %s: %s", path, exc)


def read_usage_log(state_dir: Path, *, limit: int = 2000) -> list[dict[str, Any]]:
    path = usage_log_path(state_dir)
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        _log.debug("dashboard: log read failed for %s: %s", path, exc)
        return []
    out: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def referenced_rate(state_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fraction of surfaced memorias later fetched/clicked (lower bound on use).

    For each distinct memoria id surfaced in `rows`, record its earliest
    surfacing time; count it referenced if usage.log has a fetch of that id at
    or after that time. Returns the rate plus raw counts so the caller can
    label it as the floor it is.
    """
    surfaced: dict[str, Any] = {}
    for r in rows:
        ts = _parse_ts(r.get("ts"))
        for h in r.get("hits") or []:
            hid = h.get("id")
            if not hid:
                continue
            cur = surfaced.get(hid)
            if cur is None or (ts is not None and (cur is None or ts < cur)):
                surfaced[hid] = ts
    if not surfaced:
        return {"referenced_rate": None, "surfaced": 0, "referenced": 0}
    uses: dict[str, list[Any]] = {}
    for u in read_usage_log(state_dir):
        uid = u.get("id")
        if uid:
            uses.setdefault(uid, []).append(_parse_ts(u.get("ts")))
    referenced = 0
    for hid, sts in surfaced.items():
        for uts in uses.get(hid, []):
            if sts is None or uts is None or uts >= sts:
                referenced += 1
                break
    return {
        "referenced_rate": round(referenced / len(surfaced), 3),
        "surfaced": len(surfaced),
        "referenced": referenced,
    }


# -------------------- grounding log (outcome-based "used in answer") --------------------
# referenced_rate counts explicit fetches (a floor). grounding.log closes the
# loop: at Stop the grounding detector (see grounding.py) compares the generated
# answer against the memorias recalled that turn and writes a per-memoria
# `used_score`. grounded_rate is the outcome-based successor — "was this actually
# used", not just "was it shown". Keyed by (session_id, turn, recall_id), it joins
# recall.log on the same key.

# used_score floor above which a recalled memoria counts as grounded-in-answer.
GROUNDED_SCORE = 0.5


def grounding_log_path(state_dir: Path) -> Path:
    return state_dir / "grounding.log"


def append_grounding_log(
    state_dir: Path,
    *,
    session_id: str,
    turn: int,
    recall_id: str,
    used_score: float,
    method: str,
    client: str | None = None,
    answer_len: int | None = None,
    recall_top_score: float | None = None,
    downstream_action: str | None = None,
    action_evidence: str | None = None,
    cap: int = 1000,
) -> None:
    """Append one grounding event (recall→use) to the JSONL ring buffer.
    Best-effort, never raises — Stop-hook telemetry must not fail the turn."""
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        entry: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "session_id": session_id,
            "turn": int(turn),
            "recall_id": (recall_id or "")[:8],
            "used_score": round(float(used_score), 4),
            "method": method,
        }
        if client is not None:
            entry["client"] = client
        if answer_len is not None:
            entry["answer_len"] = int(answer_len)
        if recall_top_score is not None:
            entry["recall_top_score"] = round(float(recall_top_score), 4)
        if downstream_action is not None:
            entry["downstream_action"] = downstream_action
        if action_evidence is not None:
            entry["action_evidence"] = action_evidence[:200]
        path = grounding_log_path(state_dir)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if path.stat().st_size > 1024 * 200:  # ~200 KB
            lines = path.read_text(encoding="utf-8").splitlines()[-cap:]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as exc:
        _log.debug("dashboard: log trim failed for %s: %s", path, exc)


def read_grounding_log(state_dir: Path, *, limit: int = 4000) -> list[dict[str, Any]]:
    path = grounding_log_path(state_dir)
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        _log.debug("dashboard: log read failed for %s: %s", path, exc)
        return []
    out: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def grounded_rate(state_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fraction of CORRELATABLE surfaced memorias that were grounded in the
    answer (used_score >= GROUNDED_SCORE). Outcome-based successor to
    referenced_rate.

    Denominator = distinct (session_id, turn, id) surfacings among `rows` that
    carry a session_id (old rows without correlation keys are excluded so they
    don't dilute the rate). A surfacing is grounded if grounding.log has an event
    for that exact (session_id, turn, recall_id) with used_score >= GROUNDED_SCORE.
    """
    surfaced: set[tuple[str, int, str]] = set()
    for r in rows:
        sid = r.get("session_id")
        turn = r.get("turn")
        if not sid or not isinstance(turn, int):
            continue
        for h in r.get("hits") or []:
            hid = h.get("id")
            if hid:
                surfaced.add((sid, turn, hid))
    if not surfaced:
        return {"grounded_rate": None, "surfaced": 0, "grounded": 0}
    grounded_keys: set[tuple[str, int, str]] = set()
    for g in read_grounding_log(state_dir):
        sid = g.get("session_id")
        turn = g.get("turn")
        rid = g.get("recall_id")
        score = g.get("used_score")
        if (
            sid and isinstance(turn, int) and rid
            and isinstance(score, (int, float)) and float(score) >= GROUNDED_SCORE
        ):
            grounded_keys.add((sid, turn, rid))
    grounded = sum(1 for k in surfaced if k in grounded_keys)
    return {
        "grounded_rate": round(grounded / len(surfaced), 3),
        "surfaced": len(surfaced),
        "grounded": grounded,
    }


# -------------------- re-ask avoidance (P1, signal b) --------------------
# A grounded recall that the user did NOT have to re-ask = a derivation memo
# saved. We measure it lexically (no MLX in the report path): for each recall
# that grounded ≥1 memoria, look at the same session's LATER prompts within a
# turn window; a near-duplicate follow-up = a re-ask (memo did NOT save it),
# its absence = a saved re-derivation. Pure read over recall.log + grounding.log.

_REASK_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")
_REASK_STOP = frozenset(
    ["the", "and", "for", "that", "with", "this", "from", "have", "are", "was", "were", "has", "not", "but", "you", "your", "una", "los", "las", "del", "por", "con", "para", "como", "que", "esta", "este", "más"]
)


def _reask_tokens(text: str) -> set[str]:
    return {t for t in _REASK_TOKEN_RE.findall((text or "").lower()) if t not in _REASK_STOP}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def reask_stats(
    state_dir: Path,
    *,
    window_turns: int = 4,
    sim_threshold: float = 0.6,
    limit: int = 500,
) -> dict[str, Any]:
    """Re-derivations memo prevented. For each grounded recall, check whether a
    near-duplicate prompt recurs in the same session within `window_turns`:
      - recurs (lexical Jaccard ≥ sim_threshold) → re-ask (NOT avoided)
      - no recurrence → reask_avoided (memo saved the re-derivation)
    Only correlatable rows (session_id+turn) are considered.
    """
    rows = dedup_double_fire(read_recall_log(state_dir, limit=limit))
    # Which (session, turn) recalls grounded at least one memoria?
    grounded_turns: set[tuple[str, int]] = set()
    for g in read_grounding_log(state_dir):
        sid = g.get("session_id")
        turn = g.get("turn")
        score = g.get("used_score")
        if sid and isinstance(turn, int) and isinstance(score, (int, float)) and float(score) >= GROUNDED_SCORE:
            grounded_turns.add((sid, turn))

    # Per-session prompt timeline keyed by turn.
    by_session: dict[str, list[tuple[int, str]]] = {}
    for r in rows:
        sid = r.get("session_id")
        turn = r.get("turn")
        prompt = (r.get("prompt") or "").strip()
        if sid and isinstance(turn, int) and prompt:
            by_session.setdefault(sid, []).append((turn, prompt))
    for lst in by_session.values():
        lst.sort(key=lambda x: x[0])

    considered = 0
    reask = 0
    for sid, turn in grounded_turns:
        timeline = by_session.get(sid) or []
        this = next((p for t, p in timeline if t == turn), None)
        if not this:
            continue
        considered += 1
        this_tok = _reask_tokens(this)
        recurred = any(
            t > turn and (t - turn) <= window_turns
            and _jaccard(this_tok, _reask_tokens(p)) >= sim_threshold
            for t, p in timeline
        )
        if recurred:
            reask += 1
    reask_avoided = considered - reask
    return {
        "considered": considered,
        "reask": reask,
        "reask_avoided": reask_avoided,
        "reask_rate": round(reask / considered, 3) if considered else None,
    }


# -------------------- helpers --------------------

def sparkline(values: list[int], width: int = 12) -> str:
    """Render `values` as a unicode-block sparkline of `width` chars.

    Buckets the series into `width` slots (averaging when len > width,
    padding with the lowest-level char when len < width). All-zero
    series renders as ▁▁▁…, not blank, so the slot stays visually present.
    """
    if not values:
        return _SPARK[0] * width
    if len(values) > width:
        # Bucket into `width` chunks.
        step = len(values) / width
        buckets: list[float] = []
        for i in range(width):
            lo = int(i * step)
            bucket_hi = max(lo + 1, int((i + 1) * step))
            chunk = values[lo:bucket_hi]
            buckets.append(sum(chunk) / len(chunk) if chunk else 0)
        series: list[float] = buckets
    else:
        series = [float(v) for v in values]
        # Pad left with zeros so the latest sample sits at the right.
        series = [0.0] * (width - len(series)) + series

    hi = max(series) or 1.0
    out = []
    for v in series:
        idx = int((v / hi) * (len(_SPARK) - 1))
        out.append(_SPARK[max(0, min(len(_SPARK) - 1, idx))])
    return "".join(out)


def _human_age(ts: str | None) -> str:
    """Render an ISO ts as 'Nm ago' / 'Nh ago' / 'Nd ago'."""
    if not ts:
        return "—"
    try:
        # Accept both Z and +00:00 forms; strip any trailing Z.
        t = ts.rstrip("Z")
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        delta = datetime.now(UTC) - dt
        secs = int(delta.total_seconds())
    except (ValueError, TypeError):
        return "—"
    if secs < 0:
        return "now"
    if secs < 60:
        return f"{secs}s ago"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"


def _human_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    if n < 1024 ** 3:
        return f"{n / (1024 ** 2):.1f} MB"
    return f"{n / (1024 ** 3):.2f} GB"


def _dir_size(p: Path) -> int:
    if not p.is_dir():
        return 0
    total = 0
    for f in p.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            continue
    return total


def _watcher_status() -> tuple[bool, str]:
    """Return (running, status_text). `running` is True iff the launchd
    job is loaded; status_text is a one-liner suitable for display."""
    uid = os.getuid()
    target = f"gui/{uid}/{_WATCH_LABEL}"
    res = subprocess.run(
        ["launchctl", "print", target],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        return False, "not installed (memo install-watcher)"
    out = res.stdout
    state = "running"
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("state ="):
            state = line.split("=", 1)[1].strip()
            break
    return True, state


# -------------------- panel builders --------------------

def _panel_corpus(memory: Any) -> Panel:
    types_counter: Counter[str] = Counter()
    projects: set[str] = set()
    # Avoid loading every body — list_recent yields meta-only rows from
    # the store layer (path/title/type/tags/created/updated).
    rows = memory.store.list_recent(limit=10_000)
    for r in rows:
        types_counter[r["type"]] += 1
        for t in r["tags"] or []:
            if t.startswith("project:"):
                projects.add(t)

    total = sum(types_counter.values())
    top3 = types_counter.most_common(3)
    types_line = "  ".join(f"[bold]{n}[/bold] {t}" for t, n in top3) or "—"
    body = Text.from_markup(
        f"[bold cyan]{total}[/bold cyan] memorias  ·  "
        f"[bold cyan]{len(projects)}[/bold cyan] proj  ·  {types_line}",
    )
    return Panel(body, title="[bold magenta]corpus[/bold magenta]",
                 border_style="magenta", padding=(0, 1))


def _panel_runtime(memory: Any) -> Panel:
    cfg = memory.cfg
    embedder_warm = memory.embedder._model is not None  # type: ignore[attr-defined]
    chat_warm = bool(getattr(getattr(memory, "_chat", None), "_loaded", None))
    rerank_warm = False
    try:
        rr = getattr(memory, "_reranker", None)
        if rr is not None:
            rerank_warm = bool(getattr(rr, "_model", None))
    except Exception as exc:
        _log.debug("dashboard: reranker warm probe failed: %s", exc)

    vault_size = _dir_size(cfg.memory_dir)
    watcher_loaded, watcher_state = _watcher_status()

    def _dot(ok: bool, label: str) -> str:
        return (
            f"[bold green]●[/bold green] {label}"
            if ok else f"[dim]○ {label}[/dim]"
        )

    mlx_line = "  ".join([
        _dot(embedder_warm, "emb"),
        _dot(rerank_warm, "rrk"),
        _dot(chat_warm, "chat"),
    ])
    watcher_line = (
        f"[green]✓ {watcher_state}[/green]"
        if watcher_loaded
        else f"[yellow]{watcher_state}[/yellow]"
    )
    body = Text.from_markup(
        f"{mlx_line}  ·  [cyan]{_human_bytes(vault_size)}[/cyan]  ·  "
        f"{watcher_line}",
    )
    return Panel(body, title="[bold blue]runtime[/bold blue]",
                 border_style="blue", padding=(0, 1))


def _panel_recent_saves(memory: Any, limit: int = 10) -> Panel:
    events = memory.history.list_recent(limit=limit, op="save")
    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(style="dim", width=8)
    tbl.add_column(style="bold yellow", width=10)
    tbl.add_column(style="bold")
    tbl.add_column(style="cyan", width=10)
    if not events:
        tbl.add_row("—", "—", Text("(no saves yet)", style="dim italic"), "—")
    for ev in events:
        tbl.add_row(
            _human_age(ev.get("ts")),
            "[" + (ev.get("record_id") or "")[:8] + "]",
            (ev.get("title") or "")[:60],
            ev.get("type") or "",
        )
    return Panel(tbl, title="[bold yellow]recent saves[/bold yellow]",
                 border_style="yellow")


def _daemon_status(state_dir: Path) -> str:
    """Return a one-line daemon status string for the TUI panels."""
    try:
        from memo.recall_server import _is_pid_alive, _read_pid
        pid = _read_pid(state_dir)
        running = pid is not None and _is_pid_alive(pid)
    except (OSError, ValueError):
        running = False

    try:
        warm_signal = state_dir / ".prewarm_ts"
        warm = (
            warm_signal.exists()
            and (time.time() - float(warm_signal.read_text().strip())) < 3600
        )
    except (OSError, ValueError):
        warm = False

    daemon_label = "[green]running[/green]" if running else "[dim]off[/dim]"
    warm_label = "[green]warm[/green]" if warm else "[yellow]cold[/yellow]"
    return f"daemon: {daemon_label} | {warm_label}"


def _panel_recent_recalls(state_dir: Path, limit: int = 8) -> Panel:
    entries = read_recall_log(state_dir, limit=limit)
    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(style="dim", width=8)
    tbl.add_column(style="cyan", width=6)   # mode column
    tbl.add_column()
    if not entries:
        tbl.add_row("—", "—", Text("(no recalls logged yet)", style="dim italic"))
    for e in entries:
        prompt = (e.get("prompt") or "").replace("\n", " ")[:60]
        hits = e.get("hits") or []
        mode_val = e.get("mode") or "—"
        scores = ", ".join(
            f"{h.get('score', 0):.2f}" for h in hits if h.get("score") is not None
        )
        if scores:
            line = Text.assemble(
                ("\"" + prompt + "\"", "white"),
                ("  → ", "dim"),
                (f"{len(hits)} hits", "bold cyan"),
                ("  @ ", "dim"),
                (scores, "magenta"),
            )
        else:
            line = Text.assemble(
                ("\"" + prompt + "\"", "white"),
                ("  → no hits", "dim"),
            )
        tbl.add_row(_human_age(e.get("ts")), mode_val, line)

    status_line = _daemon_status(state_dir)
    title = f"[bold green]recent recalls[/bold green]  [{status_line}]"
    return Panel(tbl, title=title, border_style="green")


def _panel_top_tags(memory: Any, limit: int = 8) -> Panel:
    rows = memory.store.list_recent(limit=10_000)
    counter: Counter[str] = Counter()
    for r in rows:
        for t in r["tags"] or []:
            counter[t] += 1
    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(style="cyan")
    tbl.add_column(justify="right", style="bold")
    if not counter:
        tbl.add_row(Text("(no tags)", style="dim italic"), "")
    for tag, n in counter.most_common(limit):
        style = "bold magenta" if tag.startswith("project:") else "cyan"
        tbl.add_row(Text(tag, style=style), str(n))
    return Panel(tbl, title="[bold cyan]top tags[/bold cyan]",
                 border_style="cyan")


def _panel_activity(memory: Any, state_dir: Path) -> Panel:
    # saves: aggregate `events.ts` per day for the last 14 days.
    events = memory.history.list_recent(limit=2000, op="save")
    buckets: dict[str, int] = {}
    now = datetime.now(UTC)
    days = 14
    for i in range(days):
        d = (now - _td(i)).date().isoformat()
        buckets[d] = 0
    for ev in events:
        ts = ev.get("ts") or ""
        try:
            day = datetime.fromisoformat(ts.rstrip("Z")).date().isoformat()
        except ValueError:
            continue
        if day in buckets:
            buckets[day] += 1
    saves_series = [buckets[k] for k in sorted(buckets.keys())]

    # recalls: same bucketing over recall.log
    recall_buckets: dict[str, int] = {k: 0 for k in buckets}
    for e in read_recall_log(state_dir, limit=1000):
        ts = e.get("ts") or ""
        try:
            day = datetime.fromisoformat(ts.rstrip("Z")).date().isoformat()
        except ValueError:
            continue
        if day in recall_buckets:
            recall_buckets[day] += 1
    recalls_series = [recall_buckets[k] for k in sorted(recall_buckets.keys())]

    body = Table.grid(padding=(0, 1))
    body.add_column(style="dim", width=14)
    body.add_column(style="bold", width=18)
    body.add_column(justify="right", style="cyan", width=10)
    body.add_row(
        f"saves/day ({days}d)",
        Text(sparkline(saves_series, width=14), style="yellow"),
        f"Σ {sum(saves_series)}",
    )
    body.add_row(
        f"recalls/day ({days}d)",
        Text(sparkline(recalls_series, width=14), style="green"),
        f"Σ {sum(recalls_series)}",
    )
    return Panel(body, title="[bold]activity[/bold]", border_style="bright_black")


def _td(days: int):
    from datetime import timedelta
    return timedelta(days=days)


# -------------------- main loop --------------------

def render(memory: Any, state_dir: Path) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="top", size=3),
        Layout(name="mid", size=8),
        Layout(name="bot", size=6),
        Layout(name="footer", size=1),
    )
    layout["top"].split_row(Layout(name="corpus"), Layout(name="runtime"))
    layout["mid"].split_row(Layout(name="saves"), Layout(name="recalls"))
    layout["bot"].split_row(Layout(name="tags"), Layout(name="activity"))

    layout["corpus"].update(_panel_corpus(memory))
    layout["runtime"].update(_panel_runtime(memory))
    layout["saves"].update(_panel_recent_saves(memory, limit=5))
    layout["recalls"].update(_panel_recent_recalls(state_dir, limit=4))
    layout["tags"].update(_panel_top_tags(memory, limit=4))
    layout["activity"].update(_panel_activity(memory, state_dir))
    now = datetime.now().strftime("%H:%M:%S")
    footer = Text.from_markup(
        f"[dim]memo · live  ·  {memory.cfg.memory_dir}  ·  [/dim][cyan]{now}[/cyan]"
        f"  [dim]·  [/dim][bold]q[/bold][dim] / [/dim][bold]ESC[/bold][dim] / Ctrl+C to quit[/dim]",
    )
    layout["footer"].update(Align.center(footer))
    return layout


def _spawn_key_reader(stop_event: threading.Event) -> None:
    """Background thread that reads single keystrokes from stdin and
    sets `stop_event` when the user presses `q`, `Q`, or ESC. Falls
    back gracefully when stdin isn't a TTY (CI, pipes) — the thread
    just exits and the user can still Ctrl+C.
    """
    import select
    import sys
    import termios
    import tty

    try:
        fd = sys.stdin.fileno()
        if not sys.stdin.isatty():
            return
        old = termios.tcgetattr(fd)
    except (OSError, termios.error):
        return

    try:
        tty.setcbreak(fd)
        while not stop_event.is_set():
            r, _, _ = select.select([fd], [], [], 0.25)
            if not r:
                continue
            ch = sys.stdin.read(1)
            if ch in ("q", "Q", "\x1b"):
                stop_event.set()
                return
    except Exception:
        # Stdin reads can fail during shutdown — never propagate.
        pass
    finally:
        with contextlib.suppress(Exception):
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def run_tui(*, refresh: float = 1.0, no_clear: bool = False) -> None:
    """Block on a Live dashboard until the user presses q / ESC / Ctrl+C."""
    import threading

    # The legacy-path warning from Memory() startup pollutes the alt
    # screen (and the user can't act on it from inside the TUI anyway).
    os.environ.setdefault("MEMO_SUPPRESS_LEGACY_WARN", "1")

    from memo.config import Config
    from memo.memory import Memory

    cfg = Config.from_env()
    mem = Memory(cfg)
    console = Console()

    stop = threading.Event()
    reader = threading.Thread(target=_spawn_key_reader, args=(stop,), daemon=True)
    reader.start()

    with Live(
        render(mem, cfg.state_dir),
        console=console,
        refresh_per_second=max(1.0 / refresh, 1.0),
        screen=not no_clear,
        transient=False,
    ) as live:
        try:
            while not stop.is_set():
                time.sleep(refresh)
                live.update(render(mem, cfg.state_dir))
        except KeyboardInterrupt:
            stop.set()
    # Reader thread will see stop and clean up termios on its own.
    reader.join(timeout=1.0)


# Keep `Group` importable for tests / external users of the layout
# (avoids an "unused import" lint when downstream users compose panels).
__all__ = [
    "Group",
    "append_recall_log",
    "read_recall_log",
    "recall_log_path",
    "render",
    "run_tui",
    "sparkline",
]
