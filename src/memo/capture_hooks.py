"""Hook integration for capture: Stop-event and incremental capture entry points.

This module contains the hook entry points and their supporting infrastructure:
- `run_capture`: Stop hook — fire once at session end
- `run_capture_incremental`: Incremental capture — periodic mid-session passes
- Watermark and state management for idempotence and throttling
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


def _state_file(state_dir: Path) -> Path:
    return state_dir / "last-capture.json"


def _load_state(state_dir: Path) -> dict[str, Any]:
    f = _state_file(state_dir)
    if not f.is_file():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(state_dir: Path, state: dict[str, Any]) -> None:
    # Atomic write: this file is shared by every session on the machine, so a
    # torn/partial write would clobber a sibling session's last-capture state.
    # Write to a .tmp then os.replace (mirrors session.py `_write`).
    state_dir.mkdir(parents=True, exist_ok=True)
    dest = _state_file(state_dir)
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    os.replace(tmp, dest)


def _watermark_file(state_dir: Path, session_id: str) -> Path:
    # session_id is a Claude Code UUID — filename-safe, matching how
    # session.py keys its per-session JSON files.
    return state_dir / ".capture_watermark" / f"{session_id}.json"


def _load_watermark(state_dir: Path, session_id: str) -> dict[str, Any]:
    """Per-session watermark, or {} on missing/corrupt (so a clobbered or
    hand-edited file degrades to a fresh full pass, never a crash)."""
    f = _watermark_file(state_dir, session_id)
    if not f.is_file():
        return {}
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _capture_lock_file(state_dir: Path, session_id: str) -> Path:
    return state_dir / ".capture_watermark" / f"{session_id}.capture.lock"


def _save_watermark(state_dir: Path, session_id: str, watermark: dict[str, Any]) -> None:
    f = _watermark_file(state_dir, session_id)
    f.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write (see _save_state): never leave a torn watermark behind.
    tmp = f.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(watermark), encoding="utf-8")
    os.replace(tmp, f)


def list_sessions_without_watermark(
    state_dir: Path,
    sessions: list[dict[str, Any]],
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return sessions that have no watermark (never captured).

    Filters a list of session dicts from list_sessions() against the
    watermark directory. Returns up to `limit` sessions that have never
    been captured, sorted by most recent first.
    """
    wm_dir = state_dir / ".capture_watermark"
    if not wm_dir.is_dir():
        return sessions[:limit]

    pending: list[dict[str, Any]] = []

    for sess in sessions:
        sid = sess.get("session_id")
        if not sid:
            continue
        wm_file = wm_dir / f"{sid}.json"
        if not wm_file.is_file():
            pending.append(sess)
            if len(pending) >= limit:
                break

    return pending


def incremental_tick_due(state_dir: Path, session_id: str, interval_s: int) -> bool:
    """True if at least `interval_s` seconds elapsed since this session's last
    incremental pass (the watermark `updated` stamp).

    Cheap by design — a small JSON read, no transcript parse and no Memory /
    MLX — so the per-prompt hook can call it on every prompt and bail when not
    due. `interval_s <= 0` disables the throttle (always due)."""
    if interval_s <= 0:
        return True
    import time

    wm = _load_watermark(state_dir, session_id)
    try:
        last = float(wm.get("updated", 0) or 0)
    except (TypeError, ValueError):
        last = 0.0
    return (time.time() - last) >= interval_s


def run_capture(
    transcript_path: Path,
    *,
    debug: bool = False,
) -> dict[str, Any]:
    """Top-level entry: read transcript, extract, dedup, save.
    Returns a result summary dict so the CLI can print + the tests can
    assert on counts. All errors absorbed (logged to stderr in debug).

    Env vars:
      MEMO_CAPTURE_CONTEXT_TURNS  — number of recent exchanges to include
          as context for the LLM (default 3). Higher = richer context but
          longer prompt; lower = cheaper, may miss multi-turn decisions.
      MEMO_CAPTURE_COOLDOWN_MIN   — minimum minutes between captures in the
          same session (default 0 = no cooldown). Set to e.g. 30 to avoid
          flooding the corpus during a long refactoring session.
    """
    import sys
    import time

    from memo.config import Config
    from memo.flags import flag_int
    from memo.memory import Memory

    from .capture_core import (
        _capture_provenance,
        _extract_and_save,
        _hash_assistant,
        _passes_prefilter,
        _read_recent_exchanges,
    )

    cfg = Config.from_env()
    state = _load_state(cfg.state_dir)

    # Cooldown: skip if we saved too recently.
    cooldown_min = float(flag_int("MEMO_CAPTURE_COOLDOWN_MIN") or 0)
    if cooldown_min > 0:
        last_save_ts = state.get("last_save_ts", 0.0)
        elapsed_min = (time.time() - float(last_save_ts)) / 60.0
        if elapsed_min < cooldown_min:
            if debug:
                print(
                    f"# memo capture: cooldown — {elapsed_min:.1f}m elapsed, need {cooldown_min}m",
                    file=sys.stderr,
                )
            return {"status": "cooldown"}

    context_turns = max(1, flag_int("MEMO_CAPTURE_CONTEXT_TURNS") or 3)
    pair = _read_recent_exchanges(transcript_path, n=context_turns)
    if pair is None:
        return {"status": "no_pair"}
    user_text, assistant_text = pair

    # Idempotence — hash the assistant text of the LAST turn only (not all
    # turns in the context window) so multi-turn context doesn't break the
    # duplicate-detection across successive Stop firings.
    last_pair = _read_recent_exchanges(transcript_path, n=1)
    h = _hash_assistant(last_pair[1] if last_pair else assistant_text)
    if state.get("last_hash") == h:
        return {"status": "duplicate_turn"}

    if not _passes_prefilter(assistant_text):
        # Stamp the state anyway so we don't keep re-checking the same turn.
        state["last_hash"] = h
        _save_state(cfg.state_dir, state)
        return {"status": "no_trigger"}

    # Lazy heavy imports: only paid past pre-filter.
    mem = Memory(cfg)
    # Claude Code transcripts are <session_id>.jsonl — the stem IS the session id.
    provenance = _capture_provenance(transcript_path.stem, transcript_path, h)
    try:
        result = _extract_and_save(
            mem, cfg, user_text, assistant_text, debug=debug, extra_base=provenance
        )
    finally:
        mem.close()

    state["last_hash"] = h
    if result["saved"]:
        state["last_save_ts"] = time.time()
    _save_state(cfg.state_dir, state)
    return {"status": "ok", **result}


# ── Incremental capture (mid-session) ───────────────────────────────────────
#
# The Stop hook only fires once, at session end. A long-running session can
# accumulate hours of durable insight that never reaches `.md` (and thus the
# local index + git sync) until Stop — and is lost entirely on a crash.
#
# Incremental capture closes that gap: a periodic, self-throttled pass mines
# only the NEW turns since a per-session watermark, reusing the exact
# extract → quality → dedup → save pipeline above. The watermark
# (`state_dir/.capture_watermark/<session_id>.json`) records how many
# (user, assistant) exchanges have been processed, so each pass is bounded to
# the turns added since the last pass and old turns are never reprocessed.


def run_capture_incremental(
    transcript_path: Path,
    session_id: str,
    *,
    debug: bool = False,
) -> dict[str, Any]:
    """Mine only NEW turns since this session's watermark, then advance it.

    Reuses the Stop-hook extract/dedup/save pipeline (`_extract_and_save`);
    the watermark is what makes it incremental. Bounded to the exchanges added
    since the previous pass; old turns are never reprocessed. Self-throttling
    is the caller's job (`incremental_tick_due`) — this always processes what
    is new. Soft-fail: returns a status dict, never raises.

    Statuses: ``no_pair`` (empty/unreadable transcript), ``no_new`` (watermark
    already current), ``no_trigger`` (new turns but no insight keyword), ``ok``.
    """
    import time

    from memo.config import Config
    from memo.memory import Memory

    from .capture_core import (
        _capture_provenance,
        _extract_and_save,
        _hash_assistant,
        _parse_exchanges,
        _passes_prefilter,
    )

    cfg = Config.from_env()

    # Cross-process lock: the idle-capture daemon and the MCP server's
    # _auto_capture can both run this for the same session concurrently. Without
    # a lock they race the load-watermark→process→stamp cycle and double-save.
    # Hold an exclusive, non-blocking flock for the whole cycle; if another
    # process already holds it, skip this run cleanly — it advances the
    # watermark, and the next due tick picks up anything newer.
    lock_path = _capture_lock_file(cfg.state_dir, session_id)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as lock_fh:
        try:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return {"status": "locked"}

        exchanges = _parse_exchanges(transcript_path)
        if not exchanges:
            return {"status": "no_pair"}

        total = len(exchanges)
        wm = _load_watermark(cfg.state_dir, session_id)
        try:
            start = int(wm.get("exchange_count", 0) or 0)
        except (TypeError, ValueError):
            start = 0
        # A negative watermark is invalid — reset to beginning.
        # A watermark ahead of the transcript means nothing new; clamp to total.
        if start < 0:
            start = 0
        elif start > total:
            start = total

        def _stamp() -> None:
            _save_watermark(
                cfg.state_dir,
                session_id,
                {"session_id": session_id, "exchange_count": total, "updated": time.time()},
            )

        new = exchanges[start:]
        if not new:
            _stamp()  # refresh `updated` so the throttle clock advances
            return {"status": "no_new", "exchange_count": total}

        combined_user = "\n\n---\n\n".join(u for u, _ in new)
        combined_assistant = "\n\n---\n\n".join(a for _, a in new)

        if not _passes_prefilter(combined_assistant):
            # Advance past these triggerless turns so we don't re-scan them.
            _stamp()
            return {"status": "no_trigger", "exchange_count": total}

        provenance = _capture_provenance(
            session_id, transcript_path, _hash_assistant(combined_assistant)
        )
        mem = Memory(cfg)
        try:
            result = _extract_and_save(
                mem, cfg, combined_user, combined_assistant, debug=debug, extra_base=provenance
            )
        finally:
            mem.close()
        _stamp()
        return {"status": "ok", "processed_turns": len(new), "exchange_count": total, **result}
