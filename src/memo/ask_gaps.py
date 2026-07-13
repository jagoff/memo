"""Phase 3 — ASK: promote AT MOST ONE recurring unmet gap per session into an
explicit question on the SessionStart briefing.

The nightly ``dream_anticipate`` pass already writes recurring knowledge gaps
memo could NOT answer into the dream receipt (``anticipated.gaps``), sorted by
frequency-then-recency, but nothing surfaces them to the user. This module reads
that receipt and, when enabled, renders the single highest-value gap as a
QUESTION in the briefing. It NEVER fabricates — a gap is only ever a prompt memo
already failed to answer from real usage; asking it back is honest.

Report-only: shadow-logs what it WOULD ask regardless of the enable flag; a
human flips MEMO_ASK_GAPS_ENABLED after reviewing ``memo ask shadow``. Deduped
per session (a session asks at most one NEW gap and never repeats one).
"""

from __future__ import annotations

import json as _json
import os as _os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SHADOW_LOG = "ask_shadow.log"
_SHADOW_CAP = 1000
_SHADOW_SIZE_LIMIT = 1_000_000


def pick_gap(gaps: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The highest-value gap — the receipt is already sorted (count, recency)."""
    return gaps[0] if gaps else None


def phrase_question(gap: dict[str, Any]) -> str:
    """Re-ask an unmet gap as a question. Pure string; never an invented answer."""
    prompt = str(gap.get("prompt") or "").strip().rstrip("?")
    count = int(gap.get("count", 1) or 1)
    return (
        f"❓ **memo keeps hitting a gap** ({count}×): {prompt}? — "  # noqa: RUF001
        "save what you know so it stops asking."
    )


def _gap_key(gap: dict[str, Any]) -> str:
    return " ".join(str(gap.get("prompt") or "").lower().split())


def _marker_file(state_dir: Path, session_id: str) -> Path:
    return Path(state_dir) / ".ask_gaps_seen" / f"{session_id}.json"


def _load_marker(state_dir: Path, session_id: str) -> dict[str, Any]:
    f = _marker_file(state_dir, session_id)
    if not f.is_file():
        return {}
    try:
        data = _json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _save_marker(state_dir: Path, session_id: str, marker: dict[str, Any]) -> None:
    f = _marker_file(state_dir, session_id)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".json.tmp")
    tmp.write_text(_json.dumps(marker), encoding="utf-8")
    _os.replace(tmp, f)


def already_asked(state_dir: Path, session_id: str, key: str) -> bool:
    return key in set(_load_marker(state_dir, session_id).get("asked", []))


def note_asked(state_dir: Path, session_id: str, key: str) -> None:
    m = _load_marker(state_dir, session_id)
    asked = set(m.get("asked", []))
    asked.add(key)
    m["asked"] = sorted(asked)
    _save_marker(state_dir, session_id, m)


def shadow_record(gap: dict[str, Any], *, rendered: bool) -> dict[str, Any]:
    return {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "prompt": str(gap.get("prompt") or "")[:200],
        "count": int(gap.get("count", 1) or 1),
        "rendered": bool(rendered),
    }


def _shadow_path(state_dir: Path) -> Path:
    return Path(state_dir) / SHADOW_LOG


def log_shadow(state_dir: Path, entry: dict[str, Any]) -> None:
    from memo.dashboard_logs import _write_jsonl_entry

    _write_jsonl_entry(
        _shadow_path(state_dir), entry, cap=_SHADOW_CAP, size_limit=_SHADOW_SIZE_LIMIT
    )


def read_shadow(state_dir: Path, *, limit: int = 1000) -> list[dict[str, Any]]:
    from memo.dashboard_logs import _read_jsonl

    return _read_jsonl(_shadow_path(state_dir), limit=limit, newest_first=True)


def _read_receipt_gaps(state_dir: Path) -> list[dict[str, Any]]:
    try:
        last = Path(state_dir) / "dream" / "last.json"
        if not last.is_file():
            return []
        data = _json.loads(last.read_text(encoding="utf-8"))
        gaps = ((data.get("anticipated") or {}).get("gaps")) or []
        return [g for g in gaps if isinstance(g, dict) and g.get("prompt")]
    except Exception:
        return []


def briefing_lines(cfg: Any, *, session_id: str) -> list[str]:
    """SessionStart render: at most one NEW high-value gap as a question. Never
    raises, never fabricates. Shadow-logs whether or not it renders."""
    try:
        from memo.flags import flag_bool

        gaps = _read_receipt_gaps(cfg.state_dir)
        gap = pick_gap(gaps)
        if gap is None:
            return []
        key = _gap_key(gap)
        if already_asked(cfg.state_dir, session_id, key):
            return []
        render = flag_bool("MEMO_ASK_GAPS_ENABLED")
        log_shadow(cfg.state_dir, shadow_record(gap, rendered=render))
        if not render:
            return []
        note_asked(cfg.state_dir, session_id, key)
        return [phrase_question(gap), ""]
    except Exception:
        return []
