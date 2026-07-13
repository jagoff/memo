"""Phase 3 — INTERJECT: a sharpened, contradiction-gated variant of the guard.

The guard (guard.py) flags ANY prior decision/preference the prompt looks to be
reversing. INTERJECT narrows that to the cases worth interrupting for: a guard
candidate that is (a) at HIGH *calibrated* confidence (Phase-1 recalibrated_band)
AND (b) already flagged in the *persisted* contradiction store (an open/competing
pair from a prior nightly scan) — i.e. memo has independent evidence this prior
decision is contested, not just lexically reversed.

Surface constraints (honest):
- Rides UserPromptSubmit (the recall hook), the SAME surface the guard rides —
  memo has NO PostToolUse hook, so a true mid-turn interrupt is impossible; this
  is the closest faithful realization (a sharper pre-turn banner).
- Uses ONLY cheap, already-on-the-path signals: recalibrated_band (one
  mtime-cached read) + contradict_store.pairs_for_ids (one sqlite SELECT the
  recall path already runs). NO new embed, NO MLX, NO scan_corpus on the 5s hook.
- "Repeats a fixed bug" is NOT realized — memo has no bug-fixed status; interject
  covers the contradiction-of-decision case only.

Pure core: interject_candidates / interject_banner take injected ``band_of`` and
``disputed_ids`` seams (wired to recalibrated_band + pairs_for_ids by the
orchestrator), so the whole decision is unit-testable without MLX or the store.
Report-only: shadow-logs regardless of the enable flag; a human flips
MEMO_INTERJECT_ENABLED after reviewing ``memo interject shadow``.
"""

from __future__ import annotations

import json as _json
import os as _os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

INTERJECT_HEADER = "⚠ INTERJECT — a prior decision at high confidence conflicts"


def interject_candidates(
    prompt: str,
    hits: list[Any],
    *,
    sim_threshold: float,
    band_of: Callable[[Any], str],
    disputed_ids: set[str],
) -> list[Any]:
    """Guard candidates that ALSO clear the calibrated-confidence gate
    (``band_of(h) == "high"``) and the persisted-contradiction gate
    (``h.id in disputed_ids``). Reuses guard.guard_candidates for the
    reversal/type/score filter — never re-derives it."""
    from memo.guard import guard_candidates

    cands = guard_candidates(prompt, hits, sim_threshold=sim_threshold)
    return [
        h
        for h in cands
        if band_of(h) == "high" and (getattr(h, "id", "") or "") in disputed_ids
    ]


def interject_banner(
    prompt: str,
    hits: list[Any],
    *,
    sim_threshold: float,
    band_of: Callable[[Any], str],
    disputed_ids: set[str],
    top: int = 1,
) -> str | None:
    """⚠ INTERJECT banner naming the high-confidence, contested prior decision(s)
    the prompt looks to reverse. None when nothing clears both gates."""
    cand = interject_candidates(
        prompt, hits, sim_threshold=sim_threshold, band_of=band_of, disputed_ids=disputed_ids
    )[:top]
    if not cand:
        return None
    lines = [INTERJECT_HEADER]
    for h in cand:
        title = (getattr(h, "title", "") or "").strip()
        lines.append(f'  You decided [{getattr(h, "id", "?")}]: "{title}"')
    lines.append(
        "  memo has this on record as contested — confirm before overriding it."
    )
    return "\n".join(lines)


def shadow_record(prompt: str, ids: list[str], *, rendered: bool) -> dict[str, Any]:
    """One shadow-log entry: what interject WOULD (or did) fire on this turn.
    ``rendered`` distinguishes an actually-shown banner (flag on, in budget, not
    silenced) from a suppressed one (the shadow the human reviews)."""
    return {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "prompt": (prompt or "")[:200],
        "ids": ids,
        "rendered": bool(rendered),
    }


SHADOW_LOG = "interject_shadow.log"
_SHADOW_CAP = 1000
_SHADOW_SIZE_LIMIT = 1_000_000


def _marker_file(state_dir: Path, session_id: str) -> Path:
    return Path(state_dir) / ".interject_seen" / f"{session_id}.json"


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


def should_render(state_dir: Path, session_id: str, *, max_per_session: int) -> bool:
    """True when interject may render THIS turn: not silenced and under budget.
    Side-effect free (does not increment — call note_rendered on an actual fire)."""
    if max_per_session <= 0:
        return False
    m = _load_marker(state_dir, session_id)
    if m.get("silenced"):
        return False
    return int(m.get("count", 0)) < max_per_session


def note_rendered(state_dir: Path, session_id: str) -> None:
    m = _load_marker(state_dir, session_id)
    m["count"] = int(m.get("count", 0)) + 1
    _save_marker(state_dir, session_id, m)


def silence(state_dir: Path, session_id: str) -> None:
    """One-key silence: suppress all further interject fires for this session."""
    m = _load_marker(state_dir, session_id)
    m["silenced"] = True
    _save_marker(state_dir, session_id, m)


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
