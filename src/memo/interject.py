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

from collections.abc import Callable
from datetime import UTC, datetime
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
