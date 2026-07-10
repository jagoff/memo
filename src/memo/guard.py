"""Guard — flag a prior decision the user looks to be reversing (pre-turn).

Pure Python, no LLM (v1). A cheap gate over already-ranked recall hits:
a hit is a guard candidate when it is a durable decision/preference, scored
above threshold, and the prompt carries a reversal signal. memo SURFACES the
prior decision; it never blocks or claims the user is wrong.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_GUARD_TYPES = frozenset({"decision", "preference"})

# Reversal signals — the user is changing a prior direction, not asking fresh.
_REVERSAL = re.compile(
    r"\b(instead|switch|actually|redo|revert|rather than|changed my mind|"
    r"no longer|en vez|en lugar|cambi|mejor us|revert[íi]|volv[ae]mos)\b",
    re.IGNORECASE,
)


def has_reversal_signal(prompt: str) -> bool:
    """True when the prompt reads like reversing a prior direction."""
    return bool(_REVERSAL.search(prompt or ""))


def guard_candidates(prompt: str, hits: list[Any], *, sim_threshold: float) -> list[Any]:
    """Prior decisions/preferences the prompt looks to be reversing, score-desc."""
    if not has_reversal_signal(prompt):
        return []
    cand = [
        h
        for h in hits
        if getattr(h, "type", None) in _GUARD_TYPES and (getattr(h, "score", None) or 0.0) >= sim_threshold
    ]
    return sorted(cand, key=lambda h: getattr(h, "score", None) or 0.0, reverse=True)


def guard_banner(
    prompt: str, hits: list[Any], *, sim_threshold: float, top: int = 1
) -> str | None:
    """⚠ banner naming the prior decision(s) the prompt looks to reverse."""
    cand = guard_candidates(prompt, hits, sim_threshold=sim_threshold)[:top]
    if not cand:
        return None
    lines = ["⚠ PRIOR DECISION — check before overriding"]
    for h in cand:
        title = (getattr(h, "title", "") or "").strip()
        lines.append(f'  You decided [{getattr(h, "id", "?")}]: "{title}"')
    lines.append("  This prompt reads like reversing it — confirm the change is intentional.")
    return "\n".join(lines)


def boost_guarded(hits: list[Any], guarded_ids: set[str], boost: float) -> list[Any]:
    """Return a new list with guard-flagged hits' scores raised by `boost`."""
    import copy
    import dataclasses

    out = []
    for h in hits:
        if getattr(h, "id", None) in guarded_ids and boost:
            base = getattr(h, "score", None) or 0.0
            try:
                out.append(dataclasses.replace(h, score=base + boost))
            except TypeError:  # not a dataclass — mutate a shallow copy
                c = copy.copy(h)
                c.score = base + boost
                out.append(c)
        else:
            out.append(h)
    return out


def log_guard_fire(state_dir: Path, *, prompt: str, ids: list[str]) -> None:
    """Append one guard-fire record to state_dir/guard.log. Best-effort."""
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        rec = {"prompt": (prompt or "")[:200], "ids": ids}
        with (state_dir / "guard.log").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass
