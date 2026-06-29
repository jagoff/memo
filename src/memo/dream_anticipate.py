"""`memo dream anticipate` — anticipatory memory pass (Phase 3).

Surfaces what you'll likely need *before* you ask: the recurring KNOWLEDGE
GAPS memo could not answer (``outcome.detect_gaps``) plus your hot recurring
queries, and pre-warms their embeddings so the next recall is fast.

It deliberately does **not** fabricate answers — naming a gap memo can't fill
is honest; inventing a memory for it is not. The gaps land in the dream receipt
and the SessionStart briefing so you (or a future `memo_save`) fill them.
OFF by default (``MEMO_DREAM_ANTICIPATE_ENABLED``). Reuses Phase-0 substrate
ideas: read-from-real-usage, surface-not-fabricate.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from memo.dashboard_logs import read_recall_log
from memo.outcome import detect_gaps


def anticipate(
    cfg: Any,
    mem: Any | None = None,
    *,
    top_gaps: int = 5,
    top_queries: int = 5,
    min_gap_count: int = 2,
    prewarm: bool = True,
) -> dict[str, Any]:
    """Compute anticipated needs. Returns a receipt fragment; never raises."""
    res: dict[str, Any] = {"gaps": [], "hot_queries": [], "prewarmed": 0}
    try:
        gaps = detect_gaps(cfg.state_dir, min_count=min_gap_count)[:top_gaps]
        entries = read_recall_log(cfg.state_dir, limit=200)
        counts: Counter[str] = Counter()
        for e in entries:
            q = (e.get("prompt") or "").strip()
            if q:
                counts[q] += 1
        hot = [q for q, _ in counts.most_common(top_queries)]

        prewarmed = 0
        if prewarm and mem is not None and getattr(mem, "embedder", None) is not None:
            seen: set[str] = set()
            for q in [g["prompt"] for g in gaps] + hot:
                if not q or q in seen:
                    continue
                seen.add(q)
                try:
                    mem.embedder.embed_query(q)
                    prewarmed += 1
                except Exception:  # noqa: S110 — warming is best-effort, never fatal
                    pass

        res = {
            "gaps": [{"prompt": g["prompt"], "count": g.get("count", 1)} for g in gaps],
            "hot_queries": hot,
            "prewarmed": prewarmed,
        }
    except Exception as exc:  # surfaced into the receipt, never silent
        res["error"] = f"{type(exc).__name__}: {exc}"
    return res


def briefing_line(fragment: dict[str, Any]) -> str:
    """One-line human summary for the SessionStart briefing / dream status."""
    gaps = fragment.get("gaps") or []
    if not gaps:
        return "anticipate: no recurring gaps"
    top = gaps[0]["prompt"]
    extra = f" (+{len(gaps) - 1} more)" if len(gaps) > 1 else ""
    return f"anticipate: {len(gaps)} unmet gap(s) — top: {top[:60]!r}{extra}"
