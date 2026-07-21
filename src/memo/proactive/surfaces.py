from __future__ import annotations

from .arbiter import Routed
from .nudge import (
    KIND_CONTINUITY,
    KIND_DEJAVU,
    KIND_HEALTH,
    KIND_RELIABILITY,
    KIND_ROI,
    Nudge,
)

_ICON = {
    KIND_RELIABILITY: "⚠️",
    KIND_CONTINUITY: "↻",
    KIND_DEJAVU: "🔁",
    KIND_HEALTH: "🧹",
    KIND_ROI: "📊",
}
_LABEL = {
    KIND_RELIABILITY: "Reliability",
    KIND_CONTINUITY: "Continuity",
    KIND_DEJAVU: "Déjà-vu",
    KIND_HEALTH: "Health",
    KIND_ROI: "ROI",
}
_ORDER = [KIND_RELIABILITY, KIND_CONTINUITY, KIND_DEJAVU, KIND_HEALTH, KIND_ROI]


def render_badge(routed: Routed) -> str:
    if routed.badge_count == 0:
        return ""
    icon = "⚠️" if routed.badge_kind == KIND_RELIABILITY else "💡"
    return f"{icon}{routed.badge_count}"


def render_urgent_line(n: Nudge) -> str:
    tail = f" · {n.action}" if n.action else ""
    return f"⚠️ memo: {n.title}{tail}"


def render_digest(routed: Routed) -> str:
    if not routed.digest:
        return "memo: nothing to surface."
    by_kind: dict[str, list[Nudge]] = {}
    for n in routed.digest:
        by_kind.setdefault(n.kind, []).append(n)
    lines: list[str] = []
    for kind in _ORDER:
        items = by_kind.get(kind)
        if not items:
            continue
        lines.append(f"{_ICON[kind]} {_LABEL[kind]} ({len(items)})")
        for n in items:
            tail = f" · {n.action}" if n.action else ""
            lines.append(f"   {n.title}{tail}")
    return "\n".join(lines)
