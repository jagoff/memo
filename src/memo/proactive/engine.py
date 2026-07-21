from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from .arbiter import Routed, route
from .detectors.continuity import detect_continuity
from .detectors.dejavu import detect_dejavu
from .detectors.health import detect_health
from .detectors.reliability import detect_reliability
from .detectors.roi import detect_roi
from .store import ProactiveStore

_FEEDBACK_WINDOW_DAYS = 30


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def push_gate(
    store: ProactiveStore, *, now: str, day: str, cooldown_h: int, daily_cap: int
) -> bool:
    if store.pushes_today(day) >= daily_cap:
        return False
    last = store.last_push_at()
    return last is None or _parse(now) - _parse(last) >= timedelta(hours=cooldown_h)


def refresh_candidates(mem: Any, store: ProactiveStore, *, now: str) -> int:
    nudges = (
        detect_reliability(mem, now=now)
        + detect_continuity(mem, now=now)
        + detect_health(mem, now=now)
        + detect_roi(mem, now=now)
        + detect_dejavu(mem, now=now)
    )
    store.put_candidates(nudges)
    return len(nudges)


def compute_routed(store: ProactiveStore, *, now: str, day: str) -> Routed:
    from memo.flags import flag_float, flag_int

    floor = flag_float("MEMO_PROACTIVE_MULT_FLOOR")
    floor = 0.2 if floor is None else floor
    digest_top = flag_int("MEMO_PROACTIVE_DIGEST_TOP")
    digest_top = 7 if digest_top is None else digest_top
    urgent_min = flag_float("MEMO_PROACTIVE_URGENT_MIN")
    urgent_min = 0.7 if urgent_min is None else urgent_min
    cooldown_h = flag_int("MEMO_PROACTIVE_PUSH_COOLDOWN_H")
    cooldown_h = 6 if cooldown_h is None else cooldown_h
    daily_cap = flag_int("MEMO_PROACTIVE_DAILY_CAP")
    daily_cap = 3 if daily_cap is None else daily_cap

    since = (_parse(now) - timedelta(days=_FEEDBACK_WINDOW_DAYS)).isoformat()

    return route(
        store.active_candidates(now),
        store.kind_multipliers(floor, since=since),
        digest_top=digest_top,
        urgent_min=urgent_min,
        can_push=push_gate(store, now=now, day=day, cooldown_h=cooldown_h, daily_cap=daily_cap),
    )
