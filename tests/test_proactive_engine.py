from pathlib import Path

from memo.proactive.engine import push_gate
from memo.proactive.store import ProactiveStore


def test_push_gate_respects_cooldown_and_cap(tmp_path: Path):
    s = ProactiveStore(tmp_path / "p.db")
    assert (
        push_gate(s, now="2026-07-21T10:00:00Z", day="2026-07-21", cooldown_h=6, daily_cap=3)
        is True
    )
    s.mark_pushed("2026-07-21T09:00:00Z")  # 1h ago < 6h cooldown
    assert (
        push_gate(s, now="2026-07-21T10:00:00Z", day="2026-07-21", cooldown_h=6, daily_cap=3)
        is False
    )


def test_push_gate_daily_cap(tmp_path: Path):
    s = ProactiveStore(tmp_path / "p.db")
    for h in (1, 2, 3):
        s.mark_pushed(f"2026-07-21T0{h}:00:00Z")
    # 3 pushes today, all >6h before 'now' → cooldown ok but cap hit
    assert (
        push_gate(s, now="2026-07-21T23:00:00Z", day="2026-07-21", cooldown_h=6, daily_cap=3)
        is False
    )
