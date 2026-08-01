from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from memo.operational_continuity import ContinuityComposer


@dataclass(frozen=True)
class _Handoff:
    id: str
    status: str
    summary: str
    from_actor: str
    to_actor: str
    project: str = "memo"
    created_at: str = "2026-07-31T11:00:00Z"


@dataclass(frozen=True)
class _Delivery:
    id: str
    message_id: str
    target_id: str
    state: str
    attempt_count: int = 1
    deadline_at: str | None = None


@dataclass(frozen=True)
class _Lease:
    id: str
    files: tuple[str, ...]


@dataclass(frozen=True)
class _Conflict:
    project: str
    file: str
    lease_ids: tuple[str, ...]


@dataclass(frozen=True)
class _Checkpoint:
    session_id: str
    summary: str
    branch: str
    head: str
    checkpointed_at: str
    recoverable_reason: str


class _Coordination:
    def handoffs(self):
        return {
            "handoff-2": _Handoff("handoff-2", "consumed", "old", "agent-a", "agent-b"),
            "handoff-1": _Handoff("handoff-1", "open", "finish signed sync", "agent-a", "agent-b"),
        }


class _DeliveryService:
    def deliveries(self):
        return [
            _Delivery("delivery-2", "message-2", "agent-b", "acknowledged"),
            _Delivery("delivery-1", "message-1", "agent-b", "known_failed", 2),
        ]


class _Presence:
    def active(self, *, project, now):
        del project, now
        return [_Lease("lease-1", ("src/memo/a.py",))]

    def conflicts(self, *, project, files, now):
        del files, now
        return [_Conflict(project, "src/memo/a.py", ("lease-1", "lease-2"))]


class _Sessions:
    def latest_recoverable(self, *, project, workspace):
        del project, workspace
        return _Checkpoint(
            "session-1",
            "sync implementation",
            "feat/memflow-absorption",
            "abc123",
            "2026-07-31T11:30:00Z",
            "client disconnected",
        )


def _composer(**overrides) -> ContinuityComposer:
    values = {
        "durable_briefing": lambda **_: (
            "Decision: Memo is the sole runtime.\nProof must be signed."
        ),
        "coordination": _Coordination(),
        "delivery": _DeliveryService(),
        "presence": _Presence(),
        "sessions": _Sessions(),
        "health": lambda: {
            "verdict": "healthy",
            "gaps": {},
            "pending": 1,
            "detail": "verified operational runtime " * 5,
        },
        "clock": lambda: datetime(2026, 7, 31, 12, tzinfo=UTC),
    }
    values.update(overrides)
    return ContinuityComposer(**values)


def test_continuity_order_cap_and_provenance() -> None:
    packet = _composer().compose(query="resume", cwd="/work/memo", max_chars=500)

    assert packet.text.index("Durable briefing") < packet.text.index("Handoffs")
    assert packet.text.index("Handoffs") < packet.text.index("Checkpoint")
    assert len(packet.text) <= 500
    assert packet.omissions
    assert all(source.id for source in packet.sources)
    assert packet.durable_available is True
    assert packet.operational_available is True


def test_continuity_is_deterministic_and_excludes_terminal_records() -> None:
    composer = _composer()
    first = composer.compose(cwd="/work/memo")
    second = composer.compose(cwd="/work/memo")

    assert first == second
    assert "finish signed sync" in first.text
    assert "old" not in first.text
    assert "delivery-1" in first.text
    assert "delivery-2" not in first.text
    assert "session-1" in first.text


class _BrokenOperational:
    def handoffs(self):
        raise RuntimeError("view unavailable")


def _raise_health():
    raise RuntimeError("health unavailable")


def test_continuity_degrades_without_operational_views() -> None:
    broken = _BrokenOperational()
    packet = _composer(
        coordination=broken,
        delivery=broken,
        presence=broken,
        sessions=broken,
        health=_raise_health,
    ).compose(cwd="/work/memo")

    assert "operational state unavailable" in packet.text.lower()
    assert packet.durable_available is True
    assert packet.operational_available is False
    assert all("memflow" not in item.lower() for item in packet.fallbacks)


def test_continuity_degrades_when_durable_briefing_fails() -> None:
    def broken_briefing(**_):
        raise RuntimeError("durable unavailable")

    packet = _composer(durable_briefing=broken_briefing).compose(cwd="/work/memo")

    assert packet.durable_available is False
    assert packet.operational_available is True
    assert "durable briefing unavailable" in packet.text.lower()
    assert len(packet.text) <= 12_000
