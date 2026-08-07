"""Lockless local multi-agent event bus, backed by the event journal.

Gated by ``MEMO_EVENT_BUS_ENABLED`` (read once at construction): with the flag
off, ``publish``/``poll_new_events`` are no-ops so the bus costs nothing in
agents that never opted in.

Unlike the original jsonl-dangling implementation, this is a *facade over
`memo.event_surface`* — the same flocked, indexed, paginated event journal the
CLI ``events`` group and terminal bridge already use. Events are ingested
through ``ingest_event(kind="agent")`` (deduped by event_id, index reconciled,
flock-protected) instead of a second hand-rolled append file, so an event
published here shows up in ``memo events list`` and survives the journal's
rotation/GC rules rather than accumulating in an unbounded ``agent_events.jsonl``.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentEvent:
    event_type: str
    agent_id: str
    data: dict[str, Any]
    timestamp: float


def _event_id(event_type: str, agent_id: str, data: dict[str, Any]) -> str:
    """Deterministic event_id so re-publish of the same logical event dedups."""
    payload = hashlib.sha256(
        f"{event_type}|{agent_id}|{sorted(data.items())}".encode()
    ).hexdigest()[:24]
    return f"agent:{payload}"


class AgentEventBus:
    """In-memory + journal-backed event bus for local multi-agent sync.

    ``publish`` writes one `agent`-kind event to the shared journal via
    ``event_surface.ingest_event`` (idempotent: same event_type/agent/data
    → same event_id, deduped) and fans it out to in-process subscribers.
    ``poll_new_events`` reads back journal events for this agent's session,
    skips the current process's own events, and advances a local cursor.
    """

    def __init__(self, state_dir: Path, agent_id: str = "agent-local") -> None:
        from memo.flags import flag_bool

        self._enabled = flag_bool("MEMO_EVENT_BUS_ENABLED")
        self.state_dir = state_dir
        self.agent_id = agent_id
        self._subscribers: list[Callable[[AgentEvent], None]] = []
        # Seen event_ids (journal entry ids) so repeated polls are idempotent —
        # the journal is append-only and list_events has no per-consumer cursor.
        self._seen_ids: set[str] = set()

    def publish(self, event_type: str, data: dict[str, Any] | None = None) -> AgentEvent:
        """Publish a state event locally and to the shared event journal."""
        from memo.event_surface import ingest_event

        event = AgentEvent(
            event_type=event_type,
            agent_id=self.agent_id,
            data=data or {},
            timestamp=time.time(),
        )
        if not self._enabled:
            return event

        # Notify in-process subscribers
        for sub in self._subscribers:
            try:
                sub(event)
            except Exception as exc:
                _logger.debug("Event bus subscriber error: %s", exc)

        # Append to the shared (flocked, receipt-indexed) event journal. The
        # event_surface layer owns dedup + recovery + rotation — no unbounded
        # second journal here.
        try:
            eid = _event_id(event.event_type, event.agent_id, event.data)
            ingest_event(
                {
                    "event_id": eid,
                    "kind": "agent",
                    "event_type": event.event_type,
                    "agent_id": event.agent_id,
                    "data": event.data,
                    "timestamp": event.timestamp,
                },
                state_dir=self.state_dir,
            )
            self._seen_ids.add(eid)
        except Exception as exc:
            _logger.debug("Failed to write agent event to journal: %s", exc)

        return event

    def subscribe(self, callback: Callable[[AgentEvent], None]) -> None:
        """Subscribe to in-process events."""
        self._subscribers.append(callback)

    def poll_new_events(self) -> list[AgentEvent]:
        """Poll the journal for events written by other local agent processes."""
        from memo.event_surface import list_events

        if not self._enabled:
            return []
        events: list[AgentEvent] = []
        try:
            for item in list_events(state_dir=self.state_dir, kind="agent", limit=200):
                eid = str(item.get("event_id") or "")
                if eid and eid in self._seen_ids:
                    continue
                agent_id = str(item.get("agent_id") or "")
                if agent_id == self.agent_id:
                    # Our own events were already delivered in-process.
                    if eid:
                        self._seen_ids.add(eid)
                    continue
                events.append(
                    AgentEvent(
                        event_type=str(item.get("event_type") or ""),
                        agent_id=agent_id,
                        data=dict(item.get("data") or {}),
                        timestamp=float(item.get("timestamp") or 0.0),
                    )
                )
                if eid:
                    self._seen_ids.add(eid)
        except Exception as exc:
            _logger.debug("Error polling agent events: %s", exc)
        return events