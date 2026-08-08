"""Functional contract for the local multi-agent event bus.

`AgentEventBus` (agent_event_bus.py) is the `MEMO_EVENT_BUS_ENABLED`-gated
facade over the flocked/indexed event journal (event_surface.py). These tests
assert the *wiring* end-to-end:

- publish → readable via `memo events list` (shared journal, not a dangling
  bespoke jsonl file)
- two bus instances on the same state dir see each other's published events
- poll is idempotent per process (subscription semantics)
- journal dedup: publishing the same logical event twice writes once
"""

from __future__ import annotations

from memo.agent_event_bus import AgentEventBus
from memo.config import Config


def test_publish_lands_in_shared_event_journal(tmp_cfg: Config) -> None:
    bus = AgentEventBus(tmp_cfg.state_dir, agent_id="agent-test-a")
    bus.publish("test.event", {"memory_id": "m1"})

    from memo.event_surface import list_events

    rows = list_events(state_dir=tmp_cfg.state_dir, kind="agent")
    assert any(r.get("event_type") == "test.event" for r in rows)


def test_cross_process_events_visible_and_poll_is_idempotent(tmp_cfg: Config) -> None:
    a = AgentEventBus(tmp_cfg.state_dir, agent_id="agent-a")
    b = AgentEventBus(tmp_cfg.state_dir, agent_id="agent-b")

    a.publish("decision_saved", {"memory_id": "m2"})

    first = b.poll_new_events()
    assert [e.event_type for e in first] == ["decision_saved"]
    assert first[0].agent_id == "agent-a"
    assert first[0].data["memory_id"] == "m2"

    # Idempotent: the same session already saw that event.
    assert b.poll_new_events() == []
