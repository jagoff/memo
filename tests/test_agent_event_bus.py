from __future__ import annotations

import tempfile
from pathlib import Path

from memo.agent_event_bus import AgentEvent, AgentEventBus


def test_agent_event_bus_publish_subscribe() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        state_dir = Path(tmpdir)
        bus_a = AgentEventBus(state_dir, agent_id="agent-claude")
        bus_b = AgentEventBus(state_dir, agent_id="agent-cursor")

        received: list[AgentEvent] = []
        bus_a.subscribe(lambda evt: received.append(evt))

        # Bus A publishes an event
        evt1 = bus_a.publish("decision_saved", {"memory_id": "mem_001"})
        assert evt1.event_type == "decision_saved"
        assert len(received) == 1
        assert received[0].data["memory_id"] == "mem_001"

        # Bus B polls new events from Bus A
        new_evts = bus_b.poll_new_events()
        assert len(new_evts) == 1
        assert new_evts[0].agent_id == "agent-claude"
        assert new_evts[0].event_type == "decision_saved"

        # Subsequent poll returns 0 new events (cursor offset maintained)
        assert len(bus_b.poll_new_events()) == 0
