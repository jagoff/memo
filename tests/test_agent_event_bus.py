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


def test_publish_reports_failure_when_the_bus_is_disabled(tmp_path, monkeypatch):
    """`published: True` must mean the event actually reached the journal."""
    from memo.agent_event_bus import AgentEventBus

    monkeypatch.setenv("MEMO_EVENT_BUS_ENABLED", "0")
    bus = AgentEventBus(tmp_path, agent_id="test")
    bus.publish("sync.completed", {})
    assert bus.enabled is False
    assert bus.last_publish_ok is False

    monkeypatch.setenv("MEMO_EVENT_BUS_ENABLED", "1")
    live = AgentEventBus(tmp_path, agent_id="test")
    live.publish("sync.completed", {"n": 1})
    assert live.enabled is True
    assert live.last_publish_ok is True


def test_mcp_event_bus_is_reused_so_poll_stays_idempotent(tmp_cfg):
    """A fresh bus per call reset the delivered-once cursor."""
    from memo.mcp_tools import _mcp_event_bus

    class _Mem:
        cfg = tmp_cfg

    mem = _Mem()
    assert _mcp_event_bus(mem) is _mcp_event_bus(mem)
