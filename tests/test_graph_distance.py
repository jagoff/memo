import pytest
from memo.memory.graph import Graph
from memo.memory.record import MemoryRecord


def test_distance_to_nearest_fact_direct():
    """A fact should have distance 0."""
    graph = Graph()
    fact = MemoryRecord(
        id="fact1",
        path="",
        title="base fact",
        type="fact",
        tags=[],
        created="2026-01-01T00:00:00",
        updated="2026-01-01T00:00:00",
        body="base fact",
    )
    graph.add_memory(fact)

    distance = graph.distance_to_nearest_fact("fact1")
    assert distance == 0


def test_distance_to_nearest_fact_one_hop():
    """A decision derived from a fact should have distance 1."""
    graph = Graph()
    fact = MemoryRecord(
        id="fact1",
        path="",
        title="base fact",
        type="fact",
        tags=[],
        created="2026-01-01T00:00:00",
        updated="2026-01-01T00:00:00",
        body="base fact",
    )
    decision = MemoryRecord(
        id="dec1",
        path="",
        title="derived",
        type="decision",
        tags=[],
        created="2026-01-01T00:00:00",
        updated="2026-01-01T00:00:00",
        body="derived",
    )

    graph.add_memory(fact)
    graph.add_memory(decision)
    graph.add_edge(from_id="dec1", to_id="fact1", weight=1.0)

    distance = graph.distance_to_nearest_fact("dec1")
    assert distance == 1


def test_distance_to_nearest_fact_unreachable():
    """A synthesis with no path to facts should return 999."""
    graph = Graph()
    synth = MemoryRecord(
        id="syn1",
        path="",
        title="isolated",
        type="synthesis",
        tags=[],
        created="2026-01-01T00:00:00",
        updated="2026-01-01T00:00:00",
        body="isolated",
    )
    graph.add_memory(synth)

    distance = graph.distance_to_nearest_fact("syn1")
    assert distance == 999


def test_distance_shortest_path():
    """BFS should return shortest path when multiple exist."""
    graph = Graph()
    fact = MemoryRecord(
        id="fact1",
        path="",
        title="fact",
        type="fact",
        tags=[],
        created="2026-01-01T00:00:00",
        updated="2026-01-01T00:00:00",
        body="fact",
    )
    dec1 = MemoryRecord(
        id="dec1",
        path="",
        title="d1",
        type="decision",
        tags=[],
        created="2026-01-01T00:00:00",
        updated="2026-01-01T00:00:00",
        body="d1",
    )
    dec2 = MemoryRecord(
        id="dec2",
        path="",
        title="d2",
        type="decision",
        tags=[],
        created="2026-01-01T00:00:00",
        updated="2026-01-01T00:00:00",
        body="d2",
    )
    synth = MemoryRecord(
        id="syn1",
        path="",
        title="syn",
        type="synthesis",
        tags=[],
        created="2026-01-01T00:00:00",
        updated="2026-01-01T00:00:00",
        body="syn",
    )

    graph.add_memory(fact)
    graph.add_memory(dec1)
    graph.add_memory(dec2)
    graph.add_memory(synth)

    # Path 1: synth -> dec1 -> fact (2 hops)
    graph.add_edge(from_id="syn1", to_id="dec1", weight=1.0)
    graph.add_edge(from_id="dec1", to_id="fact1", weight=1.0)

    # Path 2: synth -> dec2 -> fact (2 hops, same length)
    graph.add_edge(from_id="syn1", to_id="dec2", weight=1.0)
    graph.add_edge(from_id="dec2", to_id="fact1", weight=1.0)

    distance = graph.distance_to_nearest_fact("syn1")
    assert distance == 2
