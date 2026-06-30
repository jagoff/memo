"""Tests for the consolidated read-only ``memo_graph`` MCP tool + ``weighted_path``."""

from __future__ import annotations

import asyncio

import pytest

from memo.server import build_server


@pytest.fixture(autouse=True)
def _isolate_codegraph(monkeypatch, tmp_path):
    """Keep the merged graph hermetic — never read the machine's real codegraph."""
    from memo import codegraph_loader

    monkeypatch.setattr(codegraph_loader, "CODEGRAPH_DB", tmp_path / "no-codegraph.db")
    codegraph_loader.reset()
    yield
    codegraph_loader.reset()


def _seed_chain(mem) -> None:
    """Seed a small A-B-C entity chain: mem1 links A,B; mem2 links B,C."""
    mem.graph.record_extraction(
        memory_id="m1",
        memory_date="2026-01-01",
        entities=[{"name": "Alpha", "type": "technology"}, {"name": "Beta", "type": "technology"}],
        extracted_at="2026-01-01T00:00:00Z",
    )
    mem.graph.record_extraction(
        memory_id="m2",
        memory_date="2026-01-01",
        entities=[{"name": "Beta", "type": "technology"}, {"name": "Gamma", "type": "technology"}],
        extracted_at="2026-01-01T00:00:00Z",
    )
    mem.graph.rebuild_edges()


# ---------------------------------------------------------------------------
# weighted_path unit tests (navigation.py)
# ---------------------------------------------------------------------------


def test_weighted_path_same_entity(mock_memory):
    _seed_chain(mock_memory)  # 'alpha' exists in the graph
    nav = mock_memory.navigator
    # self-path for a PRESENT entity
    assert nav.weighted_path("alpha", "alpha") == {"path": ["alpha"], "edges": []}
    # self-query for an ABSENT entity is None, not a phantom self-path (fix #8)
    assert nav.weighted_path("ghost", "ghost") is None


def test_weighted_path_missing_entity(mock_memory):
    _seed_chain(mock_memory)
    assert mock_memory.navigator.weighted_path("alpha", "nonexistent") is None


def test_weighted_path_two_hops_with_weights(mock_memory):
    _seed_chain(mock_memory)
    result = mock_memory.navigator.weighted_path("alpha", "gamma")
    assert result is not None
    assert result["path"] == ["alpha", "beta", "gamma"]
    assert len(result["edges"]) == 2
    first = result["edges"][0]
    assert first["from"] == "alpha"
    assert first["to"] == "beta"
    assert first["weight"] == pytest.approx(1.0)
    assert result["edges"][1]["from"] == "beta"
    assert result["edges"][1]["to"] == "gamma"


# ---------------------------------------------------------------------------
# Tool registration + per-verb output
# ---------------------------------------------------------------------------


def test_memo_graph_registered_on_agent_profile(mock_memory, monkeypatch):
    monkeypatch.delenv("MEMO_MCP_PROFILE", raising=False)
    monkeypatch.delenv("MEMO_MCP_SLIM", raising=False)
    server = build_server(memory=mock_memory)
    assert asyncio.run(server.get_tool("memo_graph")) is not None


def _call(mem, **kwargs):
    server = build_server(memory=mem)
    tool = asyncio.run(server.get_tool("memo_graph"))
    return tool.fn(**kwargs)


def test_memo_graph_path_verb(mock_memory):
    _seed_chain(mock_memory)
    out = _call(mock_memory, verb="path", a="alpha", b="gamma")
    assert out["verb"] == "path"
    assert out["result"]["path"] == ["alpha", "beta", "gamma"]


def test_memo_graph_why_verb(mock_memory):
    _seed_chain(mock_memory)
    out = _call(mock_memory, verb="why", a="alpha", b="gamma")
    assert out["verb"] == "why"
    assert out["result"]["path"] == ["alpha", "beta", "gamma"]
    assert len(out["result"]["edges"]) == 2


def test_memo_graph_neighbors_verb(mock_memory):
    _seed_chain(mock_memory)
    out = _call(mock_memory, verb="neighbors", entity="beta")
    assert out["verb"] == "neighbors"
    assert set(out["result"]["direct_neighbors"]) == {"alpha", "gamma"}


def test_memo_graph_explore_verb(mock_memory):
    _seed_chain(mock_memory)
    out = _call(mock_memory, verb="explore", entity="beta")
    assert out["verb"] == "explore"
    assert out["result"]["entity"] == "beta"
    assert out["result"]["degree"] >= 2


def test_memo_graph_communities_verb(mock_memory):
    _seed_chain(mock_memory)
    out = _call(mock_memory, verb="communities", limit=5)
    assert out["verb"] == "communities"
    assert isinstance(out["result"], list)
    assert len(out["result"]) <= 5


def test_memo_graph_unknown_verb_returns_error(mock_memory):
    out = _call(mock_memory, verb="bogus")
    assert "error" in out
    assert "why" in out["verbs"]


def test_memo_graph_path_requires_endpoints(mock_memory):
    out = _call(mock_memory, verb="path", a="alpha")
    assert "error" in out
