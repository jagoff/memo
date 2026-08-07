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


def test_memo_graph_code_navigation_includes_evidence(mock_memory):
    _seed_chain(mock_memory)
    out = _call(
        mock_memory,
        verb="path",
        a="alpha",
        b="gamma",
        include_code=True,
    )
    assert out["code_evidence"]["schema"] == "memo.code_evidence.v1"
    assert out["code_evidence"]["recording_status"] == "missing"


def test_memo_graph_why_verb(mock_memory):
    _seed_chain(mock_memory)
    out = _call(mock_memory, verb="why", a="alpha", b="gamma")
    assert out["verb"] == "why"
    assert out["result"]["path"] == ["alpha", "beta", "gamma"]
    assert len(out["result"]["edges"]) == 2
    assert out["result"]["evidence_memory_ids"] == ["m1", "m2"]


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


def test_memo_graph_architecture_verb(mock_memory, monkeypatch):
    monkeypatch.setattr(
        mock_memory,
        "code_context_pack",
        lambda cwd, **kwargs: {
            "schema": "memo.code_context_pack.v1",
            "cwd": cwd,
            **kwargs,
        },
    )

    out = _call(
        mock_memory,
        verb="architecture",
        cwd="/tmp/repo",
        entity="src/memo",
        scope="src",
        mode="verify",
        cursor="next-page",
        max_chars=4_000,
    )

    assert out["verb"] == "architecture"
    assert out["result"]["schema"] == "memo.code_context_pack.v1"
    assert out["result"]["focus"] == "src/memo"
    assert out["result"]["scope"] == "src"
    assert out["result"]["mode"] == "verify"
    assert out["result"]["cursor"] == "next-page"


def test_memo_graph_architecture_requires_cwd(mock_memory):
    out = _call(mock_memory, verb="architecture")
    assert out == {"error": "architecture requires cwd"}


def test_memo_graph_unknown_verb_returns_error(mock_memory):
    out = _call(mock_memory, verb="bogus")
    assert "error" in out
    assert "why" in out["verbs"]


def test_memo_graph_path_requires_endpoints(mock_memory):
    out = _call(mock_memory, verb="path", a="alpha")
    assert "error" in out


def _seed_hub(mem, *, bridges: int) -> None:
    """Seed one hub entity joined to a neighbour by ``bridges`` memories."""
    for i in range(bridges):
        mem.graph.record_extraction(
            memory_id=f"hub{i}",
            memory_date="2026-01-01",
            entities=[
                {"name": "Hub", "type": "technology"},
                {"name": "Spoke", "type": "technology"},
            ],
            extracted_at="2026-01-01T00:00:00Z",
        )
    mem.graph.rebuild_edges()


def test_memo_graph_neighbors_bounds_bridge_ids(mock_memory):
    """`limit` must bound the payload, not just the neighbour count.

    Regression: on the live corpus `verb="neighbors", limit=4` returned every
    bridging memory id for every neighbour (~15k tokens), even though the tool
    is documented as the cheap id/title-only traversal.
    """
    _seed_hub(mock_memory, bridges=40)

    out = _call(mock_memory, verb="neighbors", entity="hub", limit=4)

    bridges = out["result"]["neighbor_memories"]["spoke"]
    assert len(bridges) <= 5, f"payload unbounded: {len(bridges)} ids"
    # The true count is preserved so the caller still sees link strength.
    assert out["result"]["neighbor_memory_counts"]["spoke"] == 40


def test_memo_graph_impact_bounds_the_symbol_walk(mock_memory, monkeypatch):
    """The traversal cap is not a response budget.

    Regression: `code_change_impact` bounds its walk at 1000 symbols, so a
    13-file working tree produced 376 symbol rows (~100k chars) and the tool
    result blew past the MCP client's token cap entirely — with `limit=3`.
    """
    monkeypatch.setattr(
        mock_memory,
        "code_change_impact",
        lambda cwd, **kw: {
            "available": True,
            "changed_files": ["a.py"],
            "symbols": [{"stable_symbol_id": f"s{i}", "distance": i} for i in range(376)],
            "impacted_paths": ["a.py"],
            "memories": [],
        },
    )

    out = _call(mock_memory, verb="impact", cwd="/repo", limit=3)

    assert len(out["result"]["symbols"]) == 3
    assert out["result"]["symbol_count"] == 376
    # Nearest-first ordering is preserved, so the kept rows are the relevant ones.
    assert [s["distance"] for s in out["result"]["symbols"]] == [0, 1, 2]
