"""The three MCP tools that could never return data, and the fit that fixed them.

Measured 2026-08-08 on the developer's live install (~11.3k memories, the
codegraph layer merged in) against the DEFAULT 10,000-token cap, calling each
tool with NO arguments through the real budget middleware:

    memo_graph_communities                 14,585 tokens  -> error dict
    memo_graph_export (format="dot")       11,365 tokens  -> error dict
    memo_graph_export (format="json")      27,413 tokens  -> error dict
    memo_embed_query (2560-dim vector)     21,924 tokens  -> error dict

Every one of them, on every call, for every caller. The count caps those tools
already carried (`limit=20` x 50 entities, `max_edges=500`) cannot fix this:
their elastic unit is a variable-length STRING (an entity name, a DOT edge
line), so no fixed count bounds the payload. The same `max_edges=500` costs
11.4k tokens as DOT and 27.4k as JSON off one graph.

So the bound is now the size itself (`mcp_budget.fit_to_budget`), and these
tests pin the outcome that matters: a real payload back, under the default
budget, with no argument-guessing by the caller.

Hermetic: the codegraph index is pinned to a nonexistent path so the merge in
`Navigator._build_adjacency_list` degrades to the seeded entity graph (see
`_no_codegraph`). Nothing here loads MLX -- `memo_embed_query`'s embedder is
replaced with a float32 vector generator.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from memo.config import Config
from memo.memory import Memory
from memo.server import build_server

# 22 disjoint cliques of 55 entities each. Sized so the CURRENT (count-capped)
# tools measure over the 10,000-token cap and the fitted ones do not:
#   - communities: 20 (the default `limit`) x 50 (the old entity cap) long
#     names is ~25k tokens; the seed must supply at least that many.
#   - export: 22 * 55*54/2 = 32,670 edges, far past the 500-edge default, so
#     the DOT/JSON payloads are bounded by the fit and not by the seed.
_ISLANDS = 22
_ISLAND_SIZE = 55
_CREATED = "2026-01-01T00:00:00+00:00"


def _entity(island: int, n: int) -> str:
    """A codegraph-length symbol name (~60 chars).

    Short names would hide the defect: 1,000 six-character entity names fit
    the cap comfortably. The live payload is expensive because the entities
    the codegraph layer contributes are real symbol names
    (`test_memo_graph_export_json_bounds_edges_and_reports_true_counts`).
    """
    return f"conformance_symbol_{island:02d}_{n:02d}_bounds_edges_and_reports_true_counts"


@pytest.fixture
def _no_codegraph(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Pin codegraph resolution at a path that does not exist, and prove it.

    `tests/conftest.py` turns cwd discovery off and drops `MEMO_CODEGRAPH_DB`,
    but `codegraph_loader._resolve_db` still falls back to the module-relative
    `CODEGRAPH_DB` -- memo's own checkout, which on the developer's machine
    holds a real 79 MB index. Measured 2026-08-08 with that fallback in play:
    4 of the 20 communities came from the live index rather than from the
    seed, so the sizes asserted here would have been machine state.

    Every layer of the resolution order is pinned (`_resolve_db`: explicit arg
    > cwd discovery > MEMO_CODEGRAPH_DB > module `CODEGRAPH_DB`), not just the
    env var, and the assertion below fails loudly if any of them ever resolves
    a real file again. That assertion is the only thing that would catch it:
    the worktree this was written in has no `.codegraph`, so deleting the pin
    here leaves every test green while the main checkout leaks.
    """
    from memo import codegraph_loader

    missing = tmp_path / "no-such-codegraph.db"
    monkeypatch.setenv("MEMO_CODEGRAPH_DISCOVERY", "0")
    monkeypatch.setenv("MEMO_CODEGRAPH_DB", str(missing))
    monkeypatch.setattr(codegraph_loader, "CODEGRAPH_DB", missing)
    resolved = codegraph_loader._resolve_db()
    assert not resolved.is_file(), f"codegraph leaked into a hermetic test: {resolved}"


@pytest.fixture
def graph_server(tmp_cfg: Config, _no_codegraph: None) -> Iterator[Any]:
    """An MCP server over a seeded entity graph, big enough to blow the cap."""
    from memo.graph import GraphStore

    store = GraphStore(tmp_cfg.graph_db)
    try:
        for island in range(_ISLANDS):
            store.record_extraction(
                memory_id=f"seed-memory-{island:03d}",
                memory_date=_CREATED,
                extracted_at=_CREATED,
                entities=[
                    {"name": _entity(island, n), "type": "concept"} for n in range(_ISLAND_SIZE)
                ],
            )
    finally:
        store.close()

    memory = Memory(tmp_cfg)
    try:
        yield build_server(memory=memory)
    finally:
        memory.close()


def _payload(result: Any) -> Any:
    """The structured payload, asserting it is not the budget's error dict.

    `structured_content` is what the budget middleware substitutes into, and
    what a real MCP client reads. A list-returning tool is wrapped by FastMCP
    as `{"result": [...]}`; the error substitution is always a bare dict with
    an `error` key, so unwrapping after the check cannot mask it.
    """
    sc = result.structured_content
    assert isinstance(sc, dict), f"expected structured content, got {type(sc)}"
    assert sc.get("error") != "response_budget_exceeded", (
        f"tool returned the budget error instead of data: {sc}"
    )
    return sc.get("result", sc)


@pytest.mark.asyncio
async def test_memo_graph_communities_returns_data_under_the_default_budget(
    graph_server: Any,
) -> None:
    """Called bare, through the real middleware, it comes back with communities.

    No `limit`, no `min_size` -- the defect was that the DEFAULT call could
    never succeed, so passing arguments here would test the wrong thing.
    """
    result = await graph_server.call_tool("memo_graph_communities", {})

    communities = _payload(result)
    assert isinstance(communities, list)
    assert len(communities) == 20, f"expected the default page of 20, got {len(communities)}"
    # Real data, not empty husks: the seeded cliques are 55 entities each and
    # `size` must still report that truth even though `entities` is sampled.
    assert all(c["size"] == _ISLAND_SIZE for c in communities)
    assert all(c["entities"] for c in communities)
    assert all(c["representative_entity"] for c in communities)


@pytest.mark.asyncio
async def test_memo_graph_communities_reports_the_true_size_when_it_samples(
    graph_server: Any,
) -> None:
    """The entity list is a sample; `size` and `entities_truncated` say so."""
    result = await graph_server.call_tool("memo_graph_communities", {})

    first = _payload(result)[0]
    assert len(first["entities"]) < first["size"]
    assert first["entities_truncated"] is True


@pytest.mark.asyncio
async def test_memo_graph_communities_max_entities_reaches_the_detail(
    graph_server: Any,
) -> None:
    """The detail the smaller default gives up is reachable by asking for it."""
    result = await graph_server.call_tool(
        "memo_graph_communities", {"limit": 3, "max_entities": _ISLAND_SIZE}
    )

    communities = _payload(result)
    assert len(communities) == 3
    assert len(communities[0]["entities"]) == _ISLAND_SIZE
    assert communities[0]["entities_truncated"] is False


@pytest.mark.asyncio
async def test_memo_graph_communities_survives_a_max_entities_that_blows_the_count_caps(
    graph_server: Any,
) -> None:
    """Raising `max_entities` costs communities off the page, not the whole result.

    20 communities x 55 long entity names is ~35k tokens -- past the cap by
    3.5x, and the exact shape the old fixed `_MAX_COMMUNITY_ENTITIES = 50`
    shipped by default. This is what `fit_to_budget` is for: no count cap can
    hold a payload whose unit is a name of arbitrary length, so the size
    decides and the caller still gets data.
    """
    result = await graph_server.call_tool("memo_graph_communities", {"max_entities": _ISLAND_SIZE})

    communities = _payload(result)
    assert communities, "the fit dropped everything"
    assert len(communities) < 20, "nothing was traded away, so the fit did not engage"
    assert len(communities[0]["entities"]) == _ISLAND_SIZE


@pytest.mark.asyncio
async def test_memo_graph_export_dot_returns_data_under_the_default_budget(
    graph_server: Any,
) -> None:
    """Bare `memo_graph_export` (format defaults to "dot") returns real DOT."""
    result = await graph_server.call_tool("memo_graph_export", {})

    payload = _payload(result)
    assert payload["format"] == "dot"
    assert payload["content"].startswith("graph memo_entities {")
    assert payload["content"].rstrip().endswith("}")
    kept = [line for line in payload["content"].split("\n") if " -- " in line]
    assert kept, "no edges survived the fit"
    # The seed has 32,670 edges; the true count is reported whatever is kept.
    assert payload["edge_count"] == _ISLANDS * _ISLAND_SIZE * (_ISLAND_SIZE - 1) // 2
    assert payload["truncated"] is True
    assert len(kept) < payload["edge_count"]


@pytest.mark.asyncio
async def test_memo_graph_export_json_returns_data_under_the_default_budget(
    graph_server: Any,
) -> None:
    """The JSON format too -- it is ~2.4x the DOT payload for the same edges.

    A single `max_edges` count cannot bound both formats: 500 edges measured
    11,365 tokens as DOT and 27,413 as JSON off the same live graph.
    """
    result = await graph_server.call_tool("memo_graph_export", {"format": "json"})

    payload = _payload(result)
    assert payload["format"] == "json"
    edges = payload["data"]["edges"]
    nodes = payload["data"]["nodes"]
    assert edges, "no edges survived the fit"
    assert nodes, "no nodes survived the fit"
    assert payload["truncated"] is True
    assert len(edges) < payload["edge_count"]
    # Still a drawable graph: every retained edge's endpoints are present.
    ids = {n["id"] for n in nodes}
    assert all(e["source"] in ids and e["target"] in ids for e in edges)


@pytest.mark.asyncio
async def test_the_graph_tools_disclose_that_codegraph_is_cwd_scoped(
    graph_server: Any,
) -> None:
    """The codegraph merge is left cwd-scoped on purpose; it must be stated.

    A code graph describes the repo you are standing in, which is not
    something MEMO_DATA_DIR can express -- so the behaviour stays. What was
    unacceptable was the silence: pointed at a brand-new EMPTY store, both
    tools returned byte-identical repo symbols with nothing in the response,
    the description, or the schema saying where they came from.

    Asserted on `Tool.description`, the field that actually crosses the MCP
    wire (verified against `list_tools()`; `parameters`, not `inputSchema`, is
    the sibling that carries the schema).
    """
    tools = {t.name: t for t in await graph_server.list_tools()}

    for name in ("memo_graph_communities", "memo_graph_export"):
        description = tools[name].description or ""
        assert "CURRENT WORKING DIRECTORY" in description, name
        assert "MEMO_DATA_DIR" in description, name
        assert "MEMO_GRAPH_USE_CODEGRAPH=0" in description, name


def _float32_vector(dim: int) -> list[float]:
    """A deterministic float32 vector with realistic repr length.

    Size is the whole point of this test, and a float32's decimal repr is what
    costs the tokens: a real MLX component reads back as -0.0012969970703125
    (18 chars), while a hand-written 0.1 would be 3. Values are round-tripped
    through `struct` so they are genuinely float32, exactly as
    `MLXEmbedder.embed_query` returns them.
    """
    raw = [((i * 2654435761) % 2_000_003) / 2_000_003.0 - 0.5 for i in range(dim)]
    return [struct.unpack("<f", struct.pack("<f", v / 32.0))[0] for v in raw]


@pytest.mark.asyncio
async def test_memo_embed_query_returns_the_whole_vector_under_the_default_budget(
    tmp_cfg: Config,
) -> None:
    """A 2560-dim vector is ~21.9k tokens on the wire -- 2.2x the default cap.

    It is exempt (`mcp_budget.CAPS`) rather than trimmed because a truncated
    vector is not a vector: the tool's entire contract is "the exact vector
    memo would use", and its size is fixed by the embedder profile, not by
    the corpus or by anything the caller can grow.

    No MLX: `embed_query` is replaced outright.
    """
    dim = 2560
    vector = _float32_vector(dim)
    memory = Memory(tmp_cfg)
    try:
        memory.embedder.embed_query = lambda text: list(vector)  # type: ignore[method-assign]
        server = build_server(memory=memory)
        result = await server.call_tool("memo_embed_query", {"text": "budget probe"})
    finally:
        memory.close()

    payload = _payload(result)
    assert payload["dim"] == dim
    assert payload["vector"] == pytest.approx(vector)
