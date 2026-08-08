"""MCP tools — knowledge-graph domain (split from server.py).

Registered by `build_server()` via `register(server, memory)`. Tool names,
signatures, defaults, docstrings and bodies are identical to the originals;
only the enclosing function and indentation changed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any

from fastmcp import FastMCP
from pydantic import Field

from memo.mcp_budget import bounded_list, cap_for, fit_to_budget
from memo.memory import Memory
from memo.server_annotations import READ_ONLY, annotated_tool

# Default per-community entity sample. Was 50 until 2026-08-08, when the
# default call was measured at 14,585 tokens on the developer's LIVE install
# (~11.3k memories with the codegraph layer merged in) against the 10,000-token
# cap -- i.e. `memo_graph_communities` returned the budget error on EVERY call
# and could not return data at all. 20 communities x 50 codegraph-length symbol
# names is simply too much detail for one tool result; at 12 the same call
# measures 4,628 tokens, and the entities a caller actually loses are reachable
# via `max_entities`. `size` still reports the community's true entity count.
#
# This is the comfort default, NOT the guarantee: `fit_to_budget` below is what
# makes an over-cap response structurally impossible, because no fixed count
# bounds a payload whose unit is a symbol name of arbitrary length.
_DEFAULT_COMMUNITY_ENTITIES = 12

# Why both tools below carry a "Scope:" paragraph naming the working directory.
# `Navigator._build_adjacency_list` / `_weighted_adjacency` fold
# `.codegraph/codegraph.db` into the entity graph, and `codegraph_loader`
# resolves that index from the CWD (nearest `.codegraph` walking up), never
# from MEMO_DATA_DIR/MEMO_STATE_DIR. Verified 2026-08-08: two brand-new,
# separately-configured, EMPTY memo stores returned byte-identical communities
# (58,337 chars, sha256 7a5d26d93f187e9f) made entirely of repo symbols
# (`conftest.py`, `prep-cesium-path.mjs`). The scoping is intentional and left
# alone -- a code graph describes the repo you are standing in, which is not
# something a memory store's path can express, and `MEMO_GRAPH_USE_CODEGRAPH=0`
# / `MEMO_CODEGRAPH_DISCOVERY=0` are the documented opt-outs -- but it was
# invisible, so the two tools that surface it now say so.


def _bounded_dot(dot: str, *, max_edges: int, cap: int) -> dict[str, Any]:
    """Trim the DOT export to `max_edges` edge lines AND to `cap` tokens.

    The live graph renders 24,264 edge lines — a whole-graph dump no tool
    result can carry. Only edge lines are dropped; the header and the closing
    brace stay, so the trimmed DOT still parses.

    `max_edges` alone is not a bound: at the 500-edge default this measured
    11,365 tokens against a 10,000-token cap, because an edge line is two
    symbol names of arbitrary length. So the count cap picks the candidate
    pool and `fit_to_budget` decides how much of it actually fits.
    """
    lines = dot.split("\n")
    edge_lines = [i for i, line in enumerate(lines) if " -- " in line]
    pool = edge_lines[:max_edges]
    edge_positions = set(edge_lines)

    def _render(kept: Sequence[int]) -> dict[str, Any]:
        keep = set(kept)
        return {
            "format": "dot",
            "content": "\n".join(
                line for i, line in enumerate(lines) if i not in edge_positions or i in keep
            ),
            "edge_count": len(edge_lines),
            "truncated": len(keep) < len(edge_lines),
        }

    _, payload = fit_to_budget(pool, cap=cap, render=_render)
    return payload


def _bounded_json(data: dict[str, Any], *, max_edges: int, cap: int) -> dict[str, Any]:
    """Trim the JSON export to `max_edges` edges AND to `cap` tokens.

    Nodes are filtered down to the endpoints of the retained edges so the
    payload stays a drawable graph rather than edges pointing at absent nodes
    plus isolates.

    The same 500-edge default that costs 11,365 tokens as DOT costs 27,413
    here — the clearest proof that one edge COUNT cannot bound both formats,
    and why the real bound is the size.
    """
    nodes = data.get("nodes") or []
    edges = data.get("edges") or []
    pool = edges[:max_edges]

    def _render(kept: Sequence[Any]) -> dict[str, Any]:
        if len(kept) == len(edges):
            # Untrimmed: hand back the graph verbatim. Filtering to endpoints
            # here too would silently drop ISOLATED nodes the caller asked for.
            payload_data: dict[str, Any] = data
        else:
            endpoints = {e.get("source") for e in kept} | {e.get("target") for e in kept}
            payload_data = {
                "nodes": [n for n in nodes if n.get("id") in endpoints],
                "edges": list(kept),
            }
        return {
            "format": "json",
            "data": payload_data,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "truncated": len(kept) < len(edges),
        }

    _, payload = fit_to_budget(pool, cap=cap, render=_render)
    return payload


def register(server: FastMCP, memory: Memory) -> None:
    @annotated_tool(server, **READ_ONLY)
    def memo_graph_path(
        source: str,
        target: str,
        max_length: int = 5,
    ) -> dict[str, Any] | None:
        """Find shortest path between two entities in the entity graph.

        Uses BFS to find the shortest path. Two entities are connected if
        they share a memory. Returns the path including intermediate entities.

        Args:
            source: Source entity name.
            target: Target entity name.
            max_length: Maximum path length to search.
        """
        path = memory.navigator.find_shortest_path(source, target, max_length=max_length)
        return path.__dict__ if path else None

    @annotated_tool(server, **READ_ONLY)
    def memo_graph_neighbors(
        entity: str,
        max_neighbors: int = 50,
    ) -> dict[str, Any]:
        """Get direct neighbors of an entity in the graph.

        Returns entities directly connected to the given entity, along with
        the memories that connect them.

        Args:
            entity: Entity name.
            max_neighbors: Maximum neighbors to return.
        """
        neighbors = memory.navigator.get_neighbors(entity, max_neighbors=max_neighbors)
        return neighbors.to_bounded_dict()

    @annotated_tool(server, **READ_ONLY)
    def memo_explore(
        entity: str,
        max_neighbors: int = 8,
        max_memories: int = 8,
    ) -> dict[str, Any]:
        """Zoom into one entity: its degree, the neighbours it connects to (with
        how many memories bridge each link), and the memories that mention it —
        one rich "what's around X" view over the merged entity + code graph.

        Args:
            entity: Entity or code-symbol name.
            max_neighbors: Max neighbours to return.
            max_memories: Max mentioning memories to return.
        """
        from memo.explore import explore_entity

        return explore_entity(
            memory, entity, max_neighbors=max_neighbors, max_memories=max_memories
        )

    @annotated_tool(server, **READ_ONLY)
    def memo_graph_communities(
        min_size: int = 2,
        limit: Annotated[
            int,
            Field(
                description="Maximum communities to return, largest first. A full "
                "page (exactly `limit` items) means more may exist — raise `limit` "
                "or `min_size` to see past it."
            ),
        ] = 20,
        max_entities: Annotated[
            int,
            Field(
                description="Entity names to sample per community. The default "
                "is a summary sample, not the whole community — `size` always "
                "reports the true count and `entities_truncated` flags the "
                "sampling. Raise it to see more; the response budget still "
                "applies, so a large value may cost communities off the page."
            ),
        ] = _DEFAULT_COMMUNITY_ENTITIES,
    ) -> list[dict[str, Any]]:
        """Detect communities (connected components) in the entity graph.

        Uses connected components to find clusters of related entities.
        Useful for discovering thematic clusters in the knowledge graph.

        Bounded on both dimensions that scale with the corpus: the list is
        capped at `limit` (largest first) and each community's `entities` at
        `max_entities`. `size` is always the community's TRUE entity count, so
        `size > len(entities)` means the entity list was trimmed and
        `entities_truncated` says so outright. Past those counts the response
        is trimmed again to fit the MCP response budget, so a default call
        always returns data instead of a `response_budget_exceeded` error.

        The return stays a LIST — memo's tool surface is a public contract
        (PyPI, MCP registries, the Claude store), and wrapping it in an
        object to carry a community COUNT would break every caller doing
        `for c in result` and change the generated output schema. The count
        of communities beyond the page is therefore not reported; a full
        page is the signal, per the `limit` field description.

        Scope: the code layer merged into this graph comes from the codegraph
        index resolved from the CURRENT WORKING DIRECTORY (nearest
        .codegraph/codegraph.db), not from MEMO_DATA_DIR/MEMO_STATE_DIR, so
        results can include symbols from a repo outside the configured memo
        store. Set MEMO_GRAPH_USE_CODEGRAPH=0 for the memory-only graph.

        Args:
            min_size: Minimum community size to include.
            limit: Maximum communities to return, largest first.
            max_entities: Entity names sampled per community.
        """
        communities = memory.navigator.detect_communities(min_size=min_size)
        kept, _ = bounded_list(communities, cap=max(0, limit), key=lambda c: -c.size)
        bounded: list[dict[str, Any]] = []
        for c in kept:
            entities, entity_meta = bounded_list(c.entities, cap=max(0, max_entities))
            bounded.append(
                {
                    "id": c.id,
                    "representative_entity": c.representative_entity,
                    "size": c.size,
                    "entities": entities,
                    "entities_truncated": entity_meta["truncated"],
                }
            )
        fitted, _payload = fit_to_budget(
            bounded, cap=cap_for("memo_graph_communities"), render=list
        )
        return fitted

    @annotated_tool(server, **READ_ONLY)
    def memo_graph_trace(
        memory_id: Annotated[
            str | None,
            Field(
                description=(
                    "Memory id (or unique id prefix) whose linked code references to "
                    "return. Provide exactly one of memory_id or code."
                ),
            ),
        ] = None,
        code: Annotated[
            str | None,
            Field(
                description=(
                    "Code reference whose linked memories to return: a codegraph:// "
                    "URI (exact, case-insensitive) or a substring of a symbol label, "
                    "qualified name, file path, or stable symbol id. Provide exactly "
                    "one of memory_id or code."
                ),
            ),
        ] = None,
        limit: Annotated[
            int,
            Field(description="Maximum linked items to return (clamped to 1-200)."),
        ] = 50,
    ) -> dict[str, Any]:
        """Trace evidence from one memory to code, or one code reference to memories.

        Provide exactly one of ``memory_id`` or ``code``. Code accepts a stable
        codegraph URI, symbol id, qualified name, or file path.
        """
        return memory.graph_trace(memory_id=memory_id, code=code, limit=limit)

    @annotated_tool(server, **READ_ONLY)
    def memo_graph_discover(
        min_community_size: Annotated[
            int,
            Field(
                description=(
                    "Minimum node count for a connected component to be reported as "
                    "a community (components above the internal 40-node region cap "
                    "are also skipped)."
                ),
            ),
        ] = 4,
        min_bridge_side: Annotated[
            int,
            Field(
                description=(
                    "Minimum node count required on each side of an articulation "
                    "bridge (floored to 1)."
                ),
            ),
        ] = 2,
        max_communities: Annotated[
            int,
            Field(description="Maximum communities to return, largest first (floored to 0)."),
        ] = 5,
        max_bridges: Annotated[
            int,
            Field(
                description=(
                    "Maximum bridges to return, largest combined sides first (floored to 0)."
                ),
            ),
        ] = 5,
        include_code: Annotated[
            bool,
            Field(
                description=(
                    "If false, exclude codegraph:// code nodes from the graph before "
                    "detecting communities and bridges."
                ),
            ),
        ] = True,
    ) -> dict[str, Any]:
        """Discover bounded curated communities and bridges with exact evidence."""
        return memory.graph_discover(
            min_community_size=min_community_size,
            min_bridge_side=min_bridge_side,
            max_communities=max_communities,
            max_bridges=max_bridges,
            include_code=include_code,
        )

    @annotated_tool(server, **READ_ONLY)
    def memo_graph_centrality(
        top: int = 20,
    ) -> dict[str, Any]:
        """Compute centrality metrics for all entities.

        Returns degree centrality (number of connections) and betweenness
        centrality (how often entity lies on shortest paths). Useful for
        identifying hub entities in the graph.

        Args:
            top: Return top N entities by degree centrality.
        """
        scores = memory.navigator.compute_centrality()
        sorted_by_degree = sorted(scores.degree.items(), key=lambda x: x[1], reverse=True)[:top]
        return {
            "top_entities": [
                {"entity": e, "degree": d, "betweenness": scores.betweenness.get(e, 0.0)}
                for e, d in sorted_by_degree
            ],
            "total_entities": len(scores.degree),
        }

    @annotated_tool(server, **READ_ONLY)
    def memo_graph_export(
        format: str = "dot",
        include_memories: bool = False,
        max_edges: Annotated[
            int,
            Field(
                description=(
                    "Ceiling on edges returned, first seen first (floored to 0) — "
                    "FEWER come back when the response budget bites, which it does "
                    "well before 500 edges on a real graph, and sooner for json "
                    "than for dot. The true graph size is always reported in "
                    "edge_count/node_count, and truncated says whether edges were "
                    "dropped. For the whole graph use the CLI: "
                    "`memo graph export -o <file>`."
                ),
            ),
        ] = 500,
    ) -> dict[str, Any]:
        """Export a bounded slice of the entity graph for visualization.

        Returns graph data in the specified format. Use with external tools
        like Graphviz (dot format) or web visualization libraries (JSON format).
        The slice is sized to the MCP response budget, so a default call
        always returns a graph instead of a `response_budget_exceeded` error.

        Scope: the code layer merged into this graph comes from the codegraph
        index resolved from the CURRENT WORKING DIRECTORY (nearest
        .codegraph/codegraph.db), not from MEMO_DATA_DIR/MEMO_STATE_DIR, so
        results can include symbols from a repo outside the configured memo
        store. Set MEMO_GRAPH_USE_CODEGRAPH=0 for the memory-only graph.

        Args:
            format: Either "dot" for Graphviz DOT format or "json" for web UI.
            include_memories: If True and format is "json", include memory IDs in edge data.
            max_edges: Ceiling on edges returned; the true sizes come back alongside.
        """
        edge_ceiling = max(0, max_edges)
        budget = cap_for("memo_graph_export")
        if format == "dot":
            return _bounded_dot(
                memory.navigator.export_graphviz(), max_edges=edge_ceiling, cap=budget
            )
        data = memory.navigator.export_json(include_memories=include_memories)
        return _bounded_json(data, max_edges=edge_ceiling, cap=budget)
