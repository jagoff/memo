"""MCP tool — one consolidated, read-only graph navigator (``memo_graph``).

A single small tool on the default agent profile that dispatches to the existing
``GraphNavigator`` / ``explore_entity`` corpus-navigation methods. Read-only,
returns compact JSON-serialisable dicts, and keeps the token surface to one tool
instead of the nine advanced ``memo_graph_*`` tools (those stay gated to the
full profile in ``server_graph.py``).

This is corpus navigation, not cognition — it carries no
``agent``/``cognitive``/``federation``/``lifecycle``/``suggest`` verb, so the
brain-like-tools architecture guard still holds.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from memo.memory import Memory
from memo.server_annotations import READ_ONLY, annotated_tool

_VERBS = ["path", "neighbors", "explore", "communities", "why"]


def register(server: FastMCP, memory: Memory) -> None:
    @annotated_tool(server, **READ_ONLY)
    def memo_graph(
        verb: str,
        a: str | None = None,
        b: str | None = None,
        entity: str | None = None,
        limit: int = 8,
        include_code: bool = False,
    ) -> dict[str, Any]:
        """Navigate the entity knowledge graph (read-only).

        One consolidated explorer over memo's corpus graph. Pick a ``verb``:

        - ``"path"``: shortest entity path from ``a`` to ``b`` (fewest hops).
        - ``"why"``: weighted shortest path ``a``->``b`` as evidence — the same
          route with each hop's edge weight (how many memories bridge it), so a
          connection is explained, not just asserted.
        - ``"neighbors"``: direct neighbours of ``entity`` (or ``a``).
        - ``"explore"``: a rich "what's around X" view of ``entity`` (or ``a``) —
          degree, neighbours, and the memories that mention it.
        - ``"communities"``: clusters of related entities (``limit`` caps count).

        By default this navigates the MEMORY graph only (entities linked through
        shared memories). Set ``include_code=True`` to also fold in the codegraph
        code-structure layer (call/extends/etc. edges between code symbols).

        Args:
            verb: One of path | neighbors | explore | communities | why.
            a: First entity (path/why source; fallback for entity).
            b: Second entity (path/why target).
            entity: Entity name for neighbors/explore.
            limit: Result cap (neighbours, mentioning memories, communities).
            include_code: Fold in the codegraph code-structure layer (default off
                → memory-only, so results are durable-memory navigation).
        """
        nav = memory.navigator
        v = (verb or "").strip().lower()
        focus = entity or a
        # Memory navigator by default: entity-only graph unless code is requested.
        uc = None if include_code else False

        if v == "path":
            if not a or not b:
                return {"error": "path requires a and b"}
            path = nav.find_shortest_path(a, b, use_codegraph=uc)
            return {"verb": "path", "result": path.__dict__ if path else None}

        if v == "why":
            if not a or not b:
                return {"error": "why requires a and b"}
            return {"verb": "why", "result": nav.weighted_path(a, b, use_codegraph=uc)}

        if v == "neighbors":
            if not focus:
                return {"error": "neighbors requires entity (or a)"}
            return {
                "verb": "neighbors",
                "result": nav.get_neighbors(focus, max_neighbors=limit, use_codegraph=uc).__dict__,
            }

        if v == "explore":
            if not focus:
                return {"error": "explore requires entity (or a)"}
            from memo.explore import explore_entity

            return {
                "verb": "explore",
                "result": explore_entity(
                    memory, focus, max_neighbors=limit, max_memories=limit, use_codegraph=uc
                ),
            }

        if v == "communities":
            communities = nav.detect_communities(min_size=2, use_codegraph=uc)
            return {
                "verb": "communities",
                "result": [c.__dict__ for c in communities[:limit]],
            }

        return {"error": f"unknown verb: {verb}", "verbs": _VERBS}
