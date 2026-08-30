"""MCP tool — one consolidated, read-only graph navigator (``memo_graph``).

A single small tool on the default agent profile that dispatches to the existing
``GraphNavigator`` / ``explore_entity`` corpus-navigation methods. Read-only,
returns compact JSON-serialisable dicts, and keeps the token surface to one tool
instead of exposing the advanced ``memo_graph_*`` tools on the default profile
(those stay gated to the full profile in ``server_graph.py``).

This is corpus navigation, not cognition — it carries no
``agent``/``cognitive``/``federation``/``lifecycle``/``suggest`` verb, so the
brain-like-tools architecture guard still holds.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from memo.mcp_budget import bounded_list
from memo.memory import Memory
from memo.server_annotations import READ_ONLY, annotated_tool

# Per-community entity cap. A community's membership is elastic (156 entities
# in the largest live one) and a caller asking for communities wants the shape,
# not every member — `entities_truncated` says when there is more.
_MAX_COMMUNITY_ENTITIES = 25

_VERBS = ["path", "neighbors", "explore", "communities", "why", "impact", "architecture"]


def _with_code_evidence(
    payload: dict[str, Any],
    *,
    include_code: bool,
) -> dict[str, Any]:
    if not include_code:
        return payload
    from memo import codegraph_loader
    from memo.code_evidence import codegraph_evidence
    from memo.code_traceability import codegraph_repo_id

    # `_resolve_db`, not the bare constant — the constant is derived from
    # `__file__` and points inside site-packages under an isolated `uv tool`
    # install, so the tool reports "missing" on a machine whose index is
    # configured and readable. See the same fix in `code_traceability`.
    db_path = codegraph_loader._resolve_db()
    repo_root = db_path.parent.parent
    payload["code_evidence"] = codegraph_evidence(
        db_path=db_path,
        repo_root=repo_root,
        repo_id=codegraph_repo_id(repo_root),
    ).to_dict()
    return payload


def _bounded_impact(result: dict[str, Any], *, limit: int) -> dict[str, Any]:
    """Trim the symbol walk to what a tool result can carry.

    `code_change_impact` caps its traversal at 1000 symbols — a bound sized for
    the graph walk, not for a response: a 13-file working tree yielded 376 rows
    (~100k chars), past the client's token cap. `impacted_paths` already gives
    the blast radius, so the rows (nearest-first) are cut to `limit` and the
    true size kept in `symbol_count`.
    """
    symbols = result.get("symbols")
    if not isinstance(symbols, list) or len(symbols) <= limit:
        return result
    bounded = dict(result)
    bounded["symbols"] = symbols[: max(0, limit)]
    bounded["symbol_count"] = len(symbols)
    return bounded


def _communities_payload(communities: list[Any], *, limit: int) -> dict[str, Any]:
    """Bound BOTH elastic dimensions of the communities verb.

    The page was a bare slice with no `total`/`truncated`, and each community
    carried its FULL entity list — measured on the live corpus: 4,327
    communities, the largest holding 156 entities, ~12k tokens at the default
    limit of 8.
    """
    kept, meta = bounded_list(communities, cap=limit, key=lambda c: -c.size)
    result = []
    for community in kept:
        entities = list(getattr(community, "entities", []) or [])
        shown = entities[:_MAX_COMMUNITY_ENTITIES]
        result.append(
            {
                **{k: v for k, v in community.__dict__.items() if k != "entities"},
                "entities": shown,
                "entities_shown": len(shown),
                "entities_truncated": len(entities) > len(shown),
            }
        )
    return {"verb": "communities", "result": result, **meta}


def _memory_navigation_result(
    memory: Memory,
    verb: str,
    *,
    a: str | None,
    b: str | None,
    focus: str | None,
    limit: int,
    include_code: bool,
) -> dict[str, Any] | None:
    nav = memory.navigator
    use_codegraph = None if include_code else False
    if verb == "path":
        if not a or not b:
            return {"error": "path requires a and b"}
        path = nav.find_shortest_path(a, b, use_codegraph=use_codegraph)
        payload = {"verb": "path", "result": path.__dict__ if path else None}
        return _with_code_evidence(payload, include_code=include_code)
    if verb == "why":
        if not a or not b:
            return {"error": "why requires a and b"}
        payload = {"verb": "why", "result": nav.why_connected(a, b, use_codegraph=use_codegraph)}
        return _with_code_evidence(payload, include_code=include_code)
    if verb == "neighbors":
        if not focus:
            return {"error": "neighbors requires entity (or a)"}
        payload = {
            "verb": "neighbors",
            "result": nav.get_neighbors(
                focus,
                max_neighbors=limit,
                use_codegraph=use_codegraph,
            ).to_bounded_dict(),
        }
        return _with_code_evidence(payload, include_code=include_code)
    if verb == "explore":
        if not focus:
            return {"error": "explore requires entity (or a)"}
        from memo.explore import explore_entity

        payload = {
            "verb": "explore",
            "result": explore_entity(
                memory,
                focus,
                max_neighbors=limit,
                max_memories=limit,
                use_codegraph=use_codegraph,
            ),
        }
        return _with_code_evidence(payload, include_code=include_code)
    if verb == "communities":
        communities = nav.detect_communities(min_size=2, use_codegraph=use_codegraph)
        # Both dimensions were unbounded-in-effect: the page was a bare slice
        # with no `total`/`truncated`, and each community carried its FULL
        # entity list. Measured on the live corpus: 4,327 communities, the
        # largest holding 156 entities — the default limit=8 rendered ~12k
        # tokens, over the response budget, and reported neither fact.
        payload = _communities_payload(communities, limit=limit)
        return _with_code_evidence(payload, include_code=include_code)
    return None


def _code_navigation_result(
    memory: Memory,
    verb: str,
    *,
    cwd: str | None,
    focus: str | None,
    depth: int,
    limit: int,
    scope: str | None,
    mode: str,
    cursor: str | None,
    max_chars: int,
) -> dict[str, Any] | None:
    if verb == "impact":
        if not cwd:
            return {"error": "impact requires cwd"}
        return {
            "verb": "impact",
            "result": _bounded_impact(
                memory.code_change_impact(cwd, depth=depth, limit=limit),
                limit=limit,
            ),
        }
    if verb == "architecture":
        if not cwd:
            return {"error": "architecture requires cwd"}
        return {
            "verb": "architecture",
            "result": memory.code_context_pack(
                cwd,
                focus=focus,
                scope=scope,
                mode=mode,
                limit=limit,
                cursor=cursor,
                max_chars=max_chars,
            ),
        }
    return None


def _memo_graph_result(
    memory: Memory,
    verb: str,
    *,
    a: str | None,
    b: str | None,
    entity: str | None,
    limit: int,
    include_code: bool,
    cwd: str | None,
    depth: int,
    scope: str | None,
    mode: str,
    cursor: str | None,
    max_chars: int,
) -> dict[str, Any]:
    normalized = (verb or "").strip().lower()
    focus = entity or a
    result = _memory_navigation_result(
        memory,
        normalized,
        a=a,
        b=b,
        focus=focus,
        limit=limit,
        include_code=include_code,
    )
    if result is not None:
        return result
    result = _code_navigation_result(
        memory,
        normalized,
        cwd=cwd,
        focus=focus,
        depth=depth,
        limit=limit,
        scope=scope,
        mode=mode,
        cursor=cursor,
        max_chars=max_chars,
    )
    if result is not None:
        return result
    return {"error": f"unknown verb: {verb}", "verbs": _VERBS}


def register(server: FastMCP, memory: Memory) -> None:
    @annotated_tool(server, **READ_ONLY)
    def memo_graph(
        verb: str,
        a: str | None = None,
        b: str | None = None,
        entity: str | None = None,
        limit: int = 8,
        include_code: bool = False,
        cwd: str | None = None,
        depth: int = 1,
        scope: str | None = None,
        mode: str = "scout",
        cursor: str | None = None,
        max_chars: int = 12_000,
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
        - ``"impact"``: changed code plus linked durable memories for ``cwd``.
        - ``"architecture"``: a bounded CodeContextPack for ``cwd``. ``entity``
          (or ``a``) selects a symbol/path focus; ``mode`` is scout, verify, or
          audit and ``cursor`` continues an incomplete page.

        By default this navigates the MEMORY graph only (entities linked through
        shared memories). Set ``include_code=True`` to also fold in the codegraph
        code-structure layer (call/extends/etc. edges between code symbols).

        Args:
            verb: One of path | neighbors | explore | communities | why |
                impact | architecture.
            a: First entity (path/why source; fallback for entity).
            b: Second entity (path/why target).
            entity: Entity name for neighbors/explore.
            limit: Result cap (neighbours, mentioning memories, communities).
            include_code: Fold in the codegraph code-structure layer (default off
                → memory-only, so results are durable-memory navigation).
            cwd: Git working tree used by the impact verb.
            depth: CodeGraph hop depth for impact (bounded to 0..3).
            scope: Repo-relative path bounding architecture findings.
            mode: Architecture evidence mode: scout | verify | audit.
            cursor: Opaque continuation cursor returned by architecture.
            max_chars: Approximate architecture finding budget.
        """
        return _memo_graph_result(
            memory,
            verb,
            a=a,
            b=b,
            entity=entity,
            limit=limit,
            include_code=include_code,
            cwd=cwd,
            depth=depth,
            scope=scope,
            mode=mode,
            cursor=cursor,
            max_chars=max_chars,
        )
