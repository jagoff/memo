"""User-facing explanations for graph-derived retrieval signal."""

from __future__ import annotations

from typing import Any

from memo.graph_signal import GraphSignalTrace


def build_graph_reason(
    memory_id: str,
    trace: GraphSignalTrace,
    *,
    relations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    reason: dict[str, Any] = {
        "memory_id": memory_id,
        "mode": trace.mode,
        "query_entities": trace.query_entities,
        "hit_entities": trace.hit_entities,
        "confidence": "derived",
    }
    if trace.neighbor_edges:
        reason["neighbor_edges"] = trace.neighbor_edges
    if relations:
        reason["relations"] = relations
    if trace.skipped:
        reason["skipped"] = trace.skipped
    return reason


def format_graph_reason(reason: dict[str, Any]) -> str:
    mode = str(reason.get("mode") or "graph")
    query_entities = [str(e) for e in (reason.get("query_entities") or [])]
    hit_entities = [str(e) for e in (reason.get("hit_entities") or [])]
    if query_entities and hit_entities:
        return f"related via graph ({mode}): {', '.join(query_entities)} -> {', '.join(hit_entities)}"
    if query_entities:
        return f"related via graph ({mode}): {', '.join(query_entities)}"
    return f"related via graph ({mode})"
