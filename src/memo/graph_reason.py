"""User-facing explanations for curated graph retrieval evidence."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from memo.graph_signal import GraphEvidenceTrace


def build_graph_reason(
    memory_id: str,
    trace: GraphEvidenceTrace,
    *,
    relations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    reason: dict[str, Any] = {
        "memory_id": memory_id,
        "projection_version": trace.projection_version,
        "mode": trace.mode,
        "query_nodes": list(trace.query_nodes),
        "hit_nodes": list(trace.hit_nodes),
        "edges": [asdict(edge) for edge in trace.edges],
        "normalized_signal": trace.normalized_signal,
        "confidence": "derived",
    }
    for edge in reason["edges"]:
        edge["evidence_ids"] = list(edge["evidence_ids"])
    if trace.hub_suppressed:
        reason["hub_suppressed"] = list(trace.hub_suppressed)
    if relations:
        reason["relations"] = relations
    return reason


def format_graph_reason(reason: dict[str, Any]) -> str:
    mode = str(reason.get("mode") or "graph")
    query_nodes = [str(value) for value in (reason.get("query_nodes") or [])]
    hit_nodes = [str(value) for value in (reason.get("hit_nodes") or [])]
    # Read legacy explanations too; old memories/eval fixtures may retain them.
    if not query_nodes:
        query_nodes = [str(value) for value in (reason.get("query_entities") or [])]
    if not hit_nodes:
        hit_nodes = [str(value) for value in (reason.get("hit_entities") or [])]
    if query_nodes and hit_nodes:
        return f"related via graph ({mode}): {', '.join(query_nodes)} -> {', '.join(hit_nodes)}"
    if query_nodes:
        return f"related via graph ({mode}): {', '.join(query_nodes)}"
    return f"related via graph ({mode})"
