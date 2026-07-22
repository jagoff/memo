"""Public rebuild and health operations for the curated graph projection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from memo.flags import flag_bool, flag_float, flag_int
from memo.graph_projection import (
    ProjectionBuildConfig,
    ProjectionBuildResult,
    ProjectionMemoryState,
)
from memo.lifecycle import IS_FORGOTTEN_KEY
from memo.memory._base import _MemoryBase


@dataclass(frozen=True)
class GraphRebuildResult:
    orphan_links_pruned: int
    entities_merged: int
    raw_edges: int
    projection: ProjectionBuildResult


class _GraphOpsMixin(_MemoryBase):
    def _projection_memory_states(self) -> dict[str, ProjectionMemoryState]:
        resolver = None
        if flag_bool("MEMO_GRAPH_CODE_TRACE_ENABLED"):
            from memo.code_traceability import CodeReferenceResolver

            resolver = CodeReferenceResolver()
        states: dict[str, ProjectionMemoryState] = {}
        for row in self.store.list_recent(limit=100_000):
            extra = row.get("extra") or {}
            states[str(row["id"])] = ProjectionMemoryState(
                id=str(row["id"]),
                type=str(row.get("type") or "note"),
                forgotten=bool(extra.get(IS_FORGOTTEN_KEY)),
                code_refs=resolver.resolve(extra) if resolver is not None else (),
            )
        return states

    @staticmethod
    def _projection_build_config() -> ProjectionBuildConfig:
        min_quality = flag_float("MEMO_GRAPH_PROJECTION_MIN_QUALITY")
        hub_ratio = flag_float("MEMO_GRAPH_HUB_MAX_DOC_FREQ_RATIO")
        return ProjectionBuildConfig(
            min_quality=0.45 if min_quality is None else min_quality,
            hub_max_doc_freq_ratio=0.25 if hub_ratio is None else hub_ratio,
        )

    def rebuild_graph(self) -> GraphRebuildResult:
        states = self._projection_memory_states()
        pruned = self.graph.prune_memory_links(set(states))
        merged = self.graph.canonicalize_existing()
        edges = self.graph.rebuild_edges()
        projection = self.graph.projection.rebuild(
            states,
            self._projection_build_config(),
        )
        return GraphRebuildResult(
            orphan_links_pruned=pruned,
            entities_merged=merged,
            raw_edges=edges,
            projection=projection,
        )

    def graph_health(self) -> dict[str, Any]:
        projection = self.graph.projection.health()
        max_age = flag_int("MEMO_GRAPH_PROJECTION_MAX_AGE_HOURS") or 36
        age = projection.get("age_hours")
        projection["stale"] = age is not None and float(age) > max_age
        return {
            "raw": self.graph.stats(),
            "edges": self.graph.edge_stats(),
            "projection": projection,
        }

    def rebuild_graph_if_due(self) -> GraphRebuildResult | None:
        if not flag_bool("MEMO_GRAPH_PROJECTION_ENABLED"):
            return None
        health = self.graph_health()["projection"]
        if not (health["dirty"] or not health["active_version"] or health["stale"]):
            return None
        return self.rebuild_graph()

    def graph_trace(
        self,
        *,
        memory_id: str | None = None,
        code: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Inspect active-projection memory↔code evidence in either direction."""
        empty: dict[str, Any] = {"code_refs": [], "memories": []}
        if bool(memory_id) == bool(code):
            return {
                "available": False,
                "reason": "exactly_one_of_memory_id_or_code_required",
                **empty,
            }
        max_age = flag_int("MEMO_GRAPH_PROJECTION_MAX_AGE_HOURS") or 36
        model = self.graph.projection.read_model(max_age)
        if not model.available:
            return {
                "available": False,
                "reason": model.skip_reason or "projection_unavailable",
                **empty,
            }
        bounded = max(1, min(int(limit), 200))
        if memory_id:
            resolved = self.resolve_id(memory_id) or memory_id
            links = model.code_links_for_memory(resolved)[:bounded]
            return {
                "available": True,
                "projection_version": model.version,
                "memory_id": resolved,
                "code_refs": [asdict(link) for link in links],
                "memories": [],
            }
        nodes = model.resolve_code(code or "")[:bounded]
        links = tuple(link for node in nodes for link in model.code_links_for_uri(node.uri))[
            :bounded
        ]
        memories: list[dict[str, Any]] = []
        seen: set[str] = set()
        for link in links:
            if link.memory_id in seen:
                continue
            seen.add(link.memory_id)
            record = self.get(link.memory_id)
            memories.append(
                {
                    "id": link.memory_id,
                    "title": getattr(record, "title", "") if record is not None else "",
                    "type": getattr(record, "type", "") if record is not None else "",
                    "relation": link.relation,
                    "evidence_id": link.evidence_id,
                }
            )
        return {
            "available": True,
            "projection_version": model.version,
            "query": code,
            "code_refs": [
                asdict(link) for node in nodes for link in model.code_links_for_uri(node.uri)[:1]
            ],
            "memories": memories,
        }

    def graph_discover(
        self,
        *,
        min_community_size: int = 4,
        min_bridge_side: int = 2,
        max_communities: int = 5,
        max_bridges: int = 5,
        max_region_size: int = 40,
        include_code: bool = True,
    ) -> dict[str, Any]:
        """Build a read-only insight packet from the active curated projection."""
        if not flag_bool("MEMO_GRAPH_DISCOVERY_ENABLED"):
            return {
                "available": False,
                "reason": "disabled",
                "projection_version": None,
                "communities": [],
                "bridges": [],
            }
        from memo.graph_discovery import discover_graph

        max_age = flag_int("MEMO_GRAPH_PROJECTION_MAX_AGE_HOURS") or 36
        model = self.graph.projection.read_model(max_age)
        return discover_graph(
            model,
            min_community_size=min_community_size,
            min_bridge_side=min_bridge_side,
            max_communities=max_communities,
            max_bridges=max_bridges,
            max_region_size=max_region_size,
            include_code=include_code,
        )


__all__ = ["GraphRebuildResult"]
