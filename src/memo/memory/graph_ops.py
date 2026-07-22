"""Public rebuild and health operations for the curated graph projection."""

from __future__ import annotations

from dataclasses import dataclass
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
        states: dict[str, ProjectionMemoryState] = {}
        for row in self.store.list_recent(limit=100_000):
            extra = row.get("extra") or {}
            states[str(row["id"])] = ProjectionMemoryState(
                id=str(row["id"]),
                type=str(row.get("type") or "note"),
                forgotten=bool(extra.get(IS_FORGOTTEN_KEY)),
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
        if not (
            health["dirty"]
            or not health["active_version"]
            or health["stale"]
        ):
            return None
        return self.rebuild_graph()


__all__ = ["GraphRebuildResult"]
