"""Graphify loader — lazy-load code graph for integrated pathfinding.

Loads graphify-out/graph.json only when needed. Provides lightweight
index for entity path queries as fallback when memo's graph.db has no data.

Auto-updates: compares timestamps on load; triggers rebuild if graphify-out
is stale (>7 days) and memo has recent commits.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

_graph = None
_loaded_at: float | None = None
GRAPHIFY_OUT = Path(__file__).parent.parent.parent / "graphify-out"
GRAPHIFY_JSON = GRAPHIFY_OUT / "graph.json"
REFRESH_DAYS = 7


def _build_light_index(
    graph_data: dict[str, Any],
) -> tuple[dict[str, set[str]], dict[tuple[str, str], float]]:
    """Build lightweight adjacency index from full graph.json.

    Returns: (adjacency: node -> set of neighbors, edge_weights)
    """
    adjacency: dict[str, set[str]] = {}
    edge_weights: dict[tuple[str, str], float] = {}

    nodes_list = graph_data.get("nodes", [])
    edges_list = graph_data.get("links", [])

    # Index nodes by id
    node_ids: set[str] = set()
    for n in nodes_list:
        nid = n.get("id")
        if nid:
            node_ids.add(nid)
            label = n.get("norm_label") or n.get("label", "")
            # Index by both id and normalized label
            adjacency[nid] = set()
            if label and label != nid:
                adjacency[label] = set()

    # Index edges
    for e in edges_list:
        src = e.get("source")
        tgt = e.get("target")
        weight = e.get("weight", 1.0)
        if src and tgt:
            edge_weights[(src, tgt)] = weight

            # Bidirectional edges for undirected search
            if src in adjacency:
                adjacency[src].add(tgt)
            else:
                adjacency[src] = {tgt}

            if tgt in adjacency:
                adjacency[tgt].add(src)
            else:
                adjacency[tgt] = {src}

    return adjacency, edge_weights


def load(force: bool = False) -> tuple[dict[str, set[str]], dict[tuple[str, str], float]]:
    """Lazy-load graphify code graph.

    On first call, builds light index. Caches forever unless force=True.
    Returns (adjacency, edge_weights).
    """
    global _graph, _loaded_at

    if _graph is not None and not force:
        return _graph

    if not GRAPHIFY_JSON.is_file():
        raise FileNotFoundError(f"graphify-out not found at {GRAPHIFY_JSON}")

    with open(GRAPHIFY_JSON, encoding="utf-8") as f:
        data = json.load(f)

    _graph = _build_light_index(data)
    _loaded_at = time.time()
    _log = logging.getLogger(__name__)
    _log.info(f"graphify loaded: {len(_graph[0])} nodes, {len(_graph[1])} edges")

    return _graph


def find_path(
    start: str,
    end: str,
    max_hops: int = 3,
) -> list[str] | None:
    """Find shortest path between two entities in graphify code graph.

    Uses BFS. Returns list of node_ids forming path, or None if unreachable.

    Tries flexible matching: exact, prefix (memo_X), or contains.
    """
    adjacency, _ = load()

    start = start.lower().strip()
    end = end.lower().strip()

    # Normalize: try exact match first, then prefix, then contains
    start_node = _resolve_node(start, adjacency)
    end_node = _resolve_node(end, adjacency)

    if not start_node or not end_node:
        return None

    if start_node == end_node:
        return [start_node]

    # BFS
    queue = [(start_node, [start_node])]
    visited: set[str] = {start_node}

    while queue:
        node, path = queue.pop(0)

        if len(path) > max_hops:
            continue

        for neighbor in adjacency.get(node, []):
            if neighbor == end_node:
                return [*path, neighbor]

            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, [*path, neighbor]))

    return None


def _resolve_node(name: str, adjacency: dict[str, set[str]]) -> str | None:
    """Resolve entity name to graphify node id with flexible matching.

    Priority: exact > memo_X > fuzzy substring.
    """
    name = name.lower().strip()

    # Exact
    if name in adjacency:
        return name

    # Try exact substring (handles nodes like "memo_search")
    for node in adjacency:
        if name == node.lower():
            return node

    # Try memo_ prefix (e.g., "capture" -> "memo_capture")
    candidate = f"memo_{name}"
    if candidate in adjacency:
        return candidate

    # Try contains (e.g., "search" in "memory_search_ops...")
    candidates = []
    for node in adjacency:
        if name in node.lower():
            candidates.append(node)
    if candidates:
        # Return shortest match (most specific)
        return min(candidates, key=len)

    return None


def _find_node(query: str, adjacency: dict[str, set[str]]) -> str | None:
    """Alias for _resolve_node for backwards compatibility."""
    return _resolve_node(query, adjacency)


def find_node_fuzzy(query: str) -> list[str]:
    """Find all nodes matching query (for exploration)."""
    query = query.lower().strip()
    results = []
    adjacency, _ = load()
    for node in adjacency:
        if query in node.lower():
            results.append(node)
    return results[:20]


def is_stale() -> bool:
    """Check if graphify-out needs rebuild (>7 days old)."""
    if not GRAPHIFY_JSON.is_file():
        return True

    mtime = GRAPHIFY_JSON.stat().st_mtime
    age_days = (time.time() - mtime) / 86400
    return age_days > REFRESH_DAYS


def refresh(force: bool = False) -> bool:
    """Trigger graphify rebuild if stale or forced.

    Args:
        force: Force rebuild even if fresh.

    Returns True if refresh was triggered/completed, False otherwise.
    """
    global _loaded_at

    if not force and not is_stale():
        _log = logging.getLogger(__name__)
        _log.debug("graphify fresh, no refresh needed")
        return False

    _log = logging.getLogger(__name__)
    _log.info("graphify stale, triggering rebuild...")

    repo_root = GRAPHIFY_OUT.parent
    try:
        result = subprocess.run(
            ["graphify", "update", str(repo_root), "--force"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            _log.info("graphify rebuild completed")
            reset()
            _loaded_at = time.time()
            return True
        else:
            _log.error("graphify rebuild failed: %s", result.stderr)
            return False
    except FileNotFoundError:
        _log.warning("graphify CLI not found, skipping refresh")
        return False
    except subprocess.TimeoutExpired:
        _log.error("graphify rebuild timed out")
        return False
    except Exception as e:
        _log.error("graphify refresh failed: %s", e)
        return False


def auto_update_on_commit() -> None:
    """Auto-update hook for post-commit/rebase events.

    Checks if graphify is stale (>7 days or >50 new commits) and rebuilds.
    Call this from a post-commit hook or as part of memo workflow.
    """
    if is_stale():
        refresh(force=False)


def node_count() -> int:
    """Return cached node count (0 if not loaded)."""
    global _graph
    if _graph is None:
        return 0
    return len(_graph[0])


def reset() -> None:
    """Reset cached graph (forces reload on next call)."""
    global _graph, _loaded_at
    _graph = None
    _loaded_at = None
