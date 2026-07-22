"""Codegraph loader — lazy-load code graph for integrated pathfinding.

Reads the codegraph SQLite index (``.codegraph/codegraph.db``) only when
needed. Provides a lightweight symbol adjacency index for entity path queries
as a fallback when memo's own knowledge graph (``graph.db``) has no data.

Unlike a build-on-demand backend, codegraph keeps its index continuously fresh
via its own file-watcher, so this loader never shells out to rebuild — it reads
the live DB and caches the adjacency in-process, reloading automatically when the
DB's mtime advances so long-lived processes don't serve a frozen graph.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

_graph: tuple[dict[str, set[str]], dict[tuple[str, str], float]] | None = None
_loaded_at: float | None = None
CODEGRAPH_DIR = Path(__file__).parent.parent.parent / ".codegraph"
CODEGRAPH_DB = CODEGRAPH_DIR / "codegraph.db"

# Symbol→symbol edge kinds worth traversing for pathfinding. ``contains`` /
# ``imports`` are structural (file→member, file→import) and only add noise to a
# symbol reachability graph, so they are excluded.
_EDGE_KINDS = ("calls", "instantiates", "extends", "decorates")


def _build_light_index(
    rows: list[tuple[str, str]],
) -> tuple[dict[str, set[str]], dict[tuple[str, str], float]]:
    """Build a lightweight undirected adjacency index from (src_name, tgt_name) rows.

    Returns: (adjacency: name -> set of neighbor names, edge_weights).
    Names are lowercased so callers can match by human symbol name.
    """
    adjacency: dict[str, set[str]] = {}
    edge_weights: dict[tuple[str, str], float] = {}

    for src_name, tgt_name in rows:
        if not src_name or not tgt_name:
            continue
        src = src_name.lower()
        tgt = tgt_name.lower()
        if src == tgt:
            continue
        edge_weights[(src, tgt)] = edge_weights.get((src, tgt), 0.0) + 1.0
        adjacency.setdefault(src, set()).add(tgt)
        adjacency.setdefault(tgt, set()).add(src)

    return adjacency, edge_weights


def load(force: bool = False) -> tuple[dict[str, set[str]], dict[tuple[str, str], float]]:
    """Lazy-load the codegraph symbol graph.

    On first call, builds the light index from ``.codegraph/codegraph.db``.
    Caches forever unless ``force=True``. Returns (adjacency, edge_weights).
    """
    global _graph, _loaded_at

    if _graph is not None and not force:
        return _graph

    if not CODEGRAPH_DB.is_file():
        raise FileNotFoundError(f"codegraph index not found at {CODEGRAPH_DB}")

    # Static, fully-parameterized query — one '?' per _EDGE_KINDS entry. The
    # bind count is asserted by test_load_builds_symbol_adjacency, which fails
    # loudly if _EDGE_KINDS and these placeholders ever drift apart.
    query = (
        "SELECT s.name, t.name FROM edges e "
        "JOIN nodes s ON e.source = s.id "
        "JOIN nodes t ON e.target = t.id "
        "WHERE e.kind IN (?, ?, ?, ?)"
    )
    conn = sqlite3.connect(f"file:{CODEGRAPH_DB}?mode=ro", uri=True)
    try:
        rows = conn.execute(query, _EDGE_KINDS).fetchall()
    finally:
        conn.close()

    _graph = _build_light_index(rows)
    _loaded_at = time.time()
    logging.getLogger(__name__).info(
        "codegraph loaded: %d nodes, %d edges", len(_graph[0]), len(_graph[1])
    )

    return _graph


def is_stale() -> bool:
    """Codegraph self-maintains its index via a file-watcher.

    There is nothing to rebuild from here, so the index is "stale" only when it
    is missing entirely.
    """
    return not CODEGRAPH_DB.is_file()


def refresh(force: bool = False) -> bool:
    """No-op kept for API parity.

    Codegraph keeps ``.codegraph/codegraph.db`` continuously up to date through
    its own watcher; there is no rebuild to trigger from memo. Always returns
    False (nothing was done).
    """
    logging.getLogger(__name__).debug("codegraph self-maintains; refresh is a no-op")
    return False


def auto_update_on_commit() -> None:
    """No-op kept for API parity (codegraph watches the working tree itself)."""
    return None


def reset() -> None:
    """Reset the cached graph (forces a reload on the next call)."""
    global _graph, _loaded_at
    _graph = None
    _loaded_at = None
