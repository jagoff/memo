"""Codegraph loader — lazy-load code graph for integrated pathfinding.

Reads the codegraph SQLite index (``.codegraph/codegraph.db``) only when
needed. Provides a lightweight symbol adjacency index for entity path queries
as a fallback when memo's own knowledge graph (``graph.db``) has no data.

Unlike a build-on-demand backend, codegraph keeps its index continuously fresh
via its own file-watcher, so this loader never shells out to rebuild — it reads
the live DB and caches the adjacency in-process, reloading automatically when the
DB's mtime advances so long-lived processes don't serve a frozen graph.

The DB is resolved per call: explicit ``db_path`` > nearest
``.codegraph/codegraph.db`` walking up from cwd (project-aware discovery,
kill-switch ``MEMO_CODEGRAPH_DISCOVERY=0``) > ``MEMO_CODEGRAPH_DB`` (env, then
Markdown config — pins daemons whose cwd is outside any repo) > module-level
``CODEGRAPH_DB`` (memo's own checkout — the historical behavior).
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from pathlib import Path

_Graph = tuple[dict[str, set[str]], dict[tuple[str, str], float]]

# Per-DB cache: resolved DB path -> (graph, DB mtime at load). A changed mtime
# invalidates the entry; distinct projects each keep their own warm graph.
_cache: dict[Path, tuple[_Graph, float]] = {}
_loaded_at: float | None = None
CODEGRAPH_DIR = Path(__file__).parent.parent.parent / ".codegraph"
CODEGRAPH_DB = CODEGRAPH_DIR / "codegraph.db"

_FALSE = {"0", "false", "no", "off"}

# Symbol→symbol edge kinds worth traversing for pathfinding. ``contains`` /
# ``imports`` are structural (file→member, file→import) and only add noise to a
# symbol reachability graph, so they are excluded.
_EDGE_KINDS = ("calls", "instantiates", "extends", "decorates")

# Hard cap on traversable edges read by one load(). Guards the recall-hook hot
# path: with discovery on, a monorepo-sized index would otherwise be scanned in
# full inside the recall assoc budget.
_DEFAULT_MAX_EDGES = 300_000


def _max_edges() -> int:
    """Edge-count cap (``MEMO_CODEGRAPH_MAX_EDGES``, default 300000).

    Read raw from the environment (hot-path leaf, like
    ``MEMO_CODEGRAPH_DISCOVERY``); registered in flags_misc.py for
    ``memo config validate``. Unparseable values fall back to the default.
    """
    raw = os.environ.get("MEMO_CODEGRAPH_MAX_EDGES", "").strip()
    try:
        return int(raw) if raw else _DEFAULT_MAX_EDGES
    except ValueError:
        return _DEFAULT_MAX_EDGES


def _discovery_enabled() -> bool:
    """Project-aware discovery kill-switch (``MEMO_CODEGRAPH_DISCOVERY``, default on).

    Read raw from the environment (like ``MEMO_GPU_XPROC_LOCK`` in mlx_gpu.py)
    to keep the recall hot path free of flags-registry imports.
    """
    return os.environ.get("MEMO_CODEGRAPH_DISCOVERY", "").strip().lower() not in _FALSE


def _discover_db(start: Path | None = None) -> Path | None:
    """Walk from ``start`` (default cwd) upward to the nearest ``.codegraph/codegraph.db``.

    One stat per ancestor directory — cheap enough for the recall assoc budget.
    Returns None when no index exists anywhere up the tree.
    """
    base = (start or Path.cwd()).resolve()
    for directory in (base, *base.parents):
        candidate = directory / ".codegraph" / "codegraph.db"
        if candidate.is_file():
            return candidate
    return None


def _db_override() -> Path | None:
    """Explicit index path (``MEMO_CODEGRAPH_DB``), or None when unset.

    Consulted only after cwd discovery fails, so a process whose cwd is outside
    any repo (a launchd daemon at ``$HOME``, a pipx/uv-tool install whose
    module-relative ``CODEGRAPH_DB`` points inside site-packages) can still be
    pinned to a real index — without overriding project-awareness when cwd
    discovery does find a nearer one. Raw env is read first (no registry import
    when the var is exported); the flags registry — which folds in the Markdown
    config layer, reaching daemons that inherit no shell env — is imported
    lazily only when the env var is unset.
    """
    raw = os.environ.get("MEMO_CODEGRAPH_DB", "").strip()
    if not raw:
        try:
            from memo.flags import flag_str
        except ImportError:  # circular-import guard on unusual import orders
            return None
        raw = (flag_str("MEMO_CODEGRAPH_DB") or "").strip()
    return Path(raw).expanduser() if raw else None


def _resolve_db(db_path: Path | None = None) -> Path:
    """Resolution order: explicit ``db_path`` > cwd discovery >
    ``MEMO_CODEGRAPH_DB`` (env/Markdown config) > ``CODEGRAPH_DB``."""
    if db_path is not None:
        return db_path
    if _discovery_enabled():
        discovered = _discover_db()
        if discovered is not None:
            return discovered
    override = _db_override()
    if override is not None:
        return override
    return CODEGRAPH_DB


def _build_light_index(
    rows: list[tuple[str, str]],
) -> _Graph:
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


def load(force: bool = False, db_path: Path | None = None) -> _Graph:
    """Lazy-load the codegraph symbol graph for the resolved DB.

    The DB comes from ``db_path`` when given, else the nearest
    ``.codegraph/codegraph.db`` above cwd, else ``CODEGRAPH_DB``. The built
    index is cached per DB path and reloaded when the DB's mtime changes;
    ``force=True`` bypasses the cache. Returns (adjacency, edge_weights).

    Size guard: when the DB holds more than ``MEMO_CODEGRAPH_MAX_EDGES``
    traversable edges, the full edge scan is skipped. If a graph was previously
    cached for this DB it is served even though its mtime is stale (stale-serve
    beats no graph inside the recall assoc budget); otherwise a RuntimeError is
    raised — consumers already degrade to None on any exception.
    """
    global _loaded_at

    db = _resolve_db(db_path)
    try:
        mtime = db.stat().st_mtime
    except OSError as exc:
        raise FileNotFoundError(f"codegraph index not found at {db}") from exc

    if not force:
        cached = _cache.get(db)
        if cached is not None and cached[1] == mtime:
            return cached[0]

    # Static, fully-parameterized query — one '?' per _EDGE_KINDS entry. The
    # bind count is asserted by test_load_builds_symbol_adjacency, which fails
    # loudly if _EDGE_KINDS and these placeholders ever drift apart.
    query = (
        "SELECT s.name, t.name FROM edges e "
        "JOIN nodes s ON e.source = s.id "
        "JOIN nodes t ON e.target = t.id "
        "WHERE e.kind IN (?, ?, ?, ?)"
    )
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        # Size guard before the full edge scan: the COUNT is cheap (served by
        # idx_edges_kind) while the SELECT below reads every traversable edge.
        (edge_count,) = conn.execute(
            "SELECT COUNT(*) FROM edges WHERE kind IN (?, ?, ?, ?)", _EDGE_KINDS
        ).fetchone()
        cap = _max_edges()
        if edge_count > cap:
            stale = _cache.get(db)
            if stale is not None:
                logging.getLogger(__name__).warning(
                    "codegraph at %s has %d traversable edges (cap %d); serving stale cache",
                    db,
                    edge_count,
                    cap,
                )
                return stale[0]
            raise RuntimeError(
                f"codegraph index at {db} has {edge_count} traversable edges, "
                f"over MEMO_CODEGRAPH_MAX_EDGES={cap}; refusing to load"
            )
        rows = conn.execute(query, _EDGE_KINDS).fetchall()
    finally:
        conn.close()

    graph = _build_light_index(rows)
    _cache[db] = (graph, mtime)
    _loaded_at = time.time()
    logging.getLogger(__name__).info(
        "codegraph loaded from %s: %d nodes, %d edges", db, len(graph[0]), len(graph[1])
    )

    return graph


def is_stale() -> bool:
    """Codegraph self-maintains its index via a file-watcher.

    There is nothing to rebuild from here, so the index is "stale" only when
    the DB resolved by the same default resolution as ``load()`` is missing.
    """
    return not _resolve_db().is_file()


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
    """Reset all cached graphs (forces a reload on the next call)."""
    global _loaded_at
    _cache.clear()
    _loaded_at = None
