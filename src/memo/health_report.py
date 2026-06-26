"""Operational health snapshot for memo — powers `memo health`.

A single read-only aggregation of corpus size, index dims, embedder
profile, health-score coverage, feedback volume, and derived warnings.
Pure (no MLX) unless `probe_embedder=True`, which times one `embed_query`.

The MCP tool `memo_health_summary` and the `memo health` CLI both render
the dict returned by `build_health_report`.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from memo.flags import flag_str

if TYPE_CHECKING:
    from memo.memory import Memory


def _count(conn, table: str, where: str = "") -> int | None:
    """COUNT(*) for `table`, optionally filtered. None if the table is absent."""
    sql = f"SELECT COUNT(*) FROM [{table}]"  # noqa: S608
    if where:
        sql += f" WHERE {where}"
    try:
        return int(conn.execute(sql).fetchone()[0])
    except Exception:
        return None


def _max_text(conn, table: str, column: str) -> str | None:
    try:
        row = conn.execute(f"SELECT MAX([{column}]) FROM [{table}]").fetchone()  # noqa: S608
        return str(row[0]) if row and row[0] is not None else None
    except Exception:
        return None


def _vec_dims(memory: Memory) -> int | None:
    try:
        return memory.store._vec_table_dims("vec")
    except Exception:
        return None


def _fts_ready(conn) -> bool:
    try:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'fts' LIMIT 1"
        ).fetchone()
        return row is not None
    except Exception:
        return False


def _db_size_bytes(memory: Memory) -> int | None:
    try:
        path = memory.cfg.db_path
        return int(path.stat().st_size) if path.exists() else None
    except Exception:
        return None


def _archived_count(memory: Memory) -> int | None:
    try:
        archival_dir = memory.cfg.memory_dir / "archived"
        if not archival_dir.is_dir():
            return 0
        return sum(1 for _ in archival_dir.glob("*.md"))
    except Exception:
        return None


def _probe_embedder_ms(memory: Memory) -> float | None:
    """Time a single query embedding (warms the model if cold)."""
    try:
        start = time.perf_counter()
        memory.embedder.embed_query("health probe")
        return round((time.perf_counter() - start) * 1000.0, 1)
    except Exception:
        return None


def build_health_report(memory: Memory, *, probe_embedder: bool = False) -> dict[str, Any]:
    """Aggregate a read-only operational health snapshot.

    Args:
        memory: a live `Memory`.
        probe_embedder: if True, time one `embed_query` call (needs a warm
            or loadable embedder). Default False keeps the report MLX-free.
    """
    cfg = memory.cfg
    conn = memory.store._conn

    memorias = _count(conn, "meta") or 0
    expected_dims = int(getattr(cfg, "embedder_dims", 0) or 0)
    vec_dims = _vec_dims(memory)
    dims_ok = vec_dims is None or vec_dims == expected_dims

    tracked = _count(conn, "memory_health") or 0
    low_conf = _count(conn, "memory_health", "confidence < 1.0") or 0
    high_roi = _count(conn, "memory_health", "roi_score > 1.0") or 0
    feedback = _count(conn, "source_feedback")

    warnings: list[str] = []
    if memorias == 0:
        warnings.append("Corpus is empty — nothing to retrieve. Save or ingest memorias.")
    if not dims_ok:
        warnings.append(
            f"Vector index dims ({vec_dims}) != embedder dims ({expected_dims}). "
            "Reindex with `memo reindex` after a model change."
        )
    if memorias > 0 and tracked == 0:
        warnings.append(
            "Health scores not populated — confidence/ROI ranking is neutral. "
            "Run `memo dream run` or `memo contradict scan`."
        )

    report: dict[str, Any] = {
        "corpus": {
            "memorias": memorias,
            "archived": _archived_count(memory),
            "db_size_bytes": _db_size_bytes(memory),
            "db_path": str(getattr(cfg, "db_path", "")),
            "latest_update": _max_text(conn, "meta", "updated"),
        },
        "index": {
            "vec_dims": vec_dims,
            "expected_dims": expected_dims,
            "dims_ok": dims_ok,
            "fts_ready": _fts_ready(conn),
            "fts_backend": flag_str("MEMO_FTS_BACKEND") or "auto",
        },
        "embedder": {
            "model": getattr(cfg, "embedder_model", None),
            "dims": expected_dims,
            "latency_ms": _probe_embedder_ms(memory) if probe_embedder else None,
        },
        "health_table": {
            "tracked": tracked,
            "low_confidence": low_conf,
            "high_roi": high_roi,
        },
        "feedback": {"records": feedback},
        "warnings": warnings,
    }
    return report
