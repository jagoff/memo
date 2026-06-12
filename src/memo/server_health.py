"""MCP tools — memory health domain (confidence + roi_score).

Registered by `build_server()` via `register(server, memory)`.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from memo.memory import Memory


def register(server: FastMCP, memory: Memory) -> None:
    @server.tool()
    def memory_health_summary(probe_embedder: bool = False) -> dict[str, Any]:
        """Operational health snapshot: corpus size, index dims, embedder
        profile, health-score coverage, feedback volume, and warnings.

        Read-only. Mirrors the `memo health` CLI.

        Args:
            probe_embedder: if True, times one embed_query (loads the model).
        """
        from memo.health_report import build_health_report

        return build_health_report(memory, probe_embedder=probe_embedder)

    @server.tool()
    def memory_health_report(
        top_n: int = 10,
    ) -> dict[str, Any]:
        """Report memory health: lowest-confidence and highest-ROI memorias.

        Returns two ranked lists drawn from the memory_health table:
        - `low_confidence`: memorias most penalised by open contradictions.
        - `high_roi`: memorias most frequently recalled (promoted by access).

        Memorias not yet in the health table have neutral scores (1.0/1.0)
        and won't appear here. Run `memo dream run` or `memo contradict scan`
        to populate scores.

        Args:
            top_n: How many memorias to return in each list (default 10).
        """
        store = memory.store
        conn = store._conn

        rows_low = conn.execute(
            "SELECT h.id, h.confidence, h.roi_score, m.title, m.type, m.updated "
            "FROM memory_health h JOIN meta m ON m.id = h.id "
            "WHERE h.confidence < 1.0 "
            "ORDER BY h.confidence ASC LIMIT ?",
            (top_n,),
        ).fetchall()

        rows_high = conn.execute(
            "SELECT h.id, h.roi_score, h.confidence, m.title, m.type, m.updated "
            "FROM memory_health h JOIN meta m ON m.id = h.id "
            "WHERE h.roi_score > 1.0 "
            "ORDER BY h.roi_score DESC LIMIT ?",
            (top_n,),
        ).fetchall()

        return {
            "low_confidence": [
                {
                    "id": r["id"],
                    "title": r["title"],
                    "type": r["type"],
                    "confidence": round(float(r["confidence"]), 3),
                    "roi_score": round(float(r["roi_score"]), 3),
                    "updated": r["updated"],
                }
                for r in rows_low
            ],
            "high_roi": [
                {
                    "id": r["id"],
                    "title": r["title"],
                    "type": r["type"],
                    "roi_score": round(float(r["roi_score"]), 3),
                    "confidence": round(float(r["confidence"]), 3),
                    "updated": r["updated"],
                }
                for r in rows_high
            ],
            "total_tracked": conn.execute("SELECT COUNT(*) FROM memory_health").fetchone()[0],
        }
