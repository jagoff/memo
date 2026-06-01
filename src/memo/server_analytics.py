"""MCP tools — analytics domain (split from server.py).

Registered by `build_server()` via `register(server, memory)`. Tool names,
signatures, defaults, docstrings and bodies are identical to the originals;
only the enclosing function and indentation changed.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from memo.memory import Memory

def register(server: FastMCP, memory: Memory) -> None:
    @server.tool()
    def memory_analytics_summary() -> dict[str, Any]:
        """Get analytics summary of the memory corpus.

        Returns comprehensive metrics including total memorias,
        entity counts, growth rate, and type distribution.
        """
        metrics = memory.analytics.compute_corpus_metrics()
        return metrics.__dict__

    @server.tool()
    def memory_analytics_growth(
        days: int = 30,
    ) -> dict[str, Any]:
        """Get growth data over time.

        Returns memoria growth data grouped by date for the
        specified number of days.

        Args:
            days: Number of days to analyze.
        """
        growth = memory.analytics.compute_growth_data(days=days)
        return growth.__dict__
