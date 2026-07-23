from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from memo.server_annotations import READ_ONLY, annotated_tool
from memo.server_write_coordinator import McpWriteCoordinator


def register(server: FastMCP, coordinator: McpWriteCoordinator) -> None:
    @annotated_tool(server, **READ_ONLY)
    def memo_write_queue_status() -> dict[str, Any]:
        """Return process-local MCP write queue depth, waits, and rejections."""
        return coordinator.snapshot()
