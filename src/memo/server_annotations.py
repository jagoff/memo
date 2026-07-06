"""Shared MCP ToolAnnotations presets + a mock-tolerant `@server.tool` wrapper.

FastMCP >=2.3 accepts `annotations=` on `tool()`. The existing server_* test
modules stub `server.tool` with a zero-arg decorator, and very old fastmcp
lacks the kwarg — `annotated_tool` falls back to a bare `server.tool()` on
TypeError, so annotations are strictly additive metadata (MCP clients use
them to skip confirmation on read-only tools and to warn on destructive ones).
"""

from __future__ import annotations

from typing import Any

READ_ONLY: dict[str, Any] = {
    "readOnlyHint": True, "destructiveHint": False,
    "idempotentHint": True, "openWorldHint": False,
}
WRITE: dict[str, Any] = {
    "readOnlyHint": False, "destructiveHint": False,
    "idempotentHint": False, "openWorldHint": False,
}
WRITE_IDEMPOTENT: dict[str, Any] = {
    "readOnlyHint": False, "destructiveHint": False,
    "idempotentHint": True, "openWorldHint": False,
}
DESTRUCTIVE: dict[str, Any] = {
    "readOnlyHint": False, "destructiveHint": True,
    "idempotentHint": False, "openWorldHint": False,
}
NETWORK_WRITE: dict[str, Any] = {
    "readOnlyHint": False, "destructiveHint": False,
    "idempotentHint": False, "openWorldHint": True,
}


def annotated_tool(server: Any, **hints: Any) -> Any:
    """`server.tool(annotations=hints)` with graceful zero-arg fallback."""
    try:
        return server.tool(annotations=dict(hints))
    except TypeError:
        return server.tool()
