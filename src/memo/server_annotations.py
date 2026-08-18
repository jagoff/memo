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
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}
WRITE: dict[str, Any] = {
    "readOnlyHint": False,
    "destructiveHint": False,
    "idempotentHint": False,
    "openWorldHint": False,
}
WRITE_IDEMPOTENT: dict[str, Any] = {
    "readOnlyHint": False,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}
DESTRUCTIVE: dict[str, Any] = {
    "readOnlyHint": False,
    "destructiveHint": True,
    "idempotentHint": False,
    "openWorldHint": False,
}
NETWORK_WRITE: dict[str, Any] = {
    "readOnlyHint": False,
    "destructiveHint": False,
    "idempotentHint": False,
    "openWorldHint": True,
}


# Names of every tool registered with a NON read-only annotation. Populated at
# registration time so the response-budget middleware can tell a mutation from a
# query: a write commits BEFORE the middleware sizes its payload, so replacing
# an over-cap write result with an error tells the caller the write failed when
# it actually landed.
MUTATING_TOOL_NAMES: set[str] = set()


def annotated_tool(server: Any, **hints: Any) -> Any:
    """`server.tool(annotations=hints)` with graceful zero-arg fallback."""
    mutating = hints.get("readOnlyHint") is False

    def _record(fn: Any) -> Any:
        name = getattr(fn, "__name__", "")
        if mutating and name:
            MUTATING_TOOL_NAMES.add(name)
        return fn

    try:
        inner = server.tool(annotations=dict(hints))
    except TypeError:
        inner = server.tool()

    def _decorator(fn: Any) -> Any:
        return inner(_record(fn))

    return _decorator
