"""MCP tool — `memo_tool_docs`: recover a pruned tool's schema by name.

Task 9 of the token-savings-proxy plan (`memo.proxy.transforms.toolschemas`)
prunes rarely-used `memo_*` tool schemas out of the cached prefix sent to the
model, to save ~11.6k tokens paid on every request whether or not a tool is
ever called. Pruning only removes a schema from what the model sees in that
one request payload — this MCP server still registers every memo_* tool
regardless, so a pruned tool is reachable, just not visible. Without a way
back in, "reachable but invisible" is functionally "gone": the model has no
parameters to call it with. `memo_tool_docs` is that way back in, and it is
kept in the prefix unconditionally (see
`memo.proxy.transforms.toolschemas._ALWAYS_KEEP`) so it is always there to
ask.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from memo.memory import Memory
from memo.server_annotations import READ_ONLY, annotated_tool


def register(server: FastMCP, memory: Memory) -> None:
    @annotated_tool(server, **READ_ONLY)
    async def memo_tool_docs(name: str) -> dict[str, Any]:
        """Look up a memo tool's live description and parameter schema by name.

        Use this when a tool you expect (from a prior session, or from
        documentation) is missing from your current tool list: memo's proxy
        prunes schemas of recently-unused memo_* tools from context to save
        tokens, but every tool stays registered and callable. Call this with
        the tool's name to get back what its full schema would have shown,
        then call the tool directly by name.

        Args:
            name: The tool name, e.g. "memo_graph".
        """
        try:
            tool = await server.get_tool(name)
        except Exception:
            tool = None
        if tool is None:
            return {"found": False, "name": name}
        return {
            "found": True,
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.parameters,
        }
