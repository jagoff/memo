"""MCP tool — `memo_tool_docs`: recover a pruned tool's schema by name.

Task 9 of the token-savings-proxy plan (`memo.proxy.transforms.toolschemas`)
prunes tool schemas out of the cached prefix sent to the model, to save
tokens paid on every request whether or not a tool is ever called. Pruning
only removes a schema from what the model sees in that one request payload —
this MCP server still registers every memo_* tool regardless, so a pruned
memo tool is reachable, just not visible. Without a way back in, "reachable
but invisible" is functionally "gone": the model has no parameters to call it
with. `memo_tool_docs` is that way back in, and it is kept in the prefix
unconditionally (see `memo.proxy.transforms.toolschemas._ALWAYS_KEEP`) so it
is always there to ask.

`ToolSchemas` prunes tools regardless of which server owns them by default
now (`MEMO_PROXY_TOOL_SCHEMAS_SCOPE=all`), not just memo_*'s — so this tool
must be able to hydrate ANY pruned tool, not only memo's own. For a memo_*
tool, `server.get_tool(name)` below is authoritative and always fresh (this
IS the FastMCP server that registered it). For anything else — another MCP
server's tool, or a Claude Code built-in that somehow got pruned — there is
no local registration to ask, so this falls back to
`memo.proxy.tool_schema_cache`: the proxy sees every tool definition on the
wire before it prunes any of them, and caches what it prunes there. The
proxy and this MCP server are two different processes agreeing only through
that file.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from memo.memory import Memory
from memo.server_annotations import READ_ONLY, annotated_tool


def register(server: FastMCP, memory: Memory) -> None:
    @annotated_tool(server, **READ_ONLY)
    async def memo_tool_docs(name: str) -> dict[str, Any]:
        """Look up a tool's live description and parameter schema by name.

        Use this when a tool you expect (from a prior session, or from
        documentation) is missing from your current tool list: memo's proxy
        prunes schemas of recently-unused tools from context to save tokens
        — memo's own tools stay registered and callable regardless; other
        servers' tools are still reachable exactly as they were before
        pruning. Call this with the tool's name to get back what its full
        schema would have shown, then call the tool directly by name.

        Args:
            name: The tool name, e.g. "memo_graph" or
                "mcp__octocode__localSearchCode".
        """
        try:
            tool = await server.get_tool(name)
        except Exception:
            tool = None
        if tool is not None:
            return {
                "found": True,
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.parameters,
            }
        try:
            from memo.proxy.tool_schema_cache import lookup

            cached = lookup(memory.cfg.state_dir, name)
        except Exception:
            cached = None
        if cached is not None:
            return {
                "found": True,
                "name": name,
                "description": cached.get("description") or "",
                "parameters": cached.get("input_schema") or {"type": "object", "properties": {}},
            }
        return {"found": False, "name": name}
