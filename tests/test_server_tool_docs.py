"""memo_tool_docs: recover a pruned tool's schema by name.

Task 9 of the token-savings-proxy plan prunes rarely-used memo_* tool
schemas from the request payload's cached prefix (see
`memo.proxy.transforms.toolschemas`). This MCP server still registers every
memo_* tool regardless — pruning only hides a schema from the model, it
does not unregister the tool — so `memo_tool_docs` is the way for a model
that no longer sees a tool's schema to recover it and still call the tool.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from memo.config import Config
from memo.memory import Memory
from memo.server import build_server


@pytest.fixture
def docs_server(tmp_cfg: Config) -> Iterator[Any]:
    memory = Memory(tmp_cfg)
    try:
        yield build_server(memory=memory)
    finally:
        memory.close()


def _payload(result: Any) -> Any:
    sc = result.structured_content
    assert isinstance(sc, dict), f"expected structured content, got {type(sc)}"
    return sc.get("result", sc)


@pytest.mark.asyncio
async def test_memo_tool_docs_returns_the_schema_of_a_real_tool(docs_server: Any) -> None:
    result = await docs_server.call_tool("memo_tool_docs", {"name": "memo_search"})
    doc = _payload(result)
    assert doc["found"] is True
    assert doc["name"] == "memo_search"
    assert isinstance(doc["description"], str) and doc["description"]
    assert isinstance(doc["parameters"], dict)
    assert doc["parameters"].get("type") == "object"


@pytest.mark.asyncio
async def test_memo_tool_docs_reports_an_unknown_tool_rather_than_raising(
    docs_server: Any,
) -> None:
    result = await docs_server.call_tool("memo_tool_docs", {"name": "not_a_real_tool"})
    doc = _payload(result)
    assert doc == {"found": False, "name": "not_a_real_tool"}


@pytest.mark.asyncio
async def test_memo_tool_docs_is_itself_registered_and_reachable(docs_server: Any) -> None:
    """The escape hatch must not be able to prune itself out of reach."""
    result = await docs_server.call_tool("memo_tool_docs", {"name": "memo_tool_docs"})
    doc = _payload(result)
    assert doc["found"] is True


@pytest.mark.asyncio
async def test_memo_tool_docs_hydrates_a_non_memo_tool_from_the_proxy_cache(
    docs_server: Any, tmp_cfg: Config
) -> None:
    """`ToolSchemas` (memo.proxy.transforms.toolschemas) now prunes tools it
    does not own, not just memo_* ones (MEMO_PROXY_TOOL_SCHEMAS_SCOPE=all).
    `server.get_tool()` can only resolve tools registered on THIS FastMCP
    server, so for a tool like an octocode MCP tool, `memo_tool_docs` must
    fall back to the proxy's on-disk schema cache — the same file the proxy
    (a different process) wrote to when it pruned that tool from the wire.
    """
    from memo.proxy.tool_schema_cache import remember

    remember(
        tmp_cfg.state_dir,
        [
            {
                "name": "mcp__octocode__localSearchCode",
                "description": "Search code locally in a given path.",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            }
        ],
    )
    result = await docs_server.call_tool(
        "memo_tool_docs", {"name": "mcp__octocode__localSearchCode"}
    )
    doc = _payload(result)
    assert doc["found"] is True
    assert doc["name"] == "mcp__octocode__localSearchCode"
    assert doc["description"] == "Search code locally in a given path."
    assert doc["parameters"]["properties"]["query"]["type"] == "string"


@pytest.mark.asyncio
async def test_memo_tool_docs_prefers_the_live_server_over_a_stale_cache_entry(
    docs_server: Any, tmp_cfg: Config
) -> None:
    """A memo_* tool is always resolved live via `server.get_tool()` first —
    that's the authoritative, always-fresh source for memo's own tools — the
    schema cache is only ever a fallback for tools memo doesn't register."""
    from memo.proxy.tool_schema_cache import remember

    remember(
        tmp_cfg.state_dir,
        [{"name": "memo_search", "description": "stale/wrong", "input_schema": {}}],
    )
    result = await docs_server.call_tool("memo_tool_docs", {"name": "memo_search"})
    doc = _payload(result)
    assert doc["found"] is True
    assert doc["description"] != "stale/wrong"
