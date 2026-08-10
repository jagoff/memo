"""The collaborative MCP tools must satisfy the output schema they advertise.

``tests/test_server_collaborative.py`` drives these tools with a MagicMock
server, so FastMCP never derives an output schema from the return annotation
and nothing ever compares the payload against it. That is the gap that let
``-> dict[str, str]`` ship on two tools whose dataclasses carry ints, floats
and lists (``SharedConnection.confidence: float``, ``.votes: int``,
``CollectiveInsight.contributors: list[str]``, ``.upvotes/.downvotes: int``).

The declared schema was ``{"additionalProperties": {"type": "string"}}``,
which cannot describe those payloads. Measured 2026-08-09 over stdio against
the installed 4.9.3 (fastmcp 3.4.6, which enforces the schema), every call to
either tool came back as::

    Output validation error: 0 is not of type 'string'

— unconditionally, for every caller. Pinned by validating the real payload
against the tool's OWN advertised schema rather than by calling and hoping:
fastmcp 3.4.5 does not enforce output schemas, so a call-only test goes green
on a lockfile that is one patch release behind the runtime users install.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from memo.config import Config
from memo.memory import Memory
from memo.server import build_server

COLLABORATIVE_CALLS: list[tuple[str, dict[str, Any]]] = [
    (
        "memo_collaborative_share_connection",
        {
            "user_id": "user-a",
            "entity_a": "Python",
            "entity_b": "MLX",
            "relationship": "used_with",
            "confidence": 0.9,
        },
    ),
    (
        "memo_collaborative_share_insight",
        {"user_id": "user-a", "content": "sqlite-vec beats a flat scan past ~10k rows"},
    ),
]


@pytest.fixture
def collab_server(tmp_cfg: Config, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    monkeypatch.setenv("MEMO_MCP_PROFILE", "full")
    memory = Memory(tmp_cfg)
    try:
        yield build_server(memory=memory)
    finally:
        memory.close()


async def _declared_schema(server: Any, name: str) -> dict[str, Any]:
    tools = {t.name: t for t in await server._list_tools()}
    assert name in tools, f"{name} is not registered under the full profile"
    schema = tools[name].output_schema
    assert schema is not None, f"{name} advertises no output schema"
    return dict(schema)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool", "args"), COLLABORATIVE_CALLS, ids=lambda v: v if isinstance(v, str) else ""
)
async def test_payload_satisfies_the_advertised_output_schema(
    collab_server: Any, tool: str, args: dict[str, Any]
) -> None:
    """What the tool returns must validate against what it says it returns."""
    import jsonschema

    schema = await _declared_schema(collab_server, tool)
    result = await collab_server.call_tool(tool, args)
    payload = result.structured_content
    assert isinstance(payload, dict), f"expected structured content, got {type(payload)}"

    jsonschema.validate(payload, schema)


@pytest.mark.asyncio
async def test_share_connection_keeps_its_numeric_fields(collab_server: Any) -> None:
    """The payload is the point: a float confidence and an int vote count."""
    result = await collab_server.call_tool(*COLLABORATIVE_CALLS[0])

    conn = result.structured_content
    assert conn["entity_a"] == "Python"
    assert conn["relationship"] == "used_with"
    assert conn["confidence"] == pytest.approx(0.9)
    assert isinstance(conn["votes"], int)


@pytest.mark.asyncio
async def test_share_insight_keeps_its_list_and_counters(collab_server: Any) -> None:
    """`contributors` stays a list and the vote counters stay ints."""
    result = await collab_server.call_tool(*COLLABORATIVE_CALLS[1])

    insight = result.structured_content
    assert insight["content"] == "sqlite-vec beats a flat scan past ~10k rows"
    assert isinstance(insight["contributors"], list)
    assert isinstance(insight["upvotes"], int)
    assert isinstance(insight["downvotes"], int)
