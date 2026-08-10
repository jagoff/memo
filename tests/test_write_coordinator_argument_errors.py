"""A write tool must tell the caller which argument it got wrong.

The coordinator masks every non-``MemoError`` a write tool raises behind
``coordinated MCP write failed safely (<Type>) — details in the memo-mcp
server log``. Masking the MESSAGE is the right default: a storage or parsing
failure can quote memory content, and the MCP client is not entitled to it.

Argument-binding errors are different in kind. FastMCP raises
``fastmcp.exceptions.ValidationError`` before the tool body runs, and its
message describes only the CALL — which required argument is missing, which
keyword is unexpected, echoing back arguments the caller itself just sent.
Measured 2026-08-09 against the installed 4.9.3, ``memo_focus_set(focus=...,
project=...)`` masked this::

    2 validation errors for call[memo_focus_set]
    summary   Missing required argument
    focus     Unexpected keyword argument

leaving the agent with nothing to correct and a server log it cannot read —
while every READ-only tool (``memo_related``, ``memo_rerank``) returned that
same pydantic detail in full. The write surface was undiscoverable by the
agents it exists for.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from memo.config import Config
from memo.memory import Memory
from memo.server import build_server

MASK = "failed safely"


@pytest.fixture
def server(tmp_cfg: Config, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    monkeypatch.setenv("MEMO_MCP_PROFILE", "full")
    memory = Memory(tmp_cfg)
    try:
        yield build_server(memory=memory)
    finally:
        memory.close()


@pytest.mark.asyncio
async def test_missing_required_argument_names_the_argument(server: Any) -> None:
    """`memo_focus_set` needs `summary`; the caller must be told exactly that."""
    with pytest.raises(Exception) as excinfo:
        await server.call_tool("memo_focus_set", {"project": "qa"})

    message = str(excinfo.value)
    assert MASK not in message, f"argument error was masked: {message}"
    assert "summary" in message


@pytest.mark.asyncio
async def test_unexpected_keyword_argument_names_the_keyword(server: Any) -> None:
    """A wrong keyword must come back as the wrong keyword, not as a type name."""
    with pytest.raises(Exception) as excinfo:
        await server.call_tool("memo_focus_set", {"project": "qa", "summary": "s", "focus": "typo"})

    message = str(excinfo.value)
    assert MASK not in message, f"argument error was masked: {message}"
    assert "focus" in message


@pytest.mark.asyncio
async def test_a_failing_tool_body_stays_masked(server: Any, monkeypatch: Any) -> None:
    """The privacy default is unchanged: a body failure still says nothing."""
    from memo import server_write_coordinator

    async def _boom() -> None:
        raise RuntimeError("secret memory body: the deploy key is hunter2")

    coordinator = server_write_coordinator.McpWriteCoordinator(capacity=4)
    with pytest.raises(Exception) as excinfo:
        await coordinator.submit(_boom)

    message = str(excinfo.value)
    assert MASK in message
    assert "hunter2" not in message
