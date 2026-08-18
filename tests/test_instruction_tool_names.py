"""Every memo_* tool named in shipped instructions must exist on the surface
those instructions are written for.

`MANDATE_TEXT` lands in AGENTS.md/CLAUDE.md for clients installed at the
`agent`/`core` profile, and `_SERVER_INSTRUCTIONS` is sent to every MCP client
on the default profile — naming a tool that is not registered there makes the
instruction unfollowable, and the failure is silent (the client just can't call
it).
"""

from __future__ import annotations

import re

import pytest

from memo.surface import AGENT_MCP_TOOLS, CORE_MCP_TOOLS

pytestmark = pytest.mark.resource_hygiene

_TOOL_RE = re.compile(r"\bmemo_[a-z_]+\b")


def _named_tools(text: str) -> set[str]:
    return {m for m in _TOOL_RE.findall(text)}


def test_mandate_text_only_names_agent_surface_tools() -> None:
    from memo.cli_mandate import MANDATE_TEXT

    missing = _named_tools(MANDATE_TEXT) - AGENT_MCP_TOOLS - CORE_MCP_TOOLS
    assert not missing, f"MANDATE_TEXT names tools absent from agent/core: {sorted(missing)}"


def test_server_instructions_only_name_agent_surface_tools() -> None:
    from memo.server import _SERVER_INSTRUCTIONS

    missing = _named_tools(_SERVER_INSTRUCTIONS) - AGENT_MCP_TOOLS - CORE_MCP_TOOLS
    assert not missing, (
        f"_SERVER_INSTRUCTIONS names tools absent from agent/core: {sorted(missing)}"
    )
