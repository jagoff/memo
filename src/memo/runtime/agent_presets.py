"""Preset registry: how to write the memo MCP config into each trending agent.

Data-only description (path per OS, top-level JSON key, entry shape). Families
A/B/C all reduce to the existing ``_write_mcp_json`` writer; only ``json_key`` and
``include_type`` differ. YAML writers (Continue/Goose) are added in a later task.
JetBrains has no writable config file → snippet-only.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from memo.runtime.mcp import _mcp_server_json, _write_mcp_json


class Writer(StrEnum):
    JSON_MAP = "json_map"
    YAML_CONTINUE = "yaml_continue"
    YAML_GOOSE = "yaml_goose"
    SNIPPET = "snippet"


@dataclass(frozen=True)
class AgentPreset:
    agent: str
    writer: Writer
    paths: Mapping[str, str]  # keys: "darwin", "linux", "win"
    json_key: str = "mcpServers"
    include_type: bool = False
    source: str = ""  # MEMO_SOURCE value; defaults to the agent slug


def _all(path: str) -> dict[str, str]:
    """Home-relative path identical across OSes."""
    return {"darwin": path, "linux": path, "win": path}


_CODE_USER = {
    "darwin": "~/Library/Application Support/Code/User",
    "linux": "~/.config/Code/User",
    "win": "%APPDATA%/Code/User",
}


def _code_child(rel: str) -> dict[str, str]:
    return {os_key: f"{base}/{rel}" for os_key, base in _CODE_USER.items()}


AGENT_PRESETS: dict[str, AgentPreset] = {
    "vscode": AgentPreset(
        "vscode", Writer.JSON_MAP, _code_child("mcp.json"),
        json_key="servers", include_type=True,
    ),
    "zed": AgentPreset(
        "zed", Writer.JSON_MAP,
        {"darwin": "~/.config/zed/settings.json", "linux": "~/.config/zed/settings.json",
         "win": "%APPDATA%/Zed/settings.json"},
        json_key="context_servers",
    ),
    "windsurf": AgentPreset("windsurf", Writer.JSON_MAP, _all("~/.codeium/windsurf/mcp_config.json")),
    "cline": AgentPreset(
        "cline", Writer.JSON_MAP,
        _code_child("globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json"),
    ),
    "roo": AgentPreset(
        "roo", Writer.JSON_MAP,
        _code_child("globalStorage/rooveterinaryinc.roo-cline/settings/mcp_settings.json"),
    ),
    "kiro": AgentPreset("kiro", Writer.JSON_MAP, _all("~/.kiro/settings/mcp.json")),
    "antigravity": AgentPreset("antigravity", Writer.JSON_MAP, _all("~/.gemini/config/mcp_config.json")),
    "warp": AgentPreset("warp", Writer.JSON_MAP, _all("~/.warp/.mcp.json")),
    "jetbrains": AgentPreset("jetbrains", Writer.SNIPPET, {}),
}


def _platform_key() -> str:
    if sys.platform == "darwin":
        return "darwin"
    if sys.platform.startswith("win"):
        return "win"
    return "linux"


def resolve_preset_path(preset: AgentPreset) -> Path | None:
    """Resolve this agent's config path on the current OS, or None if unknown."""
    import os

    tmpl = preset.paths.get(_platform_key())
    if not tmpl:
        return None
    if tmpl.startswith("~/"):
        return Path.home() / tmpl[2:]
    return Path(os.path.expandvars(tmpl))


def _snippet_result(preset: AgentPreset, server: Any) -> dict[str, Any]:
    env = {**server.env, "MEMO_SOURCE": preset.source or preset.agent}
    body = {"mcpServers": {server.name: _mcp_server_json(Path(server.command), env, include_type=False)}}
    return {
        "ok": True,
        "agent": preset.agent,
        "action": "snippet",
        "snippet": json.dumps(body, ensure_ascii=False, indent=2),
        "note": "Paste in Settings | Tools | AI Assistant | MCP (or use 'Import from Claude').",
    }


def install_from_preset(preset: AgentPreset, server: Any, *, write: bool) -> dict[str, Any]:
    """Install the memo MCP server into one agent per its preset.

    Returns a result dict shaped for ``cli_install_mcp._report``.
    """
    if preset.writer is Writer.SNIPPET:
        return _snippet_result(preset, server)

    path = resolve_preset_path(preset)
    if path is None:
        return {"ok": False, "agent": preset.agent, "skipped": True,
                "error": f"no config path for this OS ({sys.platform})"}

    env = {**server.env, "MEMO_SOURCE": preset.source or preset.agent}
    target = dataclasses.replace(server, env=env)

    if not write:
        return {"ok": True, "agent": preset.agent, "action": "dry-run",
                "would": "write", "path": str(path)}

    if preset.writer is Writer.JSON_MAP:
        action = _write_mcp_json(path, target, json_key=preset.json_key, include_type=preset.include_type)
    else:
        # YAML dispatch is wired in Task 2 once the writers exist (keeps this task's
        # mypy gate clean — no import of not-yet-defined functions).
        return {"ok": False, "agent": preset.agent, "error": f"writer {preset.writer.value} not available"}
    return {"ok": True, "agent": preset.agent, "action": action, "path": str(path)}
