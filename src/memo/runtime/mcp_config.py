"""Scan known MCP config files for fragile memo-mcp launch paths.

A config that hardcodes a venv-internal binary path (pipx/uv site) breaks the
moment the runtime is reinstalled or switched. The stable target is the shim
``~/.local/bin/<bin>`` or the bare name on PATH. Format-agnostic: matches the
raw text, so JSON, JSONC, and YAML all work without a parser.
"""

from __future__ import annotations

import re
from pathlib import Path

# Configs that commonly launch memo-mcp as a stdio MCP server.
KNOWN_MCP_CONFIGS: tuple[str, ...] = (
    "~/.claude.json",
    "~/.config/devin/config.json",
    "~/.config/opencode/opencode.jsonc",
    "~/.config/mcp-gateway/gateway.yaml",
)

# Absolute path ending in /memo or /memo-mcp (the launched binary).
_MEMO_BIN_PATH = re.compile(r"(/[^\s\"':,]*?/(?:memo-mcp|memo))(?=[\s\"':,]|$)")

# Fragile: points inside a managed venv instead of the stable shim.
_VENV_INTERNAL = re.compile(r"/(?:pipx/venvs|\.venv|site-packages)/")


def extract_memo_command_paths(text: str) -> list[str]:
    """All absolute ``/…/memo`` or ``/…/memo-mcp`` paths mentioned in a config file."""
    return sorted({m.group(1) for m in _MEMO_BIN_PATH.finditer(text)})


def classify_command_path(path: str) -> str | None:
    """Return an issue label, or None if the path is fine.

    - "venv-internal": points inside a venv (breaks on reinstall/runtime change)
    - "missing":       file does not exist on disk
    """
    if _VENV_INTERNAL.search(path):
        return "venv-internal"
    if not Path(path).exists():
        return "missing"
    return None


def scan_mcp_configs(
    config_paths: tuple[str, ...] = KNOWN_MCP_CONFIGS,
    *,
    shim_dir: str = "~/.local/bin",
) -> list[dict[str, str]]:
    """Inspect known MCP config files for fragile/broken memo-mcp command paths."""
    findings: list[dict[str, str]] = []
    shim = str(Path(shim_dir).expanduser())
    for cfg_str in config_paths:
        cfg_path = Path(cfg_str).expanduser()
        try:
            if not cfg_path.exists():
                continue
            text = cfg_path.read_text(encoding="utf-8")
        except OSError:
            continue
        for cmd in extract_memo_command_paths(text):
            issue = classify_command_path(cmd)
            if issue is None:
                continue
            bin_name = cmd.rsplit("/", 1)[-1]
            findings.append(
                {
                    "config": str(cfg_path),
                    "command": cmd,
                    "issue": issue,
                    "suggestion": f"{shim}/{bin_name}",
                }
            )
    return findings
