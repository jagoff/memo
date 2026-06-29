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
    "~/.codex/config.toml",
    "~/.config/devin/config.json",
    "~/.config/opencode/opencode.jsonc",
    "~/.config/mcp-gateway/gateway.yaml",
    "~/.codeium/windsurf/mcp_config.json",
    "~/Library/Application Support/Windsurf/User/mcp_config.json",
)

# Absolute path ending in /memo or /memo-mcp (the launched binary).
_MEMO_BIN_PATH = re.compile(r"(/[^\s\"':,]*?/(?:memo-mcp|memo))(?=[\s\"':,]|$)")

# Fragile: points inside a managed venv instead of the stable shim.
_VENV_INTERNAL = re.compile(r"/(?:pipx/venvs|uv/tools|\.venv|site-packages)/")

# A memo path is a whole token only when followed by a delimiter or end of text;
# the lookahead stops ``/…/memo`` from matching inside ``/…/memo-mcp``.
_PATH_BOUNDARY = r"(?=[\s\"':,]|$)"


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


def repair_mcp_configs(
    config_paths: tuple[str, ...] = KNOWN_MCP_CONFIGS,
    *,
    shim_dir: str = "~/.local/bin",
    apply: bool = True,
) -> list[dict[str, str]]:
    """Repoint fragile/broken memo-mcp command paths to the stable shim.

    Mechanical and reversible: for each :func:`scan_mcp_configs` finding whose
    shim target actually EXISTS, rewrite every occurrence of the dead path in
    the config to the shim and back the original up to ``<config>.bak``. A
    finding whose shim target is missing is reported (``skipped-no-target``) but
    never written — repointing to a path that does not exist would only swap one
    broken launch for another. ``apply=False`` plans the same changes without
    touching disk (``would-repair``).

    Returns one dict per finding with the scan fields plus ``status`` —
    ``repaired`` / ``would-repair`` / ``skipped-no-target``.
    """
    by_config: dict[str, list[dict[str, str]]] = {}
    for finding in scan_mcp_configs(config_paths, shim_dir=shim_dir):
        by_config.setdefault(finding["config"], []).append(finding)

    repairs: list[dict[str, str]] = []
    for cfg_str, findings in by_config.items():
        cfg_path = Path(cfg_str)
        try:
            original = cfg_path.read_text(encoding="utf-8")
        except OSError:
            continue
        text = original
        applied: list[dict[str, str]] = []
        # Longest path first so /…/memo never rewrites inside /…/memo-mcp.
        for finding in sorted(findings, key=lambda f: len(f["command"]), reverse=True):
            if not Path(finding["suggestion"]).exists():
                repairs.append({**finding, "status": "skipped-no-target"})
                continue
            pattern = re.compile(re.escape(finding["command"]) + _PATH_BOUNDARY)
            rewritten = pattern.sub(finding["suggestion"], text)
            if rewritten != text:
                text = rewritten
                applied.append(finding)
        if not applied or text == original:
            continue
        status = "repaired" if apply else "would-repair"
        if apply:
            cfg_path.with_suffix(cfg_path.suffix + ".bak").write_text(original, encoding="utf-8")
            cfg_path.write_text(text, encoding="utf-8")
        repairs.extend({**finding, "status": status} for finding in applied)
    return repairs
