"""`memo install-mcp` — register the memo MCP server into any agent.

Thin wrapper over the shared ``consciousness_contracts.agent_install`` contract
(the same one memflow uses), so memo is installable into Codex, Claude Code,
Claude Desktop, Windsurf, Gemini, Cursor, opencode, Devin — and any future agent
via ``--config-path``. MCP-only and idempotent; for the heavier skill+plugin+MCP
flow use ``memo install-slash``.

The server command is the resolved ISOLATED runtime (`_resolved_memo_mcp`), never
a project `.venv` — a mixed runtime is the usual cause of "works in CLI, broken
in MCP" (see CLAUDE.md). Dry-run by default; pass ``--write`` to apply.
"""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path
from typing import Any

import click

from memo.runtime.detect import _resolved_memo_mcp
from memo.runtime.mcp import _mcp_server_env

CONSTRAINED_CLIENTS: frozenset[str] = frozenset({"codex", "opencode"})


def _effective_profile(profile: str, agent: str) -> str:
    if profile:
        return profile
    return "core" if agent in CONSTRAINED_CLIENTS else ""


def _resolve_isolated_memo_mcp() -> Path | None:
    """Resolve memo-mcp from the ISOLATED runtime, not whatever transient venv
    launched this process.

    An installer writes the command path into persistent agent configs, so a
    project `.venv` path is the documented footgun ("works in CLI, broken in
    MCP" — CLAUDE.md). Prefer the known isolated locations; only fall back to the
    invoked-process resolution, and reject it when it lives under a `.venv`.
    """
    home = Path.home()
    candidates = [
        home / ".local" / "bin" / "memo-mcp",
        home / ".local" / "pipx" / "venvs" / "mlx-memo" / "bin" / "memo-mcp",
        Path("/opt/homebrew/bin/memo-mcp"),
        Path("/usr/local/bin/memo-mcp"),
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    on_path = shutil.which("memo-mcp")
    if on_path and ".venv" not in on_path:
        return Path(on_path)
    fallback = _resolved_memo_mcp()
    if fallback is not None and ".venv" not in str(fallback):
        return fallback
    return None


def _build_server() -> Any:
    """Construct the contract's AgentMcpServer for memo, or raise if the
    contract / isolated runtime is unavailable."""
    try:
        from consciousness_contracts import AgentMcpServer
    except ImportError as exc:  # pragma: no cover - optional dep
        raise click.ClickException(
            "consciousness-contracts not installed; install it to use `install-mcp`."
        ) from exc
    memo_mcp = _resolve_isolated_memo_mcp()
    if memo_mcp is None:
        raise click.ClickException(
            "isolated memo-mcp not found. Install memo as an isolated tool first: "
            "`pipx install mlx-memo` or `uv tool install mlx-memo` (a project .venv "
            "must not be written into agent configs)."
        )
    return AgentMcpServer(name="memo", command=str(memo_mcp), env=_mcp_server_env())


def _report(result: dict[str, Any]) -> None:
    agent = result.get("agent", "?")
    if not result.get("ok", False):
        if result.get("skipped"):
            click.echo(f"  - {agent}: skipped ({result.get('error')})")
        else:
            click.echo(f"  ✗ {agent}: {result.get('error')}")
        return
    action = result.get("action", "")
    if result.get("strategy") == "cli":
        if action == "dry-run":
            click.echo(f"  {agent}: would run `{' '.join(result['argv'])}`")
        else:
            click.echo(f"  ✓ {agent}: {action}")
    else:
        path = result.get("path", "")
        verb = result.get("would", action) if action == "dry-run" else action
        click.echo(f"  ✓ {agent}: {verb} → {path}")


@click.command(name="install-mcp")
@click.option(
    "--agent",
    "agents",
    multiple=True,
    help="Agent to wire (repeatable). Use 'all' for every known agent. "
    "Known: codex, claude-code, claude-desktop, windsurf, gemini, cursor, "
    "opencode, devin.",
)
@click.option("--config-path", default="", help="Generic: write into this JSON config path (under $HOME).")
@click.option("--json-key", default="mcpServers", help="Generic: server-map key in the config (default mcpServers).")
@click.option("--write", is_flag=True, help="Apply changes (default: dry-run).")
@click.option("--with-mandate", is_flag=True, help="Also write the 'consult memo first' mandate.")
@click.option(
    "--profile",
    default="",
    type=click.Choice(["", "core", "slim", "default"], case_sensitive=False),
    help="MCP surface profile. 'core'/'slim' expose ~25 tools (~2.4k tokens); "
         "'default' exposes all 118 tools (~35k tokens). "
         "Constrained clients (codex, opencode) default to 'core' automatically.",
)
def install_mcp(
    agents: tuple[str, ...],
    config_path: str,
    json_key: str,
    write: bool,
    with_mandate: bool,
    profile: str,
) -> None:
    """Register the memo MCP server into one or more agents."""
    try:
        from consciousness_contracts import (
            SUPPORTED_AGENTS,
            generic_preset,
            register_agent_mcp,
        )
    except ImportError as exc:
        raise click.ClickException(
            "consciousness-contracts not installed in memo's runtime. Install it "
            "into the isolated env: `pipx inject mlx-memo -e "
            "<path>/consciousness-contracts` (or `uv tool install mlx-memo --with "
            "<path>/consciousness-contracts`)."
        ) from exc

    server = _build_server()

    profile_note = f" [profile: {profile}]" if profile and profile != "default" else ""
    click.echo(f"memo MCP → {server.command}{'' if write else '  (dry-run)'}{profile_note}")

    if config_path:
        preset = generic_preset(config_path=config_path, json_key=json_key)
        _report(register_agent_mcp("generic", server, write=write, preset=preset))
    else:
        selected = list(agents) or ["all"]
        if "all" in selected:
            selected = list(SUPPORTED_AGENTS)
        for agent in selected:
            eff_profile = _effective_profile(profile, agent)
            if eff_profile and eff_profile != "default":
                agent_env = dict(server.env)
                agent_env["MEMO_MCP_PROFILE"] = eff_profile
                agent_server = dataclasses.replace(server, env=agent_env)
            else:
                agent_server = server
            _report(register_agent_mcp(agent, agent_server, write=write))

    if with_mandate:
        click.echo("mandate (consult memo first):")
        from memo.cli_mandate import write_mandates_for_clients

        target_agents = list(agents) or ["all"]
        if "all" in target_agents:
            target_agents = ["codex", "devin", "opencode", "windsurf", "cursor"]
        for rel, status in write_mandates_for_clients(target_agents, dry_run=not write):
            click.echo(f"  {rel:<22} {status}")
