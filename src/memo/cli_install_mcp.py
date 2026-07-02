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
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

from memo.runtime.agent_presets import (
    AGENT_PRESETS,
    Writer,
    install_from_preset,
    resolve_preset_path,
)
from memo.runtime.detect import _resolved_memo_mcp
from memo.runtime.mcp import (
    _config_path,
    _mcp_add_command,
    _mcp_server_env,
    _run_agent_command,
    _write_mcp_json,
)


def _claude_desktop_config_path() -> Path:
    """Per-OS Claude Desktop MCP config path."""
    if sys.platform == "darwin":
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / "Claude"
            / "claude_desktop_config.json"
        )
    if sys.platform.startswith("win"):
        base = os.environ.get("APPDATA")
        if base:
            return Path(base) / "Claude" / "claude_desktop_config.json"
    # Linux / other POSIX
    return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"


CONSTRAINED_CLIENTS: frozenset[str] = frozenset({"codex", "opencode"})
_FALLBACK_SUPPORTED_AGENTS: tuple[str, ...] = (
    "claude-code",
    "claude-desktop",
    "codex",
    "cursor",
    "devin",
    "gemini",
    "opencode",
    "windsurf",
)


@dataclass(frozen=True)
class _AgentMcpServer:
    name: str
    command: str
    env: dict[str, str]


@dataclass(frozen=True)
class _GenericPreset:
    config_path: str
    json_key: str = "mcpServers"


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
    """Construct an AgentMcpServer-compatible value.

    `consciousness_contracts` is a local/dev integration package. The public
    `mlx-memo` install path must not require it just to print or write an MCP
    config, so fall back to the tiny dataclass shape the installer needs.
    """
    try:
        from consciousness_contracts import AgentMcpServer
    except ImportError:  # pragma: no cover - depends on optional local package
        AgentMcpServer = _AgentMcpServer
    memo_mcp = _resolve_isolated_memo_mcp()
    if memo_mcp is None:
        raise click.ClickException(
            "isolated memo-mcp not found. Install memo as an isolated tool first: "
            "`pipx install mlx-memo` or `uv tool install mlx-memo` (a project .venv "
            "must not be written into agent configs)."
        )
    return AgentMcpServer(name="memo", command=str(memo_mcp), env=_mcp_server_env())


def _generic_preset(config_path: str, json_key: str = "mcpServers") -> _GenericPreset:
    return _GenericPreset(config_path=config_path, json_key=json_key)


def _fallback_register_agent_mcp(
    agent: str,
    server: Any,
    *,
    write: bool = False,
    preset: _GenericPreset | None = None,
) -> dict[str, Any]:
    """Minimal installer used when consciousness_contracts is absent."""
    try:
        if agent == "generic":
            if preset is None:
                return {"ok": False, "agent": agent, "error": "missing generic preset"}
            path = _config_path(preset.config_path)
            if not write:
                return {
                    "ok": True,
                    "agent": agent,
                    "action": "dry-run",
                    "would": "write",
                    "path": str(path),
                }
            action = _write_mcp_json(path, server, json_key=preset.json_key, include_type=True)
            return {"ok": True, "agent": agent, "action": action, "path": str(path)}

        if agent in {"claude-code", "codex", "devin", "opencode"}:
            argv = _mcp_add_command(agent, Path(server.command), server.env)
            if not write:
                return {
                    "ok": True,
                    "agent": agent,
                    "action": "dry-run",
                    "strategy": "cli",
                    "argv": [str(arg) for arg in argv],
                }
            _run_agent_command(argv, dry_run=False)
            return {"ok": True, "agent": agent, "action": "installed", "strategy": "cli"}

        config_targets: dict[str, tuple[Path, bool]] = {
            "claude-desktop": (_claude_desktop_config_path(), False),
            "cursor": (Path.home() / ".cursor" / "mcp.json", True),
            "gemini": (Path.home() / ".gemini" / "settings.json", True),
            "windsurf": (Path.home() / ".codeium" / "windsurf" / "mcp_config.json", False),
        }
        if agent in config_targets:
            path, include_type = config_targets[agent]
            if not write:
                return {
                    "ok": True,
                    "agent": agent,
                    "action": "dry-run",
                    "would": "write",
                    "path": str(path),
                }
            action = _write_mcp_json(path, server, json_key="mcpServers", include_type=include_type)
            return {"ok": True, "agent": agent, "action": action, "path": str(path)}

        return {"ok": False, "agent": agent, "skipped": True, "error": "unsupported agent"}
    except (click.ClickException, subprocess.SubprocessError, OSError) as exc:
        return {"ok": False, "agent": agent, "error": str(exc)}


def _agent_present(agent: str) -> bool:
    """True if this agent looks installed on the current machine (best-effort)."""
    preset = AGENT_PRESETS.get(agent)
    if preset is not None:
        if preset.writer is Writer.SNIPPET:
            return False
        path = resolve_preset_path(preset)
        return bool(path and path.parent.exists())
    cli_bins = {"claude-code": "claude", "codex": "codex", "devin": "devin", "opencode": "opencode"}
    binary = cli_bins.get(agent)
    if binary:
        return shutil.which(binary) is not None
    desktop_dirs = {
        "claude-desktop": _claude_desktop_config_path().parent,
        "devin-desktop": Path.home() / ".devin",
    }
    directory = desktop_dirs.get(agent)
    return bool(directory and directory.exists())


def _report(result: dict[str, Any]) -> None:
    agent = result.get("agent", "?")
    if not result.get("ok", False):
        if result.get("skipped"):
            click.echo(f"  - {agent}: skipped ({result.get('error')})")
        else:
            click.echo(f"  ✗ {agent}: {result.get('error')}")
        return
    if result.get("action") == "snippet":
        click.echo(f"  {agent}: paste this into the client's MCP config —")
        click.echo(result["snippet"])
        click.echo(f"  ({result['note']})")
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
    "Known: codex, claude-code, claude-desktop, devin-desktop, gemini, cursor, "
    "opencode, devin, vscode, antigravity, windsurf, zed, cline, roo, kiro, warp, "
    "continue, goose, jetbrains.",
)
@click.option(
    "--config-path", default="", help="Generic: write into this JSON config path (under $HOME)."
)
@click.option(
    "--json-key",
    default="mcpServers",
    help="Generic: server-map key in the config (default mcpServers).",
)
@click.option("--write", is_flag=True, help="Apply changes (default: dry-run).")
@click.option(
    "--only-present",
    is_flag=True,
    help="Only install for agents that look installed on this machine (config dir/binary present).",
)
@click.option("--with-mandate", is_flag=True, help="Also write the 'consult memo first' mandate.")
@click.option(
    "--profile",
    default="",
    type=click.Choice(["", "core", "slim", "default"], case_sensitive=False),
    help="MCP surface profile. 'core'/'slim' expose ~30 tools (~2.8k tokens); "
    "'default' exposes all 123 tools (~15k tokens). "
    "Constrained clients (codex, opencode) default to 'core' automatically.",
)
def install_mcp(
    agents: tuple[str, ...],
    config_path: str,
    json_key: str,
    write: bool,
    only_present: bool,
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
    except ImportError:
        SUPPORTED_AGENTS = _FALLBACK_SUPPORTED_AGENTS
        generic_preset = _generic_preset
        register_agent_mcp = _fallback_register_agent_mcp

    server = _build_server()

    profile_note = f" [profile: {profile}]" if profile and profile != "default" else ""
    click.echo(f"memo MCP → {server.command}{'' if write else '  (dry-run)'}{profile_note}")

    if config_path:
        preset = generic_preset(config_path=config_path, json_key=json_key)
        if profile and profile != "default":
            gen_env = dict(server.env)
            gen_env["MEMO_MCP_PROFILE"] = profile
            gen_server = dataclasses.replace(server, env=gen_env)
        else:
            gen_server = server
        _report(register_agent_mcp("generic", gen_server, write=write, preset=preset))
    else:
        selected = list(agents) or ["all"]
        if "all" in selected:
            selected = list(dict.fromkeys((*SUPPORTED_AGENTS, *AGENT_PRESETS)))
        if only_present:
            kept = [a for a in selected if _agent_present(a)]
            for a in selected:
                if a not in kept:
                    click.echo(f"  - {a}: skipped (not present)")
            selected = kept
        for agent in selected:
            eff_profile = _effective_profile(profile, agent)
            if eff_profile and eff_profile != "default":
                agent_env = dict(server.env)
                agent_env["MEMO_MCP_PROFILE"] = eff_profile
                agent_server = dataclasses.replace(server, env=agent_env)
            else:
                agent_server = server
            if agent in AGENT_PRESETS:
                _report(install_from_preset(AGENT_PRESETS[agent], agent_server, write=write))
            elif agent == "devin-desktop":
                _report(_fallback_register_agent_mcp(agent, agent_server, write=write))
            else:
                _report(register_agent_mcp(agent, agent_server, write=write))

    if with_mandate:
        click.echo("mandate (consult memo first):")
        from memo.cli_mandate import _CLIENT_FILES, write_mandates_for_clients

        target_agents = list(agents) or ["all"]
        if "all" in target_agents:
            target_agents = list(_CLIENT_FILES)
        for rel, status in write_mandates_for_clients(target_agents, dry_run=not write):
            click.echo(f"  {rel:<28} {status}")

    if write:
        _seed_install_memory()


def _seed_install_memory() -> None:
    """First-run seed: save one REAL memory recording the install, so the
    first briefing can demonstrate recall on something the user just did.
    Idempotent via ``state_dir/.install_seed.json``; never fails the install
    (no embedder / no MLX on this box → silently skipped)."""
    import json as _json
    import socket
    from datetime import date as _date

    try:
        from memo.config import Config

        cfg = Config.from_env()
        stamp = cfg.state_dir / ".install_seed.json"
        if stamp.exists():
            return
        try:
            from importlib.metadata import version as _pkg_version

            ver = _pkg_version("mlx-memo")
        except Exception:
            ver = "unknown"
        from memo.memory import Memory

        cfg.state_dir.mkdir(parents=True, exist_ok=True)
        rec = Memory(cfg).save(
            content=(
                f"Installed memo {ver} on {_date.today().isoformat()} on "
                f"{socket.gethostname()}. memo now recalls relevant memories "
                "on every prompt and briefs at session start."
            ),
            title=f"memo {ver} installed",
            type_="note",
            tags=["memo-install-seed"],
        )
        stamp.write_text(
            _json.dumps({"id": rec.id, "ts": _date.today().isoformat(), "shown": False}),
            encoding="utf-8",
        )
    except Exception:  # noqa: S110  # onboarding decoration — never fail an install
        pass
