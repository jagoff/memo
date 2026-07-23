"""Runtime + install plumbing facade for the memo CLI.

This module keeps the historical import surface stable while delegating the
heavier responsibilities to focused modules in ``memo.runtime``.
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from memo.cli_common import console
from memo.cli_mandate import write_mandates_for_clients
from memo.runtime.codex import (
    _codex_home,
    _codex_read_app_server_response,
    _codex_send_app_server_request,
    _copy_slash_skill,
    _install_codex_plugin,
)
from memo.runtime.detect import (
    _env_root_for_bin,
    _install_mode,
    _path_is_relative_to,
    _resolve_command,
    _resolved_memo_mcp,
    _runtime_install_report,
    _safe_resolve,
)
from memo.runtime.mcp import (
    _MCP_ENV_FORWARD_KEYS,
    _MISSING_MCP_OK_ERRORS,
    _agent_asset_root,
    _devin_desktop_mcp_config_path,
    _env_flags,
    _format_command,
    _install_devin_desktop_mcp,
    _mcp_add_command,
    _mcp_server_env,
    _mcp_server_json,
    _run_agent_command,
)
from memo.runtime.migrate import _consolidate_sidecar_dbs, migrate_vault
from memo.runtime.shell_wrapper import _WRAPPER_SNIPPET_ZSH, install_shell_wrapper
from memo.runtime.update import self_update

__all__ = [
    "_MCP_ENV_FORWARD_KEYS",
    "_MISSING_MCP_OK_ERRORS",
    "_WRAPPER_SNIPPET_ZSH",
    "_agent_asset_root",
    "_codex_home",
    "_codex_read_app_server_response",
    "_codex_send_app_server_request",
    "_consolidate_sidecar_dbs",
    "_copy_slash_skill",
    "_devin_desktop_mcp_config_path",
    "_env_flags",
    "_env_root_for_bin",
    "_format_command",
    "_install_codex_plugin",
    "_install_devin_desktop_mcp",
    "_install_mode",
    "_mcp_add_command",
    "_mcp_server_env",
    "_mcp_server_json",
    "_path_is_relative_to",
    "_resolve_command",
    "_resolved_memo_mcp",
    "_run_agent_command",
    "_runtime_install_report",
    "_safe_resolve",
    "init_cmd",
    "install_shell_wrapper",
    "install_slash",
    "mcp_command",
    "migrate_vault",
    "self_update",
]


def _init_is_interactive() -> bool:
    """True when `memo init` can safely run its interactive picker.

    The picker needs a real TTY; `MEMO_NONINTERACTIVE=1` (set by hooks) also
    opts out. Extracted so tests can simulate an interactive terminal.
    """
    import sys

    from memo.flags import flag_bool

    if flag_bool("MEMO_NONINTERACTIVE"):
        return False
    return sys.stdin.isatty() and sys.stdout.isatty()


@click.command(name="init")
@click.option("--force", is_flag=True, help="Overwrite existing config without confirmation.")
def init_cmd(force: bool) -> None:
    """(Re)configure where memo stores memories."""
    from memo.config_md import config_dir, index_path
    from memo.setup.config_io import _resolve_config_path

    markdown_index = index_path()
    markdown_dir = config_dir()
    cfg_path = _resolve_config_path()
    existing_config = next(
        (
            path
            for path in (markdown_index, markdown_dir, cfg_path)
            if path.is_file() or path.is_dir()
        ),
        None,
    )
    if (
        existing_config is not None
        and not force
        and not click.confirm(
            f"Config exists at {existing_config}. Overwrite?",
            default=False,
        )
    ):
        console.print("[yellow]aborted[/yellow]")
        return
    # The picker is interactive; in a non-TTY / non-interactive context (piped
    # install, CI) exit cleanly instead of crashing inside prompt_toolkit.
    if not _init_is_interactive():
        console.print(
            "[yellow]memo init needs an interactive terminal.[/yellow] "
            "Set MEMO_DATA_DIR (and optionally MEMO_VAULT_PATH) to configure "
            "non-interactively, or run `memo init` from a TTY."
        )
        raise click.exceptions.Exit(1)
    from memo.cli import _run_picker_and_save

    _run_picker_and_save()


@click.command(name="mcp-command")
@click.option(
    "--client",
    type=click.Choice(
        ["claude-code", "claude-desktop", "codex", "devin", "devin-desktop", "opencode", "json"]
    ),
    default="claude-code",
    show_default=True,
    help="Emit a client-specific MCP registration command or raw JSON config.",
)
def mcp_command(client: str) -> None:
    memo_mcp = _require_isolated_memo_mcp()
    env = _mcp_server_env()
    if client == "json":
        click.echo(
            json.dumps(
                {"mcpServers": {"memo": _mcp_server_json(memo_mcp, env, include_type=True)}},
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if client == "claude-desktop":
        click.echo(
            json.dumps(
                {"mcpServers": {"memo": _mcp_server_json(memo_mcp, env, include_type=False)}},
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if client == "devin-desktop":
        click.echo(
            json.dumps(
                {
                    "mcpServers": {
                        "memo": _mcp_server_json(
                            memo_mcp,
                            {**env, "MEMO_SOURCE": "devin-desktop"},
                            include_type=True,
                        )
                    }
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if client == "codex":
        click.echo(
            _format_command(_mcp_add_command("codex", memo_mcp, {**env, "MEMO_SOURCE": "codex"}))
        )
        return
    if client == "devin":
        click.echo(
            _format_command(_mcp_add_command("devin", memo_mcp, {**env, "MEMO_SOURCE": "devin"}))
        )
        return
    if client == "opencode":
        click.echo(
            _format_command(
                _mcp_add_command("opencode", memo_mcp, {**env, "MEMO_SOURCE": "opencode"})
            )
        )
        return
    click.echo(_format_command(_mcp_add_command("claude-code", memo_mcp, env)))


@click.command(name="install-slash")
@click.option(
    "--client",
    "clients",
    multiple=True,
    type=click.Choice(
        ["all", "claude-code", "codex", "devin", "devin-desktop", "opencode", "blackbox"]
    ),
    help="Client to configure. Repeatable. Defaults to all supported agent clients.",
)
@click.option(
    "--repo",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Path to the memo checkout containing plugin/skill assets.",
)
@click.option("--dry-run", is_flag=True, help="Print the commands without changing configs.")
@click.option(
    "--best-effort",
    is_flag=True,
    help="Warn and continue if a client CLI is missing or rejects configuration.",
)
def install_slash(
    clients: tuple[str, ...],
    repo: Path | None,
    dry_run: bool,
    best_effort: bool,
) -> None:
    selected = set(clients or ("all",))
    if "all" in selected:
        selected.remove("all")
        selected.update({"claude-code", "codex", "devin", "devin-desktop", "opencode"})

    needs_assets = bool(selected & {"claude-code", "codex", "devin"})
    root = _agent_asset_root(repo) if needs_assets else None
    memo_mcp = _require_isolated_memo_mcp()
    env = _mcp_server_env()

    failures: list[str] = []

    def run_client(label: str, fn) -> None:
        try:
            fn()
        except click.ClickException as exc:
            if not best_effort:
                raise
            failures.append(label)
            console.print(f"[yellow]![/yellow] {label}: {exc.message}")

    def install_codex() -> None:
        assert root is not None
        console.print("[bold]Codex[/bold]")
        _copy_slash_skill(root, _codex_home() / "skills" / "memo" / "SKILL.md", dry_run=dry_run)
        _install_codex_plugin(root, dry_run=dry_run)
        _run_agent_command(["codex", "mcp", "remove", "memo"], dry_run=dry_run, best_effort=True)
        _run_agent_command(
            _mcp_add_command("codex", memo_mcp, {**env, "MEMO_SOURCE": "codex"}), dry_run=dry_run
        )
        console.print(
            "[yellow]![/yellow] Codex CLI's TUI slash menu currently lists only built-in "
            "slash commands. memo is installed as a model-visible skill and MCP server, "
            "but `/memo` may not appear in that menu."
        )

    def install_claude_code() -> None:
        assert root is not None
        console.print("[bold]Claude Code[/bold]")
        _run_agent_command(
            ["claude", "plugin", "marketplace", "add", root],
            dry_run=dry_run,
            ok_errors=("already", "exists"),
        )
        _run_agent_command(
            ["claude", "plugin", "install", "memo@memo", "-s", "user"],
            dry_run=dry_run,
            ok_errors=("already", "installed", "exists"),
        )
        _run_agent_command(
            ["claude", "mcp", "remove", "-s", "user", "memo"], dry_run=dry_run, best_effort=True
        )
        _run_agent_command(_mcp_add_command("claude-code", memo_mcp, env), dry_run=dry_run)

    def install_devin() -> None:
        assert root is not None
        console.print("[bold]Devin[/bold]")
        _copy_slash_skill(
            root,
            Path.home() / ".config" / "devin" / "skills" / "memo" / "SKILL.md",
            dry_run=dry_run,
        )
        _run_agent_command(
            ["devin", "mcp", "remove", "-s", "user", "memo"], dry_run=dry_run, best_effort=True
        )
        _run_agent_command(
            _mcp_add_command("devin", memo_mcp, {**env, "MEMO_SOURCE": "devin"}), dry_run=dry_run
        )

    def install_devin_desktop() -> None:
        console.print("[bold]Devin Desktop[/bold]")
        _install_devin_desktop_mcp(
            memo_mcp, {**env, "MEMO_SOURCE": "devin-desktop"}, dry_run=dry_run
        )
        console.print("[dim]Restart Devin Desktop after editing config.[/dim]")

    def install_opencode() -> None:
        console.print("[bold]OpenCode[/bold]")
        _run_agent_command(
            _mcp_add_command("opencode", memo_mcp, {**env, "MEMO_SOURCE": "opencode"}),
            dry_run=dry_run,
        )

    if "codex" in selected:
        run_client("Codex", install_codex)
    if "claude-code" in selected:
        run_client("Claude Code", install_claude_code)
    if "devin" in selected:
        run_client("Devin", install_devin)
    if "opencode" in selected:
        run_client("OpenCode", install_opencode)
    if "devin-desktop" in selected:
        run_client("Devin Desktop", install_devin_desktop)

    mandate_clients = [
        client
        for client in selected
        if client in {"codex", "devin", "devin-desktop", "opencode", "cursor", "blackbox"}
    ]
    if mandate_clients:
        console.print("[bold]Mandate[/bold]")
        for rel, status in write_mandates_for_clients(
            mandate_clients, cwd=Path.cwd(), dry_run=dry_run
        ):
            console.print(f"  {rel:<22} {status}")

    # Startup-banner shims — wrap agent binaries to show [MEMO ver] at launch.
    # No --client at all defaults to "all" (see `selected` above), so it must
    # install the same shim set as an explicit `--client all`.
    all_requested = not clients or "all" in clients
    _shim_agents = tuple(
        a
        for a in ("codex", "devin", "opencode", "gemini", "blackbox")
        if a in selected or all_requested
    )
    if _shim_agents:
        from memo.runtime.shims import _DEFAULT_BIN_DIR, install_path_snippet, install_shims

        console.print("[bold]Startup-banner shims[/bold]")
        shim_results = install_shims(_shim_agents, _DEFAULT_BIN_DIR, dry_run=dry_run)
        for r in shim_results:
            kind, path = r.split(":", 1)
            icon = "[green]✓[/green]" if kind == "wrote" else "[dim]✓[/dim]"
            console.print(f"  {icon} {path}")
        path_status = install_path_snippet(_DEFAULT_BIN_DIR, dry_run=dry_run)
        if path_status.startswith("written"):
            console.print(f"  [green]✓[/green] PATH snippet → {path_status.split(':', 1)[1]}")
        elif path_status == "already":
            console.print("  [dim]✓ ~/.memo/bin already in PATH snippet[/dim]")
        else:
            console.print(
                f'  [yellow]![/yellow] PATH: {path_status} — add manually: export PATH="$HOME/.memo/bin:$PATH"'
            )

    if failures:
        console.print(
            "[yellow]![/yellow] agent-client install finished with skipped clients: "
            + ", ".join(failures)
        )
        console.print(
            "[dim]Install those clients, then rerun: "
            "memo install-slash --client claude-code --client codex --client opencode --client devin-desktop[/dim]"
        )
    else:
        console.print(
            "[green]✓[/green] agent-client install complete. Open a new agent session to reload."
        )


def _require_isolated_memo_mcp() -> Path:
    """Return a persistent-safe memo-mcp path or raise with the unsafe fallback.

    Commands here emit text copied into long-lived agent configs. A project
    `.venv/bin/memo-mcp` works only while this checkout exists and has matching
    deps, so prefer the isolated resolver shared with `install-mcp`.
    """
    from memo.cli_install_mcp import _resolve_isolated_memo_mcp

    fallback = _resolved_memo_mcp()
    if fallback is not None and not _is_project_venv_path(fallback):
        return fallback

    memo_mcp = _resolve_isolated_memo_mcp()
    if memo_mcp is not None:
        return memo_mcp

    found = f" Found: {fallback}" if fallback is not None else ""
    raise click.ClickException(
        "isolated memo-mcp not found. Install memo as an isolated tool first: "
        "`pipx install mlx-memo` or `uv tool install mlx-memo`; a project .venv "
        f"must not be written into agent configs.{found}"
    )


def _is_project_venv_path(path: Path) -> bool:
    return any(part in {".venv", "venv"} for part in path.parts)
