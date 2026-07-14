"""`memo config` command group — inspect + validate feature flags.

Surfaces the central `flags.py` registry: list every documented `MEMO_*`
flag with its type/default/group, show which are currently active, and
validate the environment for misconfigured or unknown vars.

Registered onto the root group in cli.py via `cli.add_command(config_group)`.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.table import Table

from memo import flags
from memo.cli_common import console


def _terminal_is_interactive() -> bool:
    import sys

    return sys.stdin.isatty() and sys.stdout.isatty()


@click.group(name="config", invoke_without_command=True)
@click.pass_context
def config_group(ctx: click.Context) -> None:
    """Inspect, validate, and edit memo configuration."""
    if ctx.invoked_subcommand is not None:
        return

    from memo.flags import flag_bool

    if flag_bool("MEMO_NONINTERACTIVE") or not _terminal_is_interactive():
        click.echo(ctx.get_help())
        return

    from memo.tui.config import run_config_tui

    raise click.exceptions.Exit(run_config_tui())


@config_group.command(name="flags")
@click.option("--group", "group_filter", default=None, help="Filter by subsystem group.")
@click.option("--active", is_flag=True, help="Only flags currently set in the environment.")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON.")
def config_flags(group_filter: str | None, active: bool, as_json: bool) -> None:
    """List documented MEMO_* flags (type, default, group, active value).

    Example: memo config flags --group recall
    """
    active_vals = flags.active_flags()
    rows = []
    for name, spec in flags.REGISTRY.items():
        if group_filter and spec.group != group_filter:
            continue
        if active and name not in active_vals:
            continue
        rows.append(
            {
                "flag": name,
                "group": spec.group,
                "kind": spec.kind,
                "default": spec.default,
                "active": active_vals.get(name),
                "help": spec.help,
            }
        )

    if as_json:
        click.echo(json.dumps(rows, indent=2, default=str))
        return

    table = Table(title="MEMO_* flags")
    table.add_column("flag", style="cyan", no_wrap=True)
    table.add_column("group", style="magenta")
    table.add_column("kind")
    table.add_column("default", style="dim")
    table.add_column("active", style="green")
    for r in rows:
        table.add_row(
            r["flag"],
            r["group"],
            r["kind"],
            "" if r["default"] is None else str(r["default"]),
            "" if r["active"] is None else str(r["active"]),
        )
    console.print(table)
    console.print(f"[dim]{len(rows)} flag(s); {len(active_vals)} active in env[/dim]")


@config_group.command(name="validate")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON.")
def config_validate(as_json: bool) -> None:
    """Parse every set MEMO_* flag; report misconfigured or unknown vars.

    Exit code 1 if any problems are found. Example: memo config validate
    """
    problems = flags.validate()
    active_vals = flags.active_flags()

    if as_json:
        click.echo(json.dumps({"active": len(active_vals), "problems": problems}, indent=2))
    elif not problems:
        console.print(f"[green]✓[/green] {len(active_vals)} flag(s) set, all valid")
    else:
        console.print(f"[red]✗ {len(problems)} problem(s):[/red]")
        for p in problems:
            console.print(f"  [yellow]{p['flag']}[/yellow]={p['value']!r} — {p['error']}")

    if problems:
        raise SystemExit(1)


@config_group.command(name="init")
@click.option("--force", is_flag=True, help="Overwrite existing Markdown config files.")
def config_init(force: bool) -> None:
    """Create Markdown config files with current defaults."""
    from memo.config import Config
    from memo.config_md import write_default_config

    cfg = Config.from_env()
    try:
        written = write_default_config(
            data_dir=cfg.data_dir, vault_path=cfg.vault_path, force=force
        )
    except FileExistsError as exc:
        raise click.ClickException(
            f"config already exists: {exc}; use --force to overwrite"
        ) from exc
    console.print(f"[green]created[/green] {len(written)} Markdown config file(s)")


@config_group.command(name="path")
def config_path() -> None:
    """Print active Markdown and legacy config paths."""
    from memo.config_md import config_dir, config_home, index_path
    from memo.setup.config_io import _resolve_config_path

    console.print(f"config_home: {config_home()}")
    console.print(f"index: {index_path()}")
    console.print(f"domains: {config_dir()}")
    console.print(f"legacy_toml: {_resolve_config_path()}")


@config_group.command(name="set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str) -> None:
    """Set a Markdown config value, for example recall.top_k 5."""
    from memo.config_md import set_value, validate_markdown_config

    try:
        path = set_value(key, value)
    except (KeyError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    problems = validate_markdown_config()
    if problems:
        raise click.ClickException("; ".join(f"{p.key}: {p.error}" for p in problems))
    console.print(f"[green]set[/green] {key} in {path}")


@config_group.command(name="unset")
@click.argument("key")
def config_unset(key: str) -> None:
    """Remove a Markdown config override."""
    from memo.config_md import unset_value, validate_markdown_config

    try:
        path = unset_value(key)
    except (KeyError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    problems = validate_markdown_config()
    if problems:
        raise click.ClickException("; ".join(f"{p.key}: {p.error}" for p in problems))
    console.print(f"[green]unset[/green] {key} in {path}")


@config_group.command(name="show")
@click.option("--effective", is_flag=True, help="Show effective values and their sources.")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON.")
def config_show(effective: bool, as_json: bool) -> None:
    """Show Markdown config values."""
    from memo import flags as memo_flags
    from memo.config_md import field_values

    active_flags = memo_flags.active_flags()
    markdown_flags = memo_flags.active_config_values()
    rows = [
        {"key": field, "value": value, "source": "markdown", "env": ""}
        for field, value in sorted(field_values().items())
    ]
    for env_name, markdown_value in sorted(markdown_flags.items()):
        env_value = active_flags.get(env_name)
        rows.append(
            {
                "key": env_name,
                "value": env_value if effective and env_value is not None else markdown_value,
                "source": "env" if effective and env_value is not None else "markdown",
                "env": env_name,
            }
        )

    if as_json:
        click.echo(json.dumps(rows, indent=2, default=str))
        return

    table = Table(title="memo config" if not effective else "memo effective config")
    table.add_column("key", style="cyan")
    table.add_column("value")
    table.add_column("source", style="magenta")
    table.add_column("env", style="dim")
    for row in rows:
        table.add_row(str(row["key"]), str(row["value"]), str(row["source"]), str(row["env"]))
    console.print(table)


@config_group.command(name="migrate")
@click.option("--force", is_flag=True, help="Overwrite existing Markdown config files.")
def config_migrate(force: bool) -> None:
    """Migrate legacy config.toml into Markdown config files."""
    from memo.config_md import write_default_config
    from memo.setup.config_io import load_config_file, snapshot_config_file

    legacy = load_config_file() or {}
    storage = legacy.get("storage") if isinstance(legacy, dict) else {}
    if not isinstance(storage, dict) or not storage.get("data_dir"):
        raise click.ClickException("legacy config.toml has no [storage].data_dir to migrate")
    backup = snapshot_config_file(label="pre-md-config")
    written = write_default_config(
        data_dir=Path(str(storage["data_dir"])),
        vault_path=Path(str(storage["vault_path"])) if storage.get("vault_path") else None,
        force=force,
    )
    console.print(f"[green]migrated[/green] legacy config to {len(written)} Markdown file(s)")
    if backup is not None:
        console.print(f"[dim]legacy backup: {backup}[/dim]")
