"""`memo federation` command group — multi-vault federation.

Extracted from cli.py (3a god-module decomposition). Registered onto the
root group in cli.py via `cli.add_command(federation_group)`.
"""

from __future__ import annotations

import json

import click
from rich.table import Table

from memo.cli_common import console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config

# -- federation commands -------------------------------------------------------


@click.group(name="federation")
def federation_group() -> None:
    """Multi-vault federation — search across multiple vaults."""
    pass


@federation_group.command(name="add-vault")
@click.argument("name")
@click.argument("path")
@click.option("--weight", type=float, default=1.0, help="Vault weight for ranking")
def federation_add_vault(name: str, path: str, weight: float) -> None:
    """Add a vault to the federation.

    Example: memo federation add-vault work-vault /path/to/work/memo --weight 1.5
    """
    cfg = Config.from_env()
    from memo.federation import FederationConfig

    config = FederationConfig(cfg.state_dir / "federation.json")
    config.add_vault(name, path, weight)

    console.print(f"[green]Added vault '{name}'[/green]")


@federation_group.command(name="list-vaults")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def federation_list_vaults(as_json: bool) -> None:
    """List all configured vaults.

    Example: memo federation list-vaults
    """
    cfg = Config.from_env()
    from memo.federation import FederationConfig

    config = FederationConfig(cfg.state_dir / "federation.json")
    vaults = config.list_vaults()

    if as_json:
        click.echo(json.dumps([v.__dict__ for v in vaults], indent=2))
        return

    if not vaults:
        console.print("[dim]No vaults configured[/dim]")
        return

    table = Table(title="Federated Vaults")
    table.add_column("Name", style="cyan")
    table.add_column("Path", style="yellow")
    table.add_column("Weight", style="green")
    table.add_column("Enabled", style="magenta")

    for v in vaults:
        table.add_row(
            v.name,
            v.path,
            str(v.weight),
            "Yes" if v.enabled else "No",
        )

    console.print(table)


@federation_group.command(name="remove-vault")
@click.argument("name")
@click.confirmation_option(prompt="Remove this vault from federation?")
def federation_remove_vault(name: str) -> None:
    """Remove a vault from the federation.

    Example: memo federation remove-vault work-vault
    """
    cfg = Config.from_env()
    from memo.federation import FederationConfig

    config = FederationConfig(cfg.state_dir / "federation.json")
    success = config.remove_vault(name)

    if success:
        console.print(f"[green]Removed vault '{name}'[/green]")
    else:
        console.print(f"[yellow]Vault '{name}' not found[/yellow]")


@federation_group.command(name="search")
@click.argument("query")
@click.option("--limit", type=int, default=10, help="Result limit")
@click.option("--mode", type=click.Choice(["vec", "bm25", "hybrid"]), default="hybrid",
              help="Search mode (default: hybrid)")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def federation_search(query: str, limit: int, mode: str, as_json: bool) -> None:
    """Search across all federated vaults.

    Example: memo federation search "MLX" --limit 20
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    results = mem.federation.search(query, limit=limit, mode=mode)

    if as_json:
        click.echo(json.dumps([r.__dict__ for r in results], indent=2))
        return

    if not results:
        console.print("[dim]No results found[/dim]")
        return

    table = Table(title=f"Federated Search Results for '{query}'")
    table.add_column("ID", style="cyan")
    table.add_column("Vault", style="yellow")
    table.add_column("Title", style="green")
    table.add_column("Score", style="magenta")

    for r in results[:20]:
        table.add_row(
            r.memoria_id[:8],
            r.vault_name,
            r.title[:40],
            f"{r.score:.3f}",
        )

    console.print(table)
    if len(results) > 20:
        console.print(f"[dim]...and {len(results) - 20} more[/dim]")
