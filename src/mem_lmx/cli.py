"""CLI — `mem-lmx` entry point.

A handful of operational commands so the user can interact with the
memory store from the shell without spinning up the MCP server:

- `mem-lmx save 'content here' --title 'X' --tag x --tag y`
- `mem-lmx search 'query' --limit 5`
- `mem-lmx list --limit 20 --type decision`
- `mem-lmx get <id>`
- `mem-lmx delete <id>`
- `mem-lmx stats`
- `mem-lmx doctor` — verify vault path, embedder loadable, sqlite-vec
  available, MLX present.

Output style:
- Default: rich table for list/search, panel for `get`, plain stats.
- `--json` flag (where applicable): emit raw JSON for piping.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from mem_lmx.config import Config

console = Console()


@click.group()
@click.version_option()
def cli() -> None:
    """mem-lmx — local MCP memory backed by Obsidian vault, MLX-native."""


@cli.command()
@click.argument("content")
@click.option("--title", default=None, help="Short title (default: first line of content)")
@click.option(
    "--type", "type_",
    type=click.Choice(
        ["decision", "fact", "bug", "feedback", "preference", "note", "manual"],
    ),
    default="note", show_default=True,
)
@click.option("--tag", "-t", "tags", multiple=True, help="Repeatable. Lower-cased + de-duplicated.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of a panel.")
def save(content: str, title: str | None, type_: str, tags: tuple[str, ...], as_json: bool) -> None:
    """Persist CONTENT to the vault + index."""
    from mem_lmx.memory import Memory

    mem = Memory(Config.from_env())
    rec = mem.save(content=content, title=title, type_=type_, tags=list(tags))
    if as_json:
        click.echo(json.dumps(rec.to_dict(), ensure_ascii=False, indent=2))
        return
    console.print(Panel.fit(
        f"[bold]{rec.title}[/bold]\n"
        f"[dim]id:[/dim] {rec.id}\n"
        f"[dim]path:[/dim] {rec.path}\n"
        f"[dim]type:[/dim] {rec.type}  [dim]tags:[/dim] {', '.join(rec.tags) or '—'}",
        title="✓ saved", border_style="green",
    ))


@cli.command()
@click.argument("query")
@click.option("--limit", default=10, type=int, show_default=True)
@click.option("--type", "type_", default=None, help="Filter by record type.")
@click.option("--json", "as_json", is_flag=True)
def search(query: str, limit: int, type_: str | None, as_json: bool) -> None:
    """Top-k semantic search."""
    from mem_lmx.memory import Memory

    mem = Memory(Config.from_env())
    hits = mem.search(query, limit=limit, type_=type_)
    if as_json:
        click.echo(json.dumps([h.to_dict() for h in hits], ensure_ascii=False, indent=2))
        return
    if not hits:
        console.print("[dim]no results[/dim]")
        return
    tbl = Table(show_lines=False, expand=True)
    tbl.add_column("score", justify="right", width=6)
    tbl.add_column("type", width=10)
    tbl.add_column("title", overflow="fold")
    tbl.add_column("tags", overflow="fold")
    for h in hits:
        tbl.add_row(
            f"{h.score:.3f}" if h.score is not None else "—",
            h.type,
            h.title,
            ", ".join(h.tags) or "—",
        )
    console.print(tbl)


@cli.command(name="list")
@click.option("--limit", default=20, type=int, show_default=True)
@click.option("--type", "type_", default=None)
@click.option("--json", "as_json", is_flag=True)
def list_cmd(limit: int, type_: str | None, as_json: bool) -> None:
    """Recent memories by `updated` desc."""
    from mem_lmx.memory import Memory

    mem = Memory(Config.from_env())
    items = mem.list(limit=limit, type_=type_)
    if as_json:
        click.echo(json.dumps([r.to_dict() for r in items], ensure_ascii=False, indent=2))
        return
    if not items:
        console.print("[dim]vacío[/dim]")
        return
    tbl = Table(show_lines=False, expand=True)
    tbl.add_column("updated", width=20)
    tbl.add_column("type", width=10)
    tbl.add_column("title", overflow="fold")
    tbl.add_column("tags", overflow="fold")
    for r in items:
        tbl.add_row(r.updated[:19], r.type, r.title, ", ".join(r.tags) or "—")
    console.print(tbl)


@cli.command()
@click.argument("id_")
@click.option("--json", "as_json", is_flag=True)
def get(id_: str, as_json: bool) -> None:
    """Fetch one memory by id."""
    from mem_lmx.memory import Memory

    mem = Memory(Config.from_env())
    rec = mem.get(id_)
    if rec is None:
        console.print(f"[red]not found:[/red] {id_}")
        sys.exit(1)
    if as_json:
        click.echo(json.dumps(rec.to_dict(), ensure_ascii=False, indent=2))
        return
    console.print(Panel.fit(
        f"[bold]{rec.title}[/bold]\n"
        f"[dim]id:[/dim] {rec.id}  [dim]type:[/dim] {rec.type}\n"
        f"[dim]tags:[/dim] {', '.join(rec.tags) or '—'}\n"
        f"[dim]created:[/dim] {rec.created}\n"
        f"[dim]updated:[/dim] {rec.updated}\n\n"
        f"{rec.body}",
        title=rec.title, border_style="cyan",
    ))


@cli.command()
@click.argument("id_")
@click.option("--yes", is_flag=True, help="Skip confirmation.")
def delete(id_: str, yes: bool) -> None:
    """Delete one memory by id."""
    from mem_lmx.memory import Memory

    mem = Memory(Config.from_env())
    if not yes:
        click.confirm(f"Delete memory {id_!r}? This removes the .md and the index entry.", abort=True)
    ok = mem.delete(id_)
    console.print(f"[{'green' if ok else 'red'}]{'✓ deleted' if ok else 'not found'}[/]: {id_}")


@cli.command()
def stats() -> None:
    """Summary stats — total records, vault path, embedder model."""
    from mem_lmx.memory import Memory

    mem = Memory(Config.from_env())
    info: dict[str, Any] = {
        "total": mem.store.count(),
        "vault_path": str(mem.cfg.vault_path),
        "memory_dir": str(mem.cfg.memory_dir),
        "db_path": str(mem.cfg.db_path),
        "embedder_model": mem.cfg.embedder_model,
        "llm_model": mem.cfg.llm_model,
    }
    for k, v in info.items():
        console.print(f"[dim]{k:14s}[/dim] {v}")


@cli.command()
def doctor() -> None:
    """Self-check: vault present, sqlite-vec loadable, MLX importable, models in cache."""
    cfg = Config.from_env()
    ok = True

    # 1. Vault dir
    if cfg.vault_path.is_dir():
        console.print(f"[green]✓[/green] vault: {cfg.vault_path}")
    else:
        console.print(f"[red]✗[/red] vault missing: {cfg.vault_path}")
        ok = False

    # 2. sqlite-vec
    try:
        import sqlite3
        import sqlite_vec  # type: ignore[import-not-found]

        conn = sqlite3.connect(":memory:")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.close()
        console.print("[green]✓[/green] sqlite-vec loadable")
    except Exception as exc:
        console.print(f"[red]✗[/red] sqlite-vec: {exc}")
        ok = False

    # 3. MLX importable
    try:
        import mlx.core  # noqa: F401
        import mlx_lm  # noqa: F401

        console.print("[green]✓[/green] mlx + mlx_lm importable")
    except Exception as exc:
        console.print(f"[red]✗[/red] mlx: {exc}")
        ok = False

    # 4. Models in HF cache
    from pathlib import Path

    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    for model in (cfg.embedder_model, cfg.llm_model, cfg.helper_model):
        cache_dir = hf_cache / f"models--{model.replace('/', '--')}"
        if cache_dir.is_dir():
            console.print(f"[green]✓[/green] cached: {model}")
        else:
            console.print(
                f"[yellow]![/yellow] not cached: {model}  "
                f"[dim](run `hf download {model}`)[/dim]",
            )

    sys.exit(0 if ok else 1)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
