"""`memo consolidate` command group — cluster consolidation proposals.

Extracted from cli.py (3a god-module decomposition). Registered onto the
root group in cli.py via `cli.add_command(consolidate_group)`.
"""

from __future__ import annotations

import json

import click
from rich.table import Table

from memo.cli_common import console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config

# -- advanced consolidation commands -------------------------------------------


@click.group(name="consolidate")
def consolidate_group() -> None:
    """Advanced consolidation with intelligent merge and archival."""
    pass


@consolidate_group.command(name="propose")
@click.option(
    "--threshold", type=float, default=0.85, help="Cosine similarity threshold (default: 0.85)"
)
@click.option(
    "--max-clusters", type=int, default=20, help="Maximum clusters to process (default: 20)"
)
@click.option("--type", "type_", help="Filter by memoria type")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def consolidate_propose(
    threshold: float,
    max_clusters: int,
    type_: str | None,
    as_json: bool,
) -> None:
    """Detect clusters and propose merge strategies (read-only).

    Example: memo consolidate propose --threshold 0.9
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    result = mem.consolidator.consolidate_all(
        threshold=threshold,
        max_clusters=max_clusters,
        type_=type_,
        auto_apply=False,
        dry_run=True,
    )

    if as_json:
        click.echo(json.dumps(result, indent=2))
        return

    clusters = result.get("clusters", [])
    proposals = result.get("proposals", [])

    console.print(f"[bold]Detected {len(clusters)} clusters[/bold]")
    console.print(f"[yellow]Generated {len(proposals)} merge proposals[/yellow]")
    console.print()

    if not proposals:
        console.print("[green]No mergeable clusters found[/green]")
        return

    for i, p in enumerate(proposals[:10], 1):
        console.print(f"[cyan]{i}. Cluster {p['cluster_id']}[/cyan]")
        console.print(f"   Strategy: {p['merge_strategy']}")
        console.print(f"   Rationale: {p['rationale']}")
        console.print(f"   Memorias to merge: {len(p['memoria_ids'])}")
        console.print()

    if len(proposals) > 10:
        console.print(f"[dim]...and {len(proposals) - 10} more proposals[/dim]")


@consolidate_group.command(name="apply")
@click.option(
    "--threshold",
    type=float,
    default=0.85,
    help="Cosine similarity threshold for the LLM pass (default: 0.85)",
)
@click.option(
    "--auto-threshold",
    "auto_threshold",
    type=float,
    default=None,
    help="Cosine floor for the LLM-free fast lane (default: MEMO_CONSOLIDATE_AUTO_THRESHOLD=0.95)",
)
@click.option(
    "--max-clusters", type=int, default=20, help="Maximum clusters to process (default: 20)"
)
@click.option("--type", "type_", help="Filter by memoria type")
@click.option("--dry-run", is_flag=True, help="Show what would happen without applying changes")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
@click.confirmation_option(prompt="This will merge memorias and archive old ones. Continue?")
def consolidate_apply(
    threshold: float,
    auto_threshold: float | None,
    max_clusters: int,
    type_: str | None,
    dry_run: bool,
    as_json: bool,
) -> None:
    """Apply merge proposals to consolidate the corpus.

    Runs a two-pass pipeline: fast lane (cosine ≥ auto_threshold, no LLM) then
    LLM pass (cosine ≥ threshold). Use --dry-run to preview first.

    Example: memo consolidate apply --dry-run
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    result = mem.consolidator.consolidate_all(
        threshold=threshold,
        max_clusters=max_clusters,
        type_=type_,
        auto_apply=True,
        dry_run=dry_run,
        auto_threshold=auto_threshold,
    )

    if as_json:
        click.echo(json.dumps(result, indent=2))
        return

    result.get("proposals", [])
    results = result.get("results", [])

    if dry_run:
        console.print("[yellow]Dry run mode - no changes applied[/yellow]")
        console.print()

    console.print(f"[bold]Processed {len(results)} consolidations[/bold]")
    console.print()

    merged_count = sum(1 for r in results if r.get("merged_id"))
    archived_count = sum(len(r.get("archived_ids", [])) for r in results)
    skipped_count = sum(len(r.get("skipped_ids", [])) for r in results)

    console.print(f"[green]✓ Merged {merged_count} memorias[/green]")
    console.print(f"[yellow]↻ Archived {archived_count} old versions[/yellow]")
    console.print(f"[dim]⊘ Skipped {skipped_count} (conflicts)[/dim]")

    for r in results[:5]:
        console.print(f"  {r.get('summary', '')}")

    if len(results) > 5:
        console.print(f"[dim]...and {len(results) - 5} more[/dim]")


@consolidate_group.command(name="list-archived")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def consolidate_list_archived(as_json: bool) -> None:
    """List all archived memorias.

    Example: memo consolidate list-archived
    """
    cfg = Config.from_env()
    archival_dir = cfg.memory_dir / "archived"

    if not archival_dir.is_dir():
        console.print("[dim]No archived memorias found[/dim]")
        return

    archived_files = list(archival_dir.glob("*.md"))

    if as_json:
        archived_data = []
        for f in archived_files:
            import frontmatter

            post = frontmatter.loads(f.read_text(encoding="utf-8"))
            archived_data.append(
                {
                    "id": f.stem,
                    "title": post.get("title", ""),
                    "archived_for": post.get("archived_for", ""),
                    "archived_at": post.get("archived_at", ""),
                }
            )
        click.echo(json.dumps(archived_data, indent=2))
        return

    table = Table(title="Archived Memorias")
    table.add_column("ID", style="cyan")
    table.add_column("Title", style="yellow")
    table.add_column("Archived For", style="green")
    table.add_column("Archived At", style="dim")

    for f in archived_files[:50]:
        import frontmatter

        post = frontmatter.loads(f.read_text(encoding="utf-8"))
        table.add_row(
            f.stem[:8],
            str(post.get("title") or "")[:40],
            str(post.get("archived_for") or "")[:8],
            str(post.get("archived_at") or "")[:10],
        )

    console.print(table)
    if len(archived_files) > 50:
        console.print(f"[dim]...and {len(archived_files) - 50} more[/dim]")
