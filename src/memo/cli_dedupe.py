from __future__ import annotations

import json

import click

from memo.cli_common import _short, console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config


@click.command(name="dedupe")
@click.option(
    "--threshold",
    type=float,
    default=0.92,
    help="Cosine threshold for near-duplicate clustering (default: 0.92)",
)
@click.option("--max-clusters", type=int, default=50, help="Max clusters to surface (default: 50)")
@click.option("--type", "type_", help="Filter by memory type")
@click.option(
    "--apply",
    "do_apply",
    is_flag=True,
    help="Interactively merge each cluster (default: list-only)",
)
@click.option("--dry-run", is_flag=True, help="With --apply: show merges without writing")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def dedupe_cmd(
    threshold: float,
    max_clusters: int,
    type_: str | None,
    do_apply: bool,
    dry_run: bool,
    as_json: bool,
) -> None:
    """Find and (optionally) merge near-duplicate memories."""
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    clusters = mem.consolidate(
        threshold=threshold,
        max_clusters=max_clusters,
        type_=type_,
    )
    dup_clusters = [c for c in clusters if c.get("relationship") in ("duplicate", "evolution")]

    if as_json:
        click.echo(json.dumps(dup_clusters, indent=2))
        return

    if not dup_clusters:
        console.print("[green]No near-duplicate clusters found at this threshold.[/green]")
        return

    console.print(f"[bold]Found {len(dup_clusters)} duplicate-like cluster(s).[/bold]")

    if not do_apply:
        for c in dup_clusters[:20]:
            console.print()
            console.print(
                f"[cyan]cluster {c.get('cluster_id', '?')}[/cyan] · "
                f"rel={c.get('relationship')} · n={len(c.get('members', []))}"
            )
            console.print(f"  [dim]{_short(c.get('summary', ''), 200)}[/dim]")
            for m in c.get("members", []):
                console.print(f"    - {m['id'][:8]} · {_short(m.get('title', ''), 70)}")
        if len(dup_clusters) > 20:
            console.print(f"[dim]…and {len(dup_clusters) - 20} more[/dim]")
        console.print()
        console.print("[dim]Re-run with --apply to merge interactively.[/dim]")
        return

    for c in dup_clusters:
        console.print()
        console.print(
            f"[cyan]cluster {c.get('cluster_id', '?')}[/cyan] · "
            f"rel={c.get('relationship')} · n={len(c.get('members', []))}"
        )
        for m in c.get("members", []):
            console.print(f"    - {m['id'][:8]} · {_short(m.get('title', ''), 70)}")

        if not click.confirm("Propose merge for this cluster?", default=True):
            continue

        proposal = mem.consolidator.propose_merge(c)
        if proposal is None:
            console.print("[red]No merge proposal generated.[/red]")
            continue
        console.print(f"[bold]merged title:[/bold] {proposal.merged_title}")
        console.print(f"[dim]strategy={proposal.merge_strategy}[/dim]")
        console.print(f"[dim]rationale={proposal.rationale}[/dim]")

        if not click.confirm("Apply merge?", default=False):
            continue

        result = mem.consolidator.apply_merge(proposal, dry_run=dry_run)
        console.print(
            f"[green]merged →[/green] "
            f"{result.merged_id[:8] if result.merged_id else 'n/a'}  "
            f"archived={len(result.archived_ids)}"
        )
