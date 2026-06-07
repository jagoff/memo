"""`memo synthesize` — run one emergent-synthesis pass over the corpus.

Reads all non-synthesis memories, clusters them at a configurable cosine
threshold, and asks the local LLM what each cluster implies that no single
memory states alone. Results are saved as ``type=synthesis`` memorias with
provenance links to the contributing sources.

Requires a local LLM (Qwen2.5-7B or equivalent). Same LLM as consolidation.
Runs automatically in `memo maintain` (opt-out with ``MEMO_SYNTHESIS_ENABLED=0``).
"""

from __future__ import annotations

import json

import click

from memo.cli_common import console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config


@click.command(name="synthesize")
@click.option("--dry-run", is_flag=True, help="Propose syntheses without saving.")
@click.option(
    "--threshold",
    type=float,
    default=None,
    help="Cosine similarity for clustering (default from MEMO_SYNTHESIS_THRESHOLD or 0.78).",
)
@click.option(
    "--min-cluster-size", type=int, default=None, help="Minimum memories per cluster (default 3)."
)
@click.option(
    "--max-clusters", type=int, default=None, help="Max clusters to process (default 20)."
)
@click.option(
    "--min-confidence",
    type=click.Choice(["low", "medium", "high"]),
    default=None,
    help="Minimum confidence to save a synthesis (default medium).",
)
@click.option("--json", "as_json", is_flag=True, help="Emit results as JSON.")
def synthesize_cmd(
    dry_run: bool,
    threshold: float | None,
    min_cluster_size: int | None,
    max_clusters: int | None,
    min_confidence: str | None,
    as_json: bool,
) -> None:
    """Generate emergent insights from related memory clusters.

    Unlike consolidation (which merges duplicates), synthesis asks:
    "what do these memories collectively imply that none states alone?"

    Results are saved as type=synthesis memorias with provenance links.
    Use --dry-run to preview without saving.

    Example:
      memo synthesize --dry-run
      memo maintain   # synthesis runs by default
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    kwargs: dict = {"dry_run": dry_run}
    if threshold is not None:
        kwargs["threshold"] = threshold
    if min_cluster_size is not None:
        kwargs["min_cluster_size"] = min_cluster_size
    if max_clusters is not None:
        kwargs["max_clusters"] = max_clusters
    if min_confidence is not None:
        kwargs["min_confidence"] = min_confidence

    results = mem.synthesize_cross_cluster(**kwargs)

    if as_json:
        click.echo(json.dumps(results, ensure_ascii=False, indent=2))
        return

    tag = "[dim](dry-run)[/dim] " if dry_run else ""
    console.print(f"{tag}[bold]memo synthesize[/bold]")

    if not results:
        console.print("  no synthesis candidates found")
        return

    saved = sum(1 for r in results if r.get("saved"))
    console.print(f"  clusters processed: {len(results)}, saved: {saved}")

    for r in results:
        title = r.get("title") or "(no insight generated)"
        conf = r.get("confidence", "?")
        n_sources = len(r.get("sources", []))
        status = (
            "[green]saved[/green]"
            if r.get("saved")
            else ("[yellow]proposed[/yellow]" if r.get("title") else "[dim]no insight[/dim]")
        )
        mid = f"  [{r['id'][:8]}]" if r.get("id") else "          "
        console.print(f"{mid} {status} [{conf}] {title} (from {n_sources} memories)")
        if r.get("rationale"):
            console.print(f"          [dim]{r['rationale']}[/dim]")
