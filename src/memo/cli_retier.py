"""`memo retier` — reclassify bulk-ingested vault notes into the `reference` tier.

memo's corpus mixes durable knowledge (decisions/facts/preferences) with
bulk-ingested vault material (Obsidian notes, CVs, course quotes, lyrics).
When everything is `type=note` the reference material drowns durable knowledge
in the auto-recall hook. This one-shot migration moves vault-sourced `note`
memorias to `type=reference` so the recall hook (which excludes the reference
tier — see `memo.tiers`) surfaces durable knowledge again. Reference material
stays fully searchable on demand via `memo_search` / `memo search`.

Only `meta.type` changes — bodies/embeddings are untouched, so it's cheap and
needs no re-embed. Dry-run by default; pass `--apply` to commit.
"""

from __future__ import annotations

import click

from memo.cli_common import console
from memo.config import Config
from memo.tiers import is_reference_candidate


@click.command(name="retier")
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    help="Commit the reclassification (default: dry-run preview).",
)
@click.option(
    "--limit", default=20, show_default=True, type=int, help="Sample rows to print in the preview."
)
def retier_cmd(apply_changes: bool, limit: int) -> None:
    """Move vault-sourced `note` memorias into the `reference` tier."""
    from memo.memory import Memory

    mem = Memory(Config.from_env())

    # Pull the whole corpus once (cheap metadata-only rows) and pick out the
    # `note` rows that look like bulk vault ingest.
    rows = mem.store.list_recent(limit=10_000_000)
    candidates = [
        r
        for r in rows
        if r.get("type") == "note"
        and is_reference_candidate(r.get("path"), r.get("tags"), r.get("title"))
    ]
    total_notes = sum(1 for r in rows if r.get("type") == "note")

    console.print(f"corpus: {len(rows)} memorias, {total_notes} of type 'note'")
    console.print(f"vault-sourced notes to retier → reference: [bold]{len(candidates)}[/bold]")
    for r in candidates[:limit]:
        console.print(
            f"  [{(r.get('id') or '')[:8]}] {(r.get('title') or '')[:70]}"
            f"  [dim]«{r.get('path') or ''}»[/dim]"
        )
    if len(candidates) > limit:
        console.print(f"  … and {len(candidates) - limit} more")

    if not apply_changes:
        console.print("\n[dim](dry-run — re-run with --apply to commit)[/dim]")
        return

    n = mem.store.bulk_update_type([r["id"] for r in candidates], "reference")
    console.print(f"\n[green]reclassified {n} memorias to type=reference[/green]")
