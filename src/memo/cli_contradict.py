"""`memo contradict` command group — contradiction radar + triage.

Extracted from cli.py (3a god-module decomposition). Registered onto the
root group in cli.py via `cli.add_command(contradict_group)`.
"""

from __future__ import annotations

import json

import click
from rich.markup import escape
from rich.table import Table

from memo.cli_common import _short, console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config
from memo.resume._utils import _sort_key

# -- contradiction radar + dedupe ---------------------------------------------


@click.group(name="contradict")
def contradict_group() -> None:
    """Detect and triage contradictions / staleness across the corpus.

    `scan` runs the LLM classifier over near-neighbor pairs and stores
    contradictions in a sidecar DB. `list` shows open pairs. `triage`
    walks them one by one and applies the user's verdict (fuse / keep
    newer / dismiss / etc.).
    """
    pass


def _fmt_pair_header(rec_a, rec_b, pair) -> str:
    rel = pair.relationship or "unknown"
    color = "red" if rel == "contradiction" else "yellow"
    return (
        f"[bold {color}]{rel.upper()}[/bold {color}] "
        f"conf={pair.confidence:.2f}  pair={pair.pair_id}"
    )


def _older_first(rec_a, rec_b):
    """Return records ordered by updated instant, not ISO text."""
    return (
        (rec_a, rec_b)
        if _sort_key(getattr(rec_a, "updated", "")) <= _sort_key(getattr(rec_b, "updated", ""))
        else (rec_b, rec_a)
    )


@contradict_group.command(name="scan")
@click.option(
    "--top-k", type=int, default=5, help="Vec neighbors to consider per memory (default: 5)"
)
@click.option(
    "--sim-floor",
    type=float,
    default=0.55,
    help="Cosine floor; pairs below are skipped (default: 0.55)",
)
@click.option(
    "--confidence", type=float, default=0.7, help="Min LLM confidence to store (default: 0.7)"
)
@click.option(
    "--min-days-apart",
    type=int,
    default=0,
    help="Skip pairs whose updates are within N days (default: 0)",
)
@click.option(
    "--max-memories",
    "max_memories",
    type=int,
    default=2000,
    help="Cap on memories visited (default: 2000)",
)
@click.option(
    "--max-pairs", type=int, default=500, help="Cap on pairs sent to the LLM (default: 500)"
)
@click.option("--since", help="Only scan memories updated on/after this ISO date")
@click.option("--type", "type_", help="Filter by memory type")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def contradict_scan(
    top_k: int,
    sim_floor: float,
    confidence: float,
    min_days_apart: int,
    max_memories: int,
    max_pairs: int,
    since: str | None,
    type_: str | None,
    as_json: bool,
) -> None:
    """Scan the corpus for contradiction/evolution pairs.

    Example: memo contradict scan --since 2026-04-01
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    console.print("[bold]Scanning corpus for contradictions…[/bold]")
    last_idx = {"n": 0}

    def progress(idx: int, total: int, title: str) -> None:
        if idx == total or idx - last_idx["n"] >= 25:
            console.print(f"[dim]  {idx}/{total}  {_short(title, 60)}[/dim]")
            last_idx["n"] = idx

    result = mem.contradict_scanner.scan_corpus(
        top_k=top_k,
        sim_floor=sim_floor,
        confidence_threshold=confidence,
        min_days_apart=min_days_apart,
        max_memories=max_memories,
        max_pairs=max_pairs,
        since=since,
        type_=type_,
        progress=progress,
    )

    payload = {
        "scanned_memories": result.scanned_memories,
        "pairs_examined": result.pairs_examined,
        "pairs_inserted": result.pairs_inserted,
        "pairs_refreshed": result.pairs_refreshed,
        "pairs_skipped_resolved": result.pairs_skipped_resolved,
        "contradictions_found": result.contradictions_found,
        "evolutions_found": result.evolutions_found,
    }
    if as_json:
        click.echo(json.dumps(payload, indent=2))
        return

    table = Table(title="Scan summary")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for k, v in payload.items():
        table.add_row(k.replace("_", " "), str(v))
    console.print(table)

    if result.contradictions_found or result.evolutions_found:
        console.print("\n[green]→[/green] Run [cyan]memo contradict triage[/cyan] to resolve them.")


@contradict_group.command(name="list")
@click.option("--limit", type=int, default=20, help="Max rows (default: 20)")
@click.option("--min-confidence", type=float, default=0.0)
@click.option(
    "--relationship",
    type=click.Choice(["contradiction", "evolution"]),
    help="Filter by relationship type",
)
@click.option(
    "--status",
    type=click.Choice(
        ["open", "fused", "kept_newer", "kept_older", "evolved", "competing", "dismissed"]
    ),
    default="open",
    help="Filter by status (default: open)",
)
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def contradict_list(
    limit: int,
    min_confidence: float,
    relationship: str | None,
    status: str,
    as_json: bool,
) -> None:
    """List contradiction pairs from the sidecar DB."""
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    if status == "open":
        pairs = mem.contradict_store.list_open(
            limit=limit,
            min_confidence=min_confidence,
            relationship=relationship,
        )
    else:
        pairs = mem.contradict_store.list_all(status=status, limit=limit)
        if relationship:
            pairs = [p for p in pairs if p.relationship == relationship]

    if as_json:
        click.echo(json.dumps([p.__dict__ for p in pairs], indent=2, default=str))
        return

    if not pairs:
        console.print(f"[green]No pairs in status '{status}'[/green]")
        return

    table = Table(title=f"Contradiction pairs · status={status}")
    table.add_column("id", justify="right")
    table.add_column("rel")
    table.add_column("conf", justify="right")
    table.add_column("a")
    table.add_column("b")
    table.add_column("rationale")
    for p in pairs:
        table.add_row(
            str(p.pair_id),
            p.relationship,
            f"{p.confidence:.2f}",
            p.memory_id_a[:8],
            p.memory_id_b[:8],
            _short(p.rationale, 70),
        )
    console.print(table)


def _display_pair_excerpt(rec, label: str, *, stale_days: int = 180) -> None:
    from memo.contradict import is_stale

    age_marker = "  [red](stale)[/red]" if is_stale(rec.updated, stale_days) else ""
    console.print(
        f"[bold cyan]{label}[/bold cyan] · {rec.id[:8]} · "
        f"[dim]{rec.type}[/dim] · updated={rec.updated[:10]}{age_marker}"
    )
    console.print(f"  [bold]{escape(rec.title)}[/bold]")
    body = (rec.body or "").strip()
    if len(body) > 600:
        body = body[:599] + "…"
    for line in body.splitlines():
        console.print(f"  {escape(line)}")
    console.print()


_TRIAGE_HELP = """
Actions for each pair:
  f = fuse (LLM-merge both → new memory, archive both)
  n = newer wins (keep newer, delete older)
  o = older wins (keep older, delete newer)
  e = evolved (legitimate evolution, mark resolved, keep both)
  d = dismiss (false positive)
  s = skip (leave as open)
  q = quit walker
""".strip()


@contradict_group.command(name="triage")
@click.option(
    "--limit", type=int, default=20, help="Max pairs to walk in this session (default: 20)"
)
@click.option(
    "--min-confidence",
    type=float,
    default=0.7,
    help="Skip pairs below this LLM confidence (default: 0.7)",
)
@click.option(
    "--relationship",
    type=click.Choice(["contradiction", "evolution"]),
    help="Only walk pairs of this relationship type",
)
@click.option(
    "--stale-days",
    type=int,
    default=180,
    help="Days threshold for the [stale] marker (default: 180)",
)
@click.option(
    "--yes-fuse", is_flag=True, help="Auto-accept fuse without an extra confirmation prompt"
)
def contradict_triage(
    limit: int,
    min_confidence: float,
    relationship: str | None,
    stale_days: int,
    yes_fuse: bool,
) -> None:
    """Interactive triage walker over open contradiction pairs.

    Example: memo contradict triage --relationship contradiction
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    pairs = mem.contradict_store.list_open(
        limit=limit,
        min_confidence=min_confidence,
        relationship=relationship,
    )
    if not pairs:
        console.print("[green]No open pairs to triage.[/green]")
        return

    console.print(f"[bold]Walking {len(pairs)} pair(s).[/bold] Type [cyan]?[/cyan] for help.")

    for pair in pairs:
        rec_a = mem.get(pair.memory_id_a)
        rec_b = mem.get(pair.memory_id_b)
        if rec_a is None or rec_b is None:
            mem.contradict_store.resolve(
                pair.pair_id,
                "dismissed",
                note="auto: one side missing at triage time",
            )
            continue

        # Orient newer as "B" so the walker is always presented with
        # the same temporal layout (older on top, newer below).
        rec_a, rec_b = _older_first(rec_a, rec_b)

        console.print()
        console.print(_fmt_pair_header(rec_a, rec_b, pair))
        if pair.rationale:
            console.print(f"[dim]rationale: {pair.rationale}[/dim]")
        console.print()
        _display_pair_excerpt(rec_a, "OLDER", stale_days=stale_days)
        _display_pair_excerpt(rec_b, "NEWER", stale_days=stale_days)

        while True:
            choice = (
                click.prompt(
                    "Action [f/n/o/e/d/s/q/?]",
                    type=str,
                    default="s",
                    show_default=False,
                )
                .strip()
                .lower()
            )
            if choice == "?":
                console.print(_TRIAGE_HELP)
                continue
            if choice in {"f", "n", "o", "e", "d", "s", "q"}:
                break
            console.print("[red]Unknown action.[/red] Use ? for help.")

        if choice == "q":
            console.print("[dim]Stopping walker.[/dim]")
            break
        if choice == "s":
            continue
        if choice == "d":
            note = click.prompt("note (optional)", default="", show_default=False) or None
            mem.contradict_store.resolve(pair.pair_id, "dismissed", note=note)
            console.print("[dim]dismissed.[/dim]")
            continue
        if choice == "e":
            mem.contradict_store.resolve(pair.pair_id, "evolved")
            console.print("[dim]marked as evolved.[/dim]")
            continue
        if choice == "n":
            if click.confirm(f"Delete OLDER {rec_a.id[:8]}?", default=False):
                mem.contradict_store.resolve(
                    pair.pair_id, "kept_newer", note=f"deleted older {rec_a.id}"
                )
                mem.delete(rec_a.id)
                console.print(f"[green]kept newer.[/green] older {rec_a.id[:8]} deleted.")
            continue
        if choice == "o":
            if click.confirm(f"Delete NEWER {rec_b.id[:8]}?", default=False):
                mem.contradict_store.resolve(
                    pair.pair_id, "kept_older", note=f"deleted newer {rec_b.id}"
                )
                mem.delete(rec_b.id)
                console.print(f"[green]kept older.[/green] newer {rec_b.id[:8]} deleted.")
            continue
        if choice == "f":
            cluster = {
                "cluster_id": pair.pair_id,
                "relationship": "duplicate",
                "rationale": pair.rationale,
                "members": [
                    {
                        "id": rec_a.id,
                        "title": rec_a.title,
                        "updated": rec_a.updated,
                        "body_preview": (rec_a.body or "")[:400],
                    },
                    {
                        "id": rec_b.id,
                        "title": rec_b.title,
                        "updated": rec_b.updated,
                        "body_preview": (rec_b.body or "")[:400],
                    },
                ],
            }
            proposal = mem.consolidator.propose_merge(cluster)
            if proposal is None:
                console.print("[red]LLM declined to propose a merge. Skipping.[/red]")
                continue
            console.print(f"[bold]Proposed merged title:[/bold] {proposal.merged_title}")
            console.print(f"[dim]strategy={proposal.merge_strategy}[/dim]")
            console.print(f"[dim]rationale={proposal.rationale}[/dim]")
            if yes_fuse or click.confirm("Apply this merge?", default=False):
                merge_result = mem.consolidator.apply_merge(proposal, dry_run=False)
                mem.contradict_store.resolve(
                    pair.pair_id,
                    "fused",
                    note=f"merged into {merge_result.merged_id}",
                )
                console.print(
                    f"[green]fused →[/green] {merge_result.merged_id[:8] if merge_result.merged_id else 'n/a'}"
                )
            continue

    stats = mem.contradict_store.stats()
    console.print()
    console.print(f"[bold]Session stats:[/bold] {stats}")


@contradict_group.command(name="stats")
def contradict_stats() -> None:
    """Show counts of pairs grouped by status."""
    cfg = Config.from_env()
    mem = _get_memory(cfg)
    stats = mem.contradict_store.stats()
    if not stats:
        console.print("[dim]No pairs recorded yet. Run `memo contradict scan` first.[/dim]")
        return
    table = Table(title="Contradiction pairs by status")
    table.add_column("status")
    table.add_column("count", justify="right")
    for k, v in sorted(stats.items()):
        table.add_row(k, str(v))
    console.print(table)


@contradict_group.command(name="reopen")
@click.argument("pair_id", type=int)
def contradict_reopen(pair_id: int) -> None:
    """Send a resolved pair back to the open queue."""
    cfg = Config.from_env()
    mem = _get_memory(cfg)
    if mem.contradict_store.reopen(pair_id):
        console.print(f"[green]pair {pair_id} reopened.[/green]")
    else:
        console.print(f"[red]pair {pair_id} not found.[/red]")
