"""`memo temporal` command group — temporal contradictions / timeline / staleness.

Extracted from cli.py (3a god-module decomposition). Registered onto the
root group in cli.py via `cli.add_command(temporal_group)`.
"""

from __future__ import annotations

import json

import click
from rich.table import Table

from memo.cli_common import console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config

# -- temporal reasoning commands ----------------------------------------------


@click.group(name="temporal")
def temporal_group() -> None:
    """Analyze temporal patterns and contradictions in memories."""
    pass


@temporal_group.group(name="facts")
def temporal_facts_group() -> None:
    """Inspect and maintain temporal fact edges."""
    pass


@temporal_facts_group.command(name="add")
@click.argument("subject")
@click.argument("predicate")
@click.argument("object_")
@click.option("--source-id", help="Source memory id for provenance.")
@click.option("--valid-at", help="ISO timestamp when this fact became valid.")
@click.option("--invalid-at", help="ISO timestamp when this fact became invalid.")
@click.option("--expired-at", help="ISO timestamp when this fact expires.")
@click.option("--confidence", type=float, default=1.0, help="Confidence score, default 1.0.")
@click.option("--supersedes", multiple=True, help="Existing fact id invalidated by this fact.")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON.")
def temporal_facts_add(
    subject: str,
    predicate: str,
    object_: str,
    source_id: str | None,
    valid_at: str | None,
    invalid_at: str | None,
    expired_at: str | None,
    confidence: float,
    supersedes: tuple[str, ...],
    as_json: bool,
) -> None:
    """Add one temporal fact edge.

    Example: memo temporal facts add memo backend sqlite --valid-at 2026-01-01
    """
    mem = _get_memory(Config.from_env())
    fact_id = mem.fact_edges.upsert_fact(
        subject=subject,
        predicate=predicate,
        object=object_,
        source_record_id=source_id,
        valid_at=valid_at,
        invalid_at=invalid_at,
        expired_at=expired_at,
        confidence=confidence,
        provenance={"surface": "cli"},
        supersedes=list(supersedes),
    )
    row = mem.fact_edges.get(fact_id)
    if as_json:
        click.echo(json.dumps(row, indent=2, ensure_ascii=False))
        return
    console.print(f"[green]fact saved[/green] [dim]{fact_id[:8]}[/dim]")


@temporal_facts_group.command(name="list")
@click.option("--subject", help="Filter by subject.")
@click.option("--predicate", help="Filter by predicate.")
@click.option("--object", "object_", help="Filter by object.")
@click.option("--source-id", help="Filter by source memory id.")
@click.option("--as-of", help="ISO timestamp for live fact filtering.")
@click.option("--include-inactive", is_flag=True, help="Include invalid/future/expired facts.")
@click.option("--limit", type=int, default=50, help="Maximum facts to show.")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON.")
def temporal_facts_list(
    subject: str | None,
    predicate: str | None,
    object_: str | None,
    source_id: str | None,
    as_of: str | None,
    include_inactive: bool,
    limit: int,
    as_json: bool,
) -> None:
    """List temporal fact edges live at --as-of."""
    mem = _get_memory(Config.from_env())
    rows = mem.fact_edges.query(
        subject=subject,
        predicate=predicate,
        object=object_,
        source_record_id=source_id,
        as_of=as_of,
        include_inactive=include_inactive,
        limit=limit,
    )
    if as_json:
        click.echo(json.dumps(rows, indent=2, ensure_ascii=False))
        return
    if not rows:
        console.print("[dim]no temporal facts[/dim]")
        return
    table = Table(title="Temporal facts")
    table.add_column("id", width=10)
    table.add_column("subject")
    table.add_column("predicate")
    table.add_column("object")
    table.add_column("valid")
    table.add_column("invalid")
    table.add_column("conf", justify="right")
    for row in rows:
        table.add_row(
            str(row["id"])[:8],
            str(row["subject"]),
            str(row["predicate"]),
            str(row["object"]),
            str(row["valid_at"])[:10],
            str(row.get("invalid_at") or "—")[:10],
            f"{float(row['confidence']):.2f}",
        )
    console.print(table)


@temporal_facts_group.command(name="invalidate")
@click.argument("fact_id")
@click.option("--invalid-at", help="ISO timestamp when this fact became invalid.")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON.")
def temporal_facts_invalidate(fact_id: str, invalid_at: str | None, as_json: bool) -> None:
    """Invalidate one temporal fact edge."""
    mem = _get_memory(Config.from_env())
    ok = mem.fact_edges.invalidate(fact_id, invalid_at=invalid_at)
    payload = {"id": fact_id, "invalidated": ok}
    if as_json:
        click.echo(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    if ok:
        console.print(f"[green]invalidated[/green] [dim]{fact_id[:8]}[/dim]")
    else:
        console.print(f"[yellow]not found[/yellow] [dim]{fact_id[:8]}[/dim]")


@temporal_group.command(name="contradictions")
@click.argument("entity")
@click.option("--type", "entity_type", help="Filter by entity type from graph")
@click.option(
    "--confidence",
    "min_confidence",
    type=float,
    default=0.7,
    help="Minimum confidence threshold (default: 0.7)",
)
@click.option(
    "--max-pairs", type=int, default=20, help="Maximum number of pairs to analyze (default: 20)"
)
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def temporal_contradictions(
    entity: str,
    entity_type: str | None,
    min_confidence: float,
    max_pairs: int,
    as_json: bool,
) -> None:
    """Detect contradictions among memories mentioning a specific entity.

    Example: memo temporal contradictions mlx
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    contradictions = mem.temporal.detect_entity_contradictions(
        entity_name=entity,
        entity_type=entity_type,
        confidence_threshold=min_confidence,
        max_pairs=max_pairs,
    )

    if as_json:
        click.echo(json.dumps([c.__dict__ for c in contradictions], indent=2))
        return

    if not contradictions:
        console.print(f"[green]No contradictions found for entity '{entity}'[/green]")
        return

    table = Table(title=f"Contradictions for '{entity}'")
    table.add_column("ID A", style="cyan")
    table.add_column("ID B", style="cyan")
    table.add_column("Title A", style="yellow")
    table.add_column("Title B", style="yellow")
    table.add_column("Relationship", style="magenta")
    table.add_column("Confidence", style="green")
    table.add_column("Rationale")

    for c in contradictions:
        table.add_row(
            c.memory_id_a[:8],
            c.memory_id_b[:8],
            c.title_a[:40],
            c.title_b[:40],
            c.relationship,
            f"{c.confidence:.2f}",
            c.rationale,
        )

    console.print(table)


@temporal_group.command(name="timeline")
@click.argument("entity")
@click.option("--type", "entity_type", help="Filter by entity type from graph")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def temporal_timeline(entity: str, entity_type: str | None, as_json: bool) -> None:
    """Build a chronological timeline of all memories mentioning an entity.

    Example: memo temporal timeline mlx
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    timeline = mem.temporal.build_entity_timeline(
        entity_name=entity,
        entity_type=entity_type,
    )

    if timeline is None:
        console.print(f"[yellow]No memories found for entity '{entity}'[/yellow]")
        return

    if as_json:
        click.echo(
            json.dumps(
                {
                    "entity_name": timeline.entity_name,
                    "entity_type": timeline.entity_type,
                    "first_seen": timeline.first_seen,
                    "last_seen": timeline.last_seen,
                    "events": [e.__dict__ for e in timeline.events],
                },
                indent=2,
            )
        )
        return

    console.print(f"[bold]Timeline for '{entity}' ({timeline.entity_type})[/bold]")
    console.print(f"First seen: {timeline.first_seen}")
    console.print(f"Last seen: {timeline.last_seen}")
    console.print()

    for event in timeline.events:
        console.print(f"[cyan]{event.date}[/cyan] [dim][{event.memory_id[:8]}][/dim]")
        console.print(f"  [yellow]{event.title}[/yellow] ({event.type})")
        console.print(f"  {event.snippet}")
        console.print()


@temporal_group.command(name="stale")
@click.option(
    "--days", type=int, default=180, help="Days since last update to consider stale (default: 180)"
)
@click.option(
    "--min-access",
    "min_access_count",
    type=int,
    default=1,
    help="Minimum access count to exclude (default: 1)",
)
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def temporal_stale(days: int, min_access_count: int, as_json: bool) -> None:
    """Find memories that may be stale based on age and lack of access.

    Example: memo temporal stale --days 90
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    stale = mem.temporal.detect_stale_memories(
        days_threshold=days,
        min_access_count=min_access_count,
    )

    if as_json:
        click.echo(json.dumps(stale, indent=2))
        return

    if not stale:
        console.print(f"[green]No stale memories found (threshold: {days} days)[/green]")
        return

    table = Table(title=f"Potentially Stale Memories (>{days} days)")
    table.add_column("ID", style="cyan")
    table.add_column("Title", style="yellow")
    table.add_column("Type", style="magenta")
    table.add_column("Updated", style="dim")
    table.add_column("Days Old", style="red")
    table.add_column("Access Count", style="green")

    for item in stale[:50]:  # Cap display
        table.add_row(
            item["id"][:8],
            item["title"][:40],
            item["type"],
            item["updated"][:10],
            str(item["days_since_update"]),
            str(item["access_count"]),
        )

    console.print(table)
    if len(stale) > 50:
        console.print(f"[dim]...and {len(stale) - 50} more[/dim]")


@temporal_group.command(name="patterns")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def temporal_patterns(as_json: bool) -> None:
    """Analyze high-level temporal patterns across the entire corpus.

    Example: memo temporal patterns
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    patterns = mem.temporal.detect_temporal_patterns()

    if as_json:
        click.echo(json.dumps(patterns, indent=2))
        return

    console.print("[bold]Temporal Patterns[/bold]")
    console.print()

    # Memories per month
    console.print("[yellow]Memories per month:[/yellow]")
    for month, count in list(patterns["memories_per_month"].items())[-12:]:
        console.print(f"  {month}: {count}")
    console.print()

    # Most active entities
    console.print("[yellow]Most active entities:[/yellow]")
    for entity, count in list(patterns["most_active_entities"].items())[:10]:
        console.print(f"  {entity}: {count} mentions")
