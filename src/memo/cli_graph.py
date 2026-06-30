"""`memo graph` command group — entity-graph navigation.

Extracted from cli.py (3a god-module decomposition). Registered onto the
root group in cli.py via `cli.add_command(graph_group)`.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.table import Table

from memo.cli_common import console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config


@click.group(name="graph")
def graph_group() -> None:
    """Navigate the entity graph with path finding and community detection."""
    pass


@graph_group.command(name="path")
@click.argument("source")
@click.argument("target")
@click.option(
    "--max-length", type=int, default=5, help="Maximum path length to search (default: 5)"
)
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
@click.option(
    "--codegraph/--no-codegraph",
    default=None,
    help="Force codegraph code graph fallback (auto-enabled if memo graph empty)",
)
def graph_path(source: str, target: str, max_length: int, as_json: bool, codegraph: bool | None) -> None:
    """Find shortest path between two entities.

    Falls back to the codegraph code graph if no path in memo memories.

    Example: memo graph path capture session
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    path = mem.navigator.find_shortest_path(source, target, max_length=max_length)

    if as_json:
        click.echo(json.dumps(path.__dict__ if path else None, indent=2))
        return

    if path is None:
        console.print(f"[yellow]No path found between '{source}' and '{target}'[/yellow]")
        return

    console.print(f"[bold]Path from '{source}' to '{target}'[/bold]")
    console.print(f"Length: {path.length}")
    console.print()
    console.print(" → ".join(path.path))
    console.print()
    console.print(f"[dim]Via {len(path.intermediate_memories)} memory(s)[/dim]")


@graph_group.command(name="neighbors")
@click.argument("entity")
@click.option(
    "--max", "max_neighbors", type=int, default=50, help="Maximum neighbors to show (default: 50)"
)
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def graph_neighbors(entity: str, max_neighbors: int, as_json: bool) -> None:
    """Show direct neighbors of an entity.

    Example: memo graph neighbors mlx
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    neighbors = mem.navigator.get_neighbors(entity, max_neighbors=max_neighbors)

    if as_json:
        click.echo(json.dumps(neighbors.__dict__, indent=2))
        return

    console.print(f"[bold]Neighbors of '{entity}'[/bold]")
    console.print(f"Degree: {neighbors.degree}")
    console.print()

    if not neighbors.direct_neighbors:
        console.print("[dim]No neighbors found[/dim]")
        return

    table = Table()
    table.add_column("Neighbor", style="cyan")
    table.add_column("Shared Memories", style="green")

    for neighbor in neighbors.direct_neighbors[:20]:
        mem_count = len(neighbors.neighbor_memories.get(neighbor, []))
        table.add_row(neighbor, str(mem_count))

    console.print(table)
    if len(neighbors.direct_neighbors) > 20:
        console.print(f"[dim]...and {len(neighbors.direct_neighbors) - 20} more[/dim]")


@graph_group.command(name="explore")
@click.argument("entity")
@click.option("--neighbors", "max_neighbors", type=int, default=8, help="Max neighbours.")
@click.option("--memories", "max_memories", type=int, default=8, help="Max mentioning memories.")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON.")
def graph_explore(entity: str, max_neighbors: int, max_memories: int, as_json: bool) -> None:
    """Zoom into one entity: what it connects to + the memories about it.

    Example: memo graph explore vecstore
    """
    from memo.explore import explore_entity

    view = explore_entity(
        _get_memory(Config.from_env()),
        entity,
        max_neighbors=max_neighbors,
        max_memories=max_memories,
    )
    if as_json:
        click.echo(json.dumps(view, indent=2, ensure_ascii=False))
        return
    console.print(f"[bold]Around '{view['entity']}'[/bold]  (degree {view['degree']})")
    if view["neighbors"]:
        table = Table(title="Connects to")
        table.add_column("Neighbor", style="cyan")
        table.add_column("Shared", style="green")
        for n in view["neighbors"]:
            table.add_row(str(n["name"]), str(n["shared"]))
        console.print(table)
    if view["memories"]:
        console.print("[bold]Memories[/bold]")
        for m in view["memories"]:
            console.print(f"  [dim]{m['id'][:8]}[/dim] {m['title']}")
    if not view["neighbors"] and not view["memories"]:
        console.print("[dim]Nothing around this entity.[/dim]")


@graph_group.command(name="communities")
@click.option("--min-size", type=int, default=2, help="Minimum community size (default: 2)")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def graph_communities(min_size: int, as_json: bool) -> None:
    """Detect communities (weighted label propagation) in the entity graph.

    Example: memo graph communities --min-size 3
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    communities = mem.navigator.detect_communities(min_size=min_size)

    if as_json:
        click.echo(json.dumps([c.__dict__ for c in communities], indent=2))
        return

    console.print(f"[bold]Found {len(communities)} communities[/bold]")
    console.print()

    for i, comm in enumerate(communities[:10], 1):
        console.print(f"[cyan]{i}. Community {comm.id}[/cyan] (size: {comm.size})")
        console.print(f"   Representative: {comm.representative_entity}")
        console.print(f"   Entities: {', '.join(comm.entities[:10])}")
        if len(comm.entities) > 10:
            console.print(f"   ...and {len(comm.entities) - 10} more")
        console.print()

    if len(communities) > 10:
        console.print(f"[dim]...and {len(communities) - 10} more communities[/dim]")


@graph_group.command(name="rebuild")
def graph_rebuild() -> None:
    """Canonicalize entities and rebuild the weighted edge table.

    Example: memo graph rebuild
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)
    merged = mem.graph.canonicalize_existing()
    edges = mem.graph.rebuild_edges()
    console.print(
        f"[green]graph rebuilt[/green]: merged {merged} duplicate entities, {edges} edges"
    )


@graph_group.command(name="centrality")
@click.option("--top", type=int, default=20, help="Top N entities by centrality (default: 20)")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def graph_centrality(top: int, as_json: bool) -> None:
    """Compute centrality metrics for all entities.

    Example: memo graph centrality --top 30
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    scores = mem.navigator.compute_centrality()

    if as_json:
        click.echo(
            json.dumps(
                {
                    "degree": scores.degree,
                    "betweenness": scores.betweenness,
                },
                indent=2,
            )
        )
        return

    console.print("[bold]Top entities by degree centrality[/bold]")
    console.print()

    table = Table()
    table.add_column("Entity", style="cyan")
    table.add_column("Degree", style="green")
    table.add_column("Betweenness", style="yellow")

    sorted_by_degree = sorted(scores.degree.items(), key=lambda x: x[1], reverse=True)[:top]
    for entity, degree in sorted_by_degree:
        betweenness = scores.betweenness.get(entity, 0.0)
        table.add_row(entity, str(degree), f"{betweenness:.3f}")

    console.print(table)


@graph_group.command(name="export")
@click.option(
    "--format",
    "format_type",
    type=click.Choice(["dot", "json"]),
    default="dot",
    help="Output format (default: dot)",
)
@click.option("--output", "-o", "output_path", help="Output file path (default: stdout)")
def graph_export(format_type: str, output_path: str | None) -> None:
    """Export the entity graph for visualization.

    Example: memo graph export --format json -o graph.json
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    if format_type == "dot":
        content = mem.navigator.export_graphviz(output_path=output_path)
        if not output_path:
            click.echo(content)
    else:  # json
        data = mem.navigator.export_json(include_memories=True)
        json_str = json.dumps(data, indent=2)
        if output_path:
            Path(output_path).write_text(json_str, encoding="utf-8")
        else:
            click.echo(json_str)

    if output_path:
        console.print(f"[green]Exported to {output_path}[/green]")
