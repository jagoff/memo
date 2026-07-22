"""`memo graph` command group — entity-graph navigation.

Extracted from cli.py (3a god-module decomposition). Registered onto the
root group in cli.py via `cli.add_command(graph_group)`.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import click
from rich.table import Table

from memo.cli_common import console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config
from memo.flags import flag_float


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
def graph_path(
    source: str, target: str, max_length: int, as_json: bool, codegraph: bool | None
) -> None:
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


@graph_group.command(name="why")
@click.argument("source")
@click.argument("target")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
@click.option(
    "--codegraph/--no-codegraph",
    default=None,
    help="Force codegraph code graph fallback (auto-enabled if memo graph empty)",
)
def graph_why(source: str, target: str, as_json: bool, codegraph: bool | None) -> None:
    """Explain why two entities are connected.

    Example: memo graph why mlx daemon
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)
    result = mem.navigator.why_connected(source, target, use_codegraph=codegraph)

    if as_json:
        click.echo(json.dumps(result, indent=2))
        return

    if not result["path"]:
        console.print(f"[yellow]No path found between '{source}' and '{target}'[/yellow]")
        return

    console.print(f"[bold]Why '{source}' connects to '{target}'[/bold]")
    console.print(" → ".join(result["path"]))
    table = Table()
    table.add_column("From", style="cyan")
    table.add_column("To", style="cyan")
    table.add_column("Weight", style="green")
    table.add_column("Evidence", style="yellow")
    for edge in result["edges"]:
        table.add_row(
            str(edge["from"]),
            str(edge["to"]),
            str(edge["weight"]),
            str(edge.get("memory_id") or ""),
        )
    console.print(table)


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


@graph_group.command(name="stats")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON.")
def graph_stats(as_json: bool) -> None:
    """Entity-graph health: entity / link / edge counts + weight distribution.

    Example: memo graph stats
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)
    health = mem.graph_health()
    if as_json:
        click.echo(json.dumps(health, indent=2, ensure_ascii=False))
        return
    s = health["raw"]
    es = health["edges"]
    projection = health["projection"]
    console.print(
        f"[bold]entities[/bold] {s['entities']}  "
        f"[bold]links[/bold] {s['links']}  "
        f"[bold]edges[/bold] {es['edges']}"
    )
    pct = (es["edges_gt1"] / es["edges"] * 100) if es["edges"] else 0.0
    console.print(
        f"edge weight: min {es['weight_min']:.0f} / mean {es['weight_mean']:.2f} / "
        f"max {es['weight_max']:.0f}  ([green]{es['edges_gt1']}[/green] = {pct:.1f}% > 1)"
    )
    active = projection["active_version"]
    if active:
        console.print(
            f"projection [cyan]{str(active)[:8]}[/cyan]: "
            f"{projection['node_count']} nodes / {projection['edge_count']} edges / "
            f"{projection['rejected_count']} rejected"
        )
    else:
        console.print("projection: [yellow]missing[/yellow]")


@graph_group.command(name="rebuild")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON.")
def graph_rebuild(as_json: bool) -> None:
    """Canonicalize entities and rebuild the weighted edge table.

    Example: memo graph rebuild
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)
    result = mem.rebuild_graph()
    if as_json:
        click.echo(json.dumps(asdict(result), indent=2, ensure_ascii=False))
        return
    console.print(
        "[green]graph rebuilt[/green]: "
        f"pruned {result.orphan_links_pruned} orphan links, "
        f"merged {result.entities_merged} duplicate entities, "
        f"{result.raw_edges} raw edges, "
        f"{result.projection.node_count} projected nodes / "
        f"{result.projection.edge_count} projected edges"
    )


@graph_group.command(name="hubs")
@click.option("--limit", type=int, default=20, show_default=True, help="Maximum hubs to show.")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON.")
def graph_hubs(limit: int, as_json: bool) -> None:
    """Show entities broad enough to behave like graph hubs.

    Example: memo graph hubs --limit 30
    """

    cfg = Config.from_env()
    mem = _get_memory(cfg)
    hubs = mem.graph.entity_hubs(limit=limit)
    threshold = flag_float("MEMO_GRAPH_HUB_MAX_DOC_FREQ_RATIO") or 0.25
    for hub in hubs:
        hub["is_hub"] = bool(hub["doc_freq_ratio"] > threshold)
    if as_json:
        click.echo(json.dumps(hubs, indent=2, ensure_ascii=False))
        return
    if not hubs:
        console.print("[dim]No graph hubs found.[/dim]")
        return
    table = Table(title="Graph hubs")
    table.add_column("Entity", style="cyan")
    table.add_column("Type")
    table.add_column("Docs", justify="right")
    table.add_column("Ratio", justify="right")
    table.add_column("Degree", justify="right")
    table.add_column("Hub", justify="center")
    for hub in hubs:
        table.add_row(
            str(hub["name"]),
            str(hub["type"]),
            str(hub["doc_freq"]),
            f"{float(hub['doc_freq_ratio']):.2f}",
            str(hub["degree"]),
            "yes" if hub["is_hub"] else "",
        )
    console.print(table)


@graph_group.group(name="relations")
def graph_relations() -> None:
    """Manage rebuildable semantic relations."""
    pass


@graph_relations.command(name="rebuild")
@click.option("--limit", type=int, default=200, show_default=True, help="Recent memories to scan.")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON.")
def graph_relations_rebuild(limit: int, as_json: bool) -> None:
    """Backfill deterministic semantic relations between recent memories.

    Example: memo graph relations rebuild --limit 500
    """

    from memo.semantic_relations import (
        DETERMINISTIC_DERIVED_FROM,
        extract_relations_batch,
        store_relations,
    )

    cfg = Config.from_env()
    mem = _get_memory(cfg)
    memories = mem.list(limit=limit, include_forgotten=False)
    pairs = [
        (source, target) for source in memories for target in memories if source.id != target.id
    ]
    relations = extract_relations_batch(pairs)
    deleted = mem.graph.delete_semantic_relations_by_derived_from(DETERMINISTIC_DERIVED_FROM)
    written = store_relations(mem.graph, relations)
    result = {
        "scanned_memories": len(memories),
        "processed_pairs": len(pairs),
        "deleted": deleted,
        "written": written,
        "derived_from": DETERMINISTIC_DERIVED_FROM,
    }
    if as_json:
        click.echo(json.dumps(result, indent=2, ensure_ascii=False))
        return
    console.print(
        "[green]semantic relations rebuilt[/green]: "
        f"{written} relation(s) from {len(pairs)} pair(s) "
        f"({deleted} previous deterministic row(s) removed)"
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
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json_str, encoding="utf-8")
        else:
            click.echo(json_str)

    if output_path:
        console.print(f"[green]Exported to {output_path}[/green]")
