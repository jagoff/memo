"""`memo collaborative` command group — shared knowledge graph.

Extracted from cli.py (3a god-module decomposition). Registered onto the
root group in cli.py via `cli.add_command(collaborative_group)`.
"""

from __future__ import annotations

import click
from rich.markup import escape

from memo.cli_common import console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config

# -- collaborative commands (gamechanger #18) -----------------------------------


@click.group(name="collaborative")
def collaborative_group() -> None:
    """Collaborative Social Memory with a Shared Knowledge Graph."""
    pass


@collaborative_group.command(name="share-connection")
@click.argument("user-id")
@click.argument("entity-a")
@click.argument("entity-b")
@click.argument("relationship")
@click.option("--confidence", type=float, default=0.7, help="Confidence score")
def collaborative_share_connection(
    user_id: str, entity_a: str, entity_b: str, relationship: str, confidence: float
) -> None:
    """Share a discovered connection with the community.

    Example: memo collaborative share-connection user123 MLX Apple "optimized for"
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    conn = mem.collaborative.share_connection(
        user_id=user_id,
        entity_a=entity_a,
        entity_b=entity_b,
        relationship=relationship,
        confidence=confidence,
    )

    console.print("[green]Connection shared[/green]")
    console.print(f"Connection ID: {conn.connection_id}")
    console.print(f"From: {conn.from_user}")
    console.print(f"{entity_a} --{relationship}--> {entity_b}")


@collaborative_group.command(name="connections")
@click.argument("entity")
def collaborative_connections(entity: str) -> None:
    """View shared connections for an entity.

    Example: memo collaborative connections MLX
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    connections = mem.collaborative.get_shared_connections(entity)

    console.print(f"[bold]Shared connections for {entity}[/bold]")
    for c in connections:
        console.print(f"  {c.entity_a} --{c.relationship}--> {c.entity_b}")
        console.print(f"    From: {c.from_user}, votes: {c.votes}, confidence: {c.confidence:.2f}")


@collaborative_group.command(name="recommend")
@click.argument("entity")
@click.option("--limit", type=int, default=10, help="Maximum number of results")
def collaborative_recommend(entity: str, limit: int) -> None:
    """Get recommended connections based on collective patterns.

    Example: memo collaborative recommend MLX
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    recommendations = mem.collaborative.get_recommended_connections(entity, limit=limit)

    console.print(f"[bold]Recommended connections for {entity}[/bold]")
    for r in recommendations:
        console.print(f"  {escape(r.entity_a)} --{escape(r.relationship)}--> {escape(r.entity_b)}")
        console.print(
            f"    From: {escape(r.from_user)}, votes: {r.votes}, confidence: {r.confidence:.2f}"
        )


@collaborative_group.command(name="share-insight")
@click.argument("user-id")
@click.argument("content")
def collaborative_share_insight(user_id: str, content: str) -> None:
    """Share an insight with the community.

    Example: memo collaborative share-insight user123 "MLX is ideal for edge because..."
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    insight = mem.collaborative.share_insight(user_id, content)

    console.print("[green]Insight shared[/green]")
    console.print(f"Insight ID: {insight.insight_id}")
    console.print(f"Content: {content[:100]}...")


@collaborative_group.command(name="insights")
@click.option("--limit", type=int, default=10, help="Maximum number of results")
def collaborative_insights(limit: int) -> None:
    """View the community's most-voted insights.

    Example: memo collaborative insights
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    insights = mem.collaborative.get_top_insights(limit=limit)

    console.print("[bold]Top insights[/bold]")
    for i, insight in enumerate(insights, 1):
        console.print(f"[cyan]{i}.[/cyan] {insight.content[:100]}...")
        console.print(f"    Upvotes: {insight.upvotes}, Downvotes: {insight.downvotes}")
