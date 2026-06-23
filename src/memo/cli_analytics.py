"""`memo analytics` command group — corpus metrics + dashboards.

Extracted from cli.py (3a god-module decomposition). Registered onto the
root group in cli.py via `cli.add_command(analytics_group)`.
"""

from __future__ import annotations

import json

import click
from rich.table import Table

from memo.cli_common import console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config

# -- analytics commands ----------------------------------------------------------


@click.group(name="analytics")
def analytics_group() -> None:
    """Memory analytics dashboard — metrics and visualizations."""
    pass


@analytics_group.command(name="summary")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def analytics_summary(as_json: bool) -> None:
    """Show analytics summary.

    Example: memo analytics summary
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    metrics = mem.analytics.compute_corpus_metrics()

    if as_json:
        click.echo(json.dumps(metrics.__dict__, indent=2))
        return

    console.print("[bold]Memory Analytics Summary[/bold]")
    console.print()
    console.print(f"Total Memories: {metrics.total_memorias}")
    console.print(f"Total Entities: {metrics.total_entities}")
    console.print(f"Growth Rate: {metrics.growth_rate:.2f} memories/day")
    console.print(f"Average Access Count: {metrics.average_access_count:.2f}")
    console.print()
    console.print("[bold]Type Distribution[/bold]")
    for t, c in metrics.type_distribution.items():
        console.print(f"  {t}: {c}")
    console.print()
    console.print("[bold]Top 10 Tags[/bold]")
    for i, (t, c) in enumerate(list(metrics.tag_frequency.items())[:10], 1):
        console.print(f"  {i}. {t}: {c}")


@analytics_group.command(name="growth")
@click.option("--days", type=int, default=30, help="Days to analyze")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def analytics_growth(days: int, as_json: bool) -> None:
    """Show growth data over time.

    Example: memo analytics growth --days 30
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    growth = mem.analytics.compute_growth_data(days=days)

    if as_json:
        click.echo(json.dumps(growth.__dict__, indent=2))
        return

    console.print(f"[bold]Growth (Last {days} Days)[/bold]")
    console.print()

    table = Table()
    table.add_column("Date", style="cyan")
    table.add_column("Count", style="green")

    for d, c in zip(growth.dates, growth.counts, strict=False):
        table.add_row(d, str(c))

    console.print(table)


@analytics_group.command(name="export-json")
@click.argument("output_path", type=click.Path())
def analytics_export_json(output_path: str) -> None:
    """Export analytics to JSON.

    Example: memo analytics export-json /path/to/analytics.json
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    from pathlib import Path

    mem.analytics.export_metrics_json(Path(output_path))

    console.print(f"[green]Exported analytics to {output_path}[/green]")


@analytics_group.command(name="export-csv")
@click.argument("output_path", type=click.Path())
def analytics_export_csv(output_path: str) -> None:
    """Export analytics to CSV.

    Example: memo analytics export-csv /path/to/analytics.csv
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    from pathlib import Path

    mem.analytics.export_metrics_csv(Path(output_path))

    console.print(f"[green]Exported analytics to {output_path}[/green]")


@analytics_group.command(name="dashboard-html")
@click.argument("output_path", type=click.Path())
def analytics_dashboard_html(output_path: str) -> None:
    """Generate HTML dashboard.

    Example: memo analytics dashboard-html /path/to/dashboard.html
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    from pathlib import Path

    mem.dashboard.generate_html_dashboard(Path(output_path))

    console.print(f"[green]Generated HTML dashboard at {output_path}[/green]")
