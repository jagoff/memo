"""`memo import` command group — import from JSON/CSV/markdown bundle.

Extracted from cli.py (3a god-module decomposition). Registered onto the
root group in cli.py via `cli.add_command(import_group)`.
"""

from __future__ import annotations

import click

from memo.cli_common import console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config

# -- import/export commands ------------------------------------------------------


@click.group(name="import")
def import_group() -> None:
    """Import memorias from other formats."""
    pass


@import_group.command(name="json")
@click.argument("input_path", type=click.Path())
@click.option("--format", help="Format (json, csv, markdown_bundle)")
def import_json(input_path: str, format: str | None) -> None:
    """Import from JSON file.

    Example: memo import json /path/to/export.json
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    from pathlib import Path
    result = mem.import_export.import_from(Path(input_path), format or "json")

    console.print("[green]Import complete[/green]")
    console.print(f"Imported: {result.imported_count}")
    console.print(f"Skipped: {result.skipped_count}")

    if result.errors:
        console.print(f"[yellow]Errors: {len(result.errors)}[/yellow]")


@import_group.command(name="csv")
@click.argument("input_path", type=click.Path())
def import_csv(input_path: str) -> None:
    """Import from CSV file.

    Example: memo import csv /path/to/export.csv
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    from pathlib import Path
    result = mem.import_export.import_from(Path(input_path), "csv")

    console.print("[green]Import complete[/green]")
    console.print(f"Imported: {result.imported_count}")
    console.print(f"Skipped: {result.skipped_count}")

    if result.errors:
        console.print(f"[yellow]Errors: {len(result.errors)}[/yellow]")


@import_group.command(name="markdown-bundle")
@click.argument("input_path", type=click.Path())
def import_markdown_bundle(input_path: str) -> None:
    """Import from Markdown bundle (zip).

    Example: memo import markdown-bundle /path/to/export.zip
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    from pathlib import Path
    result = mem.import_export.import_from(Path(input_path), "markdown_bundle")

    console.print("[green]Import complete[/green]")
    console.print(f"Imported: {result.imported_count}")
    console.print(f"Skipped: {result.skipped_count}")

    if result.errors:
        console.print(f"[yellow]Errors: {len(result.errors)}[/yellow]")
