"""`memo export` command group — export to JSON/CSV/markdown bundle.

Extracted from cli.py (3a god-module decomposition). Registered onto the
root group in cli.py via `cli.add_command(export_group)`.
"""

from __future__ import annotations

from pathlib import Path

import click

from memo.cli_common import console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config


def _export_target(output_path: str) -> Path:
    """Check the user-supplied destination before any export work starts.

    Left to the writer, a missing parent surfaces as a FileNotFoundError
    traceback from the atomic-write tempfile, and an existing directory only
    fails at the final rename with IsADirectoryError, after the whole corpus has
    been serialized. Refuse rather than create the tree: the usual cause is a
    typo in a hand-typed path.
    """
    out_p = Path(output_path)
    if out_p.is_dir():
        raise click.ClickException(f"output path is a directory, not a file: {out_p}")
    if not out_p.parent.is_dir():
        raise click.ClickException(
            f"output directory does not exist: {out_p.parent} "
            f"(create it with: mkdir -p {out_p.parent})"
        )
    return out_p


@click.group(name="export")
def export_group() -> None:
    """Export memories to other formats."""
    pass


@export_group.command(name="json")
@click.argument("output_path", type=click.Path())
def export_json(output_path: str) -> None:
    """Export to JSON file.

    Example: memo export json /path/to/export.json
    """
    out_p = _export_target(output_path)
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    result = mem.import_export.export_to(out_p, "json")

    console.print("[green]Export complete[/green]")
    console.print(f"Exported: {result.exported_count}")
    console.print(f"Output: {result.output_path}")


@export_group.command(name="passport")
@click.argument("output_path", type=click.Path())
def export_passport(output_path: str) -> None:
    """Export a versioned, vendor-neutral passport (memo.passport.v1).

    Higher fidelity than `json`: a stable schema header + the provenance /
    verification `extra` bag, so another memo (or tool) can validate and
    re-import with the canonical record intact.

    Example: memo export passport /path/to/brain.passport
    """
    out_p = _export_target(output_path)
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    result = mem.import_export.export_to(out_p, "passport")

    console.print("[green]Passport exported[/green]")
    console.print(f"Exported: {result.exported_count}")
    console.print(f"Output: {result.output_path}")


@export_group.command(name="csv")
@click.argument("output_path", type=click.Path())
def export_csv(output_path: str) -> None:
    """Export to CSV file.

    Example: memo export csv /path/to/export.csv
    """
    out_p = _export_target(output_path)
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    result = mem.import_export.export_to(out_p, "csv")

    console.print("[green]Export complete[/green]")
    console.print(f"Exported: {result.exported_count}")
    console.print(f"Output: {result.output_path}")


@export_group.command(name="markdown-bundle")
@click.argument("output_path", type=click.Path())
def export_markdown_bundle(output_path: str) -> None:
    """Export to Markdown bundle (zip).

    Example: memo export markdown-bundle /path/to/export.zip
    """
    out_p = _export_target(output_path)
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    result = mem.import_export.export_to(out_p, "markdown_bundle")

    console.print("[green]Export complete[/green]")
    console.print(f"Exported: {result.exported_count}")
    console.print(f"Output: {result.output_path}")
