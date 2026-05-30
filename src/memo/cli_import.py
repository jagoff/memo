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


@import_group.command(name="whatsapp")
@click.option("--include-chat", "include_chats", multiple=True,
              help="Chat JID to ingest (repeatable). Opt-in allowlist.")
@click.option("--exclude-chat", "exclude_chats", multiple=True,
              help="Chat JID to skip (repeatable).")
@click.option("--all-chats", is_flag=True,
              help="Ingest every chat (minus exclusions). Required if no --include-chat.")
@click.option("--retention-days", type=int, default=180, show_default=True,
              help="Only include messages newer than N days.")
@click.option("--since", help="Floor date YYYY-MM-DD (in addition to retention).")
@click.option("--notes-dir", "notes_dir", type=click.Path(), default=None,
              help="Override output folder (default: <vault>/Obsidian/Whatsapp).")
@click.option("--dry-run", is_flag=True, help="Report counts, write nothing.")
@click.option("--index/--no-index", default=True, show_default=True,
              help="Run `memo ingest` on the notes folder so the chat can find them.")
@click.option("--db", "db_path", type=click.Path(),
              help="Override bridge messages.db path.")
@click.option("--json", "as_json", is_flag=True, help="Emit the summary as JSON.")
def import_whatsapp(
    include_chats: tuple[str, ...],
    exclude_chats: tuple[str, ...],
    all_chats: bool,
    retention_days: int,
    since: str | None,
    notes_dir: str | None,
    dry_run: bool,
    index: bool,
    db_path: str | None,
    as_json: bool,
) -> None:
    """Ingest WhatsApp conversations as readable per-contact notes.

    Writes one Markdown note per chat to `<vault>/Obsidian/Whatsapp/<contacto>.md`
    (transcript grouped by date) and — unless --no-index — runs `memo ingest` on
    that folder so the notes are searchable by `memo search`/`ask` and the
    synapse :8765 chat (source=vault-ingest). Notes are regenerated in full each
    run (idempotent). Scope is opt-in.

    Examples:

      memo import whatsapp --include-chat 549XXX@s.whatsapp.net --dry-run
      memo import whatsapp --include-chat 549XXX@s.whatsapp.net
      memo import whatsapp --all-chats --retention-days 90
    """
    import json as _json
    import subprocess
    from pathlib import Path

    from memo import whatsapp_ingest

    cfg = Config.from_env()
    mem = _get_memory(cfg)

    try:
        summary = whatsapp_ingest.run(
            mem,
            bridge_db=Path(db_path) if db_path else None,
            since=since,
            retention_days=retention_days,
            include_chats=include_chats,
            exclude_chats=exclude_chats,
            all_chats=all_chats,
            notes_dir=Path(notes_dir) if notes_dir else None,
            dry_run=dry_run,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc

    # Index the notes folder so the :8765 chat can retrieve them.
    summary["indexed"] = False
    if index and not dry_run and summary["notes_written"]:
        try:
            subprocess.run(
                ["memo", "ingest", summary["notes_dir"], "--name", "whatsapp"],
                check=True,
            )
            summary["indexed"] = True
        except Exception as exc:  # noqa: BLE001
            summary["index_error"] = str(exc)

    if as_json:
        console.print_json(_json.dumps(summary))
        return

    console.print("[green]WhatsApp ingest complete[/green]")
    console.print(f"Messages read:   {summary['messages_read']}")
    console.print(f"Chats:           {len(summary['chats'])}")
    if summary["dry_run"]:
        console.print("[yellow]dry-run — nothing written[/yellow]")
    else:
        console.print(f"Notes written:   {summary['notes_written']}  → {summary['notes_dir']}")
        console.print(f"Indexed:         {summary['indexed']}")
    if summary.get("index_error"):
        console.print(f"[yellow]index error: {summary['index_error']}[/yellow]")


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
