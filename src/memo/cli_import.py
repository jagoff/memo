"""`memo import` command group — import from JSON/CSV/markdown bundle.

Extracted from cli.py (3a god-module decomposition). Registered onto the
root group in cli.py via `cli.add_command(import_group)`.
"""

from __future__ import annotations

import click

from memo.cli_common import console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config
from memo.import_export import ImportResult

# -- import/export commands ------------------------------------------------------

_ERROR_SAMPLE_MAX = 5


def _report_import_result(result: ImportResult) -> None:
    """Print the import summary; exit non-zero when any record errored.

    Shows a short sample of the collected per-record error messages so a
    failed migration is diagnosable (previously only `Errors: <count>` was
    printed and the command still exited 0, silently green for scripts).
    """
    console.print("[green]Import complete[/green]")
    console.print(f"Imported: {result.imported_count}")
    console.print(f"Skipped: {result.skipped_count}")

    if result.errors:
        console.print(f"[yellow]Errors: {len(result.errors)}[/yellow]")
        for err in result.errors[:_ERROR_SAMPLE_MAX]:
            console.print(f"[yellow]  - {err}[/yellow]")
        remaining = len(result.errors) - _ERROR_SAMPLE_MAX
        if remaining > 0:
            console.print(f"[yellow]  … and {remaining} more[/yellow]")
        raise SystemExit(1)


@click.group(name="import")
def import_group() -> None:
    """Import memories from other formats."""
    pass


@import_group.command(name="json")
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--format", help="Format (json, csv, markdown_bundle)")
def import_json(input_path: str, format: str | None) -> None:
    """Import from JSON file.

    Example: memo import json /path/to/export.json
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    from pathlib import Path

    result = mem.import_export.import_from(Path(input_path), format or "json")
    _report_import_result(result)


@import_group.command(name="passport")
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False))
def import_passport(input_path: str) -> None:
    """Import a memo.passport.v1 file (validated, high-fidelity).

    Preserves content/title/type/tags/created and the provenance /
    verification `extra` bag; ids and derived indexes are rebuilt by this
    store. Rejects a malformed / wrong-schema passport before writing.

    Example: memo import passport /path/to/brain.passport
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    from pathlib import Path

    result = mem.import_export.import_from(Path(input_path), "passport")
    _report_import_result(result)


@import_group.command(name="csv")
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False))
def import_csv(input_path: str) -> None:
    """Import from CSV file.

    Example: memo import csv /path/to/export.csv
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    from pathlib import Path

    result = mem.import_export.import_from(Path(input_path), "csv")
    _report_import_result(result)


@import_group.command(name="markdown-bundle")
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False))
def import_markdown_bundle(input_path: str) -> None:
    """Import from Markdown bundle (zip).

    Example: memo import markdown-bundle /path/to/export.zip
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    from pathlib import Path

    result = mem.import_export.import_from(Path(input_path), "markdown_bundle")
    _report_import_result(result)


# -- cold-start history importers (Codex / opencode / ChatGPT / Claude.ai) -------


def _echo_import_summary(summary: dict, as_json: bool) -> None:
    import json as _json

    if as_json:
        console.print_json(_json.dumps(summary))
        return
    if summary.get("status") == "no_files":
        console.print(f"[yellow]no files found under {summary.get('root')}[/yellow]")
        return
    console.print("[green]Import complete[/green]")
    if "files_total" in summary:
        console.print(
            f"Files: {summary.get('files_processed', 0)} processed, "
            f"{summary.get('files_skipped', 0)} already up to date"
        )
    console.print(f"Candidates: {summary.get('candidates', 0)}")
    console.print(f"Saved: {len(summary.get('saved', []))}")
    console.print(f"Skipped (dup): {summary.get('skipped_dup', 0)}")
    if summary.get("dry_run"):
        console.print("[yellow]dry-run — nothing written[/yellow]")


@import_group.command(name="codex")
@click.option(
    "--path", "root_path", default=None, help="Rollouts root (default: ~/.codex/sessions)."
)
@click.option(
    "--since", "since_days", type=int, default=None, help="Only files modified in the last N days."
)
@click.option("--limit", "file_limit", type=int, default=None, help="Cap on files (newest first).")
@click.option("--dry-run", is_flag=True, help="Extract, don't save.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON summary.")
def import_codex(root_path, since_days, file_limit, dry_run, as_json) -> None:
    """Mine Codex CLI rollouts (~/.codex/sessions) into memories.

    Same prefilter → helper-LLM extract → dedup pipeline as `memo
    mine-history`; resumable via per-file cursors in
    state_dir/import-history.json. First run on a long history is slow
    (helper LLM is the bottleneck) — start with `--limit 10 --since 30`.
    """
    from pathlib import Path

    from memo.history_importers import run_codex_import

    root = Path(root_path).expanduser() if root_path else None
    summary = run_codex_import(
        root=root, since_days=since_days, file_limit=file_limit, dry_run=dry_run
    )
    _echo_import_summary(summary, as_json)


@import_group.command(name="opencode")
@click.option(
    "--db",
    "db_path",
    type=click.Path(),
    default=None,
    help="opencode.db path (default: ~/.local/share/opencode/opencode.db).",
)
@click.option("--dry-run", is_flag=True, help="Extract, don't save.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON summary.")
def import_opencode(db_path, dry_run, as_json) -> None:
    """Mine opencode's SQLite session store into memories."""
    from pathlib import Path

    from memo.history_importers import iter_opencode_exchanges, run_file_import

    db = (
        Path(db_path).expanduser()
        if db_path
        else Path.home() / ".local" / "share" / "opencode" / "opencode.db"
    )
    if not db.is_file():
        raise click.ClickException(f"opencode db not found: {db}")
    summary = run_file_import(iter_opencode_exchanges(db), dry_run=dry_run, source_name=db.name)
    _echo_import_summary(summary, as_json)


@import_group.command(name="chatgpt")
@click.argument("export_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--dry-run", is_flag=True, help="Extract, don't save.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON summary.")
def import_chatgpt(export_path, dry_run, as_json) -> None:
    """Mine a ChatGPT data-export conversations.json into memories."""
    from pathlib import Path

    from memo.history_importers import iter_chatgpt_exchanges, run_file_import

    p = Path(export_path)
    summary = run_file_import(iter_chatgpt_exchanges(p), dry_run=dry_run, source_name=p.name)
    _echo_import_summary(summary, as_json)


@import_group.command(name="claude-export")
@click.argument("export_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--dry-run", is_flag=True, help="Extract, don't save.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON summary.")
def import_claude_export(export_path, dry_run, as_json) -> None:
    """Mine a Claude.ai data-export conversations.json into memories."""
    from pathlib import Path

    from memo.history_importers import iter_claude_export_exchanges, run_file_import

    p = Path(export_path)
    summary = run_file_import(iter_claude_export_exchanges(p), dry_run=dry_run, source_name=p.name)
    _echo_import_summary(summary, as_json)


# -- competitor-store migrators (adoption funnel) --------------------------------


@import_group.command(name="mem0")
@click.argument("dump_path", type=click.Path(exists=True, dir_okay=False))
def import_mem0(dump_path: str) -> None:
    """Migrate a Mem0 export dump (JSON) into memo.

    Example: memo import mem0 mem0_export.json
    """
    import json as _json
    from pathlib import Path

    from memo.store_migrators import mem0_to_import_records

    cfg = Config.from_env()
    mem = _get_memory(cfg)
    data = _json.loads(Path(dump_path).read_text(encoding="utf-8"))
    result = mem.import_export.importer.import_records(mem0_to_import_records(data))
    console.print("[green]Import complete[/green]")
    console.print(f"Imported: {result.imported_count}")
    console.print(f"Skipped: {result.skipped_count}")
    if result.errors:
        console.print(f"[yellow]Errors: {len(result.errors)}[/yellow]")


@import_group.command(name="zep")
@click.argument("dump_path", type=click.Path(exists=True, dir_okay=False))
def import_zep(dump_path: str) -> None:
    """Migrate a Zep facts dump (JSON) into memo. Invalidated facts are skipped.

    Example: memo import zep zep_facts.json
    """
    import json as _json
    from pathlib import Path

    from memo.store_migrators import zep_to_import_records

    cfg = Config.from_env()
    mem = _get_memory(cfg)
    data = _json.loads(Path(dump_path).read_text(encoding="utf-8"))
    result = mem.import_export.importer.import_records(zep_to_import_records(data))
    console.print("[green]Import complete[/green]")
    console.print(f"Imported: {result.imported_count}")
    console.print(f"Skipped: {result.skipped_count}")
    if result.errors:
        console.print(f"[yellow]Errors: {len(result.errors)}[/yellow]")
