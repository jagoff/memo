"""Core memory verbs for the memo CLI — CRUD + lifecycle.

Extracted from cli.py (top-level command grouping); retrieval moved to
cli_search.py and knowledge-graph verbs to cli_entities.py. Each command is a
standalone @click.command registered onto the root group in cli.py.
Commands: save, list, get, update, reindex, delete, history, ocr-image,
provenance, lint, restore.
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from typing import Any

import click
from rich.panel import Panel
from rich.table import Table

from memo.cli_common import _resolved, console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config


@click.command()
@click.argument("content")
@click.option("--title", default=None, help="Short title (default: first line of content)")
@click.option(
    "--type",
    "type_",
    type=click.Choice(
        ["decision", "fact", "bug", "feedback", "preference", "note", "manual"],
    ),
    default="note",
    show_default=True,
)
@click.option("--tag", "-t", "tags", multiple=True, help="Repeatable. Lower-cased + de-duplicated.")
@click.option(
    "--auto-derive",
    is_flag=True,
    help="When title/type/tags missing, ask Qwen2.5-3B helper to derive them. "
    "Adds ~1-2s latency on first call.",
)
@click.option(
    "--no-project-tag",
    "no_project_tag",
    is_flag=True,
    help="Skip the auto `project:<repo>` tag derived from the current git toplevel.",
)
@click.option(
    "--defer-embed",
    is_flag=True,
    help="Save markdown + BM25 index only; run `memo reindex` later for semantic search.",
)
@click.option(
    "--meta",
    "meta_pairs",
    multiple=True,
    metavar="KEY=VALUE",
    help="Repeatable. Adds an entry to the `extra` metadata bag persisted "
    "to frontmatter + meta.extra_json. Synapse uses this to attach "
    "provenance (`--meta synapse_trace_id=...`, `--meta synapse_agent_id=...`).",
)
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of a panel.")
def save(
    content: str,
    title: str | None,
    type_: str,
    tags: tuple[str, ...],
    auto_derive: bool,
    no_project_tag: bool,
    defer_embed: bool,
    meta_pairs: tuple[str, ...],
    as_json: bool,
) -> None:
    """Persist CONTENT to the vault + index. Pass `-` to read CONTENT from stdin."""

    if content == "-":
        content = sys.stdin.read()
    extra: dict[str, Any] | None = None
    if meta_pairs:
        extra = {}
        for pair in meta_pairs:
            if "=" not in pair:
                raise click.BadParameter(
                    f"--meta expects KEY=VALUE, got {pair!r}",
                    param_hint="--meta",
                )
            key, _, value = pair.partition("=")
            key = key.strip()
            if not key:
                raise click.BadParameter(
                    f"--meta key cannot be empty: {pair!r}",
                    param_hint="--meta",
                )
            extra[key] = value
    mem = _get_memory(Config.from_env())
    rec = mem.save(
        content=content,
        title=title,
        type_=type_,
        tags=list(tags),
        auto_derive=auto_derive,
        auto_project=not no_project_tag,
        defer_embed=defer_embed,
        extra=extra,
    )
    if as_json:
        click.echo(json.dumps(rec.to_dict(), ensure_ascii=False, indent=2))
        return
    console.print(
        Panel.fit(
            f"[bold]{rec.title}[/bold]\n"
            f"[dim]id:[/dim] {rec.id}\n"
            f"[dim]path:[/dim] {rec.path}\n"
            f"[dim]type:[/dim] {rec.type}  [dim]tags:[/dim] {', '.join(rec.tags) or '—'}",
            title="✓ saved",
            border_style="green",
        )
    )


@click.command(name="list")
@click.option("--limit", default=20, type=int, show_default=True)
@click.option("--type", "type_", default=None)
@click.option("--json", "as_json", is_flag=True)
def list_cmd(limit: int, type_: str | None, as_json: bool) -> None:
    """Recent memories by `updated` desc."""

    mem = _get_memory(Config.from_env())
    items = mem.list(limit=limit, type_=type_)
    if as_json:
        click.echo(json.dumps([r.to_dict() for r in items], ensure_ascii=False, indent=2))
        return
    if not items:
        console.print("[dim]empty[/dim]")
        return
    tbl = Table(show_lines=False, expand=True)
    tbl.add_column("updated", width=20)
    tbl.add_column("type", width=10)
    tbl.add_column("title", overflow="fold")
    tbl.add_column("tags", overflow="fold")
    for r in items:
        tbl.add_row(r.updated[:19], r.type, r.title, ", ".join(r.tags) or "—")
    console.print(tbl)


@click.command()
@click.argument("id_")
@click.option("--json", "as_json", is_flag=True)
def get(id_: str, as_json: bool) -> None:
    """Fetch one memory by id."""

    mem = _get_memory(Config.from_env())
    rec = _resolved(lambda: mem.get(id_))
    if rec is None:
        console.print(f"[red]not found:[/red] {id_}")
        sys.exit(1)
    # Feedback loop: a fetch is a "useful" signal → feeds recall preferences.
    with contextlib.suppress(Exception):
        mem.contextual.record_click(rec.id)
    if as_json:
        click.echo(json.dumps(rec.to_dict(), ensure_ascii=False, indent=2))
        return
    console.print(
        Panel.fit(
            f"[bold]{rec.title}[/bold]\n"
            f"[dim]id:[/dim] {rec.id}  [dim]type:[/dim] {rec.type}\n"
            f"[dim]tags:[/dim] {', '.join(rec.tags) or '—'}\n"
            f"[dim]created:[/dim] {rec.created}\n"
            f"[dim]updated:[/dim] {rec.updated}\n\n"
            f"{rec.body}",
            title=rec.title,
            border_style="cyan",
        )
    )


@click.command(name="edit")
@click.argument("id_")
@click.option("--title", default=None)
@click.option(
    "--type",
    "type_",
    type=click.Choice(
        ["decision", "fact", "bug", "feedback", "preference", "note", "manual"],
    ),
    default=None,
)
@click.option("--tag", "-t", "tags", multiple=True, help="Replaces existing tags.")
@click.option(
    "--content",
    default=None,
    help="Replace body. Use '-' to read from stdin.",
)
@click.option("--json", "as_json", is_flag=True)
def update(
    id_: str,
    title: str | None,
    type_: str | None,
    tags: tuple[str, ...],
    content: str | None,
    as_json: bool,
) -> None:
    """Patch fields on an existing memory. Re-embeds only if body changed."""

    if content == "-":
        content = sys.stdin.read()

    mem = _get_memory(Config.from_env())
    rec = _resolved(
        lambda: mem.update(
            id_,
            title=title,
            type_=type_,
            tags=list(tags) if tags else None,
            content=content,
        )
    )
    if rec is None:
        console.print(f"[red]not found:[/red] {id_}")
        sys.exit(1)
    if as_json:
        click.echo(json.dumps(rec.to_dict(), ensure_ascii=False, indent=2))
        return
    console.print(
        Panel.fit(
            f"[bold]{rec.title}[/bold]\n"
            f"[dim]id:[/dim] {rec.id}  [dim]type:[/dim] {rec.type}\n"
            f"[dim]tags:[/dim] {', '.join(rec.tags) or '—'}\n"
            f"[dim]updated:[/dim] {rec.updated}",
            title="✓ updated",
            border_style="yellow",
        )
    )


@click.command()
@click.option(
    "--force",
    is_flag=True,
    help="Re-embed ALL indexed entries regardless of body_hash. "
    "Use after embedder swap or composition change.",
)
@click.option(
    "--rebuild",
    is_flag=True,
    help="Truncate the markdown-derivable index (meta/vec/fts) and replay the "
    "whole thing from disk — the markdown-is-truth reset. Preserves feedback / "
    "access / health signal. Prefer this over `rm memvec.db`.",
)
@click.option("--json", "as_json", is_flag=True)
def reindex(force: bool, rebuild: bool, as_json: bool) -> None:
    """Re-scan memory dir, re-embed entries with body_hash mismatch.

    Run after editing memory `.md` files directly in Obsidian, or after
    restoring memories from a backup. Use `--force` to re-embed every
    entry (slower; needed after model/composition changes). Use `--rebuild`
    to drop and replay the whole index from the `.md` source of truth
    without losing user-signal data — the safe alternative to deleting the DB.
    """

    import os

    if rebuild:
        os.environ.setdefault("MEMO_SKIP_MODEL_VERSION_CHECK", "1")
    mem = _get_memory(Config.from_env())
    counts = mem.reindex(force=force, rebuild=rebuild)
    if as_json:
        click.echo(json.dumps(counts, indent=2))
        return
    console.print(
        f"checked: [cyan]{counts['checked']}[/cyan]  "
        f"reindexed: [yellow]{counts['reindexed']}[/yellow]  "
        f"added: [green]{counts['added']}[/green]  "
        f"skipped: [dim]{counts['skipped']}[/dim]",
    )


@click.command()
@click.argument("id_")
@click.option("--yes", is_flag=True, help="Skip confirmation.")
def delete(id_: str, yes: bool) -> None:
    """Delete one memory by id."""

    mem = _get_memory(Config.from_env())
    if not yes:
        click.confirm(
            f"Delete memory {id_!r}? This removes the .md and the index entry.", abort=True
        )
    ok = _resolved(lambda: mem.delete(id_))
    console.print(f"[{'green' if ok else 'red'}]{'✓ deleted' if ok else 'not found'}[/]: {id_}")


@click.command()
@click.option("--limit", default=20, type=int, show_default=True)
@click.option(
    "--op",
    default=None,
    type=click.Choice(["save", "update", "delete"]),
    help="Filter to one op type.",
)
@click.option(
    "--id",
    "record_id",
    default=None,
    help="Filter to events for one record (full id or unique prefix).",
)
@click.option("--json", "as_json", is_flag=True)
def history(limit: int, op: str | None, record_id: str | None, as_json: bool) -> None:
    """Recent save/update/delete events. Append-only audit log."""

    mem = _get_memory(Config.from_env())
    if record_id and len(record_id) < 32:
        # Resolve prefix → full id (audit log stores full ids).
        resolved = mem.resolve_id(record_id)
        if resolved is None:
            console.print(f"[red]not found:[/red] {record_id}")
            sys.exit(1)
        record_id = resolved
    rows = mem.history.list_recent(limit=limit, op=op, record_id=record_id)
    if as_json:
        click.echo(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
        return
    if not rows:
        console.print("[dim]no events[/dim]")
        return
    tbl = Table(show_lines=False, expand=True)
    tbl.add_column("ts", width=20)
    tbl.add_column("op", width=7)
    tbl.add_column("id", width=10)
    tbl.add_column("title", overflow="fold")
    tbl.add_column("delta", overflow="fold")
    for r in rows:
        delta = ""
        if r.get("delta"):
            delta = ", ".join(f"{k}" for k in r["delta"])
        tbl.add_row(
            (r["ts"] or "")[:19],
            r["op"],
            (r["record_id"] or "")[:8],
            r["title"] or "—",
            delta or "—",
        )
    console.print(tbl)


@click.command(name="ocr-image")
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--json", "as_json", is_flag=True)
def ocr_image(image_path: str, as_json: bool) -> None:
    """Extract text from an image using Apple Vision OCR.

    Results cached by SHA256 under `<state_dir>/ocr_cache`. Returns the
    raw extracted text on stdout (or JSON envelope with `--json`).
    Empty output indicates Vision unavailable or no text recognized.
    """

    from memo.ocr import extract_text_cached, vision_available

    cfg = Config.from_env()
    cache_dir = cfg.state_dir / "ocr_cache"
    if not vision_available():
        if as_json:
            click.echo(json.dumps({"text": "", "error": "vision unavailable"}))
        else:
            console.print("[yellow]Apple Vision not available[/yellow]")
        return
    text = extract_text_cached(Path(image_path), cache_dir=cache_dir)
    if as_json:
        click.echo(json.dumps({"text": text, "cached": True}))
    else:
        click.echo(text)


@click.command()
@click.argument("id_", metavar="ID")
@click.option("--json", "as_json", is_flag=True)
def provenance(id_: str, as_json: bool) -> None:
    """Provenance trail for one memory.

    Returns the current synapse_*/agent_* keys plus every save/update
    event carrying its own provenance snapshot. Useful to audit which
    agent / trace_id / route_reason produced each version of a memory.
    """

    mem = _get_memory(Config.from_env())
    payload = mem.provenance(id_)
    if payload is None:
        console.print(f"[red]not found:[/red] {id_}")
        sys.exit(1)
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return
    cur = payload.get("current") or {}
    if cur:
        console.print(
            Panel.fit(
                "\n".join(f"[dim]{k}:[/dim] {v}" for k, v in cur.items()),
                title=f"provenance {payload['id'][:8]}",
                border_style="cyan",
            )
        )
    else:
        console.print(f"[dim]no provenance for {payload['id'][:8]} (current state)[/dim]")
    events = payload.get("events") or []
    if not events:
        return
    tbl = Table(show_lines=False, expand=True, title="history")
    tbl.add_column("ts", width=20)
    tbl.add_column("op", width=7)
    tbl.add_column("provenance", overflow="fold")
    for ev in events:
        prov = ev.get("provenance") or {}
        prov_str = ", ".join(f"{k}={v}" for k, v in prov.items()) if prov else "—"
        tbl.add_row((ev.get("ts") or "")[:19], ev.get("op") or "", prov_str)
    console.print(tbl)


@click.command()
@click.option(
    "--category",
    default=None,
    type=click.Choice(["legacy_extra", "few_tags", "body_skinny", "untitled"]),
    help="Show only one category. Default: summary of all.",
)
@click.option(
    "--limit",
    default=20,
    type=int,
    show_default=True,
    help="Max entries per category in the report.",
)
@click.option("--json", "as_json", is_flag=True)
def lint(category: str | None, limit: int, as_json: bool) -> None:
    """Surface memories with quality issues. Read-only — does not edit
    anything. Use to plan a manual cleanup pass.
    """

    mem = _get_memory(Config.from_env())
    report = mem.lint()
    if category:
        report = {category: report.get(category, [])}
    if as_json:
        click.echo(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return
    for cat, rows in report.items():
        n = len(rows)
        if n == 0:
            console.print(f"[green]✓[/green] {cat}: 0")
            continue
        console.print(f"[yellow]{cat}[/yellow]: {n}")
        for entry in rows[:limit]:
            console.print(
                f"  · {entry['id'][:8]} · {entry['title'][:60]} · [dim]{entry['reason']}[/dim]"
            )
        if n > limit:
            console.print(f"  · …and {n - limit} more")


@click.command()
@click.argument("zip_path", type=click.Path(exists=True))
@click.option(
    "--reindex",
    is_flag=True,
    help="After restoring .md files, run `memo reindex` to "
    "rebuild the index from disk (use when restoring without "
    "the bundled state DBs, or across embedder model versions).",
)
@click.option("--yes", is_flag=True, help="Skip confirmation.")
def restore(zip_path: str, reindex: bool, yes: bool) -> None:
    """Restore from a backup zip created by `memo backup`.

    Extracts memory `.md` files into the vault and (optionally) the
    state DBs. **Will overwrite** matching files in the vault and
    state dir — confirmation required unless `--yes`.
    """
    import zipfile

    cfg = Config.from_env()
    cfg.ensure_dirs()

    with zipfile.ZipFile(zip_path, "r") as zf:
        try:
            manifest = json.loads(zf.read("manifest.json"))
        except KeyError:
            manifest = None
        if manifest:
            console.print(
                f"backup created: {manifest.get('created')}  "
                f"memories: {manifest.get('n_md')}  "
                f"embedder: {manifest.get('embedder_model')}",
            )
        if not yes:
            click.confirm(
                f"Extract into {cfg.data_dir} + {cfg.state_dir}? "
                "Existing files will be overwritten.",
                abort=True,
            )
        # Stream entries.
        n_md = n_db = 0
        for info in zf.infolist():
            if info.filename == "manifest.json":
                continue
            data = zf.read(info)
            if info.filename.startswith("memory/"):
                rel = info.filename[len("memory/") :]
                dest = cfg.data_dir / rel
                if not dest.resolve().is_relative_to(cfg.data_dir.resolve()):
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                n_md += 1
            elif info.filename.startswith("state/"):
                rel = info.filename[len("state/") :]
                dest = cfg.state_dir / rel
                if not dest.resolve().is_relative_to(cfg.state_dir.resolve()):
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                n_db += 1

    console.print(
        f"[green]✓[/green] restored {n_md} memories + {n_db} state DB(s) into {cfg.data_dir}",
    )

    if reindex:
        mem = _get_memory(Config.from_env())
        # Force re-embed in case the bundled DB is from a different
        # embedder model — rebuilds vectors from .md authoritative state.
        counts = mem.reindex(force=True)
        console.print(
            f"reindex: checked {counts['checked']}  reindexed {counts['reindexed']}  "
            f"added {counts['added']}  skipped {counts['skipped']}",
        )
