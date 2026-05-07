"""CLI — `memo` entry point.

A handful of operational commands so the user can interact with the
memory store from the shell without spinning up the MCP server:

- `memo save 'content here' --title 'X' --tag x --tag y`
- `memo search 'query' --limit 5`
- `memo list --limit 20 --type decision`
- `memo get <id>`
- `memo delete <id>`
- `memo stats`
- `memo doctor` — verify vault path, embedder loadable, sqlite-vec
  available, MLX present.

Output style:
- Default: rich table for list/search, panel for `get`, plain stats.
- `--json` flag (where applicable): emit raw JSON for piping.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from memo.config import Config

console = Console()


def _resolved(thunk):
    """Run `thunk()` translating `AmbiguousIdError` into a friendly print
    + exit code 2. Used by every CLI verb that takes an id-or-prefix
    argument (`get`, `update`, `delete`).
    """
    from memo.memory import AmbiguousIdError

    try:
        return thunk()
    except AmbiguousIdError as exc:
        console.print(f"[red]ambiguous id prefix[/red] {exc.prefix!r} matches:")
        for m in exc.matches[:8]:
            console.print(f"  · {m}")
        if len(exc.matches) > 8:
            console.print(f"  · …and {len(exc.matches) - 8} more")
        sys.exit(2)


@click.group()
@click.version_option()
def cli() -> None:
    """memo — local MCP memory backed by Obsidian vault, MLX-native."""


@cli.command()
@click.argument("content")
@click.option("--title", default=None, help="Short title (default: first line of content)")
@click.option(
    "--type", "type_",
    type=click.Choice(
        ["decision", "fact", "bug", "feedback", "preference", "note", "manual"],
    ),
    default="note", show_default=True,
)
@click.option("--tag", "-t", "tags", multiple=True, help="Repeatable. Lower-cased + de-duplicated.")
@click.option("--auto-derive", is_flag=True,
              help="When title/type/tags missing, ask Qwen2.5-3B helper to derive them. "
                   "Adds ~1-2s latency on first call.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of a panel.")
def save(content: str, title: str | None, type_: str, tags: tuple[str, ...],
         auto_derive: bool, as_json: bool) -> None:
    """Persist CONTENT to the vault + index. Pass `-` to read CONTENT from stdin."""
    from memo.memory import Memory

    if content == "-":
        content = sys.stdin.read()
    mem = Memory(Config.from_env())
    rec = mem.save(content=content, title=title, type_=type_,
                   tags=list(tags), auto_derive=auto_derive)
    if as_json:
        click.echo(json.dumps(rec.to_dict(), ensure_ascii=False, indent=2))
        return
    console.print(Panel.fit(
        f"[bold]{rec.title}[/bold]\n"
        f"[dim]id:[/dim] {rec.id}\n"
        f"[dim]path:[/dim] {rec.path}\n"
        f"[dim]type:[/dim] {rec.type}  [dim]tags:[/dim] {', '.join(rec.tags) or '—'}",
        title="✓ saved", border_style="green",
    ))


@cli.command()
@click.argument("query")
@click.option("--limit", default=10, type=int, show_default=True)
@click.option("--type", "type_", default=None, help="Filter by record type.")
@click.option("--mode", default="hybrid",
              type=click.Choice(["hybrid", "vec", "bm25"]), show_default=True,
              help="hybrid = RRF fusion of vec + bm25 (default). vec = semantic only. bm25 = keyword only.")
@click.option("--json", "as_json", is_flag=True)
def search(query: str, limit: int, type_: str | None, mode: str, as_json: bool) -> None:
    """Top-k search — hybrid (semantic + keyword) by default."""
    from memo.memory import Memory

    mem = Memory(Config.from_env())
    hits = mem.search(query, limit=limit, type_=type_, mode=mode)
    if as_json:
        click.echo(json.dumps([h.to_dict() for h in hits], ensure_ascii=False, indent=2))
        return
    if not hits:
        console.print("[dim]no results[/dim]")
        return
    tbl = Table(show_lines=False, expand=True)
    tbl.add_column("score", justify="right", width=6)
    tbl.add_column("type", width=10)
    tbl.add_column("title", overflow="fold")
    tbl.add_column("tags", overflow="fold")
    for h in hits:
        tbl.add_row(
            f"{h.score:.3f}" if h.score is not None else "—",
            h.type,
            h.title,
            ", ".join(h.tags) or "—",
        )
    console.print(tbl)


@cli.command()
@click.argument("question")
@click.option("--k", default=5, type=int, show_default=True,
              help="Top-K memorias to feed the LLM as context.")
@click.option("--type", "type_", default=None, help="Restrict the retrieval to one record type.")
@click.option("--json", "as_json", is_flag=True)
def ask(question: str, k: int, type_: str | None, as_json: bool) -> None:
    """RAG over the memory archive — synthesises a prose answer with
    inline `[id]` citations using MLXChat 7B over the top-K hybrid hits.
    """
    from memo.memory import Memory

    mem = Memory(Config.from_env())
    out = mem.ask(question, k=k, type_=type_)
    if as_json:
        click.echo(json.dumps(out, ensure_ascii=False, indent=2))
        return
    console.print(Panel.fit(
        out["answer"] or "[dim](sin respuesta)[/dim]",
        title=f"❓ {question[:60]}", border_style="cyan",
    ))
    if out["sources"]:
        console.print("[dim]fuentes:[/dim]")
        for s in out["sources"]:
            console.print(
                f"  [dim][{s['id_short']}][/dim] {s['title'][:60]}  "
                f"[dim](score {s['score']:.3f})[/dim]"
            )


@cli.command(name="list")
@click.option("--limit", default=20, type=int, show_default=True)
@click.option("--type", "type_", default=None)
@click.option("--json", "as_json", is_flag=True)
def list_cmd(limit: int, type_: str | None, as_json: bool) -> None:
    """Recent memories by `updated` desc."""
    from memo.memory import Memory

    mem = Memory(Config.from_env())
    items = mem.list(limit=limit, type_=type_)
    if as_json:
        click.echo(json.dumps([r.to_dict() for r in items], ensure_ascii=False, indent=2))
        return
    if not items:
        console.print("[dim]vacío[/dim]")
        return
    tbl = Table(show_lines=False, expand=True)
    tbl.add_column("updated", width=20)
    tbl.add_column("type", width=10)
    tbl.add_column("title", overflow="fold")
    tbl.add_column("tags", overflow="fold")
    for r in items:
        tbl.add_row(r.updated[:19], r.type, r.title, ", ".join(r.tags) or "—")
    console.print(tbl)


@cli.command()
@click.argument("id_")
@click.option("--json", "as_json", is_flag=True)
def get(id_: str, as_json: bool) -> None:
    """Fetch one memory by id."""
    from memo.memory import Memory

    mem = Memory(Config.from_env())
    rec = _resolved(lambda: mem.get(id_))
    if rec is None:
        console.print(f"[red]not found:[/red] {id_}")
        sys.exit(1)
    if as_json:
        click.echo(json.dumps(rec.to_dict(), ensure_ascii=False, indent=2))
        return
    console.print(Panel.fit(
        f"[bold]{rec.title}[/bold]\n"
        f"[dim]id:[/dim] {rec.id}  [dim]type:[/dim] {rec.type}\n"
        f"[dim]tags:[/dim] {', '.join(rec.tags) or '—'}\n"
        f"[dim]created:[/dim] {rec.created}\n"
        f"[dim]updated:[/dim] {rec.updated}\n\n"
        f"{rec.body}",
        title=rec.title, border_style="cyan",
    ))


@cli.command()
@click.argument("id_")
@click.option("--title", default=None)
@click.option(
    "--type", "type_",
    type=click.Choice(
        ["decision", "fact", "bug", "feedback", "preference", "note", "manual"],
    ),
    default=None,
)
@click.option("--tag", "-t", "tags", multiple=True, help="Replaces existing tags.")
@click.option(
    "--content", default=None,
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
    from memo.memory import Memory

    if content == "-":
        content = sys.stdin.read()

    mem = Memory(Config.from_env())
    rec = _resolved(lambda: mem.update(
        id_,
        title=title,
        type_=type_,
        tags=list(tags) if tags else None,
        content=content,
    ))
    if rec is None:
        console.print(f"[red]not found:[/red] {id_}")
        sys.exit(1)
    if as_json:
        click.echo(json.dumps(rec.to_dict(), ensure_ascii=False, indent=2))
        return
    console.print(Panel.fit(
        f"[bold]{rec.title}[/bold]\n"
        f"[dim]id:[/dim] {rec.id}  [dim]type:[/dim] {rec.type}\n"
        f"[dim]tags:[/dim] {', '.join(rec.tags) or '—'}\n"
        f"[dim]updated:[/dim] {rec.updated}",
        title="✓ updated", border_style="yellow",
    ))


@cli.command()
@click.option("--force", is_flag=True,
              help="Re-embed ALL indexed entries regardless of body_hash. "
                   "Use after embedder swap or composition change.")
@click.option("--json", "as_json", is_flag=True)
def reindex(force: bool, as_json: bool) -> None:
    """Re-scan memory dir, re-embed entries with body_hash mismatch.

    Run after editing memory `.md` files directly in Obsidian, or after
    restoring memories from a backup. Use `--force` to re-embed every
    entry (slower; needed after model/composition changes).
    """
    from memo.memory import Memory

    mem = Memory(Config.from_env())
    counts = mem.reindex(force=force)
    if as_json:
        click.echo(json.dumps(counts, indent=2))
        return
    console.print(
        f"checked: [cyan]{counts['checked']}[/cyan]  "
        f"reindexed: [yellow]{counts['reindexed']}[/yellow]  "
        f"added: [green]{counts['added']}[/green]  "
        f"skipped: [dim]{counts['skipped']}[/dim]",
    )


@cli.command()
@click.argument("id_")
@click.option("--yes", is_flag=True, help="Skip confirmation.")
def delete(id_: str, yes: bool) -> None:
    """Delete one memory by id."""
    from memo.memory import Memory

    mem = Memory(Config.from_env())
    if not yes:
        click.confirm(f"Delete memory {id_!r}? This removes the .md and the index entry.", abort=True)
    ok = _resolved(lambda: mem.delete(id_))
    console.print(f"[{'green' if ok else 'red'}]{'✓ deleted' if ok else 'not found'}[/]: {id_}")


@cli.command()
@click.option("--limit", default=20, type=int, show_default=True)
@click.option("--op", default=None,
              type=click.Choice(["save", "update", "delete"]),
              help="Filter to one op type.")
@click.option("--id", "record_id", default=None,
              help="Filter to events for one record (full id or unique prefix).")
@click.option("--json", "as_json", is_flag=True)
def history(limit: int, op: str | None, record_id: str | None, as_json: bool) -> None:
    """Recent save/update/delete events. Append-only audit log."""
    from memo.memory import Memory

    mem = Memory(Config.from_env())
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
            delta = ", ".join(f"{k}" for k in r["delta"].keys())
        tbl.add_row(
            (r["ts"] or "")[:19], r["op"], (r["record_id"] or "")[:8],
            r["title"] or "—", delta or "—",
        )
    console.print(tbl)


@cli.command(name="extract-entities")
@click.option("--all", "all_", is_flag=True, help="Process every memoria in the store.")
@click.option("--id", "id_", default=None, multiple=True,
              help="Repeatable. Process specific memoria id(s) (full or prefix).")
@click.option("--force", is_flag=True,
              help="Re-extract even if memoria already has entity links (default skips).")
@click.option("--json", "as_json", is_flag=True)
def extract_entities(all_: bool, id_: tuple[str, ...], force: bool, as_json: bool) -> None:
    """Extract named entities (person/project/technology/file/org/concept)
    from memoria bodies via Qwen2.5-3B and write them to the graph DB.

    Cost: ~0.5-1s per memoria. 223-doc corpus ≈ 2-4 min.
    """
    from memo.memory import Memory

    if not all_ and not id_:
        click.echo("pass --all or one or more --id <prefix>", err=True)
        sys.exit(2)

    mem = Memory(Config.from_env())
    resolved_ids: list[str] | None = None
    if id_:
        resolved_ids = []
        for raw in id_:
            r = _resolved(lambda raw=raw: mem.resolve_id(raw))
            if r is None:
                console.print(f"[red]not found:[/red] {raw}")
                sys.exit(1)
            resolved_ids.append(r)

    counts = mem.extract_entities(
        ids=resolved_ids, all_=all_, skip_already_indexed=not force,
    )
    if as_json:
        click.echo(json.dumps(counts, indent=2))
        return
    console.print(
        f"processed: [cyan]{counts['processed']}[/cyan]  "
        f"entities: [green]{counts['entities_extracted']}[/green]  "
        f"links: [green]{counts['links_written']}[/green]  "
        f"skipped: [dim]{counts['skipped']}[/dim]  "
        f"errors: [red]{counts['errors']}[/red]",
    )


@cli.command()
@click.option("--limit", default=30, type=int, show_default=True)
@click.option("--type", "type_", default=None,
              type=click.Choice(["person", "project", "technology", "file", "org", "concept"]),
              help="Filter by entity type.")
@click.option("--json", "as_json", is_flag=True)
def entities(limit: int, type_: str | None, as_json: bool) -> None:
    """Top entities by mention count."""
    from memo.memory import Memory

    mem = Memory(Config.from_env())
    rows = mem.graph.top_entities(limit=limit, type_=type_)
    if as_json:
        click.echo(json.dumps(rows, indent=2))
        return
    if not rows:
        console.print("[dim]no entities indexed — run `memo extract-entities --all` first[/dim]")
        return
    tbl = Table(show_lines=False, expand=True)
    tbl.add_column("count", justify="right", width=6)
    tbl.add_column("type", width=12)
    tbl.add_column("name", overflow="fold")
    tbl.add_column("first_seen", width=10)
    tbl.add_column("last_seen", width=10)
    for r in rows:
        tbl.add_row(
            str(r["mention_count"]), r["type"], r["name"],
            (r["first_seen"] or "")[:10], (r["last_seen"] or "")[:10],
        )
    console.print(tbl)


@cli.command()
@click.argument("name")
@click.option("--type", "type_", default=None,
              type=click.Choice(["person", "project", "technology", "file", "org", "concept"]))
@click.option("--json", "as_json", is_flag=True)
def entity(name: str, type_: str | None, as_json: bool) -> None:
    """Memorias that mention an entity."""
    from memo.memory import Memory

    mem = Memory(Config.from_env())
    ids = mem.graph.entity_memorias(name, type_=type_)
    if as_json:
        click.echo(json.dumps(ids, indent=2))
        return
    if not ids:
        console.print(f"[dim]no memorias mention {name!r}{f' ({type_})' if type_ else ''}[/dim]")
        return
    console.print(f"[bold]{len(ids)}[/bold] memoria(s) mention [cyan]{name}[/cyan]:")
    for mid in ids[:50]:
        rec = mem.store.get(mid)
        if rec:
            console.print(f"  · [{mid[:8]}] {rec['title'][:60]} [dim]({rec['updated'][:10]})[/dim]")
    if len(ids) > 50:
        console.print(f"  · …and {len(ids) - 50} more")


@cli.command()
@click.option("--threshold", default=0.85, type=float, show_default=True,
              help="Cosine similarity floor for clustering. 0.85 conservative, 0.92+ near-identical only.")
@click.option("--max-clusters", default=20, type=int, show_default=True,
              help="Cap LLM calls — only the largest N clusters get summarised.")
@click.option("--type", "type_", default=None, help="Restrict clustering to one record type.")
@click.option("--json", "as_json", is_flag=True)
def consolidate(threshold: float, max_clusters: int, type_: str | None, as_json: bool) -> None:
    """Find clusters of near-duplicate memorias and propose actions.

    Read-only: surfaces a list of {summary, relationship, members} per
    cluster. The user reviews, then uses `memo update` / `memo delete`
    to apply. NEVER auto-modifies the store.

    Latency on 7B: ~3-5s per cluster summarised.
    """
    from memo.memory import Memory

    mem = Memory(Config.from_env())
    clusters = mem.consolidate(
        threshold=threshold, max_clusters=max_clusters, type_=type_,
    )
    if as_json:
        click.echo(json.dumps(clusters, ensure_ascii=False, indent=2))
        return
    if not clusters:
        console.print(f"[green]✓[/green] no clusters at threshold ≥{threshold}")
        return
    for c in clusters:
        relation_color = {
            "duplicate": "red", "evolution": "yellow",
            "facets": "cyan", "unrelated": "dim",
        }.get(c["relationship"], "white")
        console.print(
            f"\n[bold]cluster {c['cluster_id']}[/bold] · "
            f"[{relation_color}]{c['relationship']}[/{relation_color}] · "
            f"{c['size']} memorias",
        )
        if c["summary"]:
            console.print(f"  [dim]summary:[/dim] {c['summary']}")
        if c["rationale"]:
            console.print(f"  [dim]por qué:[/dim] {c['rationale']}")
        for m in c["members"]:
            console.print(f"    · [{m['id_short']}] {m['title'][:60]} [dim]({m['updated'][:10]})[/dim]")


@cli.command()
@click.option("--category", default=None,
              type=click.Choice(["legacy_extra", "few_tags", "body_skinny", "untitled"]),
              help="Show only one category. Default: summary of all.")
@click.option("--limit", default=20, type=int, show_default=True,
              help="Max entries per category in the report.")
@click.option("--json", "as_json", is_flag=True)
def lint(category: str | None, limit: int, as_json: bool) -> None:
    """Surface memorias with quality issues. Read-only — does not edit
    anything. Use to plan a manual cleanup pass.
    """
    from memo.memory import Memory

    mem = Memory(Config.from_env())
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


@cli.command()
@click.option("--out", "out_path", default=None, type=click.Path(),
              help="Output zip path. Default: ./memo-backup-<YYYYMMDD-HHMMSS>.zip")
def backup(out_path: str | None) -> None:
    """Snapshot memory dir + sqlite-vec DB + history DB into a zip.

    Use before risky operations (mass migration, embedder swap, schema
    change). The zip is portable: extract on another machine, set
    `MEMO_VAULT_PATH` to the matching vault, run `memo restore <zip>`
    to absorb everything back. Vault `.md` files are kept as the
    storage of record so the backup is self-contained.
    """
    import datetime as _dt
    import zipfile

    cfg = Config.from_env()
    cfg.ensure_dirs()
    out = out_path or f"memo-backup-{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.zip"
    out_p = __import__("pathlib").Path(out).resolve()

    n_md = 0
    with zipfile.ZipFile(out_p, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1) Memory .md files (relative to vault).
        if cfg.memory_dir.is_dir():
            for md in sorted(cfg.memory_dir.rglob("*.md")):
                rel = md.relative_to(cfg.vault_path)
                zf.write(md, arcname=f"memory/{rel}")
                n_md += 1
        # 2) State DBs (vec + history). Stored at the root.
        for db in (cfg.db_path, cfg.history_db):
            if db.is_file():
                zf.write(db, arcname=f"state/{db.name}")
        # 3) Manifest with paths so restore can sanity-check.
        manifest = {
            "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "vault_path": str(cfg.vault_path),
            "memory_subdir": cfg.memory_subdir,
            "embedder_model": cfg.embedder_model,
            "embedder_dims": cfg.embedder_dims,
            "memo_version": __import__("memo").__version__,
            "n_md": n_md,
        }
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))

    size_kb = out_p.stat().st_size // 1024
    console.print(f"[green]✓[/green] backup: {out_p} ({n_md} memorias, {size_kb} KB)")


@cli.command()
@click.argument("zip_path", type=click.Path(exists=True))
@click.option("--reindex", is_flag=True,
              help="After restoring .md files, run `memo reindex` to "
                   "rebuild the index from disk (use when restoring without "
                   "the bundled state DBs, or across embedder model versions).")
@click.option("--yes", is_flag=True, help="Skip confirmation.")
def restore(zip_path: str, reindex: bool, yes: bool) -> None:
    """Restore from a backup zip created by `memo backup`.

    Extracts memory `.md` files into the vault and (optionally) the
    state DBs. **Will overwrite** matching files in the vault and
    state dir — confirmation required unless `--yes`.
    """
    import zipfile
    from pathlib import Path as _Path

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
                f"memorias: {manifest.get('n_md')}  "
                f"embedder: {manifest.get('embedder_model')}",
            )
        if not yes:
            click.confirm(
                f"Extract into {cfg.vault_path} + {cfg.state_dir}? "
                "Existing files will be overwritten.", abort=True,
            )
        # Stream entries.
        n_md = n_db = 0
        for info in zf.infolist():
            if info.filename == "manifest.json":
                continue
            data = zf.read(info)
            if info.filename.startswith("memory/"):
                rel = info.filename[len("memory/"):]
                dest = cfg.vault_path / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                n_md += 1
            elif info.filename.startswith("state/"):
                rel = info.filename[len("state/"):]
                dest = cfg.state_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                n_db += 1

    console.print(
        f"[green]✓[/green] restored {n_md} memorias + {n_db} state DB(s) "
        f"into {cfg.vault_path}",
    )

    if reindex:
        from memo.memory import Memory
        mem = Memory(Config.from_env())
        # Force re-embed in case the bundled DB is from a different
        # embedder model — rebuilds vectors from .md authoritative state.
        counts = mem.reindex(force=True)
        console.print(
            f"reindex: checked {counts['checked']}  reindexed {counts['reindexed']}  "
            f"added {counts['added']}  skipped {counts['skipped']}",
        )


@cli.command()
def stats() -> None:
    """Summary stats — total records, vault path, embedder model."""
    from memo.memory import Memory

    mem = Memory(Config.from_env())
    info: dict[str, Any] = {
        "total": mem.store.count(),
        "vault_path": str(mem.cfg.vault_path),
        "memory_dir": str(mem.cfg.memory_dir),
        "db_path": str(mem.cfg.db_path),
        "embedder_model": mem.cfg.embedder_model,
        "llm_model": mem.cfg.llm_model,
    }
    for k, v in info.items():
        console.print(f"[dim]{k:14s}[/dim] {v}")


@cli.command()
@click.option("--gc", "do_gc", is_flag=True, help="Detect orphans between store and disk.")
@click.option("--fix", is_flag=True, help="With --gc: drop orphan store rows. .md files are never deleted automatically.")
def doctor(do_gc: bool, fix: bool) -> None:
    """Self-check: vault present, sqlite-vec loadable, MLX importable, models in cache.

    `--gc` reports orphans (store rows whose `.md` is gone, `.md` files
    whose `id` isn't in the store). `--gc --fix` removes orphan store
    rows; orphan `.md` files are listed but never deleted automatically.
    """
    cfg = Config.from_env()
    ok = True

    # 1. Vault dir
    if cfg.vault_path.is_dir():
        console.print(f"[green]✓[/green] vault: {cfg.vault_path}")
    else:
        console.print(f"[red]✗[/red] vault missing: {cfg.vault_path}")
        ok = False

    # 2. sqlite-vec
    try:
        import sqlite3

        import sqlite_vec  # type: ignore[import-not-found]

        conn = sqlite3.connect(":memory:")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.close()
        console.print("[green]✓[/green] sqlite-vec loadable")
    except Exception as exc:
        console.print(f"[red]✗[/red] sqlite-vec: {exc}")
        ok = False

    # 3. MLX importable
    try:
        import mlx.core  # noqa: F401
        import mlx_lm  # noqa: F401

        console.print("[green]✓[/green] mlx + mlx_lm importable")
    except Exception as exc:
        console.print(f"[red]✗[/red] mlx: {exc}")
        ok = False

    # 4. Models in HF cache
    from pathlib import Path

    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    for model in (cfg.embedder_model, cfg.llm_model, cfg.helper_model):
        cache_dir = hf_cache / f"models--{model.replace('/', '--')}"
        if cache_dir.is_dir():
            console.print(f"[green]✓[/green] cached: {model}")
        else:
            console.print(
                f"[yellow]![/yellow] not cached: {model}  "
                f"[dim](run `hf download {model}`)[/dim]",
            )

    if do_gc:
        from memo.memory import Memory

        mem = Memory(cfg)
        report = mem.gc(fix=fix)
        n_store = len(report["orphan_store"])
        n_disk = len(report["orphan_disk"])
        if n_store == 0 and n_disk == 0:
            console.print("[green]✓[/green] no orphans")
        else:
            if n_store:
                verb = "dropped" if fix else "found"
                console.print(
                    f"[yellow]{verb} {n_store} orphan store row(s)[/yellow] "
                    f"(in store, .md missing)",
                )
                for oid in report["orphan_store"][:20]:
                    console.print(f"  · {oid}")
                if n_store > 20:
                    console.print(f"  · …and {n_store - 20} more")
            if n_disk:
                console.print(
                    f"[yellow]found {n_disk} orphan .md file(s)[/yellow] "
                    f"(on disk, not in store — try `memo reindex`)",
                )
                for p in report["orphan_disk"][:20]:
                    console.print(f"  · {p}")
                if n_disk > 20:
                    console.print(f"  · …and {n_disk - 20} more")

    sys.exit(0 if ok else 1)


# ── Ambient memory hooks (v0.3.0) ──────────────────────────────────────────
#
# `recall-hook` and `prewarm` are designed to be wired into Claude Code's
# `UserPromptSubmit` and `SessionStart` hooks respectively (see
# `hooks/hooks.json` in this plugin / repo). They turn memo from a manual
# memory store into an **ambient** context layer: the agent automatically
# sees relevant memories before answering, with zero `/memo` invocations
# from the user.
#
# Both commands fail SILENTLY by design — a hook crash must never block
# Claude Code's prompt submission. On any error (DB locked, model load
# failure, malformed stdin) we exit 0 with empty stdout, and Claude Code
# proceeds without injection.


@cli.command(name="recall-hook")
def recall_hook() -> None:
    """UserPromptSubmit hook — inject relevant memorias as additionalContext.

    Reads a JSON object from stdin (Claude Code hook format), embeds the
    `prompt` field via the MLX embedder, runs hybrid search, and outputs
    the top-k results as `additionalContext` in `hookSpecificOutput`.

    Configure via env vars (all optional, sensible defaults for v0.3.0):

      MEMO_RECALL_DISABLE         — set to "1" to make this a no-op.
      MEMO_RECALL_TOP_K           — default 3
      MEMO_RECALL_MIN_SIM         — default 0.6 (cosine similarity floor)
      MEMO_RECALL_MIN_PROMPT_CHARS — default 12 (skip very short prompts)
      MEMO_RECALL_BODY_CHARS      — default 240 (snippet length per result)
      MEMO_RECALL_SKIP_SLASH      — default "1" (skip if prompt starts with /)

    Output (stdout, JSON):
      `{}` — no injection (no results / disabled / error / silent fail)
      `{"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
                               "additionalContext": "<markdown>"}}`
    """
    import json as _json
    import os
    import sys as _sys

    # Always exit 0 — hooks must not block Claude Code on memo failures.
    def _bail(reason: str = "") -> None:
        if reason and os.environ.get("MEMO_RECALL_DEBUG") == "1":
            print(f"# memo recall-hook: {reason}", file=_sys.stderr)
        print("{}")
        _sys.exit(0)

    if os.environ.get("MEMO_RECALL_DISABLE") == "1":
        _bail("disabled via MEMO_RECALL_DISABLE")
        return

    # Read stdin (Claude Code passes hook input as JSON).
    try:
        raw = _sys.stdin.read()
        if not raw.strip():
            _bail("empty stdin")
            return
        payload = _json.loads(raw)
    except _json.JSONDecodeError as exc:
        _bail(f"stdin parse fail: {exc}")
        return

    prompt = (payload.get("prompt") or "").strip()
    min_chars = int(os.environ.get("MEMO_RECALL_MIN_PROMPT_CHARS", "12"))
    if len(prompt) < min_chars:
        _bail(f"prompt too short ({len(prompt)} < {min_chars})")
        return

    # Skip slash commands by default — the user is invoking another skill,
    # injecting recall context would be noise. Override with
    # MEMO_RECALL_SKIP_SLASH=0 if you want recall on /memo:memo etc.
    if os.environ.get("MEMO_RECALL_SKIP_SLASH", "1") == "1" and prompt.startswith("/"):
        _bail("slash command, skip recall")
        return

    top_k = int(os.environ.get("MEMO_RECALL_TOP_K", "3"))
    min_sim = float(os.environ.get("MEMO_RECALL_MIN_SIM", "0.6"))
    body_chars = int(os.environ.get("MEMO_RECALL_BODY_CHARS", "240"))

    # Suppress HF download progress bars on stderr — they'd contaminate
    # the hook's debug output and confuse users tailing logs. The model
    # is already downloaded for any working memo install; this only
    # silences first-run cache-check noise.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    # Defer the heavy import — only paid if we get past the early-exits.
    # Use vec mode (NOT hybrid): for ambient injection we want a
    # *confidence threshold*, and only vec mode produces true cosine
    # similarity in [0, 1] that's interpretable as a confidence score.
    # Hybrid uses RRF fusion which produces tiny scores (~0.01-0.05) that
    # can't be filtered with an absolute threshold.
    try:
        from memo.memory import Memory
        mem = Memory(Config.from_env())
        mode = os.environ.get("MEMO_RECALL_MODE", "vec")
        hits = mem.search(prompt, limit=top_k, mode=mode)
    except Exception as exc:  # noqa: BLE001
        _bail(f"search failed: {exc}")
        return

    # Filter by similarity floor. With mode="vec", `score` is cosine
    # similarity ∈ [-1, 1] (typically [0, 1] for L2-normalised embeddings).
    # 0.6 is the empirical confidence floor on the 223-doc corpus:
    #   - "qué decidí sobre MLX vs Ollama" → 3 hits @ 0.71-0.74 (all relevant)
    #   - "how to bake apple pie" → 3 hits @ 0.51-0.56 (literal-word noise,
    #     "apple-mcp" memoria matched). Threshold 0.6 cuts these out.
    # Tune via MEMO_RECALL_MIN_SIM if your corpus has different density.
    relevant = [h for h in hits if h.score is None or h.score >= min_sim]
    if not relevant:
        _bail(f"no hits above min_sim={min_sim}")
        return

    # Format as markdown additionalContext. Be terse — context budget is
    # capped at 10k chars by Claude Code; we want each prompt to inject
    # ~500-1500 chars at most so the user's actual prompt isn't drowned.
    lines = [
        "## Relevant memories from your past (memo)",
        "",
    ]
    for h in relevant:
        score_tag = f" (score {h.score:.2f})" if h.score is not None else ""
        body = (h.body or "").strip().replace("\n", " ")
        if len(body) > body_chars:
            body = body[:body_chars].rstrip() + "…"
        lines.append(f"**[{h.id[:8]}] {h.title}**{score_tag}")
        if h.tags:
            lines.append(f"_tags_: {', '.join(h.tags)}")
        if body:
            lines.append(f"> {body}")
        lines.append("")
    lines.append("_Use `/memo:memo get <id>` to see full content._")

    output = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": "\n".join(lines),
        }
    }
    print(_json.dumps(output, ensure_ascii=False))
    _sys.exit(0)


@cli.command(name="ingest")
@click.argument("vault_path", type=click.Path(exists=True, file_okay=False, resolve_path=True))
@click.option("--name", default=None, help="Vault label (default: dirname). Used as path prefix in store.")
@click.option("--force", is_flag=True, help="Re-embed even if body unchanged.")
@click.option("--dry-run", is_flag=True, help="Walk + report counts, don't embed/write.")
@click.option("--exclude", multiple=True, help="Glob to exclude (relative to vault). Repeat. Default: .obsidian/.git/.trash/.makemd/.smart-env/.space/04-Archive/99-obsidian-system/")
def ingest(vault_path: str, name: str | None, force: bool, dry_run: bool, exclude: tuple[str, ...]) -> None:
    """Bulk-ingest all .md from a vault into the memo index.

    Walks `<vault_path>/**/*.md`, embeds each, stores under path
    `<name>/<rel-path>`. Files with `id:` in frontmatter are skipped
    (those are curated memorias managed by `memo reindex`).

    The user's .md files are NOT modified — we synthesize ids from
    path hash and write only to `~/.local/share/memo/memvec.db`.

    Idempotent: re-running skips files whose body_hash matches the
    indexed value. Use --force to re-embed everything (e.g. after
    embedder model swap).

    Default exclusions skip Obsidian system dirs (.obsidian/, .trash/,
    etc.) and memo's own memory subdir (`04-Archive/99-obsidian-system/`)
    so we don't double-index curated memorias.
    """
    import hashlib
    from datetime import datetime, timezone
    from pathlib import Path

    import frontmatter

    from memo.embedder import MLXEmbedder
    from memo.store import VecStore

    cfg = Config.from_env()
    cfg.ensure_dirs()

    vault = Path(vault_path).resolve()
    # When ingesting the configured vault, paths are stored relative to
    # `cfg.vault_path` directly so `_read_body` can resolve them via
    # `cfg.vault_path / rel_path`. Prefixing with `vault.name` here would
    # double the basename ("Notes/Notes/foo.md") because `cfg.vault_path`
    # already ends in that name. For external vaults we keep the label
    # prefix as a multi-vault discriminator (read path support: TBD).
    is_principal_vault = vault == cfg.vault_path
    label = "" if is_principal_vault else (name or vault.name)

    # Default exclusions — Obsidian dotdirs + memo's own memory subdir
    # to avoid double-indexing the curated memorias managed by reindex.
    default_excludes = (
        ".obsidian", ".git", ".trash", ".makemd", ".smart-env", ".space",
        ".claude", ".devin", "04-Archive/99-obsidian-system",
    )
    exclude_patterns = list(exclude) + list(default_excludes)

    def _excluded(rel: Path) -> bool:
        s = str(rel)
        return any(s.startswith(pat) or f"/{pat}/" in f"/{s}/" for pat in exclude_patterns)

    md_files = []
    for p in vault.rglob("*.md"):
        rel = p.relative_to(vault)
        if _excluded(rel):
            continue
        md_files.append(p)
    md_files.sort()

    console.print(f"[cyan]found[/cyan] {len(md_files)} .md in {label} (after exclusions)")

    if dry_run:
        console.print("[dim](dry-run — exiting before embed/write)[/dim]")
        # Show first few + a few from deep dirs
        for p in md_files[:5]:
            console.print(f"  · {p.relative_to(vault)}")
        if len(md_files) > 5:
            console.print(f"  · …and {len(md_files) - 5} more")
        return

    # Lazy-load heavy stuff after the dry-run gate
    embedder = MLXEmbedder(model_path=cfg.embedder_model, expected_dims=cfg.embedder_dims)
    store = VecStore(cfg.db_path, dims=cfg.embedder_dims)

    skipped_id = skipped_empty = skipped_unchanged = added = updated = errors = 0

    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
    ) as progress:
        task_id = progress.add_task(f"embed {label}", total=len(md_files))

        for path in md_files:
            try:
                rel = path.relative_to(vault)
                store_path = f"{label}/{rel}" if label else str(rel)

                raw = path.read_text(encoding="utf-8", errors="replace")

                # Parse frontmatter (tolerant — files without frontmatter
                # treated as plain markdown).
                try:
                    fm = frontmatter.loads(raw)
                except Exception:
                    fm = frontmatter.Post(raw)

                # Skip curated memorias (have explicit id) — those are
                # managed by `memo reindex` from memory_dir.
                if fm.metadata.get("id"):
                    skipped_id += 1
                    progress.advance(task_id)
                    continue

                body = fm.content.strip()
                if not body:
                    skipped_empty += 1
                    progress.advance(task_id)
                    continue

                # Skip very short notes — they're typically tag-only stubs
                # (`#tagA #tagB` + a 1-line question) that produce
                # noise embeddings near the centroid. They match
                # generic queries with high false-positive rate.
                # Tunable via MEMO_INGEST_MIN_CHARS env (default 200).
                import os as _os_min  # local import — avoids tedious refactor
                min_chars = int(_os_min.environ.get("MEMO_INGEST_MIN_CHARS", "200"))
                if len(body) < min_chars:
                    skipped_empty += 1  # bucket together with empty
                    progress.advance(task_id)
                    continue

                # Synthesize stable id from path. sha256[:32] = 128-bit,
                # collision risk negligible for any realistic vault size.
                id_ = hashlib.sha256(store_path.encode("utf-8")).hexdigest()[:32]

                # Title: explicit frontmatter > first H1 > filename stem.
                title = (
                    fm.metadata.get("title")
                    or _extract_first_h1(body)
                    or path.stem.replace("-", " ").replace("_", " ")
                )
                title = str(title).strip() or path.stem

                # Tags: frontmatter tags + directory parts (skipping
                # numeric prefixes used in PARA folders like "01-Projects").
                tags: list[str] = []
                fm_tags = fm.metadata.get("tags") or []
                if isinstance(fm_tags, str):
                    fm_tags = [t.strip() for t in fm_tags.split(",")]
                for t in fm_tags:
                    if t and str(t) not in tags:
                        tags.append(str(t))
                for part in rel.parent.parts:
                    if part and part not in tags:
                        tags.append(part)

                body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]

                # Idempotence check
                existing = store.get(id_)
                if existing and existing["body_hash"] == body_hash and not force:
                    skipped_unchanged += 1
                    progress.advance(task_id)
                    continue

                # Embed: title + body, capped at max_content_chars.
                composed = f"{title}\n\n{body}"
                if len(composed) > cfg.max_content_chars:
                    composed = composed[: cfg.max_content_chars]
                # `embedder.embed()` is batched — takes Sequence[str] and
                # returns list[list[float]]. Passing a bare string iterates
                # per-char (str IS a Sequence of chars), producing wrong-
                # dim outputs. Wrap in a list and take [0].
                embedding = embedder.embed([composed])[0]

                # Loud-fail guard. Past silent-failure mode (string
                # iterated as chars, partial Metal recovery, etc.)
                # produced records with wrong-dim or zero-norm vectors
                # that polluted the index without surfacing in the
                # ingest summary. Centralised in `assert_valid_embedding`
                # so save/update/ingest share one definition of "valid".
                # Strict mode (env var MEMO_INGEST_STRICT=1) re-raises
                # — for CI runs / golden-corpus rebuilds where any
                # rejection should be surfaced loudly.
                from memo.embedder import assert_valid_embedding as _assert_emb
                try:
                    _assert_emb(embedding, cfg.embedder_dims, context=str(path))
                except ValueError as _ve:
                    errors += 1
                    import os as _os_strict
                    if _os_strict.environ.get("MEMO_INGEST_STRICT") == "1":
                        raise
                    if _os_strict.environ.get("MEMO_INGEST_DEBUG") == "1":
                        console.print(f"[red]reject:[/] {_ve}")
                    progress.advance(task_id)
                    continue

                now = datetime.now(timezone.utc).isoformat()
                # Preserve created if known (existing row), else now.
                created = existing["created"] if existing else now

                store.upsert(
                    id_=id_,
                    path=store_path,
                    title=title[:200],  # title is meta.title field, keep snug
                    type_="note",
                    tags=tags,
                    created=created,
                    updated=now,
                    body_hash=body_hash,
                    embedding=embedding,
                    extra={"source": "vault-ingest", "vault": label, "abs_path": str(path)},
                    body_text=body,
                )

                if existing:
                    updated += 1
                else:
                    added += 1
            except Exception as exc:  # noqa: BLE001
                errors += 1
                import os as _os
                if _os.environ.get("MEMO_INGEST_DEBUG") == "1":
                    console.print(f"[red]err[/] {path}: {exc}")
            finally:
                progress.advance(task_id)

    console.print(
        f"\n[green]done[/] "
        f"added={added} updated={updated} "
        f"skipped_unchanged={skipped_unchanged} "
        f"skipped_id={skipped_id} skipped_empty={skipped_empty} "
        f"errors={errors}"
    )


def _extract_first_h1(body: str) -> str | None:
    """Return text of the first `# H1` line, or None."""
    for line in body.splitlines():
        s = line.strip()
        if s.startswith("# ") and not s.startswith("##"):
            return s[2:].strip()
        if s and not s.startswith("#"):
            # First non-heading line of content — no H1.
            return None
    return None


@cli.command(name="prewarm")
def prewarm() -> None:
    """SessionStart hook — pre-load the MLX embedder so first recall is fast.

    Loads the embedder model into memory + warms the lazy `mlx_lm.load()`
    path. Subsequent `recall-hook` calls in the session benefit from the
    OS file cache (the model weights stay in RAM-backed disk cache for
    minutes after a load), shaving cold-load latency from ~2s to ~500ms.

    Designed to be invoked async at SessionStart so the user doesn't see
    the delay. Failures are silent (the recall-hook will just be slower).
    """
    import os
    import sys as _sys

    if os.environ.get("MEMO_RECALL_DISABLE") == "1":
        _sys.exit(0)
    try:
        from memo.embedder import MLXEmbedder
        cfg = Config.from_env()
        emb = MLXEmbedder(model_path=cfg.embedder_model, expected_dims=cfg.embedder_dims)
        emb.embed(["warmup"])  # batch=1; forces MLX load + first forward pass
        # Reranker prewarm — same rationale as the embedder. Skipped
        # when disabled to keep the SessionStart hook below its
        # 30s budget on machines that opted out of rerank entirely.
        if cfg.reranker_enabled:
            from memo.reranker import MLXReranker
            r = MLXReranker(model_path=cfg.reranker_model)
            r.warmup()
    except Exception as exc:  # noqa: BLE001
        if os.environ.get("MEMO_RECALL_DEBUG") == "1":
            print(f"# memo prewarm failed: {exc}", file=_sys.stderr)
    _sys.exit(0)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
