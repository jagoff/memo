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


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
