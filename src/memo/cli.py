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
import os
import re
import shlex
import shutil
import sys
from datetime import UTC
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from memo.config import Config

# Imported at module scope (not lazily) so tests can `patch("memo.cli.run_picker", ...)`.
# `run_picker` itself defers the heavy `questionary` import until called.
from memo.setup import run_picker, write_config_file

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
@click.version_option(package_name="mlx-memo")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """memo — local MCP memory backed by markdown vault, MLX-native."""
    _first_run_gate(ctx)


# Subcommands that must NEVER trigger the first-run picker — either
# because they're part of setup/diagnostics, they don't need storage,
# or they run from non-interactive hooks (the TTY check + the
# `MEMO_NONINTERACTIVE=1` env var in `hooks.json` handle the latter,
# but listing the names here is a belt-and-suspenders defence in case
# something invokes them from an interactive shell while debugging).
_FIRST_RUN_GATE_SKIP_COMMANDS = {
    "init", "doctor", "migrate-vault",
    "mcp-command", "prewarm", "recall-hook", "capture-stop", "session", "ingest",
}


def _first_run_gate(ctx: click.Context) -> None:
    """If the user hasn't configured `memo` yet, run the picker first.

    Resolution: skip when invoked from hooks (MEMO_NONINTERACTIVE=1 or
    non-TTY), when an env var already configures `data_dir`, when a
    config file already exists, or when the legacy `MEMO_VAULT_PATH`
    pair is set (back-compat path). Also skip for setup/diagnostic
    subcommands so the user can always recover via `memo doctor`.
    """
    import os
    import sys as _sys

    if ctx.invoked_subcommand in (None, *_FIRST_RUN_GATE_SKIP_COMMANDS):
        return
    if os.environ.get("MEMO_NONINTERACTIVE") == "1":
        return
    # Both stdin and stdout must be a TTY for the picker to make sense.
    if not (_sys.stdin.isatty() and _sys.stdout.isatty()):
        return
    if "MEMO_DATA_DIR" in os.environ:
        return
    if "MEMO_VAULT_PATH" in os.environ and "MEMO_MEMORY_SUBDIR" in os.environ:
        return
    # Re-resolve the config file at gate-firing time (env may have
    # changed between import and invocation, e.g. in tests).
    from memo.setup.config_io import _resolve_config_path
    if _resolve_config_path().is_file():
        return
    _run_picker_and_save()


def _run_picker_and_save() -> None:
    """Drive the interactive picker → persist to TOML → return.

    Caller is expected to be the first-run gate (or `memo init`). Picker
    aborts (Ctrl-C / ESC) raise `click.exceptions.Exit(130)` so the
    surrounding CLI invocation halts cleanly.
    """
    console.print(
        "[bold]memo first-run setup[/bold] — pick where memorias should live.\n",
    )
    try:
        result = run_picker()
    except KeyboardInterrupt:
        console.print(
            "[yellow]aborted.[/yellow] Re-run any memo command to retry, "
            "or run `memo init` to configure.",
        )
        raise click.exceptions.Exit(130) from None
    cfg_path = write_config_file(
        data_dir=result.data_dir,
        vault_path=result.vault_path,
    )
    result.data_dir.mkdir(parents=True, exist_ok=True)
    console.print(
        f"[green]✓[/green] data_dir = {result.data_dir}",
    )
    if result.vault_path is not None:
        console.print(
            f"[green]✓[/green] vault_path = {result.vault_path}  "
            "[dim](used by `memo ingest`)[/dim]",
        )
    console.print(f"[dim]config saved: {cfg_path}[/dim]")


def _safe_resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path


def _resolve_command(name: str) -> tuple[Path | None, Path | None]:
    raw = shutil.which(name)
    if not raw:
        return None, None
    raw_path = Path(raw)
    return raw_path, _safe_resolve(raw_path)


def _env_root_for_bin(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.parent.name == "bin":
        return path.parent.parent
    return None


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _install_mode(root: Path | None) -> str:
    if root is None:
        return "unknown"
    parts = set(root.parts)
    root_s = str(root)
    if "pipx" in parts and "venvs" in parts:
        return "pipx"
    if "uv" in parts and "tools" in parts:
        return "uv tool"
    if "Cellar" in parts or root_s.startswith("/opt/homebrew/"):
        return "homebrew"
    if root.name in {".venv", "venv"} or (root / "pyvenv.cfg").is_file():
        return "venv"
    return "unknown"


def _runtime_install_report(cwd: Path | None = None) -> dict[str, Any]:
    """Describe how the active `memo`/`memo-mcp` installation is wired.

    The operationally safe shape is an isolated tool install (pipx, uv tool,
    Homebrew). A project-local `.venv` works for development, but using it as
    the MCP runtime couples memory state and MLX deps to whichever repo had
    the venv active when the client was configured.
    """
    cwd = _safe_resolve(cwd or Path.cwd())
    memo_cmd, memo_resolved = _resolve_command("memo")
    mcp_cmd, mcp_resolved = _resolve_command("memo-mcp")
    py_resolved = _safe_resolve(Path(sys.executable))

    memo_root = _env_root_for_bin(memo_resolved)
    mcp_root = _env_root_for_bin(mcp_resolved)
    py_root = _env_root_for_bin(py_resolved)
    primary_root = memo_root or mcp_root or py_root
    mode = _install_mode(primary_root)

    warnings: list[str] = []
    if memo_resolved is None:
        warnings.append("`memo` is not on PATH")
    if mcp_resolved is None:
        warnings.append("`memo-mcp` is not on PATH; MCP clients cannot start it")
    if memo_root is not None and mcp_root is not None and memo_root != mcp_root:
        warnings.append(
            "`memo` and `memo-mcp` resolve to different environments; "
            "reinstall with `pipx install --force mlx-memo` or `uv tool install --force mlx-memo`"
        )
    if mode == "venv" and primary_root is not None:
        if _path_is_relative_to(primary_root, cwd):
            warnings.append(
                f"running from project venv {primary_root}; prefer an isolated "
                "tool install so MCP is not tied to this repo"
            )
        else:
            warnings.append(
                f"running from venv {primary_root}; verify this is memo's own "
                "dedicated environment, not another project's venv"
            )
    elif mode == "unknown":
        warnings.append(
            "install mode is unknown; recommended: `pipx install mlx-memo` "
            "or `uv tool install mlx-memo`"
        )

    return {
        "mode": mode,
        "root": str(primary_root) if primary_root else None,
        "memo_cmd": str(memo_cmd) if memo_cmd else None,
        "memo_resolved": str(memo_resolved) if memo_resolved else None,
        "mcp_cmd": str(mcp_cmd) if mcp_cmd else None,
        "mcp_resolved": str(mcp_resolved) if mcp_resolved else None,
        "python": str(py_resolved),
        "warnings": warnings,
    }


def _print_runtime_install_report(report: dict[str, Any]) -> None:
    mode = report["mode"]
    root = report.get("root") or "(unknown)"
    if report["warnings"]:
        console.print(f"[yellow]![/yellow] install mode: {mode}  [dim]{root}[/dim]")
    else:
        console.print(f"[green]✓[/green] install mode: {mode}  [dim]{root}[/dim]")

    for key, label in (
        ("memo_cmd", "memo"),
        ("mcp_cmd", "memo-mcp"),
    ):
        raw = report.get(key)
        resolved = report.get(key.replace("_cmd", "_resolved"))
        if raw and resolved and raw != resolved:
            console.print(f"[dim]{label:14s}[/dim] {raw} -> {resolved}")
        elif raw:
            console.print(f"[dim]{label:14s}[/dim] {raw}")
        else:
            console.print(f"[dim]{label:14s}[/dim] (not found)")
    console.print(f"[dim]{'python':14s}[/dim] {report['python']}")
    for warning in report["warnings"]:
        console.print(f"[yellow]![/yellow] {warning}")


def _resolved_memo_mcp() -> Path | None:
    _, resolved = _resolve_command("memo-mcp")
    if resolved is not None:
        return resolved
    fallback = _safe_resolve(Path(sys.executable)).with_name("memo-mcp")
    return fallback if fallback.exists() else None


@cli.command(name="init")
@click.option("--force", is_flag=True, help="Overwrite existing config without confirmation.")
def init_cmd(force: bool) -> None:
    """(Re)configure where memo stores memorias.

    Runs the interactive picker. On first run, the picker also fires
    automatically — `memo init` is for explicitly re-configuring later
    (e.g. moving to a new path, switching to/from an Obsidian vault).
    """
    from memo.setup.config_io import _resolve_config_path

    cfg_path = _resolve_config_path()
    if cfg_path.is_file() and not force and not click.confirm(
        f"Config file exists at {cfg_path}. Overwrite?",
        default=False,
    ):
        console.print("[yellow]aborted[/yellow]")
        return
    _run_picker_and_save()


@cli.command(name="migrate-vault")
@click.argument("new_data_dir", required=False, type=click.Path(file_okay=False, resolve_path=True))
@click.option("--from", "from_dir", default=None,
              type=click.Path(exists=True, file_okay=False, resolve_path=True),
              help="Source memory_dir. Defaults to current cfg.memory_dir.")
@click.option("--force", is_flag=True, help="Overwrite destination even if non-empty.")
@click.option("--yes", is_flag=True, help="Skip confirmation.")
def migrate_vault(
    new_data_dir: str | None, from_dir: str | None, force: bool, yes: bool,
) -> None:
    """Move memorias to a new data_dir; rewrites config + reindexes.

    Copies all `.md` files (preserving mtime via `shutil.copy2`),
    updates `~/.config/memo/config.toml` with the new `data_dir`,
    deletes `memvec.db`, and runs `memo reindex` from the new location.

    The original `.md` files are NOT deleted — once you've verified the
    migration with `memo search`, you can `rm -rf <old-dir>` manually.
    History DB is preserved (it's append-only audit; old paths in it
    just become historical references).
    """
    import shutil
    from pathlib import Path as _Path

    from memo.memory import Memory

    cfg = Config.from_env()

    # 1. Resolve source.
    src = _Path(from_dir).resolve() if from_dir else cfg.memory_dir
    if not src.is_dir():
        console.print(f"[red]✗[/red] source dir does not exist: {src}")
        sys.exit(1)

    # 2. Resolve destination + (optional) new vault_path.
    if new_data_dir:
        dst = _Path(new_data_dir).resolve()
        chosen_vault = cfg.vault_path
    else:
        # No arg → run the picker so user can pick (incl. an Obsidian vault).
        try:
            result = run_picker()
        except KeyboardInterrupt:
            console.print("[yellow]aborted[/yellow]")
            sys.exit(130)
        dst = result.data_dir
        chosen_vault = result.vault_path

    if dst == src:
        console.print(f"[red]✗[/red] source and destination are the same: {src}")
        sys.exit(1)

    md_files = sorted(src.rglob("*.md"))
    if dst.exists() and any(dst.iterdir()) and not force:
        console.print(
            f"[red]✗[/red] destination is non-empty: {dst}\n"
            "  Use --force to overwrite.",
        )
        sys.exit(1)

    if not yes:
        click.confirm(
            f"Copy {len(md_files)} memorias from\n  {src}\n→ {dst}\n"
            "and rebuild memvec.db. Source files will be left in place. "
            "Proceed?",
            abort=True,
        )

    # 3. Copy files (preserving mtime).
    dst.mkdir(parents=True, exist_ok=True)
    n_copied = 0
    for md in md_files:
        rel = md.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(md, target)
        n_copied += 1
    console.print(f"[green]✓[/green] copied {n_copied} files → {dst}")

    # 4. Update config + drop stale DB.
    cfg_path = write_config_file(data_dir=dst, vault_path=chosen_vault)
    console.print(f"[green]✓[/green] config: {cfg_path}")
    if cfg.db_path.is_file():
        cfg.db_path.unlink()
        console.print("[green]✓[/green] removed stale memvec.db")

    # 5. Reindex from new location. Re-build Config so from_env picks up
    # the freshly-written file (env vars / explicit kwargs cleared).
    new_cfg = Config.from_env()
    mem = Memory(new_cfg)
    counts = mem.reindex()
    console.print(
        f"[green]✓[/green] reindex: checked {counts['checked']}  "
        f"added {counts['added']}  reindexed {counts['reindexed']}  "
        f"skipped {counts['skipped']}",
    )
    console.print(
        f"\n[dim]Source files at {src} were left untouched. "
        "After verifying the migration with `memo search`, you can rm them.[/dim]",
    )


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
@click.option("--no-project-tag", "no_project_tag", is_flag=True,
              help="Skip the auto `project:<repo>` tag derived from the current git toplevel.")
@click.option("--defer-embed", is_flag=True,
              help="Save markdown + BM25 index only; run `memo reindex` later for semantic search.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of a panel.")
def save(content: str, title: str | None, type_: str, tags: tuple[str, ...],
         auto_derive: bool, no_project_tag: bool, defer_embed: bool, as_json: bool) -> None:
    """Persist CONTENT to the vault + index. Pass `-` to read CONTENT from stdin."""
    from memo.memory import Memory

    if content == "-":
        content = sys.stdin.read()
    mem = Memory(Config.from_env())
    rec = mem.save(content=content, title=title, type_=type_,
                   tags=list(tags), auto_derive=auto_derive,
                   auto_project=not no_project_tag,
                   defer_embed=defer_embed)
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
            delta = ", ".join(f"{k}" for k in r["delta"])
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
        # 1) Memory .md files (relative to memory_dir).
        if cfg.memory_dir.is_dir():
            for md in sorted(cfg.memory_dir.rglob("*.md")):
                rel = md.relative_to(cfg.memory_dir)
                zf.write(md, arcname=f"memory/{rel}")
                n_md += 1
        # 2) State DBs (vec + history). Stored at the root.
        for db in (cfg.db_path, cfg.history_db):
            if db.is_file():
                zf.write(db, arcname=f"state/{db.name}")
        # 3) Manifest with paths so restore can sanity-check.
        manifest = {
            "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "data_dir": str(cfg.data_dir),
            "vault_path": str(cfg.vault_path) if cfg.vault_path else None,
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
                f"Extract into {cfg.data_dir} + {cfg.state_dir}? "
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
                dest = cfg.data_dir / rel
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
        f"into {cfg.data_dir}",
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
        "data_dir": str(mem.cfg.data_dir),
        "vault_path": str(mem.cfg.vault_path) if mem.cfg.vault_path else "(unset)",
        "db_path": str(mem.cfg.db_path),
        "model_profile": mem.cfg.model_profile,
        "embedder_model": mem.cfg.embedder_model,
        "llm_model": mem.cfg.llm_model,
    }
    for k, v in info.items():
        console.print(f"[dim]{k:14s}[/dim] {v}")


@cli.command()
@click.option("--gc", "do_gc", is_flag=True, help="Detect orphans between store and disk.")
@click.option("--fix", is_flag=True, help="With --gc: drop orphan store rows. .md files are never deleted automatically.")
@click.option(
    "--strict-runtime",
    is_flag=True,
    help="Exit non-zero if memo/memo-mcp are not running from an isolated tool install.",
)
def doctor(do_gc: bool, fix: bool, strict_runtime: bool) -> None:
    """Self-check: vault present, sqlite-vec loadable, MLX importable, models in cache.

    `--gc` reports orphans (store rows whose `.md` is gone, `.md` files
    whose `id` isn't in the store). `--gc --fix` removes orphan store
    rows; orphan `.md` files are listed but never deleted automatically.
    """
    cfg = Config.from_env()
    ok = True

    runtime_report = _runtime_install_report()
    _print_runtime_install_report(runtime_report)
    if strict_runtime and runtime_report["warnings"]:
        ok = False

    # 1. Data dir (memorias)
    if cfg.data_dir.is_dir():
        console.print(f"[green]✓[/green] data_dir: {cfg.data_dir}")
    else:
        # Data dir is auto-created by `ensure_dirs()`; missing here means
        # something went wrong with permissions.
        console.print(f"[red]✗[/red] data_dir missing: {cfg.data_dir}")
        ok = False
    # Optional vault_path (only relevant for `memo ingest`).
    if cfg.vault_path is not None:
        if cfg.vault_path.is_dir():
            console.print(f"[green]✓[/green] vault_path: {cfg.vault_path}")
        else:
            console.print(f"[yellow]![/yellow] vault_path set but missing: {cfg.vault_path}")

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


@cli.command(name="mcp-command")
@click.option(
    "--client",
    type=click.Choice(["claude-code", "json"]),
    default="claude-code",
    show_default=True,
    help="Emit a Claude Code command or raw MCP JSON config.",
)
def mcp_command(client: str) -> None:
    """Print MCP config pinned to the resolved `memo-mcp` executable.

    This avoids accidentally registering a `memo-mcp` from another active
    project venv. Pair with `memo doctor --strict-runtime` when debugging a
    client that starts the wrong server.
    """
    memo_mcp = _resolved_memo_mcp()
    if memo_mcp is None:
        console.print(
            "[red]memo-mcp not found.[/red] Install memo as an isolated tool: "
            "`pipx install mlx-memo` or `uv tool install mlx-memo`.",
        )
        sys.exit(1)
    if client == "json":
        click.echo(json.dumps({
            "mcpServers": {
                "memo": {
                    "type": "stdio",
                    "command": str(memo_mcp),
                    "args": [],
                    "env": {},
                },
            },
        }, ensure_ascii=False, indent=2))
        return
    click.echo(f"claude mcp add memo -s user {shlex.quote(str(memo_mcp))}")


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
    `prompt` field via the MLX embedder, runs search, and outputs
    the top-k results as `additionalContext` in `hookSpecificOutput`.

    Configure via env vars (all optional, sensible defaults for v0.3.x):

      MEMO_RECALL_DISABLE          — set to "1" to make this a no-op.
      MEMO_RECALL_TOP_K            — default 3
      MEMO_RECALL_MIN_SIM          — default 0.6. Note: the floor is
        absolute over `score`. For mode=vec, `score` is cosine ∈ [0, 1].
        For mode=hybrid+rerank, `score` is fused (typically [0.2, 0.95])
        — drop the floor to ~0.4 there or hits get over-filtered.
      MEMO_RECALL_MIN_PROMPT_CHARS — default 12 (skip very short prompts)
      MEMO_RECALL_BODY_CHARS       — default 240 (snippet length per result)
      MEMO_RECALL_SKIP_SLASH       — default "1" (skip if prompt starts with /)
      MEMO_RECALL_MODE             — default "vec". "hybrid" enables
        cross-encoder rerank on top of RRF fusion. Higher precision,
        higher latency (≤5s warm with the auto-capped pool).
      MEMO_RECALL_RERANK_INPUT_K   — default 10 (only used when MODE=hybrid).
        How many fused candidates to feed the reranker; lower for tighter
        latency, higher for better recall on diffuse queries.

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
    token_budget = int(os.environ.get("MEMO_RECALL_TOKEN_BUDGET", "0") or 0)
    project_boost = float(os.environ.get("MEMO_RECALL_PROJECT_BOOST", "0.15"))

    # Read cwd from the hook payload (Claude Code passes it) so we can
    # derive the project tag the user is currently working under.
    payload_cwd = payload.get("cwd")

    # Suppress HF download progress bars on stderr — they'd contaminate
    # the hook's debug output and confuse users tailing logs. The model
    # is already downloaded for any working memo install; this only
    # silences first-run cache-check noise.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    # Defer the heavy import — only paid if we get past the early-exits.
    #
    # Mode selection:
    # - `vec` (default): pure cosine similarity. `score` is in [0, 1]
    #   and the `MEMO_RECALL_MIN_SIM` floor (0.6) cuts noise reliably.
    # - `hybrid`: reciprocal rank fusion + cross-encoder rerank when
    #   the reranker is enabled in config. Higher precision but spends
    #   the full hook budget on inference. Auto-shrinks the rerank
    #   input pool (`MEMO_RECALL_RERANK_INPUT_K`, default 10) so the
    #   warm latency stays under the 5s hook timeout.
    # - `bm25`: keyword-only, useful for queries with literal tag
    #   names or filenames where the embedder under-recalls.
    mode = os.environ.get("MEMO_RECALL_MODE", "vec")
    if mode == "hybrid":
        # Hook latency budget is tight: cap the rerank pool unless the
        # user explicitly set MEMO_RERANK_INPUT_K elsewhere. Setdefault
        # respects an upstream override (CI bench, custom shell rc).
        os.environ.setdefault(
            "MEMO_RERANK_INPUT_K",
            os.environ.get("MEMO_RECALL_RERANK_INPUT_K", "10"),
        )
    # Widen the pool when a project boost is active — we need enough
    # candidates so that off-project hits can be re-ranked below
    # on-project ones without starving the final top_k.
    project_tag = None
    if project_boost > 0:
        try:
            from memo.project import current_project_tag
            project_tag = current_project_tag(payload_cwd)
        except Exception:
            project_tag = None
    search_k = top_k * 3 if project_tag else top_k
    try:
        from memo.memory import Memory
        mem = Memory(Config.from_env())
        hits = mem.search(prompt, limit=search_k, mode=mode)
    except Exception as exc:
        _bail(f"search failed: {exc}")
        return

    # Apply project boost — additive on the raw score, then re-sort.
    if project_tag:
        for h in hits:
            if h.score is not None and project_tag in (h.tags or []):
                h.score = h.score + project_boost
        hits.sort(key=lambda h: (h.score or 0.0), reverse=True)
    # Trim back to top_k after boost-aware re-sort.
    hits = hits[:top_k]

    # Filter by similarity floor. With mode="vec", `score` is cosine
    # similarity ∈ [-1, 1] (typically [0, 1] for L2-normalised embeddings).
    # 0.6 is the empirical confidence floor on the 223-doc corpus:
    #   - "qué decidí sobre MLX vs Ollama" → 3 hits @ 0.71-0.74 (all relevant)
    #   - "how to bake apple pie" → 3 hits @ 0.51-0.56 (literal-word noise,
    #     "apple-mcp" memoria matched). Threshold 0.6 cuts these out.
    # Tune via MEMO_RECALL_MIN_SIM if your corpus has different density.
    relevant = [h for h in hits if h.score is None or h.score >= min_sim]

    # Telemetry: append every recall (with or without hits) to the
    # JSONL ring buffer consumed by `memo tui`. Best-effort; failures
    # are swallowed inside the helper.
    try:
        from memo.dashboard import append_recall_log
        append_recall_log(
            Config.from_env().state_dir,
            prompt=prompt,
            hits=[{"id": h.id, "score": h.score, "title": h.title} for h in relevant],
        )
    except Exception:
        pass

    if not relevant:
        _bail(f"no hits above min_sim={min_sim}")
        return

    # Format as markdown additionalContext. Be terse — context budget is
    # capped at 10k chars by Claude Code; we want each prompt to inject
    # ~500-1500 chars at most so the user's actual prompt isn't drowned.
    #
    # If MEMO_RECALL_TOKEN_BUDGET is set, pack memorias greedily by
    # score until the budget is met. Token estimate is 1 token ≈ 4 chars
    # (English/Spanish prose); good-enough rule-of-thumb that avoids a
    # tiktoken dep. Last memoria gets head-truncated to fit instead of
    # being dropped wholesale.
    header = "## Relevant memories from your past (memo)"
    footer = "_Use `/memo:memo get <id>` to see full content._"
    lines = [header, ""]
    used_chars = 0  # chars of formatted block body, excluding header/footer

    def _est_tokens(s: str) -> int:
        return max(1, len(s) // 4)

    budget_chars = token_budget * 4 if token_budget > 0 else None

    for h in relevant:
        score_tag = f" (score {h.score:.2f})" if h.score is not None else ""
        body = (h.body or "").strip().replace("\n", " ")
        if len(body) > body_chars:
            body = body[:body_chars].rstrip() + "…"
        block_lines = [f"**[{h.id[:8]}] {h.title}**{score_tag}"]
        if h.tags:
            block_lines.append(f"_tags_: {', '.join(h.tags)}")
        if body:
            block_lines.append(f"> {body}")
        block_lines.append("")
        block = "\n".join(block_lines)

        if budget_chars is None:
            lines.extend(block_lines)
            continue

        remaining = budget_chars - used_chars
        if remaining <= 0:
            break
        if len(block) <= remaining:
            lines.extend(block_lines)
            used_chars += len(block)
        else:
            # Truncate the body in this final block to fit the budget.
            if body:
                # Reserve space for header line + tags + closing "…"
                head_len = len(block_lines[0]) + 1
                tags_len = (len(block_lines[1]) + 1) if h.tags else 0
                avail = max(0, remaining - head_len - tags_len - 3)
                if avail > 20:
                    trunc_body = body[:avail].rstrip() + "…"
                    block_lines[-2 if h.tags else -1] = f"> {trunc_body}"
                    lines.extend(block_lines)
            break

    lines.append(footer)
    if token_budget > 0 and os.environ.get("MEMO_RECALL_DEBUG") == "1":
        approx = _est_tokens("\n".join(lines))
        print(f"# memo recall-hook: ~{approx} tokens (budget {token_budget})", file=_sys.stderr)

    output = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": "\n".join(lines),
        }
    }
    print(_json.dumps(output, ensure_ascii=False))
    _sys.exit(0)


@cli.group(name="as-of")
def as_of_group() -> None:
    """Time-machine — query the corpus as it existed at any past date.

    Subcommands: `search`, `ask`, `list`. All take `--date YYYY-MM-DD`
    (or a full ISO timestamp). The snapshot is reconstructed by
    replaying `history.db` events in reverse from "now".
    """
    pass


def _parse_as_of_date(s: str) -> str:
    """Accept date-only (`2026-03-01`) or full ISO. Return ISO with
    a stable noon-UTC anchor for date-only inputs."""
    from datetime import UTC
    from datetime import datetime as _dt
    s = s.strip()
    if len(s) == 10:  # YYYY-MM-DD
        return f"{s}T23:59:59+00:00"  # end-of-day to be inclusive
    try:
        dt = _dt.fromisoformat(s.rstrip("Z"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.isoformat()
    except ValueError as exc:
        raise click.ClickException(
            f"Could not parse --date {s!r}. Use YYYY-MM-DD or ISO 8601.",
        ) from exc


@as_of_group.command(name="search")
@click.argument("query")
@click.option("--date", "as_of", required=True, help="YYYY-MM-DD or full ISO 8601.")
@click.option("--limit", default=10, type=int, show_default=True)
@click.option("--type", "type_", default=None, help="Filter by record type.")
@click.option("--mode", default="hybrid",
              type=click.Choice(["hybrid", "vec", "bm25"]), show_default=True)
@click.option("--json", "as_json", is_flag=True)
def as_of_search(
    query: str, as_of: str, limit: int, type_: str | None, mode: str, as_json: bool,
) -> None:
    """Search the corpus as it existed on a past date."""
    from memo.memory import Memory
    from memo.time_machine import reconstruct

    mem = Memory(Config.from_env())
    snap = reconstruct(mem, as_of=_parse_as_of_date(as_of))
    hits = snap.search(query, limit=limit, mode=mode)
    if type_:
        hits = [h for h in hits if h.type == type_]

    if as_json:
        click.echo(json.dumps(
            {
                "as_of": snap.as_of.isoformat(),
                "snapshot_size": len(snap),
                "results": [h.to_dict() for h in hits],
            },
            ensure_ascii=False, indent=2,
        ))
        return

    if not hits:
        console.print(f"[dim]no results in snapshot @ {snap.as_of.date().isoformat()}[/dim]")
        return
    tbl = Table(show_lines=False, expand=True,
                title=f"snapshot @ {snap.as_of.date().isoformat()} · {len(snap)} memorias existían")
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


@as_of_group.command(name="ask")
@click.argument("question")
@click.option("--date", "as_of", required=True, help="YYYY-MM-DD or full ISO 8601.")
@click.option("--k", default=5, type=int, show_default=True)
@click.option("--json", "as_json", is_flag=True)
def as_of_ask(question: str, as_of: str, k: int, as_json: bool) -> None:
    """RAG question against a past snapshot of the corpus."""
    from memo.memory import Memory
    from memo.time_machine import reconstruct

    mem = Memory(Config.from_env())
    snap = reconstruct(mem, as_of=_parse_as_of_date(as_of))
    out = snap.ask(question, k=k)

    if as_json:
        click.echo(json.dumps(out, ensure_ascii=False, indent=2))
        return

    console.print(Panel.fit(
        out["answer"] or "[dim](sin respuesta)[/dim]",
        title=f"✓ as-of {snap.as_of.date().isoformat()} ({len(snap)} memorias in scope)",
        border_style="magenta",
    ))
    if out.get("sources"):
        console.print("\n[dim]sources:[/dim]")
        for s in out["sources"]:
            console.print(f"  [bold]{s['id_short']}[/bold]  {s['title']}  [dim]({s['type']})[/dim]")


@as_of_group.command(name="list")
@click.option("--date", "as_of", required=True, help="YYYY-MM-DD or full ISO 8601.")
@click.option("--type", "type_", default=None)
@click.option("--limit", default=20, type=int, show_default=True)
@click.option("--json", "as_json", is_flag=True)
def as_of_list(as_of: str, type_: str | None, limit: int, as_json: bool) -> None:
    """List memorias that existed in a past snapshot (most-recent first)."""
    from memo.memory import Memory
    from memo.time_machine import reconstruct

    mem = Memory(Config.from_env())
    snap = reconstruct(mem, as_of=_parse_as_of_date(as_of))
    rows = snap.list(type_=type_)[:limit]

    if as_json:
        click.echo(json.dumps(
            {
                "as_of": snap.as_of.isoformat(),
                "snapshot_size": len(snap),
                "records": [
                    {"id": r.id, "title": r.title, "type": r.type, "tags": r.tags,
                     "updated": r.updated}
                    for r in rows
                ],
            },
            ensure_ascii=False, indent=2,
        ))
        return

    if not rows:
        console.print(f"[dim]empty snapshot @ {snap.as_of.date().isoformat()}[/dim]")
        return
    tbl = Table(show_lines=False, expand=True,
                title=f"snapshot @ {snap.as_of.date().isoformat()} · {len(snap)} memorias")
    tbl.add_column("id", width=10)
    tbl.add_column("type", width=10)
    tbl.add_column("title", overflow="fold")
    tbl.add_column("updated", width=12)
    for r in rows:
        tbl.add_row(r.id[:8], r.type, r.title, (r.updated or "—")[:10])
    console.print(tbl)


@cli.command(name="diff")
@click.option("--from", "from_date", required=True,
              help="Start date — YYYY-MM-DD or full ISO 8601.")
@click.option("--to", "to_date", required=False, default=None,
              help="End date (default: now).")
@click.option("--json", "as_json", is_flag=True)
def diff_cmd(from_date: str, to_date: str | None, as_json: bool) -> None:
    """Diff the corpus between two snapshots.

    Shows added / removed / updated memorias plus a summary line. Useful
    for "what changed since last Monday" or "what evolved between two
    releases".
    """
    from datetime import UTC
    from datetime import datetime as _dt

    from memo.memory import Memory
    from memo.time_machine import diff as _diff

    to_iso = _dt.now(UTC).isoformat() if to_date is None else _parse_as_of_date(to_date)
    from_iso = _parse_as_of_date(from_date)

    mem = Memory(Config.from_env())
    d = _diff(mem, from_ts=from_iso, to_ts=to_iso)

    if as_json:
        click.echo(json.dumps({
            "from_ts": d.from_ts.isoformat(),
            "to_ts": d.to_ts.isoformat(),
            "added": [{"id": r.id, "title": r.title, "type": r.type} for r in d.added],
            "removed": [{"id": r.id, "title": r.title, "type": r.type} for r in d.removed],
            "updated": d.updated,
        }, ensure_ascii=False, indent=2))
        return

    console.print(Panel.fit(
        f"{d.from_ts.date().isoformat()}  →  {d.to_ts.date().isoformat()}\n"
        f"[bold]{d.summary()}[/bold]",
        title="corpus diff",
        border_style="cyan",
    ))
    if d.added:
        console.print(f"\n[green]+ added ({len(d.added)})[/green]")
        for r in d.added[:20]:
            console.print(f"  [green]+[/green] [{r.id[:8]}] {r.title}  [dim]({r.type})[/dim]")
    if d.removed:
        console.print(f"\n[red]- removed ({len(d.removed)})[/red]")
        for r in d.removed[:20]:
            console.print(f"  [red]-[/red] [{r.id[:8]}] {r.title}  [dim]({r.type})[/dim]")
    if d.updated:
        console.print(f"\n[yellow]~ updated ({len(d.updated)})[/yellow]")
        for u in d.updated[:20]:
            console.print(
                f"  [yellow]~[/yellow] [{u['id'][:8]}] {u['title']}  "
                f"[dim](fields: {', '.join(u['changed_fields'])})[/dim]",
            )


@cli.command(name="tui")
@click.option("--refresh", type=float, default=1.0, show_default=True,
              help="Refresh interval in seconds.")
@click.option("--no-clear", is_flag=True,
              help="Don't take over the terminal screen — render inline (handy for tmux/screen).")
def tui(refresh: float, no_clear: bool) -> None:
    """Live terminal dashboard — corpus stats, recent saves/recalls, MLX warm-state,
    watcher status, top tags, 14-day sparklines. Ctrl+C to exit.

    Reads from the existing `history.db` (saves) and a JSONL recall log
    written by `memo recall-hook`. Read-only — does not modify the
    corpus.
    """
    from memo.dashboard import run_tui

    run_tui(refresh=refresh, no_clear=no_clear)


@cli.command(name="watch")
@click.option("--delay", default=2.0, type=float, show_default=True,
              help="Debounce window in seconds — coalesces bursts of edits into one reindex.")
@click.option("--debug", is_flag=True, help="Print every reindex result to stderr.")
def watch(delay: float, debug: bool) -> None:
    """Auto-reindex on `.md` change. Foreground; Ctrl+C to stop.

    Watches `cfg.memory_dir` recursively. Files saved in Obsidian (or
    any editor) trigger a debounced `Memory.reindex()` call so the
    sqlite-vec index stays in sync without manual `memo reindex` runs.

    Run as a daemon via `memo install-watcher` (launchd plist).
    """
    from memo.watcher import run_watcher

    run_watcher(delay=delay, debug=debug)


@cli.command(name="install-watcher")
@click.option("--bin", "memo_bin", default=None,
              help="Absolute path to the `memo` binary (default: auto-detect via shutil.which).")
@click.option("--no-load", is_flag=True,
              help="Write the plist but don't `launchctl bootstrap` it.")
def install_watcher(memo_bin: str | None, no_load: bool) -> None:
    """Install + load the file-watcher as a launchd daemon.

    Generates `~/Library/LaunchAgents/com.memo.watch.plist`, loads
    it via `launchctl bootstrap`, and verifies it's running. Restart on
    crash is enabled (`KeepAlive=true`). Logs land in
    `~/Library/Logs/memo/`.
    """
    import shutil as _shutil
    import subprocess

    from memo.watcher import _PLIST_LABEL, install_plist

    if memo_bin is None:
        memo_bin = _shutil.which("memo") or ""
        if not memo_bin:
            raise click.ClickException(
                "Could not locate `memo` on PATH. Pass --bin /abs/path/to/memo.",
            )

    plist_path = install_plist(memo_bin)
    console.print(f"[dim]wrote:[/dim] {plist_path}")

    if no_load:
        console.print(
            "[yellow]Skipped load (--no-load). To activate manually:[/yellow]\n"
            f"  launchctl bootstrap gui/$(id -u) {plist_path}",
        )
        return

    uid = os.getuid()
    domain = f"gui/{uid}"
    target = f"{domain}/{_PLIST_LABEL}"

    # Unload first if already present, to pick up plist changes.
    subprocess.run(
        ["launchctl", "bootout", target],
        check=False, capture_output=True,
    )
    res = subprocess.run(
        ["launchctl", "bootstrap", domain, str(plist_path)],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        raise click.ClickException(
            f"launchctl bootstrap failed: {res.stderr.strip() or res.stdout.strip()}",
        )

    # Verify.
    verify = subprocess.run(
        ["launchctl", "print", target],
        capture_output=True, text=True,
    )
    if verify.returncode != 0:
        raise click.ClickException(
            "Plist loaded but `launchctl print` could not find it. "
            "Inspect `~/Library/Logs/memo/watch.err.log`.",
        )

    console.print(Panel.fit(
        f"[bold]watcher loaded[/bold]\n"
        f"[dim]label:[/dim] {_PLIST_LABEL}\n"
        f"[dim]plist:[/dim] {plist_path}\n"
        f"[dim]logs:[/dim] ~/Library/Logs/memo/watch.{{out,err}}.log",
        title="✓ install-watcher",
        border_style="green",
    ))


@cli.command(name="uninstall-watcher")
def uninstall_watcher_cmd() -> None:
    """Unload + remove the file-watcher launchd job."""
    import subprocess

    from memo.watcher import _PLIST_LABEL, uninstall_plist

    uid = os.getuid()
    target = f"gui/{uid}/{_PLIST_LABEL}"
    subprocess.run(
        ["launchctl", "bootout", target],
        check=False, capture_output=True,
    )
    existed = uninstall_plist()
    if existed:
        console.print("[green]✓ watcher uninstalled.[/green]")
    else:
        console.print("[yellow]No plist found to remove.[/yellow]")


@cli.command(name="mine-history")
@click.option("--path", "root_path", default=None,
              help="Transcripts root (default: ~/.claude/projects).")
@click.option("--since", "since_days", type=int, default=None,
              help="Only process transcripts modified in the last N days.")
@click.option("--limit", "file_limit", type=int, default=None,
              help="Cap on number of transcripts to process (newest first).")
@click.option("--dry-run", is_flag=True,
              help="Walk + extract, don't save. Useful for cost estimation.")
@click.option("--debug", is_flag=True, help="Print per-file/per-candidate info to stderr.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON summary instead of a panel.")
def mine_history(
    root_path: str | None, since_days: int | None, file_limit: int | None,
    dry_run: bool, debug: bool, as_json: bool,
) -> None:
    """Mine past Claude Code conversations for actionable insights.

    Walks `~/.claude/projects/<hash>/*.jsonl`, runs the same prefilter +
    helper-LLM extraction + embedding-based dedup as the live capture
    hook, and saves what's new. Resumable: per-file processed-line
    counts are tracked under `~/.local/share/memo/mine-history.json`.

    Tips:
        - First run on a long history is slow (helper LLM is the bottleneck).
          Use `--limit 10 --since 30` to start with the freshest sessions.
        - `--dry-run` reports candidate counts without writing.
    """
    from pathlib import Path as _Path

    from memo.transcript_miner import mine_transcripts

    root = _Path(root_path).expanduser() if root_path else None

    console_progress = None
    if not as_json:
        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
        )
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        )
        progress.start()
        task = progress.add_task("mining transcripts", total=None)

        def cb(idx: int, total: int, p: _Path) -> None:
            progress.update(
                task, total=total, completed=idx, description=f"[{idx + 1}/{total}] {p.name}",
            )

        console_progress = (progress, task, cb)

    try:
        summary = mine_transcripts(
            root=root, since_days=since_days, file_limit=file_limit,
            dry_run=dry_run, debug=debug,
            progress_cb=console_progress[2] if console_progress else None,
        )
    finally:
        if console_progress:
            console_progress[0].stop()

    if as_json:
        click.echo(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    status = summary.get("status")
    if status == "no_files":
        console.print(f"[yellow]No transcripts found under {summary['root']}.[/yellow]")
        return

    saved = summary.get("saved", [])
    body = (
        f"[dim]root:[/dim] {summary['root']}\n"
        f"[dim]files:[/dim] {summary['files_processed']}/{summary['files_total']} processed"
        f" ([dim]{summary['files_skipped']} skipped — already mined[/dim])\n"
        f"[dim]candidates:[/dim] {summary['candidates']}\n"
        f"[bold green]saved:[/bold green] {len(saved)}"
        f"{' [yellow](dry-run)[/yellow]' if summary['dry_run'] else ''}\n"
        f"[dim]skipped duplicates:[/dim] {summary['skipped_dup']}"
    )
    console.print(Panel.fit(body, title="✓ mine-history", border_style="green"))


@cli.command(name="ingest")
@click.argument("vault_path", type=click.Path(exists=True, file_okay=False, resolve_path=True))
@click.option("--name", default=None, help="Vault label (default: dirname). Used as path prefix in store.")
@click.option("--force", is_flag=True, help="Re-embed even if body unchanged.")
@click.option("--dry-run", is_flag=True, help="Walk + report counts, don't embed/write.")
@click.option("--exclude", multiple=True, help="Glob to exclude (relative to vault). Repeat. Default: .obsidian/.git/.trash/.makemd/.smart-env/.space/99-obsidian/")
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
    etc.) and memo's own memory subdir (`99-obsidian/`)
    so we don't double-index curated memorias.
    """
    import hashlib
    from datetime import datetime
    from pathlib import Path

    import frontmatter

    from memo.embedder import MLXEmbedder
    from memo.store import VecStore

    cfg = Config.from_env()
    cfg.ensure_dirs()

    vault = Path(vault_path).resolve()
    # `cfg.vault_path` is the user's "primary" Obsidian vault (set via
    # `memo init`'s Obsidian branch, or `MEMO_VAULT_PATH`). When we're
    # ingesting that exact vault, paths are stored without a label
    # prefix (e.g. `01-Projects/foo.md`); external vaults get a
    # `<label>/` prefix so multiple vaults coexist in one store.
    # For users who haven't set `cfg.vault_path` (non-Obsidian
    # workflows), every ingest is treated as external.
    is_principal_vault = cfg.vault_path is not None and vault == cfg.vault_path
    label = "" if is_principal_vault else (name or vault.name)

    # Default exclusions — Obsidian dotdirs + memo's own memory subdir
    # to avoid double-indexing the curated memorias managed by reindex.
    default_excludes = (
        ".obsidian", ".git", ".trash", ".makemd", ".smart-env", ".space",
        ".claude", ".devin", "99-obsidian",
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

    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

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
                #
                # EXCEPTION — high-signal short notes. Some short notes
                # exist precisely BECAUSE they pin an atomic fact: a
                # payment URL, a CBU, a one-off shell command, an API
                # endpoint. Discarding them by char count loses notes
                # the user explicitly created as quick-lookup pins. We
                # detect them via _is_high_signal and bypass the
                # threshold. The same loud-fail guards still apply
                # downstream (dim + norm asserts), so adding these
                # doesn't open the door to malformed embeddings.
                import os as _os_min  # local import — avoids tedious refactor
                min_chars = int(_os_min.environ.get("MEMO_INGEST_MIN_CHARS", "200"))
                if len(body) < min_chars and not _is_high_signal(body, fm.metadata.get("tags")):
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

                now = datetime.now(UTC).isoformat()
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
            except Exception as exc:
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


_HIGH_SIGNAL_TAGS = frozenset({
    # Notes pinned to lookup-style facts. Lowercase compare; surface
    # forms like "Link" / "LINKS" / "Pago" all match. Spanish + English
    # variants because the vault mixes both.
    "link", "links", "url", "urls",
    "dato", "datos", "data",
    "ref", "refs", "referencia", "referencias", "reference",
    "comando", "comandos", "command", "commands", "cmd", "snippet",
    "pago", "pagos", "payment",
    "credencial", "credenciales", "credential", "credentials",
    "endpoint", "endpoints", "api",
    "telefono", "teléfono", "phone", "tel",
    "cbu", "alias", "iban",
})

# Match http(s):// URLs — anchored end on whitespace, ), >, ], or "
# (common markdown wrappers). Permissive enough to catch trailing
# punctuation cases without dragging adjacent text in.
_URL_RE = re.compile(r"https?://[^\s)>\]\"]+")


def _is_high_signal(body: str, fm_tags: Any) -> bool:
    """Short notes worth indexing despite being below MIN_CHARS.

    A note is high-signal if any of:
    - frontmatter tags include `link` / `dato` / `ref` / `comando` /
      `pago` / `endpoint` / `cbu` / etc.
    - body contains an http(s) URL
    - body contains a fenced code block (```)

    The user uses these notes as atomic-fact pins (a payment URL, a
    CBU, a one-off shell command). Filtering them by char count
    dropped them from the index even when their title perfectly
    matched a future query. Real example: `Pagar escuela Grecia.md`
    with a 67-char body containing the payment URL.
    """
    if not body:
        return False

    raw_tags: list[str] = []
    if isinstance(fm_tags, list):
        raw_tags = [str(t).strip().lower() for t in fm_tags if t]
    elif isinstance(fm_tags, str):
        raw_tags = [t.strip().lower() for t in fm_tags.split(",") if t.strip()]
    if any(t in _HIGH_SIGNAL_TAGS for t in raw_tags):
        return True

    if _URL_RE.search(body):
        return True

    return "```" in body


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


@cli.command(name="capture-stop")
def capture_stop() -> None:
    """Stop hook — passive auto-extract of insights from the last turn.

    Reads the Stop hook payload from stdin (Claude Code format), pulls
    the last (user, assistant) exchange from the transcript, asks the
    helper LLM (Qwen2.5-3B) to extract any actionable insights, dedups
    against the existing corpus, and saves survivors via Memory.save().

    Hook input (stdin, JSON):
      {"transcript_path": "/path/to/...jsonl", ...}

    Hook output (stdout):
      `{}`  — always. Capture is silent; the user discovers new
      memorias via `memo list` or the next ambient recall.

    Env vars:
      MEMO_CAPTURE_DISABLE  — set to "1" to make this a no-op.
      MEMO_CAPTURE_DEBUG    — set to "1" to print extraction progress
                              to stderr (helpful while tuning the
                              extraction prompt or trigger keywords).

    Failure modes are absorbed. The hook never blocks the user — at
    worst you get no auto-save for that turn.
    """
    import json as _json
    import os
    import sys as _sys
    from pathlib import Path

    if os.environ.get("MEMO_CAPTURE_DISABLE") == "1":
        print("{}")
        _sys.exit(0)

    debug = os.environ.get("MEMO_CAPTURE_DEBUG") == "1"

    try:
        raw = _sys.stdin.read()
        payload = _json.loads(raw) if raw.strip() else {}
    except _json.JSONDecodeError:
        print("{}")
        _sys.exit(0)

    transcript_path = payload.get("transcript_path")
    if not transcript_path:
        print("{}")
        _sys.exit(0)

    try:
        from memo.capture import run_capture
        run_capture(Path(transcript_path), debug=debug)
    except Exception as exc:
        if debug:
            print(f"# memo capture-stop failed: {exc}", file=_sys.stderr)

    print("{}")
    _sys.exit(0)


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
    except Exception as exc:
        if os.environ.get("MEMO_RECALL_DEBUG") == "1":
            print(f"# memo prewarm failed: {exc}", file=_sys.stderr)
    _sys.exit(0)


# ── Session checkpoints (v0.4.0) ───────────────────────────────────────────
#
# `memo session ...` — short-lived "what was I working on" snapshots, written
# on every Claude Code Stop hook. Survive a closed/crashed session so the
# next SessionStart can show a picker of recent work. Storage is sidecar
# JSON in `state_dir/sessions/`, NOT memorias (different lifecycle, different
# query pattern — looked up by recency, never by semantic similarity).


@cli.group(name="session")
def session_group() -> None:
    """Internal session-snapshot ops — hook targets, not user-facing.

    User-facing entry is `memo resume` (list / inspect). This group
    holds the wiring the hooks call: `checkpoint` (Stop hook), `recent`
    (SessionStart additionalContext), and `prune` (LRU cleanup). Stays
    namespaced so `memo --help` doesn't surface plumbing as if it were
    everyday CLI.
    """


@session_group.command(name="checkpoint")
@click.option("--session-id", default=None, help="Override session_id (default: read from stdin payload).")
@click.option("--cwd", default=None, help="Override cwd (default: read from stdin payload, fallback os.getcwd).")
@click.option("--transcript-path", default=None, help="Override transcript path.")
@click.option("--lru-cap", default=50, type=int, show_default=True)
@click.option("--json", "as_json", is_flag=True, help="Print the persisted snapshot as JSON.")
def session_checkpoint(
    session_id: str | None, cwd: str | None,
    transcript_path: str | None, lru_cap: int, as_json: bool,
) -> None:
    """Stop hook entrypoint — upsert a session snapshot from stdin JSON.

    Reads the Stop hook payload from stdin (Claude Code passes
    `{"session_id", "transcript_path", "cwd", ...}`). Falls back to
    flags/cwd if stdin is empty (lets you run it manually for testing).

    Always exits 0 — like the other hooks, a checkpoint failure must
    not block Claude Code. On any exception we swallow + print `{}`.
    """
    import json as _json
    import os as _os
    import sys as _sys

    if _os.environ.get("MEMO_SESSION_DISABLE") == "1":
        if as_json:
            click.echo("{}")
        sys.exit(0)

    payload: dict[str, Any] = {}
    # Stdin is a TTY when run interactively → don't block on read.
    if not _sys.stdin.isatty():
        try:
            raw = _sys.stdin.read()
            if raw.strip():
                payload = _json.loads(raw)
        except _json.JSONDecodeError:
            payload = {}

    sid = session_id or payload.get("session_id")
    cwd_resolved = cwd or payload.get("cwd") or _os.getcwd()
    transcript = transcript_path or payload.get("transcript_path")

    if not sid:
        # Without a session_id we can't key the snapshot. Fail silently
        # so the hook still exits 0.
        if as_json:
            click.echo("{}")
        sys.exit(0)

    try:
        from memo.session import checkpoint as _checkpoint

        cfg = Config.from_env()
        cfg.ensure_dirs()
        snap = _checkpoint(
            cfg.state_dir,
            session_id=sid,
            cwd=cwd_resolved,
            transcript_path=transcript,
            lru_cap=lru_cap,
        )
    except Exception as exc:
        if _os.environ.get("MEMO_SESSION_DEBUG") == "1":
            print(f"# memo session checkpoint failed: {exc}", file=_sys.stderr)
        if as_json:
            click.echo("{}")
        sys.exit(0)

    if as_json:
        click.echo(_json.dumps(snap, ensure_ascii=False, indent=2))


@cli.command(name="resume")
@click.argument("session_id", required=False)
@click.option("--limit", default=10, type=int, show_default=True,
              help="Max sessions to show (only used when SESSION_ID is omitted).")
@click.option("--project", default=None, help="Filter to one project basename.")
@click.option("--cwd", "cwd_filter", default=None,
              help="Filter to sessions for this exact cwd (resolved). "
                   "Used by the shell wrapper to ask 'what was open here?' "
                   "without manual path comparison.")
@click.option("--json", "as_json", is_flag=True)
def resume(
    session_id: str | None, limit: int,
    project: str | None, cwd_filter: str | None, as_json: bool,
) -> None:
    """Recent sessions to retomar — picker for the SessionStart flow.

    With no argument, prints a table of the most recent sessions
    (cwd / branch / summary / id). Pass SESSION_ID (full or unique
    prefix ≥4 chars) to inspect one session in detail.

    Storage is sidecar JSON under `~/.local/share/memo/sessions/`,
    auto-written by the Stop hook (`memo session checkpoint`) and
    LRU-capped at 50.
    """
    from memo.session import format_relative, get_session, list_sessions

    cfg = Config.from_env()

    # Detail view — one session.
    if session_id:
        snap = get_session(cfg.state_dir, session_id)
        if snap is None:
            console.print(f"[red]not found:[/red] {session_id}")
            sys.exit(1)
        if as_json:
            click.echo(json.dumps(snap, ensure_ascii=False, indent=2))
            return
        mods = snap.get("modified_files") or []
        mods_line = ", ".join(mods[:5])
        if len(mods) > 5:
            mods_line += f", …(+{len(mods) - 5})"
        sid = snap.get("session_id") or ""
        console.print(Panel.fit(
            f"[bold]{snap.get('summary') or snap.get('last_user_msg') or 'session'}[/bold]\n"
            f"[dim]session_id:[/dim] {sid}\n"
            f"[dim]project:[/dim]    {snap.get('project') or '—'}\n"
            f"[dim]cwd:[/dim]        {snap.get('cwd') or '—'}\n"
            f"[dim]branch:[/dim]     {snap.get('branch') or '—'}\n"
            f"[dim]head:[/dim]       {snap.get('head_commit') or '—'}\n"
            f"[dim]modified:[/dim]   {mods_line or '—'}\n"
            f"[dim]transcript:[/dim] {snap.get('transcript_path') or '—'}\n"
            f"[dim]created:[/dim]    {snap.get('created')}  ({format_relative(snap.get('created'))})\n"
            f"[dim]updated:[/dim]    {snap.get('updated')}  ({format_relative(snap.get('updated'))})\n"
            f"[dim]turns:[/dim]      {snap.get('turn_count')}\n\n"
            f"{snap.get('last_user_msg') or ''}",
            title="session", border_style="cyan",
        ))
        if sid:
            console.print(
                f"\n[bold green]Para retomar:[/bold green]  "
                f"[cyan]claude --resume {sid}[/cyan]\n"
                f"[dim](copy-paste; corré el comando desde "
                f"`{snap.get('cwd') or '?'}`)[/dim]",
            )
        return

    # List view — picker.
    rows = list_sessions(
        cfg.state_dir, limit=limit, project=project, cwd=cwd_filter,
    )
    if as_json:
        click.echo(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    if not rows:
        console.print("[dim]no sessions yet — run a checkpoint first[/dim]")
        return

    # When the caller passed an explicit --cwd, the list is already
    # filtered to that cwd — printing a "Última en este proyecto"
    # banner on top of a homogeneous list would be redundant.
    if cwd_filter:
        same_cwd = []
    else:
        # Bias: if there's a session for the current cwd, surface it on top
        # with the exact resume command. The whole point of the picker is
        # crash recovery — if you crashed and reopened terminal in the same
        # project, the very first thing you want to see is "click here to
        # resume", not a generic chronological list.
        import os as _os
        from pathlib import Path as _Path
        cur_cwd = str(_Path(_os.getcwd()).resolve())
        same_cwd = [r for r in rows if (r.get("cwd") or "") == cur_cwd]
    if same_cwd:
        top = same_cwd[0]
        sid = top.get("session_id") or ""
        console.print(
            f"[bold green]Última en este proyecto[/bold green]  "
            f"[dim]({format_relative(top.get('updated'))})[/dim]: "
            f"{(top.get('summary') or top.get('last_user_msg') or '—')[:80]}",
        )
        console.print(
            f"[bold green]Para retomar:[/bold green]  "
            f"[cyan]claude --resume {sid}[/cyan]\n",
        )

    tbl = Table(show_lines=False, expand=True)
    tbl.add_column("when", width=10)
    tbl.add_column("project", width=14, overflow="fold")
    tbl.add_column("branch", width=14, overflow="fold")
    tbl.add_column("turns", justify="right", width=5)
    tbl.add_column("summary", overflow="fold")
    tbl.add_column("session_id", overflow="fold")
    for r in rows:
        tbl.add_row(
            format_relative(r.get("updated")),
            r.get("project") or "—",
            r.get("branch") or "—",
            str(r.get("turn_count") or 0),
            (r.get("summary") or r.get("last_user_msg") or "—")[:80],
            r.get("session_id") or "—",
        )
    console.print(tbl)
    console.print(
        "[dim]Detalle: `memo resume <id|prefix>`  ·  "
        "Retomar: `claude --resume <session_id>` (copy desde la tabla).[/dim]",
    )


@session_group.command(name="recent")
@click.option("--limit", default=5, type=int, show_default=True)
def session_recent(limit: int) -> None:
    """SessionStart hook entrypoint — emit `additionalContext` markdown
    listing recent sessions. Same exit-0-silent contract as recall-hook."""
    import json as _json
    import os as _os
    import sys as _sys

    if _os.environ.get("MEMO_SESSION_DISABLE") == "1":
        print("{}")
        _sys.exit(0)

    try:
        from memo.session import format_relative, list_sessions
        cfg = Config.from_env()
        rows = list_sessions(cfg.state_dir, limit=limit)
    except Exception as exc:
        if _os.environ.get("MEMO_SESSION_DEBUG") == "1":
            print(f"# memo session recent failed: {exc}", file=_sys.stderr)
        print("{}")
        _sys.exit(0)

    if not rows:
        print("{}")
        _sys.exit(0)

    # Highlight the most recent session for the current cwd. The whole
    # point of the picker is crash recovery: if Claude Code died and
    # the user reopened the terminal in the same project, the first
    # signal they want is "you crashed in <X>, retomar con <comando>".
    from pathlib import Path as _Path
    cur_cwd = str(_Path(_os.getcwd()).resolve())
    same_cwd = [r for r in rows if (r.get("cwd") or "") == cur_cwd]
    top = same_cwd[0] if same_cwd else None

    lines: list[str] = ["## Sesiones recientes (memo)", ""]

    if top:
        sid = top.get("session_id") or ""
        when = format_relative(top.get("updated"))
        summary = (
            top.get("summary") or top.get("last_user_msg") or "—"
        ).replace("\n", " ")[:120]
        lines.append(f"**Última en este proyecto** ({when}): {summary}")
        lines.append("")
        lines.append("```")
        lines.append(f"claude --resume {sid}")
        lines.append("```")
        lines.append("")

    lines.extend([
        "| # | cuándo | proyecto | branch | resumen | session_id |",
        "|---|--------|----------|--------|---------|------------|",
    ])
    for i, r in enumerate(rows, start=1):
        summary = (r.get("summary") or r.get("last_user_msg") or "—").replace("|", "·").replace("\n", " ")
        lines.append(
            f"| {i} | {format_relative(r.get('updated'))} | "
            f"{(r.get('project') or '—')[:20]} | "
            f"{(r.get('branch') or '—')[:16]} | "
            f"{summary[:60]} | "
            f"`{r.get('session_id') or ''}` |"
        )
    lines.append("")
    lines.append(
        "_Para detalle: `memo resume <id|prefix>`. "
        "Para retomar otra: `claude --resume <session_id>`._"
    )

    output = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": "\n".join(lines),
        }
    }
    print(_json.dumps(output, ensure_ascii=False))


@session_group.command(name="prune")
@click.option("--cap", default=50, type=int, show_default=True)
def session_prune(cap: int) -> None:
    """Delete oldest sessions beyond `cap`. Idempotent."""
    from memo.session import prune_lru

    cfg = Config.from_env()
    n = prune_lru(cfg.state_dir, cap=cap)
    console.print(f"[green]✓[/green] pruned {n} session(s); cap={cap}")


# ── Shell wrapper for crash recovery (v0.4.x) ──────────────────────────────
#
# Companion to `memo resume`: when the user types `claude` (no args) in a
# project that has recent session checkpoints, the wrapper offers to retomar
# one before opening a fresh session. Designed for the post-reboot case
# where iTerm2 / Terminal restored the cwd but Claude itself is closed.
# `command claude` is used everywhere to bypass the shell function and
# avoid recursion.

_WRAPPER_SNIPPET_ZSH = r"""# >>> memo session-resume wrapper >>>
# Auto-suggest resuming recent Claude Code sessions for the current cwd.
# Generated by `memo install-shell-wrapper`. Do not edit by hand — re-run
# the install command instead. To remove: delete this file and the
# matching `source` line in your shell rc.
#
# If you previously had `alias claude='claude --flag1 --flag2'`, migrate
# to:   MEMO_CLAUDE_EXTRA_ARGS=(--flag1 --flag2)
# Those flags are forwarded to every `command claude` invocation.

function claude() {
    local extra_args=("${MEMO_CLAUDE_EXTRA_ARGS[@]}")

    # 1. Args present → pass-through, no prompt (covers `claude --resume X`,
    #    `claude -p "..."`, `claude --help`, etc.).
    if (( $# > 0 )); then
        command claude "${extra_args[@]}" "$@"
        return
    fi
    # 2. Stdin not a TTY → can't prompt safely.
    if [[ ! -t 0 ]]; then
        command claude "${extra_args[@]}"
        return
    fi
    # 3. Required tools missing → degrade silently.
    if ! command -v memo >/dev/null 2>&1 || ! command -v jq >/dev/null 2>&1; then
        command claude "${extra_args[@]}"
        return
    fi
    # 4. Ask memo what's recent for this cwd.
    local raw count
    raw=$(memo resume --json --limit 5 --cwd "$PWD" 2>/dev/null) || raw=""
    if [[ -z "$raw" ]]; then
        command claude "${extra_args[@]}"
        return
    fi
    count=$(echo "$raw" | jq 'length' 2>/dev/null)
    if [[ -z "$count" || "$count" == "0" ]]; then
        command claude "${extra_args[@]}"
        return
    fi

    # 5. Selector.
    echo ""
    if [[ "$count" == "1" ]]; then
        local sid summary
        sid=$(echo "$raw" | jq -r '.[0].session_id')
        summary=$(echo "$raw" | jq -r '.[0].summary // .[0].last_user_msg // "—"')
        echo "Sesión reciente en este cwd:"
        printf "  %s\n" "${summary:0:120}"
        local ans
        printf "Retomar? [Y/n] "
        read ans
        if [[ -z "$ans" || "$ans" == [yY]* ]]; then
            command claude --resume "$sid" "${extra_args[@]}"
        else
            command claude "${extra_args[@]}"
        fi
    else
        echo "Sesiones recientes en este cwd ($count):"
        echo "$raw" | jq -r 'to_entries | .[] | "  [\(.key + 1)] \((.value.summary // .value.last_user_msg // "—") | .[0:120])"'
        echo "  [n] nueva sesión"
        local choice
        printf "Elegí: "
        read choice
        if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= count )); then
            local sid
            sid=$(echo "$raw" | jq -r ".[$((choice - 1))].session_id")
            command claude --resume "$sid" "${extra_args[@]}"
        else
            command claude "${extra_args[@]}"
        fi
    fi
}
# <<< memo session-resume wrapper <<<
"""


@cli.command(name="install-shell-wrapper")
@click.option("--print", "do_print", is_flag=True,
              help="Print the wrapper snippet to stdout. Default mode "
                   "when neither --print nor --write is set.")
@click.option("--write", "do_write", is_flag=True,
              help="Write ~/.zsh/memo-wrapper.zsh and append the matching "
                   "`source` line to ~/.zshrc (idempotent).")
@click.option("--shell", "shell_kind",
              type=click.Choice(["zsh", "bash"]), default="zsh",
              show_default=True,
              help="Target shell. zsh is the macOS default.")
@click.option("--force", is_flag=True,
              help="Overwrite ~/.zsh/memo-wrapper.zsh even if its content "
                   "differs from what we would write.")
def install_shell_wrapper(
    do_print: bool, do_write: bool, shell_kind: str, force: bool,
) -> None:
    """Install or print the `claude` shell wrapper for crash recovery.

    With no flags (or --print): emit the snippet to stdout for review.
    With --write: install to `~/.zsh/memo-wrapper.zsh` and append a
    `source` line to `~/.zshrc` if not already present.

    The wrapper makes `claude` (no args) prompt to retomar a recent
    memo session checkpoint when the current cwd has any. With args
    it falls through to the real claude. Detects a pre-existing
    `alias claude=...` and warns the user to migrate to
    `MEMO_CLAUDE_EXTRA_ARGS` so flags compose with the wrapper.
    """
    # Bash compat note: zsh's `[[ ... =~ ... ]]` and `${var:0:N}` work
    # in bash >=3, so we currently emit one snippet for both. The
    # `--shell` flag is still consumed below to pick the rc file
    # (.zshrc vs .bashrc) and is kept as an explicit dispatch point
    # for the day we need bash-specific snippet tweaks
    # (e.g. `read -k 1` → `read -n 1`).
    snippet = _WRAPPER_SNIPPET_ZSH

    # Default to --print when neither flag is set; spelled out so the
    # output flows through one path.
    if not do_write:
        click.echo(snippet)
        if not do_print:
            click.echo(
                "\n(pasale `--write` para instalar en "
                "~/.zsh/memo-wrapper.zsh + ~/.zshrc.)",
                err=True,
            )
        return

    from pathlib import Path as _Path

    home = _Path.home()
    wrapper_dir = home / ".zsh"
    wrapper_path = wrapper_dir / "memo-wrapper.zsh"
    rc_path = home / (".zshrc" if shell_kind == "zsh" else ".bashrc")
    source_line = f"[[ -f {wrapper_path} ]] && source {wrapper_path}"

    wrapper_dir.mkdir(parents=True, exist_ok=True)

    # Step 1 — write the wrapper file (idempotent + force-aware).
    if wrapper_path.is_file():
        existing = wrapper_path.read_text(encoding="utf-8")
        if existing == snippet:
            console.print(
                f"[dim]✓ {wrapper_path} ya está al día[/dim]",
            )
        elif not force:
            console.print(
                f"[red]✗[/red] {wrapper_path} existe con contenido distinto. "
                f"Pasale [bold]--force[/bold] para sobrescribir.",
            )
            sys.exit(2)
        else:
            wrapper_path.write_text(snippet, encoding="utf-8")
            console.print(f"[yellow]↻[/yellow] {wrapper_path} sobrescrito (--force)")
    else:
        wrapper_path.write_text(snippet, encoding="utf-8")
        console.print(f"[green]✓[/green] {wrapper_path} creado")

    # Step 2 — append `source` line to rc if missing.
    rc_existing = rc_path.read_text(encoding="utf-8") if rc_path.is_file() else ""
    if source_line in rc_existing:
        console.print(f"[dim]✓ {rc_path} ya tiene la línea source[/dim]")
    else:
        with rc_path.open("a", encoding="utf-8") as fh:
            if rc_existing and not rc_existing.endswith("\n"):
                fh.write("\n")
            fh.write("\n# memo session-resume wrapper\n")
            fh.write(f"{source_line}\n")
        console.print(f"[green]✓[/green] {rc_path} ← appended source line")

    # Step 3 — detect pre-existing `alias claude=...` and warn the
    # user. A shell wrapper function shadows an alias of the same
    # name, so the alias would silently lose any flags it bundled.
    # Migration target: `MEMO_CLAUDE_EXTRA_ARGS=(--flag1 --flag2)`.
    import re as _re

    try:
        rc_text = rc_path.read_text(encoding="utf-8")
    except OSError:
        rc_text = ""
    alias_match = _re.search(r"^\s*alias\s+claude\s*=.*$", rc_text, _re.MULTILINE)
    if alias_match:
        console.print(
            f"[yellow]heads-up:[/yellow] found a pre-existing `{alias_match.group(0).strip()}` "
            f"in {rc_path}.\n"
            f"  The wrapper function will shadow it. To preserve those flags,\n"
            f"  remove the alias and use [bold]MEMO_CLAUDE_EXTRA_ARGS[/bold]:\n"
            f"    [dim]export MEMO_CLAUDE_EXTRA_ARGS=(--your-flag --other-flag)[/dim]",
        )


# -- temporal reasoning commands ----------------------------------------------


@cli.group(name="temporal")
def temporal_group() -> None:
    """Analyze temporal patterns and contradictions in memories."""
    pass


@temporal_group.command(name="contradictions")
@click.argument("entity")
@click.option("--type", "entity_type", help="Filter by entity type from graph")
@click.option("--confidence", "min_confidence", type=float, default=0.7,
              help="Minimum confidence threshold (default: 0.7)")
@click.option("--max-pairs", type=int, default=20,
              help="Maximum number of pairs to analyze (default: 20)")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def temporal_contradictions(
    entity: str, entity_type: str | None, min_confidence: float,
    max_pairs: int, as_json: bool,
) -> None:
    """Detect contradictions among memorias mentioning a specific entity.

    Example: memo temporal contradictions mlx
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    contradictions = mem.temporal.detect_entity_contradictions(
        entity_name=entity,
        entity_type=entity_type,
        confidence_threshold=min_confidence,
        max_pairs=max_pairs,
    )

    if as_json:
        click.echo(json.dumps([c.__dict__ for c in contradictions], indent=2))
        return

    if not contradictions:
        console.print(f"[green]No contradictions found for entity '{entity}'[/green]")
        return

    table = Table(title=f"Contradictions for '{entity}'")
    table.add_column("ID A", style="cyan")
    table.add_column("ID B", style="cyan")
    table.add_column("Title A", style="yellow")
    table.add_column("Title B", style="yellow")
    table.add_column("Relationship", style="magenta")
    table.add_column("Confidence", style="green")
    table.add_column("Rationale")

    for c in contradictions:
        table.add_row(
            c.memoria_id_a[:8],
            c.memoria_id_b[:8],
            c.title_a[:40],
            c.title_b[:40],
            c.relationship,
            f"{c.confidence:.2f}",
            c.rationale,
        )

    console.print(table)


@temporal_group.command(name="timeline")
@click.argument("entity")
@click.option("--type", "entity_type", help="Filter by entity type from graph")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def temporal_timeline(entity: str, entity_type: str | None, as_json: bool) -> None:
    """Build a chronological timeline of all memorias mentioning an entity.

    Example: memo temporal timeline mlx
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    timeline = mem.temporal.build_entity_timeline(
        entity_name=entity,
        entity_type=entity_type,
    )

    if timeline is None:
        console.print(f"[yellow]No memorias found for entity '{entity}'[/yellow]")
        return

    if as_json:
        click.echo(json.dumps({
            "entity_name": timeline.entity_name,
            "entity_type": timeline.entity_type,
            "first_seen": timeline.first_seen,
            "last_seen": timeline.last_seen,
            "events": [e.__dict__ for e in timeline.events],
        }, indent=2))
        return

    console.print(f"[bold]Timeline for '{entity}' ({timeline.entity_type})[/bold]")
    console.print(f"First seen: {timeline.first_seen}")
    console.print(f"Last seen: {timeline.last_seen}")
    console.print()

    for event in timeline.events:
        console.print(f"[cyan]{event.date}[/cyan] [dim][{event.memoria_id[:8]}][/dim]")
        console.print(f"  [yellow]{event.title}[/yellow] ({event.type})")
        console.print(f"  {event.snippet}")
        console.print()


@temporal_group.command(name="stale")
@click.option("--days", type=int, default=180,
              help="Days since last update to consider stale (default: 180)")
@click.option("--min-access", "min_access_count", type=int, default=0,
              help="Minimum access count to exclude (default: 0)")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def temporal_stale(days: int, min_access_count: int, as_json: bool) -> None:
    """Find memorias that may be stale based on age and lack of access.

    Example: memo temporal stale --days 90
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    stale = mem.temporal.detect_stale_memorias(
        days_threshold=days,
        min_access_count=min_access_count,
    )

    if as_json:
        click.echo(json.dumps(stale, indent=2))
        return

    if not stale:
        console.print(f"[green]No stale memorias found (threshold: {days} days)[/green]")
        return

    table = Table(title=f"Potentially Stale Memorias (>{days} days)")
    table.add_column("ID", style="cyan")
    table.add_column("Title", style="yellow")
    table.add_column("Type", style="magenta")
    table.add_column("Updated", style="dim")
    table.add_column("Days Old", style="red")
    table.add_column("Access Count", style="green")

    for item in stale[:50]:  # Cap display
        table.add_row(
            item["id"][:8],
            item["title"][:40],
            item["type"],
            item["updated"][:10],
            str(item["days_since_update"]),
            str(item["access_count"]),
        )

    console.print(table)
    if len(stale) > 50:
        console.print(f"[dim]...and {len(stale) - 50} more[/dim]")


@temporal_group.command(name="patterns")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def temporal_patterns(as_json: bool) -> None:
    """Analyze high-level temporal patterns across the entire corpus.

    Example: memo temporal patterns
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    patterns = mem.temporal.detect_temporal_patterns()

    if as_json:
        click.echo(json.dumps(patterns, indent=2))
        return

    console.print("[bold]Temporal Patterns[/bold]")
    console.print()

    # Memorias per month
    console.print("[yellow]Memorias per month:[/yellow]")
    for month, count in list(patterns["memorias_per_month"].items())[-12:]:
        console.print(f"  {month}: {count}")
    console.print()

    # Most active entities
    console.print("[yellow]Most active entities:[/yellow]")
    for entity, count in list(patterns["most_active_entities"].items())[:10]:
        console.print(f"  {entity}: {count} mentions")


def _get_memory(cfg: Config) -> Any:
    """Helper to get Memory instance, used by temporal commands."""
    from memo.memory import Memory
    return Memory(cfg)


# -- advanced consolidation commands -------------------------------------------


@cli.group(name="consolidate")
def consolidate_group() -> None:
    """Advanced consolidation with intelligent merge and archival."""
    pass


@consolidate_group.command(name="propose")
@click.option("--threshold", type=float, default=0.85,
              help="Cosine similarity threshold (default: 0.85)")
@click.option("--max-clusters", type=int, default=20,
              help="Maximum clusters to process (default: 20)")
@click.option("--type", "type_", help="Filter by memoria type")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def consolidate_propose(
    threshold: float, max_clusters: int, type_: str | None, as_json: bool,
) -> None:
    """Detect clusters and propose merge strategies (read-only).

    Example: memo consolidate propose --threshold 0.9
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    result = mem.consolidator.consolidate_all(
        threshold=threshold,
        max_clusters=max_clusters,
        type_=type_,
        auto_apply=False,
        dry_run=True,
    )

    if as_json:
        click.echo(json.dumps(result, indent=2))
        return

    clusters = result.get("clusters", [])
    proposals = result.get("proposals", [])

    console.print(f"[bold]Detected {len(clusters)} clusters[/bold]")
    console.print(f"[yellow]Generated {len(proposals)} merge proposals[/yellow]")
    console.print()

    if not proposals:
        console.print("[green]No mergeable clusters found[/green]")
        return

    for i, p in enumerate(proposals[:10], 1):
        console.print(f"[cyan]{i}. Cluster {p['cluster_id']}[/cyan]")
        console.print(f"   Strategy: {p['merge_strategy']}")
        console.print(f"   Rationale: {p['rationale']}")
        console.print(f"   Memorias to merge: {len(p['memoria_ids'])}")
        console.print()

    if len(proposals) > 10:
        console.print(f"[dim]...and {len(proposals) - 10} more proposals[/dim]")


@consolidate_group.command(name="apply")
@click.option("--threshold", type=float, default=0.85,
              help="Cosine similarity threshold (default: 0.85)")
@click.option("--max-clusters", type=int, default=20,
              help="Maximum clusters to process (default: 20)")
@click.option("--type", "type_", help="Filter by memoria type")
@click.option("--dry-run", is_flag=True,
              help="Show what would happen without applying changes")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
@click.confirmation_option(prompt="This will merge memorias and archive old ones. Continue?")
def consolidate_apply(
    threshold: float, max_clusters: int, type_: str | None,
    dry_run: bool, as_json: bool,
) -> None:
    """Apply merge proposals to consolidate the corpus.

    Example: memo consolidate apply --dry-run
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    result = mem.consolidator.consolidate_all(
        threshold=threshold,
        max_clusters=max_clusters,
        type_=type_,
        auto_apply=True,
        dry_run=dry_run,
    )

    if as_json:
        click.echo(json.dumps(result, indent=2))
        return

    result.get("proposals", [])
    results = result.get("results", [])

    if dry_run:
        console.print("[yellow]Dry run mode - no changes applied[/yellow]")
        console.print()

    console.print(f"[bold]Processed {len(results)} consolidations[/bold]")
    console.print()

    merged_count = sum(1 for r in results if r.get("merged_id"))
    archived_count = sum(len(r.get("archived_ids", [])) for r in results)
    skipped_count = sum(len(r.get("skipped_ids", [])) for r in results)

    console.print(f"[green]✓ Merged {merged_count} memorias[/green]")
    console.print(f"[yellow]↻ Archived {archived_count} old versions[/yellow]")
    console.print(f"[dim]⊘ Skipped {skipped_count} (conflicts)[/dim]")

    for r in results[:5]:
        console.print(f"  {r.get('summary', '')}")

    if len(results) > 5:
        console.print(f"[dim]...and {len(results) - 5} more[/dim]")


@consolidate_group.command(name="list-archived")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def consolidate_list_archived(as_json: bool) -> None:
    """List all archived memorias.

    Example: memo consolidate list-archived
    """
    cfg = Config.from_env()
    archival_dir = cfg.memory_dir / "archived"

    if not archival_dir.is_dir():
        console.print("[dim]No archived memorias found[/dim]")
        return

    archived_files = list(archival_dir.glob("*.md"))

    if as_json:
        archived_data = []
        for f in archived_files:
            import frontmatter
            post = frontmatter.loads(f.read_text(encoding="utf-8"))
            archived_data.append({
                "id": f.stem,
                "title": post.get("title", ""),
                "archived_for": post.get("archived_for", ""),
                "archived_at": post.get("archived_at", ""),
            })
        click.echo(json.dumps(archived_data, indent=2))
        return

    table = Table(title="Archived Memorias")
    table.add_column("ID", style="cyan")
    table.add_column("Title", style="yellow")
    table.add_column("Archived For", style="green")
    table.add_column("Archived At", style="dim")

    for f in archived_files[:50]:
        import frontmatter
        post = frontmatter.loads(f.read_text(encoding="utf-8"))
        table.add_row(
            f.stem[:8],
            post.get("title", "")[:40],
            post.get("archived_for", "")[:8],
            post.get("archived_at", "")[:10],
        )

    console.print(table)
    if len(archived_files) > 50:
        console.print(f"[dim]...and {len(archived_files) - 50} more[/dim]")


# -- graph navigation commands ------------------------------------------------


@cli.group(name="graph")
def graph_group() -> None:
    """Navigate the entity graph with path finding and community detection."""
    pass


@graph_group.command(name="path")
@click.argument("source")
@click.argument("target")
@click.option("--max-length", type=int, default=5,
              help="Maximum path length to search (default: 5)")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def graph_path(source: str, target: str, max_length: int, as_json: bool) -> None:
    """Find shortest path between two entities.

    Example: memo graph path memo obsidian-rag
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
    console.print(f"[dim]Via {len(path.intermediate_memorias)} memoria(s)[/dim]")


@graph_group.command(name="neighbors")
@click.argument("entity")
@click.option("--max", "max_neighbors", type=int, default=50,
              help="Maximum neighbors to show (default: 50)")
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
    table.add_column("Shared Memorias", style="green")

    for neighbor in neighbors.direct_neighbors[:20]:
        mem_count = len(neighbors.neighbor_memorias.get(neighbor, []))
        table.add_row(neighbor, str(mem_count))

    console.print(table)
    if len(neighbors.direct_neighbors) > 20:
        console.print(f"[dim]...and {len(neighbors.direct_neighbors) - 20} more[/dim]")


@graph_group.command(name="communities")
@click.option("--min-size", type=int, default=2,
              help="Minimum community size (default: 2)")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def graph_communities(min_size: int, as_json: bool) -> None:
    """Detect communities (connected components) in the entity graph.

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


@graph_group.command(name="centrality")
@click.option("--top", type=int, default=20,
              help="Top N entities by centrality (default: 20)")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def graph_centrality(top: int, as_json: bool) -> None:
    """Compute centrality metrics for all entities.

    Example: memo graph centrality --top 30
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    scores = mem.navigator.compute_centrality()

    if as_json:
        click.echo(json.dumps({
            "degree": scores.degree,
            "betweenness": scores.betweenness,
        }, indent=2))
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
@click.option("--format", "format_type", type=click.Choice(["dot", "json"]), default="dot",
              help="Output format (default: dot)")
@click.option("--output", "-o", "output_path",
              help="Output file path (default: stdout)")
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
        data = mem.navigator.export_json(include_memorias=True)
        json_str = json.dumps(data, indent=2)
        if output_path:
            from pathlib import Path
            Path(output_path).write_text(json_str, encoding="utf-8")
        else:
            click.echo(json_str)

    if output_path:
        console.print(f"[green]Exported to {output_path}[/green]")


# -- contextual recall commands -----------------------------------------------


@cli.group(name="contextual")
def contextual_group() -> None:
    """Contextual recall with conversation history and preference learning."""
    pass


@contextual_group.command(name="search")
@click.argument("query")
@click.option("--limit", type=int, default=10,
              help="Max results (default: 10)")
@click.option("--mode", type=click.Choice(["vec", "bm25", "hybrid"]), default="hybrid",
              help="Search mode (default: hybrid)")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def contextual_search(query: str, limit: int, mode: str, as_json: bool) -> None:
    """Search with contextual re-ranking based on conversation history.

    Example: memo contextual search "MLX performance"
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    results = mem.contextual.search_with_context(
        query=query,
        limit=limit,
        mode=mode,
    )

    if as_json:
        click.echo(json.dumps([r.__dict__ for r in results], indent=2))
        return

    if not results:
        console.print("[yellow]No results found[/yellow]")
        return

    table = Table(title=f"Contextual Search Results for '{query}'")
    table.add_column("ID", style="cyan")
    table.add_column("Title", style="yellow")
    table.add_column("Original Score", style="dim")
    table.add_column("Contextual Score", style="green")
    table.add_column("Boost Factors", style="magenta")

    for r in results[:20]:
        boosts = ", ".join(f"{k}={v:.2f}" for k, v in r.boost_factors.items())
        table.add_row(
            r.memoria_id[:8],
            r.title[:40],
            f"{r.original_score:.3f}",
            f"{r.contextual_score:.3f}",
            boosts or "—",
        )

    console.print(table)
    if len(results) > 20:
        console.print(f"[dim]...and {len(results) - 20} more[/dim]")


@contextual_group.command(name="record-search")
@click.argument("query")
@click.argument("memoria_ids", nargs=-1, required=True)
def contextual_record_search(query: str, memoria_ids: tuple[str, ...]) -> None:
    """Record a search in the conversation history for learning.

    Example: memo contextual record-search "MLX" abc123 def456
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    mem.contextual.record_search(query, list(memoria_ids))
    console.print(f"[green]Recorded search with {len(memoria_ids)} recalled memorias[/green]")


@contextual_group.command(name="record-click")
@click.argument("memoria_id")
def contextual_record_click(memoria_id: str) -> None:
    """Record that the user clicked/viewed a memoria (for preference learning).

    Example: memo contextual record-click abc123
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    mem.contextual.record_click(memoria_id)
    console.print(f"[green]Recorded click for memoria {memoria_id[:8]}[/green]")


@contextual_group.command(name="preferences")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def contextual_preferences(as_json: bool) -> None:
    """Show learned user preferences for memory recall.

    Example: memo contextual preferences
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    prefs = mem.contextual.context.get_preferences()

    if as_json:
        click.echo(json.dumps(prefs.__dict__, indent=2))
        return

    console.print("[bold]User Preferences[/bold]")
    console.print()

    console.print(f"[yellow]Recency Weight:[/yellow] {prefs.recency_weight:.2f}")
    console.print(f"[yellow]Diversity Weight:[/yellow] {prefs.diversity_weight:.2f}")
    console.print(f"[yellow]Last Updated:[/yellow] {prefs.last_updated or 'Never'}")
    console.print()

    console.print("[yellow]Preferred Memory Types:[/yellow]")
    if prefs.preferred_types:
        for type_, score in sorted(prefs.preferred_types.items(), key=lambda x: x[1], reverse=True):
            console.print(f"  {type_}: {score:.2f}")
    else:
        console.print("  [dim]No preferences learned yet[/dim]")
    console.print()

    console.print("[yellow]Preferred Entities:[/yellow]")
    if prefs.preferred_entities:
        for entity, score in sorted(prefs.preferred_entities.items(), key=lambda x: x[1], reverse=True)[:10]:
            console.print(f"  {entity}: {score:.2f}")
        if len(prefs.preferred_entities) > 10:
            console.print(f"  [dim]...and {len(prefs.preferred_entities) - 10} more[/dim]")
    else:
        console.print("  [dim]No preferences learned yet[/dim]")


@contextual_group.command(name="history")
@click.option("--limit", type=int, default=10,
              help="Number of recent prompts to show (default: 10)")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def contextual_history(limit: int, as_json: bool) -> None:
    """Show recent conversation history used for contextual recall.

    Example: memo contextual history --limit 5
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    history = mem.contextual.context.get_recent_context(n=limit)

    if as_json:
        click.echo(json.dumps([c.__dict__ for c in history], indent=2))
        return

    if not history:
        console.print("[dim]No conversation history yet[/dim]")
        return

    console.print(f"[bold]Recent {len(history)} Prompts[/bold]")
    console.print()

    for i, ctx in enumerate(history, 1):
        console.print(f"[cyan]{i}. {ctx.timestamp}[/cyan]")
        console.print(f"   Prompt: {ctx.prompt[:80]}")
        console.print(f"   Recalled: {len(ctx.recalled_memorias)} memoria(s)")
        console.print()


@contextual_group.command(name="reset-preferences")
@click.confirmation_option(prompt="This will reset all learned preferences. Continue?")
def contextual_reset_preferences() -> None:
    """Reset all learned user preferences.

    Example: memo contextual reset-preferences
    """
    cfg = Config.from_env()

    # Reset preferences file
    prefs_file = cfg.state_dir / "user_preferences.json"
    if prefs_file.is_file():
        prefs_file.unlink()

    # Reload will create fresh defaults
    console.print("[green]Preferences reset successfully[/green]")


# -- cross-reference commands -------------------------------------------------


@cli.group(name="links")
def links_group() -> None:
    """Cross-reference and backlink system for memories."""
    pass


@links_group.command(name="backlinks")
@click.argument("memoria_id")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def links_backlinks(memoria_id: str, as_json: bool) -> None:
    """Show all memorias that reference this one.

    Example: memo links backlinks abc123
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    backlinks = mem.crossref.get_backlinks(memoria_id)

    if as_json:
        click.echo(json.dumps([b.__dict__ for b in backlinks], indent=2))
        return

    if not backlinks:
        console.print(f"[dim]No backlinks found for memoria {memoria_id[:8]}[/dim]")
        return

    table = Table(title=f"Backlinks to {memoria_id[:8]}")
    table.add_column("Source ID", style="cyan")
    table.add_column("Link Type", style="yellow")
    table.add_column("Context", style="dim")

    for bl in backlinks[:20]:
        table.add_row(
            bl.source_id[:8],
            bl.link_type,
            bl.context[:60] if bl.context else "—",
        )

    console.print(table)
    if len(backlinks) > 20:
        console.print(f"[dim]...and {len(backlinks) - 20} more[/dim]")


@links_group.command(name="outlinks")
@click.argument("memoria_id")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def links_outlinks(memoria_id: str, as_json: bool) -> None:
    """Show all memorias that this one references.

    Example: memo links outlinks abc123
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    outlinks = mem.crossref.get_outlinks(memoria_id)

    if as_json:
        click.echo(json.dumps([o.__dict__ for o in outlinks], indent=2))
        return

    if not outlinks:
        console.print(f"[dim]No outlinks found for memoria {memoria_id[:8]}[/dim]")
        return

    console.print(f"[bold]Outlinks from {memoria_id[:8]}[/bold]")
    console.print()

    for ol in outlinks:
        console.print(f"  [cyan]{ol.target}[/cyan]")


@links_group.command(name="suggest")
@click.argument("content")
@click.option("--title", help="Title of the memoria being saved")
@click.option("--tags", multiple=True, help="Tags of the memoria being saved")
@click.option("--limit", type=int, default=5,
              help="Max suggestions (default: 5)")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def links_suggest(content: str, title: str | None, tags: tuple[str, ...], limit: int, as_json: bool) -> None:
    """Suggest links to existing memorias based on content.

    Example: memo links suggest "MLX performance optimization" --title "MLX"
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    suggestions = mem.link_suggester.suggest_links(
        content=content,
        title=title or "",
        tags=list(tags),
        limit=limit,
    )

    if as_json:
        click.echo(json.dumps([s.__dict__ for s in suggestions], indent=2))
        return

    if not suggestions:
        console.print("[dim]No link suggestions found[/dim]")
        return

    table = Table(title="Link Suggestions")
    table.add_column("ID", style="cyan")
    table.add_column("Title", style="yellow")
    table.add_column("Similarity", style="green")
    table.add_column("Reason", style="dim")

    for s in suggestions:
        table.add_row(
            s.memoria_id[:8],
            s.title[:40],
            f"{s.similarity:.3f}",
            s.reason,
        )

    console.print(table)


@links_group.command(name="format")
@click.argument("memoria_id")
@click.option("--title", help="Display title for the link")
def links_format(memoria_id: str, title: str | None) -> None:
    """Format a memoria ID as a wikilink.

    Example: memo links format abc123 --title "My Memory"
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    wikilink = mem.link_suggester.format_wikilink(memoria_id, title)
    click.echo(wikilink)


@links_group.command(name="reindex")
@click.confirmation_option(prompt="This will rebuild the entire cross-reference index. Continue?")
def links_reindex() -> None:
    """Rebuild the cross-reference index from all memorias.

    Example: memo links reindex
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    # Delete existing index
    cfg.crossref_db.unlink(missing_ok=True)

    # Re-index all memorias
    all_records = mem.list(limit=10000)
    indexed = 0

    for rec in all_records:
        body = rec.body or ""
        if body:
            mem.crossref.index_wikilinks(rec.id, body)
            indexed += 1

    console.print(f"[green]Reindexed {indexed} memorias[/green]")


# -- lifecycle management commands ---------------------------------------------


@cli.group(name="lifecycle")
def lifecycle_group() -> None:
    """Memory lifecycle management — archival, promotion, expiration."""
    pass


@lifecycle_group.command(name="report")
@click.option("--limit", type=int, default=100,
              help="Max memorias to analyze (default: 100)")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def lifecycle_report(limit: int, as_json: bool) -> None:
    """Generate a lifecycle report on the corpus.

    Shows statistics on archival candidates, promotion/demotion candidates,
    expiration candidates, and access patterns.

    Example: memo lifecycle report --limit 50
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    report = mem.lifecycle.get_lifecycle_report(limit=limit)

    if as_json:
        click.echo(json.dumps(report, indent=2))
        return

    console.print("[bold]Lifecycle Report[/bold]")
    console.print()
    console.print(f"Total memorias: {report['total']}")
    console.print(f"Average access count: {report['avg_access_count']:.2f}")
    console.print()
    console.print(f"[yellow]Archive candidates:[/yellow] {report['archive_candidates']}")
    console.print(f"[yellow]Promotion candidates:[/yellow] {report['promotion_candidates']}")
    console.print(f"[yellow]Demotion candidates:[/yellow] {report['demotion_candidates']}")
    console.print(f"[yellow]Expiration candidates:[/yellow] {report['expiration_candidates']}")
    console.print(f"[yellow]Never accessed:[/yellow] {report['never_accessed']}")


@lifecycle_group.command(name="apply")
@click.option("--dry-run", is_flag=True,
              help="Show what would happen without applying changes")
@click.option("--limit", type=int, default=100,
              help="Max memorias to process (default: 100)")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
@click.confirmation_option(prompt="This will archive/delete memorias based on lifecycle rules. Continue?")
def lifecycle_apply(dry_run: bool, limit: int, as_json: bool) -> None:
    """Apply lifecycle rules to the corpus.

    Archives inactive memorias, expires temporary memories, and reports
    promotion/demotion candidates. Use --dry-run first to preview.

    Example: memo lifecycle apply --dry-run
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    if dry_run:
        console.print("[yellow]Dry run mode - no changes will be applied[/yellow]")
        console.print()

    actions = mem.lifecycle.apply_lifecycle_rules(dry_run=dry_run, limit=limit)

    if as_json:
        click.echo(json.dumps(actions, indent=2))
        return

    console.print("[bold]Lifecycle Actions[/bold]")
    console.print()
    console.print(f"[green]Archived:[/green] {actions['archived']}")
    console.print(f"[green]Promoted:[/green] {actions['promoted']}")
    console.print(f"[yellow]Demoted:[/yellow] {actions['demoted']}")
    console.print(f"[red]Expired:[/red] {actions['expired']}")
    console.print(f"[red]Deleted:[/red] {actions['deleted']}")
    console.print(f"[dim]Skipped:[/dim] {actions['skipped']}")


@lifecycle_group.command(name="access-count")
@click.argument("memoria_id")
def lifecycle_access_count(memoria_id: str) -> None:
    """Show access count for a specific memoria.

    Example: memo lifecycle access-count abc123
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    count = mem.lifecycle.get_access_count(memoria_id)
    console.print(f"Access count: {count}")


@lifecycle_group.command(name="list-inactive")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def lifecycle_list_inactive(as_json: bool) -> None:
    """List all archived/inactive memorias.

    Example: memo lifecycle list-inactive
    """
    cfg = Config.from_env()
    inactive_dir = cfg.memory_dir / "inactive"

    if not inactive_dir.is_dir():
        console.print("[dim]No inactive memorias found[/dim]")
        return

    files = list(inactive_dir.glob("*.md"))

    if as_json:
        inactive_data = []
        for f in files:
            inactive_data.append({
                "id": f.stem,
                "path": str(f),
            })
        click.echo(json.dumps(inactive_data, indent=2))
        return

    table = Table(title="Inactive Memorias")
    table.add_column("ID", style="cyan")
    table.add_column("Path", style="dim")

    for f in files[:50]:
        table.add_row(f.stem[:8], str(f.name))

    console.print(table)
    if len(files) > 50:
        console.print(f"[dim]...and {len(files) - 50} more[/dim]")


# -- proactive suggestions commands ---------------------------------------------


@cli.group(name="suggest")
def suggest_group() -> None:
    """Proactive memory suggestions from conversation analysis."""
    pass


@suggest_group.command(name="analyze")
@click.argument("transcript_path", type=click.Path(exists=True))
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def suggest_analyze(transcript_path: str, as_json: bool) -> None:
    """Analyze a transcript and suggest memories to save.

    Example: memo suggest analyze /path/to/transcript.jsonl
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    from pathlib import Path

    from memo.capture import _read_last_exchange

    # For now, just use the last exchange as a sample
    # In a full implementation, would analyze the full transcript
    pair = _read_last_exchange(Path(transcript_path))
    if pair is None:
        console.print("[yellow]Could not read transcript[/yellow]")
        return

    user_text, assistant_text = pair
    turns = [{"user": user_text, "assistant": assistant_text}]

    suggestions = mem.proactive.analyze_conversation(turns, limit=3)

    if as_json:
        click.echo(json.dumps([s.__dict__ for s in suggestions], indent=2))
        return

    if not suggestions:
        console.print("[dim]No suggestions found[/dim]")
        return

    console.print(f"[bold]Found {len(suggestions)} suggestions[/bold]")
    console.print()

    for i, s in enumerate(suggestions, 1):
        console.print(f"[cyan]{i}. {s.title}[/cyan]")
        console.print(f"   Type: {s.type}")
        console.print(f"   Confidence: {s.confidence:.2f}")
        console.print(f"   Tags: {', '.join(s.tags)}")
        console.print(f"   Rationale: {s.rationale}")
        console.print(f"   Snippet: {s.body_snippet[:100]}")
        console.print()


@suggest_group.command(name="feedback-stats")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def suggest_feedback_stats(as_json: bool) -> None:
    """Show statistics on suggestion feedback (acceptance rate).

    Example: memo suggest feedback-stats
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    stats = mem.proactive.get_feedback_stats()

    if as_json:
        click.echo(json.dumps(stats, indent=2))
        return

    console.print("[bold]Suggestion Feedback Stats[/bold]")
    console.print()
    console.print(f"Total suggestions: {stats['total']}")
    console.print(f"Accepted: {stats['accepted']}")
    console.print(f"Rejected: {stats['rejected']}")
    console.print(f"Acceptance rate: {stats['acceptance_rate']:.2%}")


@suggest_group.command(name="patterns")
@click.argument("transcript_path", type=click.Path(exists=True))
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def suggest_patterns(transcript_path: str, as_json: bool) -> None:
    """Detect patterns in a transcript (recurring themes, decisions, etc.).

    Example: memo suggest patterns /path/to/transcript.jsonl
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    from pathlib import Path

    patterns = mem.proactive.detect_patterns(Path(transcript_path))

    if as_json:
        click.echo(json.dumps(patterns, indent=2))
        return

    console.print("[bold]Conversation Patterns[/bold]")
    console.print()
    console.print(f"Total turns: {patterns['total_turns']}")
    console.print(f"Decision points: {patterns['decision_points']}")
    console.print(f"Technical discoveries: {patterns['technical_discoveries']}")
    console.print(f"Recurring themes: {', '.join(patterns['recurring_themes'])}")


# -- versioning commands ------------------------------------------------------


@cli.group(name="version")
def version_group() -> None:
    """Memory versioning — track changes, visualize diffs, rollback."""
    pass


@version_group.command(name="history")
@click.argument("memoria_id")
@click.option("--limit", type=int, default=10,
              help="Max versions to show (default: 10)")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def version_history(memoria_id: str, limit: int, as_json: bool) -> None:
    """Show version history for a memoria.

    Example: memo version history abc123
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    versions = mem.versioning.get_version_history(memoria_id, limit=limit)

    if as_json:
        click.echo(json.dumps([v.__dict__ for v in versions], indent=2))
        return

    if not versions:
        console.print(f"[dim]No version history for memoria {memoria_id[:8]}[/dim]")
        return

    console.print(f"[bold]Version History for {memoria_id[:8]}[/bold]")
    console.print()

    table = Table()
    table.add_column("Version ID", style="cyan")
    table.add_column("Timestamp", style="dim")
    table.add_column("Title", style="yellow")
    table.add_column("Type", style="green")
    table.add_column("Reason", style="magenta")

    for v in versions[:20]:
        table.add_row(
            str(v.version_id),
            v.timestamp[:19],
            v.title[:40],
            v.type,
            v.reason or "—",
        )

    console.print(table)
    if len(versions) > 20:
        console.print(f"[dim]...and {len(versions) - 20} more[/dim]")


@version_group.command(name="diff")
@click.argument("memoria_id")
@click.option("--version-a", type=int, help="First version ID (default: latest)")
@click.option("--version-b", type=int, help="Second version ID (default: latest-1)")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def version_diff(memoria_id: str, version_a: int | None, version_b: int | None, as_json: bool) -> None:
    """Show diff between two versions of a memoria.

    Example: memo version diff abc123 --version-a 1 --version-b 2
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    diff = mem.versioning.diff_versions(memoria_id, version_a, version_b)

    if as_json:
        click.echo(json.dumps(diff.__dict__ if diff else None, indent=2))
        return

    if diff is None:
        console.print("[yellow]Could not generate diff[/yellow]")
        return

    console.print(f"[bold]Diff for {memoria_id[:8]}[/bold]")
    console.print(f"[dim]v{diff.version_a} → v{diff.version_b}[/dim]")
    console.print()
    console.print(diff.unified_diff)


@version_group.command(name="rollback")
@click.argument("memoria_id")
@click.argument("version_id", type=int)
@click.option("--reason", help="Reason for the rollback")
@click.confirmation_option(prompt="This will restore the memoria to the specified version. Continue?")
def version_rollback(memoria_id: str, version_id: int, reason: str | None) -> None:
    """Rollback a memoria to a previous version.

    Example: memo version rollback abc123 1 --reason "Mistake in update"
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    success = mem.versioning.rollback_to_version(memoria_id, version_id, reason)

    if success:
        console.print(f"[green]Rolled back {memoria_id[:8]} to version {version_id}[/green]")
    else:
        console.print("[red]Failed to rollback[/red]")


# -- query composition commands ------------------------------------------------


@cli.group(name="query")
def query_group() -> None:
    """Query composition and saved queries."""
    pass


@query_group.command(name="save")
@click.argument("name")
@click.argument("query_text")
@click.option("--type", "type_filter", help="Filter by memoria type")
@click.option("--tags", "tags_filter", multiple=True, help="Filter by tags")
@click.option("--date-from", help="Start date (ISO format)")
@click.option("--date-to", help="End date (ISO format)")
@click.option("--mode", "search_mode", type=click.Choice(["vec", "bm25", "hybrid"]), default="hybrid",
              help="Search mode (default: hybrid)")
@click.option("--limit", type=int, default=10, help="Result limit")
@click.option("--description", help="Query description")
@click.option("--execute", is_flag=True, help="Execute the query after saving")
def query_save(
    name: str,
    query_text: str,
    type_filter: str | None,
    tags_filter: tuple[str, ...],
    date_from: str | None,
    date_to: str | None,
    search_mode: str,
    limit: int,
    description: str | None,
    execute: bool,
) -> None:
    """Save a query for reuse.

    Example: memo query save "MLX decisions" "MLX" --type decision --execute
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    if execute:
        result = mem.query_composer.compose_and_save(
            name=name,
            query_text=query_text,
            type_filter=type_filter,
            tags_filter=list(tags_filter),
            date_from=date_from,
            date_to=date_to,
            search_mode=search_mode,
            limit=limit,
            description=description,
        )
        console.print(f"[green]Saved and executed query '{name}'[/green]")
        console.print(f"Results: {result.count}")
    else:
        mem.query_composer.query_store.save_query(
            name=name,
            query_text=query_text,
            type_filter=type_filter,
            tags_filter=list(tags_filter),
            date_from=date_from,
            date_to=date_to,
            search_mode=search_mode,
            limit=limit,
            description=description,
        )
        console.print(f"[green]Saved query '{name}'[/green]")


@query_group.command(name="list")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def query_list(as_json: bool) -> None:
    """List all saved queries.

    Example: memo query list
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    queries = mem.query_composer.query_store.list_queries()

    if as_json:
        click.echo(json.dumps([q.__dict__ for q in queries], indent=2))
        return

    if not queries:
        console.print("[dim]No saved queries[/dim]")
        return

    table = Table(title="Saved Queries")
    table.add_column("Name", style="cyan")
    table.add_column("Query Text", style="yellow")
    table.add_column("Type Filter", style="green")
    table.add_column("Mode", style="magenta")
    table.add_column("Description", style="dim")

    for q in queries[:20]:
        table.add_row(
            q.name,
            q.query_text[:40],
            q.type_filter or "—",
            q.search_mode,
            q.description or "—",
        )

    console.print(table)
    if len(queries) > 20:
        console.print(f"[dim]...and {len(queries) - 20} more[/dim]")


@query_group.command(name="run")
@click.argument("name")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def query_run(name: str, as_json: bool) -> None:
    """Execute a saved query.

    Example: memo query run "MLX decisions"
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    query = mem.query_composer.query_store.get_query(name)
    if not query:
        console.print(f"[yellow]Query '{name}' not found[/yellow]")
        return

    result = mem.query_composer.execute_query(query)

    if as_json:
        # Convert results to dict format
        results_dict = [r.__dict__ for r in result.results]
        click.echo(json.dumps({
            "query_name": result.query_name,
            "count": result.count,
            "executed_at": result.executed_at,
            "results": results_dict,
        }, indent=2))
        return

    console.print(f"[bold]Query: {name}[/bold]")
    console.print(f"Results: {result.count}")
    console.print()

    for r in result.results[:10]:
        console.print(f"  [cyan]{r.id[:8]}[/cyan] {r.title}")
        console.print(f"    {r.body[:100]}")

    if len(result.results) > 10:
        console.print(f"  [dim]...and {len(result.results) - 10} more[/dim]")


@query_group.command(name="delete")
@click.argument("name")
@click.confirmation_option(prompt="Delete this saved query?")
def query_delete(name: str) -> None:
    """Delete a saved query.

    Example: memo query delete "MLX decisions"
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    success = mem.query_composer.query_store.delete_query(name)

    if success:
        console.print(f"[green]Deleted query '{name}'[/green]")
    else:
        console.print(f"[yellow]Query '{name}' not found[/yellow]")


# -- federation commands -------------------------------------------------------


@cli.group(name="federation")
def federation_group() -> None:
    """Multi-vault federation — search across multiple vaults."""
    pass


@federation_group.command(name="add-vault")
@click.argument("name")
@click.argument("path")
@click.option("--weight", type=float, default=1.0, help="Vault weight for ranking")
def federation_add_vault(name: str, path: str, weight: float) -> None:
    """Add a vault to the federation.

    Example: memo federation add-vault work-vault /path/to/work/memo --weight 1.5
    """
    cfg = Config.from_env()
    from memo.federation import FederationConfig

    config = FederationConfig(cfg.state_dir / "federation.json")
    config.add_vault(name, path, weight)

    console.print(f"[green]Added vault '{name}'[/green]")


@federation_group.command(name="list-vaults")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def federation_list_vaults(as_json: bool) -> None:
    """List all configured vaults.

    Example: memo federation list-vaults
    """
    cfg = Config.from_env()
    from memo.federation import FederationConfig

    config = FederationConfig(cfg.state_dir / "federation.json")
    vaults = config.list_vaults()

    if as_json:
        click.echo(json.dumps([v.__dict__ for v in vaults], indent=2))
        return

    if not vaults:
        console.print("[dim]No vaults configured[/dim]")
        return

    table = Table(title="Federated Vaults")
    table.add_column("Name", style="cyan")
    table.add_column("Path", style="yellow")
    table.add_column("Weight", style="green")
    table.add_column("Enabled", style="magenta")

    for v in vaults:
        table.add_row(
            v.name,
            v.path,
            str(v.weight),
            "Yes" if v.enabled else "No",
        )

    console.print(table)


@federation_group.command(name="remove-vault")
@click.argument("name")
@click.confirmation_option(prompt="Remove this vault from federation?")
def federation_remove_vault(name: str) -> None:
    """Remove a vault from the federation.

    Example: memo federation remove-vault work-vault
    """
    cfg = Config.from_env()
    from memo.federation import FederationConfig

    config = FederationConfig(cfg.state_dir / "federation.json")
    success = config.remove_vault(name)

    if success:
        console.print(f"[green]Removed vault '{name}'[/green]")
    else:
        console.print(f"[yellow]Vault '{name}' not found[/yellow]")


@federation_group.command(name="search")
@click.argument("query")
@click.option("--limit", type=int, default=10, help="Result limit")
@click.option("--mode", type=click.Choice(["vec", "bm25", "hybrid"]), default="hybrid",
              help="Search mode (default: hybrid)")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def federation_search(query: str, limit: int, mode: str, as_json: bool) -> None:
    """Search across all federated vaults.

    Example: memo federation search "MLX" --limit 20
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    results = mem.federation.search(query, limit=limit, mode=mode)

    if as_json:
        click.echo(json.dumps([r.__dict__ for r in results], indent=2))
        return

    if not results:
        console.print("[dim]No results found[/dim]")
        return

    table = Table(title=f"Federated Search Results for '{query}'")
    table.add_column("ID", style="cyan")
    table.add_column("Vault", style="yellow")
    table.add_column("Title", style="green")
    table.add_column("Score", style="magenta")

    for r in results[:20]:
        table.add_row(
            r.memoria_id[:8],
            r.vault_name,
            r.title[:40],
            f"{r.score:.3f}",
        )

    console.print(table)
    if len(results) > 20:
        console.print(f"[dim]...and {len(results) - 20} more[/dim]")


# -- sync & backup commands ------------------------------------------------------


@cli.group(name="backup")
def backup_group() -> None:
    """Backup management — create, list, restore backups."""
    pass


@backup_group.command(name="create")
@click.option("--compress/--no-compress", default=True, help="Compress backup")
@click.option("--name", help="Backup name")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def backup_create(compress: bool, name: str | None, as_json: bool) -> None:
    """Create a backup of the entire vault.

    Example: memo backup create --name "pre-migration"
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    metadata = mem.backup.create_backup(compress=compress, name=name)

    if as_json:
        click.echo(json.dumps(metadata.__dict__, indent=2))
        return

    console.print("[bold]Backup Created[/bold]")
    console.print()
    console.print(f"Timestamp: {metadata.timestamp}")
    console.print(f"Memorias: {metadata.memoria_count}")
    console.print(f"Checksum: {metadata.checksum[:16]}...")
    console.print(f"Size: {metadata.compressed_size:,} bytes (compressed)")
    console.print(f"Original: {metadata.original_size:,} bytes")


@backup_group.command(name="list")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def backup_list(as_json: bool) -> None:
    """List all available backups.

    Example: memo backup list
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    backups = mem.backup.list_backups()

    if as_json:
        click.echo(json.dumps([b.__dict__ for b in backups], indent=2))
        return

    if not backups:
        console.print("[dim]No backups found[/dim]")
        return

    table = Table(title="Available Backups")
    table.add_column("Name", style="cyan")
    table.add_column("Timestamp", style="yellow")
    table.add_column("Size", style="green")

    for b in backups[:20]:
        table.add_row(
            b.timestamp[:19],
            b.timestamp[:19],
            f"{b.compressed_size:,} bytes",
        )

    console.print(table)
    if len(backups) > 20:
        console.print(f"[dim]...and {len(backups) - 20} more[/dim]")


@backup_group.command(name="restore")
@click.argument("backup_name")
@click.option("--no-memorias", "skip_memorias", is_flag=True, help="Skip memoria files")
@click.option("--no-dbs", "skip_dbs", is_flag=True, help="Skip databases")
@click.confirmation_option(prompt="This will restore from backup. Current data may be overwritten. Continue?")
def backup_restore(backup_name: str, skip_memorias: bool, skip_dbs: bool) -> None:
    """Restore from a backup.

    Example: memo backup restore backup_2026-01-01-12-00-00
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    success = mem.backup.restore_backup(
        backup_name,
        restore_memorias=not skip_memorias,
        restore_dbs=not skip_dbs,
    )

    if success:
        console.print(f"[green]Restored from '{backup_name}'[/green]")
    else:
        console.print("[red]Failed to restore[/red]")


@cli.group(name="sync")
def sync_group() -> None:
    """Multi-vault sync — sync between vaults."""
    pass


@sync_group.command(name="diff")
@click.option("--remote", help="Path to remote vault")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def sync_diff(remote: str | None, as_json: bool) -> None:
    """Compute diff between local and remote vaults.

    Example: memo sync diff --remote /path/to/remote/memo
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    from pathlib import Path

    from memo.sync import SyncManager
    remote_path = Path(remote) if remote else None

    sync_mgr = SyncManager(mem, remote_path=remote_path)
    diff = sync_mgr.compute_diff()

    if as_json:
        click.echo(json.dumps(diff.__dict__, indent=2))
        return

    console.print("[bold]Sync Diff[/bold]")
    console.print()
    console.print(f"New: {len(diff.new)}")
    console.print(f"Modified: {len(diff.modified)}")
    console.print(f"Deleted: {len(diff.deleted)}")
    console.print(f"Conflicts: {len(diff.conflicts)}")


@sync_group.command(name="push")
@click.option("--remote", help="Path to remote vault")
def sync_push(remote: str | None) -> None:
    """Push local changes to remote vault.

    Example: memo sync push --remote /path/to/remote/memo
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    from pathlib import Path

    from memo.sync import SyncManager
    remote_path = Path(remote) if remote else None

    sync_mgr = SyncManager(mem, remote_path=remote_path)
    diff = sync_mgr.sync(direction="push")

    console.print("[bold]Push Sync[/bold]")
    console.print(f"Modified: {len(diff.modified)}")
    console.print(f"Deleted: {len(diff.deleted)}")


@sync_group.command(name="pull")
@click.option("--remote", help="Path to remote vault")
def sync_pull(remote: str | None) -> None:
    """Pull remote changes to local vault.

    Example: memo sync pull --remote /path/to/remote/memo
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    from pathlib import Path

    from memo.sync import SyncManager
    remote_path = Path(remote) if remote else None

    sync_mgr = SyncManager(mem, remote_path=remote_path)
    diff = sync_mgr.sync(direction="pull")

    console.print("[bold]Pull Sync[/bold]")
    console.print(f"New: {len(diff.new)}")
    console.print(f"Modified: {len(diff.modified)}")


@sync_group.command(name="both")
@click.option("--remote", help="Path to remote vault")
def sync_both(remote: str | None) -> None:
    """Sync both directions (bidirectional).

    Example: memo sync both --remote /path/to/remote/memo
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    from pathlib import Path

    from memo.sync import SyncManager
    remote_path = Path(remote) if remote else None

    sync_mgr = SyncManager(mem, remote_path=remote_path)
    diff = sync_mgr.sync(direction="both")

    console.print("[bold]Bidirectional Sync[/bold]")
    console.print(f"New: {len(diff.new)}")
    console.print(f"Modified: {len(diff.modified)}")
    console.print(f"Deleted: {len(diff.deleted)}")
    console.print(f"Conflicts: {len(diff.conflicts)}")


# -- encryption commands ----------------------------------------------------------


@cli.group(name="encrypt")
def encrypt_group() -> None:
    """Memory encryption — encrypt sensitive memorias."""
    pass


@encrypt_group.command(name="unlock")
@click.argument("password")
def encrypt_unlock(password: str) -> None:
    """Unlock the vault with password.

    Derives master key from password and stores in memory.

    Example: memo encrypt unlock mypassword
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    success = mem.encryption.unlock(password)

    if success:
        console.print("[green]Vault unlocked[/green]")
    else:
        console.print("[red]Failed to unlock vault[/red]")


@encrypt_group.command(name="lock")
def encrypt_lock() -> None:
    """Lock the vault (clear master key from memory).

    Example: memo encrypt lock
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    mem.encryption.lock()
    console.print("[green]Vault locked[/green]")


@encrypt_group.command(name="status")
def encrypt_status() -> None:
    """Check if vault is unlocked.

    Example: memo encrypt status
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    if mem.encryption.is_unlocked():
        console.print("[green]Vault is unlocked[/green]")
    else:
        console.print("[yellow]Vault is locked[/yellow]")


# -- sharing commands -----------------------------------------------------------


@cli.group(name="share")
def share_group() -> None:
    """Memory sharing — share memorias with others."""
    pass


@share_group.command(name="with-user")
@click.argument("memoria_id")
@click.argument("shared_with")
@click.option("--permission", type=click.Choice(["read", "comment", "edit", "admin"]), default="read",
              help="Permission level")
@click.option("--expires-days", type=int, help="Days until expiration")
def share_with_user(memoria_id: str, shared_with: str, permission: str, expires_days: int | None) -> None:
    """Share a memoria with a user.

    Example: memo share with-user abc123 user@example.com --permission comment
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    share = mem.sharing.share_with_user(
        memoria_id=memoria_id,
        shared_with=shared_with,
        permission=permission,
        expires_days=expires_days,
    )

    console.print(f"[green]Shared {memoria_id[:8]} with {shared_with}[/green]")
    console.print(f"Permission: {permission}")
    if share.expires_at:
        console.print(f"Expires: {share.expires_at}")


@share_group.command(name="unshare")
@click.argument("memoria_id")
@click.argument("shared_with")
def share_unshare(memoria_id: str, shared_with: str) -> None:
    """Unshare a memoria from a user.

    Example: memo share unshare abc123 user@example.com
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    success = mem.sharing.unshare_with_user(memoria_id, shared_with)

    if success:
        console.print(f"[green]Unshared {memoria_id[:8]} from {shared_with}[/green]")
    else:
        console.print("[yellow]Share not found[/yellow]")


@share_group.command(name="create-link")
@click.argument("memoria_id")
@click.option("--permission", type=click.Choice(["read", "comment", "edit"]), default="read",
              help="Permission level")
@click.option("--expires-hours", type=int, default=24, help="Hours until expiration")
@click.option("--password", help="Optional password protection")
def share_create_link(memoria_id: str, permission: str, expires_hours: int, password: str | None) -> None:
    """Create a temporary sharing link.

    Example: memo share create-link abc123 --permission read --expires-hours 48
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    link = mem.sharing.create_link(
        memoria_id=memoria_id,
        permission=permission,
        expires_hours=expires_hours,
        password=password,
    )

    console.print("[green]Share link created[/green]")
    console.print(f"Link: {link}")
    console.print(f"Expires in {expires_hours} hours")


@share_group.command(name="list")
@click.argument("memoria_id")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def share_list(memoria_id: str, as_json: bool) -> None:
    """List all shares for a memoria.

    Example: memo share list abc123
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    shares = mem.sharing.share_store.get_shares(memoria_id)

    if as_json:
        click.echo(json.dumps([s.__dict__ for s in shares], indent=2))
        return

    if not shares:
        console.print("[dim]No shares found[/dim]")
        return

    table = Table(title=f"Shares for {memoria_id[:8]}")
    table.add_column("Shared With", style="cyan")
    table.add_column("Permission", style="yellow")
    table.add_column("Shared At", style="green")
    table.add_column("Expires", style="magenta")

    for s in shares[:20]:
        table.add_row(
            s.shared_with,
            s.permission,
            s.shared_at[:19],
            s.expires_at[:19] if s.expires_at else "Never",
        )

    console.print(table)
    if len(shares) > 20:
        console.print(f"[dim]...and {len(shares) - 20} more[/dim]")


@share_group.command(name="comment")
@click.argument("memoria_id")
@click.argument("content")
@click.option("--author", default="user", help="Comment author")
@click.option("--parent", help="Parent comment ID for replies")
def share_comment(memoria_id: str, content: str, author: str, parent: str | None) -> None:
    """Add a comment to a memoria.

    Example: memo share comment abc123 "This is a comment" --author "John Doe"
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    comment = mem.sharing.add_comment(
        memoria_id=memoria_id,
        author=author,
        content=content,
        parent_id=parent,
    )

    console.print("[green]Comment added[/green]")
    console.print(f"Author: {comment.author}")
    console.print(f"Content: {comment.content}")


@share_group.command(name="comments")
@click.argument("memoria_id")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def share_comments(memoria_id: str, as_json: bool) -> None:
    """List all comments for a memoria.

    Example: memo share comments abc123
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    comments = mem.sharing.get_comments(memoria_id)

    if as_json:
        click.echo(json.dumps([c.__dict__ for c in comments], indent=2))
        return

    if not comments:
        console.print("[dim]No comments found[/dim]")
        return

    table = Table(title=f"Comments for {memoria_id[:8]}")
    table.add_column("Author", style="cyan")
    table.add_column("Content", style="yellow")
    table.add_column("Created", style="green")

    for c in comments[:20]:
        table.add_row(
            c.author,
            c.content[:50],
            c.created_at[:19],
        )

    console.print(table)
    if len(comments) > 20:
        console.print(f"[dim]...and {len(comments) - 20} more[/dim]")


# -- analytics commands ----------------------------------------------------------


@cli.group(name="analytics")
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
    console.print(f"Total Memorias: {metrics.total_memorias}")
    console.print(f"Total Entities: {metrics.total_entities}")
    console.print(f"Growth Rate: {metrics.growth_rate:.2f} memorias/day")
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


# -- import/export commands ------------------------------------------------------


@cli.group(name="import")
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


@cli.group(name="export")
def export_group() -> None:
    """Export memorias to other formats."""
    pass


@export_group.command(name="json")
@click.argument("output_path", type=click.Path())
def export_json(output_path: str) -> None:
    """Export to JSON file.

    Example: memo export json /path/to/export.json
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    from pathlib import Path
    result = mem.import_export.export_to(Path(output_path), "json")

    console.print("[green]Export complete[/green]")
    console.print(f"Exported: {result.exported_count}")
    console.print(f"Output: {result.output_path}")


@export_group.command(name="csv")
@click.argument("output_path", type=click.Path())
def export_csv(output_path: str) -> None:
    """Export to CSV file.

    Example: memo export csv /path/to/export.csv
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    from pathlib import Path
    result = mem.import_export.export_to(Path(output_path), "csv")

    console.print("[green]Export complete[/green]")
    console.print(f"Exported: {result.exported_count}")
    console.print(f"Output: {result.output_path}")


@export_group.command(name="markdown-bundle")
@click.argument("output_path", type=click.Path())
def export_markdown_bundle(output_path: str) -> None:
    """Export to Markdown bundle (zip).

    Example: memo export markdown-bundle /path/to/export.zip
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    from pathlib import Path
    result = mem.import_export.export_to(Path(output_path), "markdown_bundle")

    console.print("[green]Export complete[/green]")
    console.print(f"Exported: {result.exported_count}")
    console.print(f"Output: {result.output_path}")


# -- autonomous agent commands (THE GAMECHANGER) --------------------------------


@cli.group(name="agent")
def agent_group() -> None:
    """Autonomous Memory Agent — razonamiento causal y síntesis de conocimiento."""
    pass


@agent_group.command(name="synthesize")
@click.argument("topic")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def agent_synthesize(topic: str, as_json: bool) -> None:
    """Sintetiza nuevo conocimiento a partir de memorias existentes.

    THE GAMECHANGER: genera insights que NO existían antes combinando
    y razonando sobre memorias existentes.

    Example: memo agent synthesize "MLX edge computing implications"
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    synthesis = mem.agent.synthesize_knowledge(topic)

    if as_json:
        click.echo(json.dumps(synthesis.model_dump(), indent=2))
        return

    console.print("[bold]Synthesis Result[/bold]")
    console.print()
    console.print("[cyan]New Insight:[/cyan]")
    console.print(synthesis.new_insight)
    console.print()
    console.print(f"[green]Confidence:[/green] {synthesis.confidence:.2f}")
    console.print(f"[green]Novelty Score:[/green] {synthesis.novelty_score:.2f}")
    console.print()
    console.print("[yellow]Supporting Memorias:[/yellow]")
    for mid in synthesis.supporting_memorias:
        console.print(f"  {mid[:8]}")


@agent_group.command(name="investigate")
@click.argument("goal")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def agent_investigate(goal: str, as_json: bool) -> None:
    """Planifica y ejecuta una investigación compleja.

    Example: memo agent investigate "implications of using MLX for edge devices"
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    plan = mem.agent.plan_investigation(goal)

    if as_json:
        click.echo(json.dumps(plan.model_dump(), indent=2))
        return

    console.print("[bold]Investigation Plan[/bold]")
    console.print()
    console.print(f"Goal: {plan.goal}")
    console.print(f"Complexity: {plan.estimated_complexity}/10")
    console.print(f"Insight Value: {plan.estimated_insight_value}/10")
    console.print()
    console.print("[bold]Steps:[/bold]")
    for i, step in enumerate(plan.steps, 1):
        console.print(f"  {i}. {step}")


@agent_group.command(name="discover")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def agent_discover(as_json: bool) -> None:
    """Descubrimiento proactivo: explora el corpus sin que el usuario lo pida.

    El agente identifica áreas del corpus que podrían contener insights
    no descubiertos y los explora proactivamente.
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    discoveries = mem.agent.proactive_discovery()

    if as_json:
        click.echo(json.dumps([d.model_dump() for d in discoveries], indent=2))
        return

    console.print(f"[bold]Proactive Discoveries[/bold] ({len(discoveries)} found)")
    console.print()

    for i, discovery in enumerate(discoveries, 1):
        console.print(f"[cyan]{i}. Insight:[/cyan]")
        console.print(f"  {discovery.new_insight[:150]}...")
        console.print(f"  [green]Novelty: {discovery.novelty_score:.2f}[/green]")


@agent_group.command(name="thoughts")
@click.option("--type", "thought_type", help="Filter by thought type")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def agent_thoughts(thought_type: str | None, as_json: bool) -> None:
    """Ver los pensamientos del agente (meta-cognición).

    Example: memo agent thoughts --type insight
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    thoughts = mem.agent.get_thoughts(thought_type)

    if as_json:
        click.echo(json.dumps([t.model_dump() for t in thoughts], indent=2))
        return

    console.print(f"[bold]Agent Thoughts[/bold] ({len(thoughts)} total)")
    console.print()

    for t in thoughts[-10:]:  # Show last 10
        console.print(f"[cyan]{t.thought_type}:[/cyan] {t.content[:100]}...")
        console.print(f"  [dim]{t.timestamp}[/dim]")


@agent_group.command(name="think")
@click.argument("thought")
@click.option("--type", "thought_type", default="hypothesis", help="Thought type")
def agent_think(thought: str, thought_type: str) -> None:
    """Registra un pensamiento del agente.

    Example: memo agent think "Maybe MLX is ideal for edge because..." --type hypothesis
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    agent_thought = mem.agent.think(thought, thought_type)

    console.print("[green]Thought registered[/green]")
    console.print(f"Type: {agent_thought.thought_type}")
    console.print(f"Content: {agent_thought.content}")


# -- multi-modal commands (gamechanger #17) ---------------------------------------


@cli.group(name="multimodal")
def multimodal_group() -> None:
    """Memoria Multi-Modal con Embeddings Universales."""
    pass


@multimodal_group.command(name="add-image")
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--memoria-id", help="ID de memoria asociada")
def multimodal_add_image(image_path: str, memoria_id: str | None) -> None:
    """Agrega imagen al corpus multi-modal.

    Example: memo multimodal add-image /path/to/image.png --memoria-id abc123
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    from pathlib import Path
    content = mem.multimodal.add_image(Path(image_path), memoria_id)

    console.print("[green]Image added[/green]")
    console.print(f"Content ID: {content.id}")
    console.print(f"Modality: {content.modality}")


@multimodal_group.command(name="add-audio")
@click.argument("audio_path", type=click.Path(exists=True))
@click.option("--memoria-id", help="ID de memoria asociada")
def multimodal_add_audio(audio_path: str, memoria_id: str | None) -> None:
    """Agrega audio al corpus multi-modal.

    Example: memo multimodal add-audio /path/to/audio.mp3 --memoria-id abc123
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    from pathlib import Path
    content = mem.multimodal.add_audio(Path(audio_path), memoria_id)

    console.print("[green]Audio added[/green]")
    console.print(f"Content ID: {content.id}")
    console.print(f"Modality: {content.modality}")


@multimodal_group.command(name="search-images")
@click.argument("query")
@click.option("--limit", type=int, default=10, help="Máximo de resultados")
def multimodal_search_images(query: str, limit: int) -> None:
    """Busca con texto, encuentra imágenes.

    Example: memo multimodal search-images "architecture diagram"
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    results = mem.multimodal.search.search_text_find_images(query, limit=limit)

    console.print(f"[bold]Results: {len(results)} images[/bold]")
    for r in results:
        console.print(f"  {r.content_id[:8]} - similarity: {r.similarity:.2f}")


@multimodal_group.command(name="search-audio")
@click.argument("query")
@click.option("--limit", type=int, default=10, help="Máximo de resultados")
def multimodal_search_audio(query: str, limit: int) -> None:
    """Busca con texto, encuentra audio.

    Example: memo multimodal search-audio "meeting notes"
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    results = mem.multimodal.search.search_text_find_audio(query, limit=limit)

    console.print(f"[bold]Results: {len(results)} audio[/bold]")
    for r in results:
        console.print(f"  {r.content_id[:8]} - similarity: {r.similarity:.2f}")


@multimodal_group.command(name="search-all")
@click.argument("query")
@click.option("--limit", type=int, default=10, help="Máximo de resultados por modalidad")
def multimodal_search_all(query: str, limit: int) -> None:
    """Busca en todas las modalidades.

    Example: memo multimodal search-all "project documentation"
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    results = mem.multimodal.search.search_all_modalities(query, limit=limit)

    console.print("[bold]Results across modalities[/bold]")
    for modality, mod_results in results.items():
        console.print(f"\n[cyan]{modality}:[/cyan] {len(mod_results)} results")
    for r in mod_results[:5]:
        console.print(f"  {r.content_id[:8]} - similarity: {r.similarity:.2f}")


# -- collaborative commands (gamechanger #18) -----------------------------------


@cli.group(name="collaborative")
def collaborative_group() -> None:
    """Memoria Social Colaborativa con Grafo de Conocimiento Compartido."""
    pass


@collaborative_group.command(name="share-connection")
@click.argument("user-id")
@click.argument("entity-a")
@click.argument("entity-b")
@click.argument("relationship")
@click.option("--confidence", type=float, default=0.7, help="Confidence score")
def collaborative_share_connection(user_id: str, entity_a: str, entity_b: str, relationship: str, confidence: float) -> None:
    """Comparte una conexión descubierta con la comunidad.

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
    """Ver conexiones compartidas para una entidad.

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
@click.option("--limit", type=int, default=10, help="Máximo de resultados")
def collaborative_recommend(entity: str, limit: int) -> None:
    """Obtiene conexiones recomendadas basadas en patrones colectivos.

    Example: memo collaborative recommend MLX
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    recommendations = mem.collaborative.get_recommended_connections(entity, limit=limit)

    console.print(f"[bold]Recommended connections for {entity}[/bold]")
    for r in recommendations:
        console.print(f"  {r.entity_a} --{r.relationship}--> {r.entity_b}")
        console.print(f"    From: {r.from_user}, votes: {r.votes}, confidence: {r.confidence:.2f}")


@collaborative_group.command(name="share-insight")
@click.argument("user-id")
@click.argument("content")
def collaborative_share_insight(user_id: str, content: str) -> None:
    """Comparte un insight con la comunidad.

    Example: memo collaborative share-insight user123 "MLX is ideal for edge because..."
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    insight = mem.collaborative.share_insight(user_id, content)

    console.print("[green]Insight shared[/green]")
    console.print(f"Insight ID: {insight.insight_id}")
    console.print(f"Content: {content[:100]}...")


@collaborative_group.command(name="insights")
@click.option("--limit", type=int, default=10, help="Máximo de resultados")
def collaborative_insights(limit: int) -> None:
    """Ver los insights más votados de la comunidad.

    Example: memo collaborative insights
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    insights = mem.collaborative.get_top_insights(limit=limit)

    console.print("[bold]Top insights[/bold]")
    for i, insight in enumerate(insights, 1):
        console.print(f"[cyan]{i}.[/cyan] {insight.content[:100]}...")
        console.print(f"    Upvotes: {insight.upvotes}, Downvotes: {insight.downvotes}")


# -- cognitive commands (gamechanger #19) ---------------------------------------


@cli.group(name="cognitive")
def cognitive_group() -> None:
    """Memoria con Estado Mental del Usuario."""
    pass


@cognitive_group.command(name="set-state")
@click.argument("mental-state")
@click.argument("context-type")
@click.option("--goal", help="Current goal")
@click.option("--focus", help="Focus area")
@click.option("--energy", type=int, default=50, help="Energy level (0-100)")
@click.option("--stress", type=int, default=30, help="Stress level (0-100)")
def cognitive_set_state(mental_state: str, context_type: str, goal: str | None, focus: str | None, energy: int, stress: int) -> None:
    """Actualiza el estado mental del usuario.

    Example: memo cognitive set-state focused work --goal "Finish MLX integration" --focus MLX
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    state = mem.cognitive.update_mental_state(
        mental_state=mental_state,
        context_type=context_type,
        current_goal=goal,
        focus_area=focus,
        energy_level=energy,
        stress_level=stress,
    )

    console.print("[green]Mental state updated[/green]")
    console.print(f"State: {state.mental_state}")
    console.print(f"Context: {state.context_type}")
    console.print(f"Goal: {state.current_goal or 'None'}")
    console.print(f"Focus: {state.focus_area or 'None'}")


@cognitive_group.command(name="get-state")
def cognitive_get_state() -> None:
    """Obtiene el estado mental actual.

    Example: memo cognitive get-state
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    state = mem.cognitive.get_mental_state()

    if not state:
        console.print("[yellow]No mental state set[/yellow]")
        return

    console.print("[bold]Current Mental State[/bold]")
    console.print(f"State: {state.mental_state}")
    console.print(f"Context: {state.context_type}")
    console.print(f"Goal: {state.current_goal or 'None'}")
    console.print(f"Focus: {state.focus_area or 'None'}")
    console.print(f"Energy: {state.energy_level}/100")
    console.print(f"Stress: {state.stress_level}/100")


@cognitive_group.command(name="history")
@click.option("--limit", type=int, default=10, help="Máximo de resultados")
def cognitive_history(limit: int) -> None:
    """Ver el historial de estados mentales.

    Example: memo cognitive history --limit 5
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    history = mem.cognitive.tracker.get_history(limit=limit)

    console.print(f"[bold]Mental State History[/bold] ({len(history)} entries)")
    for h in history:
        console.print(f"[cyan]{h.timestamp}[/cyan]")
        console.print(f"  {h.mental_state} | {h.context_type}")
        console.print(f"  Goal: {h.current_goal or 'None'}, Focus: {h.focus_area or 'None'}")


@cognitive_group.command(name="suggestions")
@click.option("--limit", type=int, default=5, help="Máximo de sugerencias")
def cognitive_suggestions(limit: int) -> None:
    """Obtiene sugerencias proactivas basadas en estado mental.

    Example: memo cognitive suggestions
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    # Use the memory's search function
    def search_func(query: str, limit: int) -> list:
        return mem.search(query, limit=limit)

    suggestions = mem.cognitive.get_proactive_suggestions(search_func, limit=limit)

    console.print(f"[bold]Proactive Suggestions[/bold] ({len(suggestions)} found)")
    for s in suggestions:
        console.print(f"[cyan]{s.memoria_id[:8]}[/cyan] - {s.relevance_reason}")
        console.print(f"  Confidence: {s.confidence:.2f}")


# -- contradiction radar + dedupe ---------------------------------------------


@cli.group(name="contradict")
def contradict_group() -> None:
    """Detect and triage contradictions / staleness across the corpus.

    `scan` runs the LLM classifier over near-neighbor pairs and stores
    contradictions in a sidecar DB. `list` shows open pairs. `triage`
    walks them one by one and applies the user's verdict (fuse / keep
    newer / dismiss / etc.).
    """
    pass


def _short(text: str, n: int = 120) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= n else text[: n - 1] + "…"


def _fmt_pair_header(rec_a, rec_b, pair) -> str:
    rel = pair.relationship
    color = "red" if rel == "contradiction" else "yellow"
    return (
        f"[bold {color}]{rel.upper()}[/bold {color}] "
        f"conf={pair.confidence:.2f}  pair={pair.pair_id}"
    )


@contradict_group.command(name="scan")
@click.option("--top-k", type=int, default=5,
              help="Vec neighbors to consider per memoria (default: 5)")
@click.option("--sim-floor", type=float, default=0.55,
              help="Cosine floor; pairs below are skipped (default: 0.55)")
@click.option("--confidence", type=float, default=0.7,
              help="Min LLM confidence to store (default: 0.7)")
@click.option("--min-days-apart", type=int, default=1,
              help="Skip pairs whose updates are within N days (default: 1)")
@click.option("--max-memorias", type=int, default=2000,
              help="Cap on memorias visited (default: 2000)")
@click.option("--max-pairs", type=int, default=500,
              help="Cap on pairs sent to the LLM (default: 500)")
@click.option("--since", help="Only scan memorias updated on/after this ISO date")
@click.option("--type", "type_", help="Filter by memoria type")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def contradict_scan(
    top_k: int, sim_floor: float, confidence: float, min_days_apart: int,
    max_memorias: int, max_pairs: int, since: str | None, type_: str | None,
    as_json: bool,
) -> None:
    """Scan the corpus for contradiction/evolution pairs.

    Example: memo contradict scan --since 2026-04-01
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    console.print("[bold]Scanning corpus for contradictions…[/bold]")
    last_idx = {"n": 0}

    def progress(idx: int, total: int, title: str) -> None:
        if idx == total or idx - last_idx["n"] >= 25:
            console.print(f"[dim]  {idx}/{total}  {_short(title, 60)}[/dim]")
            last_idx["n"] = idx

    result = mem.contradict_scanner.scan_corpus(
        top_k=top_k,
        sim_floor=sim_floor,
        confidence_threshold=confidence,
        min_days_apart=min_days_apart,
        max_memorias=max_memorias,
        max_pairs=max_pairs,
        since=since,
        type_=type_,
        progress=progress,
    )

    payload = {
        "scanned_memorias": result.scanned_memorias,
        "pairs_examined": result.pairs_examined,
        "pairs_inserted": result.pairs_inserted,
        "pairs_refreshed": result.pairs_refreshed,
        "pairs_skipped_resolved": result.pairs_skipped_resolved,
        "contradictions_found": result.contradictions_found,
        "evolutions_found": result.evolutions_found,
    }
    if as_json:
        click.echo(json.dumps(payload, indent=2))
        return

    table = Table(title="Scan summary")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for k, v in payload.items():
        table.add_row(k.replace("_", " "), str(v))
    console.print(table)

    if result.contradictions_found or result.evolutions_found:
        console.print(
            "\n[green]→[/green] Run [cyan]memo contradict triage[/cyan] to resolve them."
        )


@contradict_group.command(name="list")
@click.option("--limit", type=int, default=20, help="Max rows (default: 20)")
@click.option("--min-confidence", type=float, default=0.0)
@click.option("--relationship", type=click.Choice(["contradiction", "evolution"]),
              help="Filter by relationship type")
@click.option("--status", type=click.Choice(
    ["open", "fused", "kept_newer", "kept_older", "evolved", "dismissed"]),
    default="open", help="Filter by status (default: open)")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def contradict_list(
    limit: int, min_confidence: float, relationship: str | None,
    status: str, as_json: bool,
) -> None:
    """List contradiction pairs from the sidecar DB."""
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    if status == "open":
        pairs = mem.contradict_store.list_open(
            limit=limit, min_confidence=min_confidence, relationship=relationship,
        )
    else:
        pairs = mem.contradict_store.list_all(status=status, limit=limit)
        if relationship:
            pairs = [p for p in pairs if p.relationship == relationship]

    if as_json:
        click.echo(json.dumps([p.__dict__ for p in pairs], indent=2, default=str))
        return

    if not pairs:
        console.print(f"[green]No pairs in status '{status}'[/green]")
        return

    table = Table(title=f"Contradiction pairs · status={status}")
    table.add_column("id", justify="right")
    table.add_column("rel")
    table.add_column("conf", justify="right")
    table.add_column("a")
    table.add_column("b")
    table.add_column("rationale")
    for p in pairs:
        table.add_row(
            str(p.pair_id),
            p.relationship,
            f"{p.confidence:.2f}",
            p.memoria_id_a[:8],
            p.memoria_id_b[:8],
            _short(p.rationale, 70),
        )
    console.print(table)


def _display_pair_excerpt(rec, label: str, *, stale_days: int = 180) -> None:
    from memo.contradict import is_stale
    age_marker = "  [red](stale)[/red]" if is_stale(rec.updated, stale_days) else ""
    console.print(
        f"[bold cyan]{label}[/bold cyan] · {rec.id[:8]} · "
        f"[dim]{rec.type}[/dim] · updated={rec.updated[:10]}{age_marker}"
    )
    console.print(f"  [bold]{rec.title}[/bold]")
    body = (rec.body or "").strip()
    if len(body) > 600:
        body = body[:599] + "…"
    for line in body.splitlines():
        console.print(f"  {line}")
    console.print()


_TRIAGE_HELP = """
Actions for each pair:
  f = fuse (LLM-merge both → new memoria, archive both)
  n = newer wins (keep newer, delete older)
  o = older wins (keep older, delete newer)
  e = evolved (legitimate evolution, mark resolved, keep both)
  d = dismiss (false positive)
  s = skip (leave as open)
  q = quit walker
""".strip()


@contradict_group.command(name="triage")
@click.option("--limit", type=int, default=20,
              help="Max pairs to walk in this session (default: 20)")
@click.option("--min-confidence", type=float, default=0.7,
              help="Skip pairs below this LLM confidence (default: 0.7)")
@click.option("--relationship", type=click.Choice(["contradiction", "evolution"]),
              help="Only walk pairs of this relationship type")
@click.option("--stale-days", type=int, default=180,
              help="Days threshold for the [stale] marker (default: 180)")
@click.option("--yes-fuse", is_flag=True,
              help="Auto-accept fuse without an extra confirmation prompt")
def contradict_triage(
    limit: int, min_confidence: float, relationship: str | None,
    stale_days: int, yes_fuse: bool,
) -> None:
    """Interactive triage walker over open contradiction pairs.

    Example: memo contradict triage --relationship contradiction
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    pairs = mem.contradict_store.list_open(
        limit=limit, min_confidence=min_confidence, relationship=relationship,
    )
    if not pairs:
        console.print("[green]No open pairs to triage.[/green]")
        return

    console.print(f"[bold]Walking {len(pairs)} pair(s).[/bold] Type [cyan]?[/cyan] for help.")

    for pair in pairs:
        rec_a = mem.get(pair.memoria_id_a)
        rec_b = mem.get(pair.memoria_id_b)
        if rec_a is None or rec_b is None:
            mem.contradict_store.resolve(
                pair.pair_id, "dismissed",
                note="auto: one side missing at triage time",
            )
            continue

        # Orient newer as "B" so the walker is always presented with
        # the same temporal layout (older on top, newer below).
        if rec_a.updated > rec_b.updated:
            rec_a, rec_b = rec_b, rec_a

        console.print()
        console.print(_fmt_pair_header(rec_a, rec_b, pair))
        if pair.rationale:
            console.print(f"[dim]rationale: {pair.rationale}[/dim]")
        console.print()
        _display_pair_excerpt(rec_a, "OLDER", stale_days=stale_days)
        _display_pair_excerpt(rec_b, "NEWER", stale_days=stale_days)

        while True:
            choice = click.prompt(
                "Action [f/n/o/e/d/s/q/?]",
                type=str, default="s", show_default=False,
            ).strip().lower()
            if choice == "?":
                console.print(_TRIAGE_HELP)
                continue
            if choice in {"f", "n", "o", "e", "d", "s", "q"}:
                break
            console.print("[red]Unknown action.[/red] Use ? for help.")

        if choice == "q":
            console.print("[dim]Stopping walker.[/dim]")
            break
        if choice == "s":
            continue
        if choice == "d":
            note = click.prompt("note (optional)", default="", show_default=False) or None
            mem.contradict_store.resolve(pair.pair_id, "dismissed", note=note)
            console.print("[dim]dismissed.[/dim]")
            continue
        if choice == "e":
            mem.contradict_store.resolve(pair.pair_id, "evolved")
            console.print("[dim]marked as evolved.[/dim]")
            continue
        if choice == "n":
            if click.confirm(f"Delete OLDER {rec_a.id[:8]}?", default=False):
                mem.delete(rec_a.id)
                mem.contradict_store.resolve(pair.pair_id, "kept_newer",
                                             note=f"deleted older {rec_a.id}")
                console.print(f"[green]kept newer.[/green] older {rec_a.id[:8]} deleted.")
            continue
        if choice == "o":
            if click.confirm(f"Delete NEWER {rec_b.id[:8]}?", default=False):
                mem.delete(rec_b.id)
                mem.contradict_store.resolve(pair.pair_id, "kept_older",
                                             note=f"deleted newer {rec_b.id}")
                console.print(f"[green]kept older.[/green] newer {rec_b.id[:8]} deleted.")
            continue
        if choice == "f":
            cluster = {
                "cluster_id": pair.pair_id,
                "relationship": "duplicate",
                "rationale": pair.rationale,
                "members": [
                    {
                        "id": rec_a.id, "title": rec_a.title,
                        "updated": rec_a.updated,
                        "body_preview": (rec_a.body or "")[:400],
                    },
                    {
                        "id": rec_b.id, "title": rec_b.title,
                        "updated": rec_b.updated,
                        "body_preview": (rec_b.body or "")[:400],
                    },
                ],
            }
            proposal = mem.consolidator.propose_merge(cluster)
            if proposal is None:
                console.print("[red]LLM declined to propose a merge. Skipping.[/red]")
                continue
            console.print(
                f"[bold]Proposed merged title:[/bold] {proposal.merged_title}"
            )
            console.print(f"[dim]strategy={proposal.merge_strategy}[/dim]")
            console.print(f"[dim]rationale={proposal.rationale}[/dim]")
            if yes_fuse or click.confirm("Apply this merge?", default=False):
                merge_result = mem.consolidator.apply_merge(proposal, dry_run=False)
                mem.contradict_store.resolve(
                    pair.pair_id, "fused",
                    note=f"merged into {merge_result.merged_id}",
                )
                console.print(
                    f"[green]fused →[/green] {merge_result.merged_id[:8] if merge_result.merged_id else 'n/a'}"
                )
            continue

    stats = mem.contradict_store.stats()
    console.print()
    console.print(f"[bold]Session stats:[/bold] {stats}")


@contradict_group.command(name="stats")
def contradict_stats() -> None:
    """Show counts of pairs grouped by status."""
    cfg = Config.from_env()
    mem = _get_memory(cfg)
    stats = mem.contradict_store.stats()
    if not stats:
        console.print("[dim]No pairs recorded yet. Run `memo contradict scan` first.[/dim]")
        return
    table = Table(title="Contradiction pairs by status")
    table.add_column("status")
    table.add_column("count", justify="right")
    for k, v in sorted(stats.items()):
        table.add_row(k, str(v))
    console.print(table)


@contradict_group.command(name="reopen")
@click.argument("pair_id", type=int)
def contradict_reopen(pair_id: int) -> None:
    """Send a resolved pair back to the open queue."""
    cfg = Config.from_env()
    mem = _get_memory(cfg)
    if mem.contradict_store.reopen(pair_id):
        console.print(f"[green]pair {pair_id} reopened.[/green]")
    else:
        console.print(f"[red]pair {pair_id} not found.[/red]")


# -- duplicate detection (exact-ish near-dups, no LLM gate) -------------------


@cli.command(name="dedupe")
@click.option("--threshold", type=float, default=0.92,
              help="Cosine threshold for near-duplicate clustering (default: 0.92)")
@click.option("--max-clusters", type=int, default=50,
              help="Max clusters to surface (default: 50)")
@click.option("--type", "type_", help="Filter by memoria type")
@click.option("--apply", "do_apply", is_flag=True,
              help="Interactively merge each cluster (default: list-only)")
@click.option("--dry-run", is_flag=True,
              help="With --apply: show merges without writing")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def dedupe_cmd(
    threshold: float, max_clusters: int, type_: str | None,
    do_apply: bool, dry_run: bool, as_json: bool,
) -> None:
    """Find and (optionally) merge near-duplicate memorias.

    Thin wrapper over `memo consolidate` with a higher default threshold,
    aimed at obvious dups (paste-restate, double-save, etc.) — not at
    semantic clustering. Use the lower-threshold `consolidate` group
    when you want LLM synthesis across loosely-related notes.
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    clusters = mem.consolidate(
        threshold=threshold,
        max_clusters=max_clusters,
        type_=type_,
    )
    dup_clusters = [c for c in clusters if c.get("relationship") in ("duplicate", "evolution")]

    if as_json:
        click.echo(json.dumps(dup_clusters, indent=2))
        return

    if not dup_clusters:
        console.print("[green]No near-duplicate clusters found at this threshold.[/green]")
        return

    console.print(f"[bold]Found {len(dup_clusters)} duplicate-like cluster(s).[/bold]")

    if not do_apply:
        for c in dup_clusters[:20]:
            console.print()
            console.print(
                f"[cyan]cluster {c.get('cluster_id', '?')}[/cyan] · "
                f"rel={c.get('relationship')} · n={len(c.get('members', []))}"
            )
            console.print(f"  [dim]{_short(c.get('summary', ''), 200)}[/dim]")
            for m in c.get("members", []):
                console.print(f"    - {m['id'][:8]} · {_short(m.get('title', ''), 70)}")
        if len(dup_clusters) > 20:
            console.print(f"[dim]…and {len(dup_clusters) - 20} more[/dim]")
        console.print()
        console.print("[dim]Re-run with --apply to merge interactively.[/dim]")
        return

    for c in dup_clusters:
        console.print()
        console.print(
            f"[cyan]cluster {c.get('cluster_id', '?')}[/cyan] · "
            f"rel={c.get('relationship')} · n={len(c.get('members', []))}"
        )
        for m in c.get("members", []):
            console.print(f"    - {m['id'][:8]} · {_short(m.get('title', ''), 70)}")

        if not click.confirm("Propose merge for this cluster?", default=True):
            continue

        proposal = mem.consolidator.propose_merge(c)
        if proposal is None:
            console.print("[red]No merge proposal generated.[/red]")
            continue
        console.print(f"[bold]merged title:[/bold] {proposal.merged_title}")
        console.print(f"[dim]strategy={proposal.merge_strategy}[/dim]")
        console.print(f"[dim]rationale={proposal.rationale}[/dim]")

        if not click.confirm("Apply merge?", default=False):
            continue

        result = mem.consolidator.apply_merge(proposal, dry_run=dry_run)
        console.print(
            f"[green]merged →[/green] "
            f"{result.merged_id[:8] if result.merged_id else 'n/a'}  "
            f"archived={len(result.archived_ids)}"
        )


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
