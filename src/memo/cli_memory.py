"""Core memory verbs for the memo CLI — CRUD + lifecycle.

Extracted from cli.py (top-level command grouping); retrieval moved to
cli_search.py and knowledge-graph verbs to cli_entities.py. Each command is a
standalone @click.command registered onto the root group in cli.py.
Commands: save, list, get, update, rename, reindex, delete, history,
ocr-image, provenance, lint, restore.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import sqlite3
import stat
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import click
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from memo.cli_common import _resolved, console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config
from memo.errors import StorageError

_RESTORE_MAX_MEMBERS = 20_000
_RESTORE_MAX_MANIFEST_BYTES = 1024 * 1024
_RESTORE_MAX_MEMORY_BYTES = 64 * 1024 * 1024
_RESTORE_MAX_STATE_BYTES = 8 * 1024 * 1024 * 1024
_RESTORE_MAX_TOTAL_BYTES = 12 * 1024 * 1024 * 1024
_RESTORE_RATIO_MIN_BYTES = 1024 * 1024
_RESTORE_MAX_COMPRESSION_RATIO = 100


def _safe_restore_relative_path(name: str, *, prefix: str) -> Path:
    """Return a canonical relative archive path, rejecting ambiguous names."""
    if not name.startswith(prefix) or "\\" in name or "\x00" in name:
        raise click.ClickException(f"invalid restore archive member: {name!r}")
    raw = name[len(prefix) :]
    posix_path = PurePosixPath(raw)
    if (
        not raw
        or posix_path.is_absolute()
        or any(part in {"", ".", ".."} for part in posix_path.parts)
        or posix_path.as_posix() != raw
    ):
        raise click.ClickException(f"invalid restore archive member: {name!r}")
    return Path(*posix_path.parts)


def _reject_symlinked_restore_destination(root: Path, relative: Path) -> Path:
    """Keep a restore below its configured root and never traverse a symlink."""
    root_resolved = root.resolve()
    candidate = root
    for part in relative.parts:
        candidate /= part
        if candidate.is_symlink():
            raise click.ClickException(f"symlinked restore destination: {candidate}")
    destination = root / relative
    if not destination.resolve(strict=False).is_relative_to(root_resolved):
        raise click.ClickException(f"restore destination escapes configured root: {relative}")
    return destination


def _write_restored_member(zf: Any, info: Any, destination: Path) -> None:
    """Stream one validated member to an atomic sibling temporary file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Re-check after mkdir so pre-existing symlink components cannot be silently
    # followed.  The final os.replace also avoids exposing partial DBs/files.
    if destination.is_symlink() or destination.parent.is_symlink():
        raise click.ClickException(f"symlinked restore destination: {destination}")
    tmp_name: str | None = None
    try:
        with (
            tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".restore-tmp",
                delete=False,
            ) as tmp,
            zf.open(info, "r") as source,
        ):
            tmp_name = tmp.name
            shutil.copyfileobj(source, tmp, length=1024 * 1024)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, destination)
        tmp_name = None
    finally:
        if tmp_name is not None:
            Path(tmp_name).unlink(missing_ok=True)


def _validate_restored_sqlite(path: Path, archive_name: str) -> None:
    """Reject corrupt/non-SQLite state before it can replace a live DB."""
    try:
        uri = f"{path.resolve().as_uri()}?mode=ro"
        with contextlib.closing(sqlite3.connect(uri, uri=True)) as connection:
            result = connection.execute("PRAGMA integrity_check").fetchone()
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
                )
            }
        if result != ("ok",):
            raise sqlite3.DatabaseError(str(result[0]) if result else "no integrity result")
        required = (
            {"events"} if Path(archive_name).name == "history.db" else {"schema_meta", "meta"}
        )
        missing = required - tables
        if missing:
            raise sqlite3.DatabaseError(
                "not a Memo state database (missing: " + ", ".join(sorted(missing)) + ")"
            )
    except sqlite3.DatabaseError as exc:
        raise click.ClickException(
            f"invalid SQLite database in restore archive: {archive_name!r}: {exc}"
        ) from exc


def _atomic_copy_restored_file(source: Path, destination: Path) -> None:
    """Copy a staged file into a same-directory temp and atomically publish it."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".restore-tmp",
    )
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as input_file, os.fdopen(descriptor, "wb") as output_file:
            descriptor = -1
            shutil.copyfileobj(input_file, output_file, length=1024 * 1024)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _copy_restored_sqlite_in_place(source: Path, destination: Path) -> None:
    """Restore a logical SQLite snapshot without replacing its live inode."""
    if source.is_symlink() or destination.is_symlink():
        raise click.ClickException("refusing symlinked SQLite restore path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_uri = f"{source.resolve(strict=True).as_uri()}?mode=ro"
    with (
        contextlib.closing(sqlite3.connect(source_uri, uri=True)) as source_connection,
        contextlib.closing(sqlite3.connect(destination, timeout=10.0)) as destination_connection,
    ):
        destination_connection.execute("PRAGMA busy_timeout = 10000")
        source_connection.backup(destination_connection)
        if destination_connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise click.ClickException(f"restored SQLite failed integrity check: {destination}")
    os.chmod(destination, 0o600)


def _move_restore_target_aside(path: Path) -> Path | None:
    """Move an existing regular restore target to a private sibling rollback slot."""
    if path.is_symlink():
        raise click.ClickException(f"symlinked restore destination: {path}")
    if not path.exists():
        return None
    descriptor, rollback_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".restore-rollback",
    )
    os.close(descriptor)
    rollback = Path(rollback_name)
    rollback.unlink()
    os.replace(path, rollback)
    return rollback


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
    help="When title/type/tags missing, ask the configured helper LLM to derive them. "
    "Adds ~1-2s latency on first call.",
)
@click.option(
    "--extract",
    is_flag=True,
    help="Decompose CONTENT into atomic facts via the helper LLM and save each "
    "as its own memory (mem0 ADD-model), instead of one blob; --tag values "
    "propagate to every fact. Falls back to a verbatim save if nothing "
    "extractable is found. Adds ~1-3s LLM latency. "
    "(Ignores --auto-derive / --defer-embed / --meta.)",
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
    "to frontmatter + meta.extra_json. Use native nested provenance through "
    "the API for trace_id, actor_id, route_reason, and evidence_uris.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of a panel.")
def save(
    content: str,
    title: str | None,
    type_: str,
    tags: tuple[str, ...],
    auto_derive: bool,
    extract: bool,
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

    from memo.flags import flag_bool

    if extract or flag_bool("MEMO_SAVE_EXTRACT"):
        from memo.capture import extract_and_save_text

        try:
            summary = extract_and_save_text(
                mem,
                mem.cfg,
                content,
                merge_tags=list(tags),
                title=title,
                type_=type_,
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        if as_json:
            click.echo(json.dumps(summary, ensure_ascii=False, indent=2))
            return
        titles = summary.get("saved_titles") or []
        verb = "verbatim (no atomic facts)" if summary["status"] == "verbatim" else "atomic facts"
        body = "\n".join(f"· {t}" for t in titles) or "[dim]—[/dim]"
        console.print(
            Panel.fit(
                f"[bold]{len(summary.get('saved') or [])} {verb}[/bold]\n{body}",
                title="✓ extracted" if summary["status"] == "extracted" else "✓ saved",
                border_style="green",
            )
        )
        return

    try:
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
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if as_json:
        click.echo(json.dumps(rec.to_dict(), ensure_ascii=False, indent=2))
        return
    action = rec.action or "saved"
    pending = "\n[yellow]index pending:[/yellow] run `memo reindex`" if rec.index_pending else ""
    console.print(
        Panel.fit(
            f"[bold]{escape(rec.title)}[/bold]\n"
            f"[dim]id:[/dim] {rec.id}\n"
            f"[dim]path:[/dim] {escape(str(rec.path))}\n"
            f"[dim]type:[/dim] {escape(rec.type)}  "
            f"[dim]tags:[/dim] {escape(', '.join(rec.tags) or '—')}"
            f"{pending}",
            title=f"✓ {action}",
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
        tbl.add_row(
            r.updated[:19], escape(r.type), escape(r.title), escape(", ".join(r.tags) or "—")
        )
    console.print(tbl)


def _record_proactive_acted(cfg: Config, id_: str) -> None:
    """Close the proactive engine's acted-feedback loop: `memo get <id>` IS the
    action the reliability nudge suggested, so a matching fetch counts as
    "acted" (`ProactiveStore.kind_multipliers`) — otherwise the feedback loop
    only ever decays (I1 review fix). Dark-flag guarded + defensive: must
    NEVER break `memo get`.
    """
    from memo.flags import flag_bool

    if not flag_bool("MEMO_PROACTIVE_ENABLED"):
        return
    with contextlib.suppress(Exception):
        from datetime import UTC, datetime

        from memo.cli_proactive import record_acted_if_matches
        from memo.proactive.store import ProactiveStore

        with ProactiveStore(cfg.state_dir / "proactive.db") as store:
            record_acted_if_matches(
                store, command_line=f"memo get {id_}", now=datetime.now(UTC).isoformat()
            )


@click.command()
@click.argument("id_")
@click.option("--json", "as_json", is_flag=True)
def get(id_: str, as_json: bool) -> None:
    """Fetch one memory by id."""

    cfg = Config.from_env()
    _record_proactive_acted(cfg, id_)

    mem = _get_memory(cfg)
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
            f"[bold]{escape(rec.title)}[/bold]\n"
            f"[dim]id:[/dim] {rec.id}  [dim]type:[/dim] {escape(rec.type)}\n"
            f"[dim]tags:[/dim] {escape(', '.join(rec.tags) or '—')}\n"
            f"[dim]created:[/dim] {rec.created}\n"
            f"[dim]updated:[/dim] {rec.updated}\n\n"
            f"{escape(rec.body or '')}",
            title=escape(rec.title),
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
    from memo.contracts import ActorIdentity

    rec = _resolved(
        lambda: mem.update(
            id_,
            title=title,
            type_=type_,
            tags=list(tags) if tags else None,
            content=content,
            actor=ActorIdentity(actor_id="memo-cli", actor_kind="human"),
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
            f"[bold]{escape(rec.title)}[/bold]\n"
            f"[dim]id:[/dim] {rec.id}  [dim]type:[/dim] {escape(rec.type)}\n"
            f"[dim]tags:[/dim] {escape(', '.join(rec.tags) or '—')}\n"
            f"[dim]updated:[/dim] {rec.updated}",
            title="✓ updated",
            border_style="yellow",
        )
    )


@click.command()
@click.argument("title")
@click.argument("id_", required=False, default=None)
@click.option("--json", "as_json", is_flag=True)
def rename(title: str, id_: str | None, as_json: bool) -> None:
    """Rename a memory's title. Without ID, renames the memory most
    recently saved on this machine — e.g. right after a `memo save`.
    """

    mem = _get_memory(Config.from_env())
    from memo.contracts import ActorIdentity

    if id_ is None:
        id_ = mem.last_saved_id()
        if id_ is None:
            console.print("[red]no recent save found[/red] — pass an ID explicitly")
            sys.exit(1)
    rec = _resolved(
        lambda: mem.update(
            id_,
            title=title,
            actor=ActorIdentity(actor_id="memo-cli", actor_kind="human"),
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
            f"[bold]{escape(rec.title)}[/bold]\n"
            f"[dim]id:[/dim] {rec.id}  [dim]type:[/dim] {escape(rec.type)}\n"
            f"[dim]updated:[/dim] {rec.updated}",
            title="✓ renamed",
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
    "access / health signal. Prefer this over manual DB deletion.",
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
    try:
        counts = mem.reindex(force=force, rebuild=rebuild)
    except StorageError as exc:
        console.print(f"[red]✗[/red] {exc}")
        raise SystemExit(1) from exc
    if as_json:
        click.echo(json.dumps(counts, indent=2))
        return
    console.print(
        f"checked: [cyan]{counts['checked']}[/cyan]  "
        f"reindexed: [yellow]{counts['reindexed']}[/yellow]  "
        f"added: [green]{counts['added']}[/green]  "
        f"skipped: [dim]{counts['skipped']}[/dim]  "
        f"facts: [cyan]{counts.get('facts', 0)}[/cyan]",
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
    from memo.contracts import ActorIdentity

    ok = _resolved(
        lambda: mem.delete(
            id_,
            actor=ActorIdentity(actor_id="memo-cli", actor_kind="human"),
        )
    )
    console.print(f"[{'green' if ok else 'red'}]{'✓ deleted' if ok else 'not found'}[/]: {id_}")
    if not ok:
        sys.exit(1)


@click.command()
@click.argument("id_")
def undo(id_: str) -> None:
    """Remove a just-captured memory. Delete, no confirmation prompt —

    capture receipts are the audit trail, so an `undo` right after a
    receipt should not stop to ask.
    """

    mem = _get_memory(Config.from_env())
    ok = _resolved(lambda: mem.delete(id_))
    console.print(f"[{'green' if ok else 'red'}]{'✓ removed' if ok else 'not found'}[/]: {id_}")
    if not ok:
        sys.exit(1)


def _rederived_title(old_title: str, old_body: str, new_body: str) -> str | None:
    """Re-derive an auto-derived title when a `fix` replaces the body.

    A memory whose title was auto-derived (title == first line of its body)
    should keep tracking the body; otherwise search/recall show the stale title
    after a body-only fix. Returns the new title when the old one was
    auto-derived, else ``None`` so an explicit user-set title is preserved.
    """
    from memo.memory.record import _derive_title

    if old_title.strip() == _derive_title(old_body).strip():
        return _derive_title(new_body) or None
    return None


@click.command()
@click.argument("id_")
@click.option("--title", default=None, help="Corrected title.")
@click.option(
    "--type",
    "type_",
    type=click.Choice(
        ["decision", "fact", "bug", "feedback", "preference", "note", "manual"],
    ),
    default=None,
    help="Corrected type.",
)
@click.option("--body", default=None, help="Corrected body content.")
def fix(id_: str, title: str | None, type_: str | None, body: str | None) -> None:
    """Correct a captured memory's title/type/body (thin wrapper over `update`)."""

    mem = _get_memory(Config.from_env())
    from memo.contracts import ActorIdentity

    # Body-only fix: re-derive an auto-derived title so it doesn't go stale.
    if body is not None and title is None:
        existing = mem.get(id_)
        if existing is not None:
            title = _rederived_title(existing.title, existing.body, body)

    rec = _resolved(
        lambda: mem.update(
            id_,
            title=title,
            type_=type_,
            content=body,
            actor=ActorIdentity(actor_id="memo-cli", actor_kind="human"),
        )
    )
    if rec is None:
        console.print(f"[red]not found:[/red] {id_}")
        sys.exit(1)
    console.print(f"[green]✓ fixed[/green]: {rec.id}")


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

    Returns the current native provenance keys plus every save/update
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
        infos = zf.infolist()
        if len(infos) > _RESTORE_MAX_MEMBERS:
            raise click.ClickException("restore archive contains too many members")

        allowed_state_names = {cfg.db_path.name, cfg.history_db.name}
        restore_plan: list[tuple[zipfile.ZipInfo, Path, str]] = []
        seen_names: set[str] = set()
        total_size = 0
        manifest_info: zipfile.ZipInfo | None = None

        # Validate the complete archive before prompting or writing anything.
        # This makes rejection fail closed even if a malicious entry follows
        # otherwise valid memory files.
        for info in infos:
            name = info.filename
            if name in seen_names:
                raise click.ClickException(f"duplicate restore archive member: {name!r}")
            seen_names.add(name)
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise click.ClickException(f"symlink archive member is not allowed: {name!r}")
            if info.is_dir():
                if name != "memory/" and name != "state/":
                    raise click.ClickException(f"unexpected restore archive member: {name!r}")
                continue
            if info.file_size < 0 or info.compress_size < 0:
                raise click.ClickException(f"invalid restore archive member size: {name!r}")
            if (
                info.file_size >= _RESTORE_RATIO_MIN_BYTES
                and info.file_size / max(info.compress_size, 1) > _RESTORE_MAX_COMPRESSION_RATIO
            ):
                raise click.ClickException(f"suspicious compression ratio: {name!r}")
            total_size += info.file_size
            if total_size > _RESTORE_MAX_TOTAL_BYTES:
                raise click.ClickException("restore archive is too large")

            if name == "manifest.json":
                if info.file_size > _RESTORE_MAX_MANIFEST_BYTES:
                    raise click.ClickException("restore manifest is too large")
                manifest_info = info
                continue
            if name.startswith("memory/"):
                if info.file_size > _RESTORE_MAX_MEMORY_BYTES:
                    raise click.ClickException(f"memory archive member is too large: {name!r}")
                relative = _safe_restore_relative_path(name, prefix="memory/")
                if relative.suffix != ".md":
                    raise click.ClickException(f"unexpected memory archive member: {name!r}")
                destination = _reject_symlinked_restore_destination(cfg.memory_dir, relative)
                restore_plan.append((info, destination, "memory"))
                continue
            if name.startswith("state/"):
                relative = _safe_restore_relative_path(name, prefix="state/")
                if len(relative.parts) != 1 or relative.name not in allowed_state_names:
                    raise click.ClickException(f"unexpected state archive member: {name!r}")
                if info.file_size > _RESTORE_MAX_STATE_BYTES:
                    raise click.ClickException(f"state archive member is too large: {name!r}")
                destination = _reject_symlinked_restore_destination(cfg.state_dir, relative)
                restore_plan.append((info, destination, "state"))
                continue
            raise click.ClickException(f"unexpected restore archive member: {name!r}")

        try:
            manifest = json.loads(zf.read(manifest_info)) if manifest_info is not None else None
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise click.ClickException(f"restore manifest is invalid JSON: {exc}") from exc
        if manifest is not None and not isinstance(manifest, dict):
            raise click.ClickException("restore manifest must be a JSON object")
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
        n_md = n_db = 0
        with tempfile.TemporaryDirectory(prefix="memo-restore-stage-") as scratch_name:
            scratch = Path(scratch_name)
            required_free = total_size * 2 + 64 * 1024 * 1024
            if shutil.disk_usage(scratch).free < required_free:
                raise click.ClickException("insufficient free disk space for safe restore staging")
            staged_plan: list[tuple[Path, Path, str]] = []
            try:
                # Fully decompress and CRC-check every member before replacing
                # any canonical file. State DBs also pass SQLite integrity.
                for index, (info, destination, kind) in enumerate(restore_plan):
                    staged = scratch / str(index)
                    _write_restored_member(zf, info, staged)
                    if staged.stat().st_size != info.file_size:
                        raise click.ClickException(
                            f"restore member size mismatch: {info.filename!r}"
                        )
                    if kind == "state":
                        _validate_restored_sqlite(staged, info.filename)
                    staged_plan.append((staged, destination, kind))
            except click.ClickException:
                raise
            except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                raise click.ClickException(f"could not stage restore archive: {exc}") from exc

            from memo.atomic_io import authority_write_lock

            file_plan = [entry for entry in staged_plan if entry[2] == "memory"]
            database_plan = [entry for entry in staged_plan if entry[2] == "state"]
            with (
                authority_write_lock(cfg.memory_dir),
                tempfile.TemporaryDirectory(
                    prefix="memo-portable-sqlite-rollback-"
                ) as rollback_name,
            ):
                rollback_root = Path(rollback_name)
                database_journal: list[tuple[Path, Path | None]] = []
                for index, (_source, destination, _kind) in enumerate(database_plan):
                    rollback = None
                    if destination.exists():
                        rollback = rollback_root / f"{index}-{destination.name}"
                        _copy_restored_sqlite_in_place(destination, rollback)
                    database_journal.append((destination, rollback))

                file_journal: list[tuple[Path, Path | None]] = []
                attempted_databases: list[tuple[Path, Path | None]] = []
                try:
                    for staged, destination, _kind in file_plan:
                        relative = destination.relative_to(cfg.memory_dir)
                        destination = _reject_symlinked_restore_destination(
                            cfg.memory_dir, relative
                        )
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        rollback = _move_restore_target_aside(destination)
                        file_journal.append((destination, rollback))
                        _atomic_copy_restored_file(staged, destination)
                        n_md += 1
                    for (staged, _destination, _kind), (
                        destination,
                        rollback,
                    ) in zip(database_plan, database_journal, strict=True):
                        attempted_databases.append((destination, rollback))
                        _copy_restored_sqlite_in_place(staged, destination)
                        n_db += 1
                except Exception as exc:
                    for destination, rollback in reversed(attempted_databases):
                        if rollback is None:
                            destination.unlink(missing_ok=True)
                        else:
                            _copy_restored_sqlite_in_place(rollback, destination)
                    for published, rollback in reversed(file_journal):
                        published.unlink(missing_ok=True)
                        if rollback is not None and rollback.exists():
                            os.replace(rollback, published)
                    if isinstance(exc, click.ClickException):
                        raise
                    raise click.ClickException(
                        f"restore failed; prior files were restored: {exc}"
                    ) from exc
                else:
                    for _published, rollback in file_journal:
                        if rollback is not None:
                            rollback.unlink(missing_ok=True)

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
            f"added {counts['added']}  skipped {counts['skipped']}  facts {counts.get('facts', 0)}",
        )
