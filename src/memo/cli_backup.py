"""`memo backup` command group — portable backups + named snapshots.

Extracted from cli.py (3a god-module decomposition). Registered onto the
root group in cli.py via `cli.add_command(backup_group)`.
"""

from __future__ import annotations

import json

import click
from rich.table import Table

from memo.cli_common import console
from memo.config import Config


def _backup_manager(cfg: Config):
    from memo.sync import BackupManager

    cfg.ensure_dirs()
    return BackupManager(
        memory_dir=cfg.memory_dir,
        db_dir=cfg.state_dir,
        backup_dir=cfg.state_dir / "backups",
    )


# -- portable + named backups --------------------------------------------------


def _portable_backup(out_path: str | None) -> None:
    """Snapshot memory dir + sqlite-vec DB + history DB into a zip.

    Use before risky operations (mass migration, embedder swap, schema
    change). The zip is portable: extract on another machine, set
    `MEMO_VAULT_PATH` to the matching vault, run `memo restore <zip>`
    to absorb everything back. Vault `.md` files are kept as the
    storage of record so the backup is self-contained.
    """
    import contextlib
    import datetime as _dt
    import tempfile
    import zipfile
    from pathlib import Path

    from memo.atomic_io import authority_write_lock
    from memo.sqlite_snapshot import snapshot_sqlite_database

    cfg = Config.from_env()
    cfg.ensure_dirs()
    out = out_path or f"memo-backup-{_dt.datetime.now(_dt.UTC).strftime('%Y%m%d-%H%M%S')}.zip"
    out_p = Path(out).resolve()
    if out_p.exists():
        raise click.ClickException(f"backup file already exists: {out_p}")
    # Reserve the archive privately before zipfile opens it. The default 0666
    # creation mode would expose the full corpus/index to other local users
    # under a common 022 umask while the backup is being written.
    out_p.touch(mode=0o600, exist_ok=False)
    out_p.chmod(0o600)

    @contextlib.contextmanager
    def _remove_partial_on_error():
        try:
            yield
        except BaseException:
            out_p.unlink(missing_ok=True)
            raise

    n_md = 0
    with (
        _remove_partial_on_error(),
        tempfile.TemporaryDirectory(prefix="memo-portable-backup-") as scratch_dir,
    ):
        snapshot_dir = Path(scratch_dir)
        # CRUD/reindex use this same authority lock, yielding one coherent
        # Markdown+SQLite checkpoint instead of a mixed-time archive.
        with (
            authority_write_lock(cfg.memory_dir),
            zipfile.ZipFile(out_p, "w", zipfile.ZIP_DEFLATED) as zf,
        ):
            # 1) Memory .md files (relative to memory_dir).
            if cfg.memory_dir.is_dir():
                memory_root = cfg.memory_dir.resolve()
                for md in sorted(cfg.memory_dir.rglob("*.md")):
                    rel = md.relative_to(cfg.memory_dir)
                    if rel.parts[:1] == ("secrets",):
                        continue
                    # A symlinked markdown file can point outside the vault;
                    # zipfile follows it and would silently exfiltrate the
                    # target. Reject symlinks in every path component and
                    # independently confine the resolved source to the root.
                    cursor = cfg.memory_dir
                    if any((cursor := cursor / part).is_symlink() for part in rel.parts):
                        continue
                    try:
                        resolved = md.resolve(strict=True)
                        resolved.relative_to(memory_root)
                    except (OSError, ValueError):
                        continue
                    if not resolved.is_file():
                        continue
                    zf.write(resolved, arcname=f"memory/{rel}")
                    n_md += 1
            # 2) State DBs (vec + history). Stored at the root. Dedup by resolved
            #    path: under single_db, history_db == db_path (one file).
            seen_dbs: set[str] = set()
            state_root = cfg.state_dir.resolve()
            for db in (cfg.db_path, cfg.history_db):
                try:
                    relative_db = db.relative_to(cfg.state_dir)
                except ValueError as exc:
                    raise click.ClickException(
                        f"refusing database outside state_dir: {db}"
                    ) from exc
                cursor = cfg.state_dir
                if any((cursor := cursor / part).is_symlink() for part in relative_db.parts):
                    raise click.ClickException(f"refusing symlinked state database: {db}")
                try:
                    resolved_db = db.resolve(strict=True)
                    resolved_db.relative_to(state_root)
                except (OSError, ValueError) as exc:
                    if db.exists() or db.is_symlink():
                        raise click.ClickException(f"refusing unsafe state database: {db}") from exc
                    continue
                key = str(resolved_db)
                if key in seen_dbs:
                    continue
                seen_dbs.add(key)
                if resolved_db.is_file():
                    snapshot = snapshot_dir / db.name
                    snapshot_sqlite_database(resolved_db, snapshot)
                    zf.write(snapshot, arcname=f"state/{db.name}")
            # 3) Manifest with paths so restore can sanity-check.
            from memo.embedder_select import active_embedder_identity

            manifest = {
                "created": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
                "data_dir": str(cfg.data_dir),
                "vault_path": str(cfg.vault_path) if cfg.vault_path else None,
                "embedder_model": active_embedder_identity(cfg),
                "embedder_dims": cfg.embedder_dims,
                "memo_version": __import__("memo").__version__,
                "n_md": n_md,
            }
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))

    size_kb = out_p.stat().st_size // 1024
    console.print(f"[green]✓[/green] backup: {out_p} ({n_md} memories, {size_kb} KB)")


@click.group(name="backup", invoke_without_command=True)
@click.option(
    "--out",
    "out_path",
    default=None,
    type=click.Path(),
    help="Output portable zip path. Default: ./memo-backup-<YYYYMMDD-HHMMSS>.zip",
)
@click.pass_context
def backup_group(ctx: click.Context, out_path: str | None) -> None:
    """Create a portable backup, or manage named backups with subcommands."""
    if ctx.invoked_subcommand is None:
        _portable_backup(out_path)
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
    metadata = _backup_manager(cfg).create_backup(compress=compress, name=name)

    if as_json:
        click.echo(json.dumps(metadata.__dict__, indent=2))
        return

    console.print("[bold]Backup Created[/bold]")
    console.print()
    console.print(f"Timestamp: {metadata.timestamp}")
    console.print(f"Memories: {metadata.memory_count}")
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
    backups = _backup_manager(cfg).list_backups()

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
    table.add_column("Status")

    for b in backups[:20]:
        # A corrupt archive gets a fabricated now() timestamp (so it sorts
        # newest) — surface the flag so the human table doesn't present it as an
        # ordinary, healthy, most-recent backup. The --json path already carries it.
        corrupt = getattr(b, "corrupted", False)
        table.add_row(
            b.name if b.name else b.timestamp[:19],
            b.timestamp[:19],
            f"{b.compressed_size:,} bytes",
            "[red]CORRUPT[/red]" if corrupt else "[green]ok[/green]",
        )

    console.print(table)
    if len(backups) > 20:
        console.print(f"[dim]...and {len(backups) - 20} more[/dim]")


@backup_group.command(name="restore")
@click.argument("backup_name")
@click.option("--no-memories", "skip_memories", is_flag=True, help="Skip memory files")
@click.option("--no-dbs", "skip_dbs", is_flag=True, help="Skip databases")
@click.confirmation_option(
    prompt="This will restore from backup. Current data may be overwritten. Continue?"
)
def backup_restore(backup_name: str, skip_memories: bool, skip_dbs: bool) -> None:
    """Restore from a backup.

    Example: memo backup restore backup_2026-01-01-12-00-00
    """
    cfg = Config.from_env()
    success = _backup_manager(cfg).restore_backup(
        backup_name,
        restore_memories=not skip_memories,
        restore_dbs=not skip_dbs,
    )

    if success:
        console.print(f"[green]Restored from '{backup_name}'[/green]")
    else:
        console.print("[red]Failed to restore[/red]")
        raise SystemExit(1)
