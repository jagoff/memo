from __future__ import annotations

import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Any

import click

from memo.cli_common import console
from memo.config import Config
from memo.setup import run_picker, write_config_file


def _q(name: str) -> str:
    """Quote a SQL identifier defensively.

    Table names here come from a hardcoded allow-list so this is belt-and-braces,
    but quoting keeps the interpolation safe if that list ever grows.
    """
    return '"' + name.replace('"', '""') + '"'


def _consolidate_sidecar_dbs() -> None:
    from memo.contradict import ContradictionStore
    from memo.crossref import CrossReferenceIndex
    from memo.graph import GraphStore
    from memo.history import HistoryStore

    cfg = Config.from_env()
    if cfg.single_db:
        console.print("[yellow]![/yellow] already in single_db mode — nothing to merge")
    main_db = cfg.db_path

    legacy_tables: dict[Path, list[str]] = {
        cfg.state_dir / "history.db": ["events", "sync_state"],
        cfg.state_dir / "graph.db": ["entities", "entity_memory"],
        cfg.state_dir / "contradictions.db": ["pairs"],
        cfg.state_dir / "crossref.db": ["backlinks"],
    }

    HistoryStore(main_db, device_id=cfg.device_id).close()
    GraphStore(main_db).close()
    ContradictionStore(main_db).close()
    CrossReferenceIndex(main_db).close()

    merged_any = False
    conn = sqlite3.connect(str(main_db), timeout=10.0)
    try:
        conn.execute("PRAGMA busy_timeout = 10000")
        for legacy, tables in legacy_tables.items():
            if not legacy.is_file():
                continue
            conn.execute("ATTACH DATABASE ? AS legacy", (str(legacy),))
            try:
                present = {
                    str(r[0])
                    for r in conn.execute(
                        "SELECT name FROM legacy.sqlite_master WHERE type='table'"
                    ).fetchall()
                }
                for tbl in tables:
                    if tbl not in present:
                        continue
                    conn.execute(
                        f"INSERT OR IGNORE INTO main.{_q(tbl)} SELECT * FROM legacy.{_q(tbl)}"  # noqa: S608
                    )
                conn.commit()
            finally:
                conn.execute("DETACH DATABASE legacy")
            bak = legacy.with_suffix(legacy.suffix + ".bak")
            legacy.replace(bak)
            console.print(
                f"[green]✓[/green] merged {legacy.name} → memvec.db, renamed → {bak.name}"
            )
            merged_any = True
    finally:
        conn.close()

    if not merged_any:
        console.print("[dim]No legacy sidecar files found to merge.[/dim]")

    existing = Config.from_env()
    write_config_file(
        data_dir=existing.data_dir,
        vault_path=existing.vault_path,
        memories_in_vault=existing.memories_in_vault,
        single_db=True,
    )
    console.print("[green]✓[/green] set single_db=1 in config — memo now uses one DB file")


def _bucket_by_project(cfg: Config) -> int:
    """Move flat-root .md files into per-project bucket folders by their
    `project:` tag (untagged -> `_global/`). Idempotent + non-destructive.
    Returns the number of files moved.
    """
    import frontmatter

    from memo.project import project_bucket

    md_root = cfg.memory_dir
    moved = 0
    # Only the FLAT root level (already-bucketed files live one level deeper
    # and are skipped, making this idempotent).
    for md in sorted(md_root.glob("*.md")):
        try:
            post = frontmatter.loads(md.read_text(encoding="utf-8"))
        except Exception:  # noqa: S112
            continue
        meta: dict[str, Any] = post.metadata
        tags = list(meta.get("tags") or [])
        bucket = project_bucket(tags)
        dest_dir = md_root / bucket
        dest = dest_dir / md.name
        if not dest.resolve().is_relative_to(md_root.resolve()):
            continue  # traversal-shaped bucket — leave the file in place
        if dest.resolve() == md.resolve():
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            continue  # name collision in bucket — leave the original in place
        md.rename(dest)
        moved += 1
    return moved


@click.command(name="migrate-vault")
@click.argument("new_data_dir", required=False, type=click.Path(file_okay=False, resolve_path=True))
@click.option(
    "--from",
    "from_dir",
    default=None,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Source memory_dir. Defaults to current cfg.memory_dir.",
)
@click.option(
    "--into-vault",
    is_flag=True,
    help="Move memories INTO the Obsidian vault (<vault>/<SYSTEM_DIR>/AI/memory) and set memories_in_vault=1 so the vault becomes the source of truth.",
)
@click.option(
    "--rollback",
    is_flag=True,
    help="Restore the config snapshot taken by the last migration and exit. Copied files are left in place (migration never deletes anything).",
)
@click.option(
    "--consolidate-db",
    is_flag=True,
    help="Merge the sidecar DBs (history/graph/contradictions/crossref) into the main memvec.db, set MEMO_SINGLE_DB=1 in config, and rename the legacy files to *.db.bak (reversible). Idempotent. Does not move any .md files.",
)
@click.option(
    "--bucket-by-project",
    is_flag=True,
    help="Move existing flat .md files into per-project folders "
    "(memory_dir/<project>/, _global/ when untagged) by their project: tag, "
    "then reindex. Non-destructive (moves only), idempotent.",
)
@click.option("--force", is_flag=True, help="Overwrite destination even if non-empty.")
@click.option("--yes", is_flag=True, help="Skip confirmation.")
def migrate_vault(
    new_data_dir: str | None,
    from_dir: str | None,
    into_vault: bool,
    rollback: bool,
    consolidate_db: bool,
    bucket_by_project: bool,
    force: bool,
    yes: bool,
) -> None:
    from memo.config import AI_SUBDIR
    from memo.memory import Memory
    from memo.setup.config_io import _resolve_config_path

    snapshot = _resolve_config_path().with_suffix(".toml.pre-migrate.bak")
    if rollback:
        if not snapshot.is_file():
            console.print(f"[red]✗[/red] no migration snapshot found at {snapshot}")
            sys.exit(1)
        shutil.copy2(snapshot, _resolve_config_path())
        console.print(f"[green]✓[/green] restored config from snapshot {snapshot}")
        console.print(
            "[dim]Copied memory files were left in place; remove them manually "
            "if you no longer want them.[/dim]"
        )
        return

    if consolidate_db:
        _consolidate_sidecar_dbs()
        return

    if bucket_by_project:
        cfg = Config.from_env()
        moved = _bucket_by_project(cfg)
        console.print(f"[green]✓[/green] bucketed {moved} memory file(s) by project")
        mem = Memory(cfg)
        try:
            mem.reindex()
        finally:
            mem.close()
        console.print(
            "[dim]reindexed (paths updated). [[id]] wikilinks are unaffected; "
            "Obsidian path-links to moved files would change.[/dim]"
        )
        return

    cfg = Config.from_env()
    src = Path(from_dir).resolve() if from_dir else cfg.memory_dir
    if not src.is_dir():
        console.print(f"[red]✗[/red] source dir does not exist: {src}")
        sys.exit(1)

    if into_vault:
        chosen_vault = cfg.vault_path
        if chosen_vault is None:
            console.print(
                "[red]✗[/red] --into-vault needs a vault: set MEMO_VAULT_PATH or run "
                "`memo init` and pick an Obsidian vault first."
            )
            sys.exit(1)
        dst = (chosen_vault / AI_SUBDIR / "memory").resolve()
    elif new_data_dir:
        dst = Path(new_data_dir).resolve()
        chosen_vault = cfg.vault_path
    else:
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
            f"[red]✗[/red] destination is non-empty: {dst}\n  Use --force to overwrite.",
        )
        sys.exit(1)

    if not yes:
        click.confirm(
            f"Copy {len(md_files)} memories from\n  {src}\n→ {dst}\n"
            "and rebuild memvec.db. Source files will be left in place. "
            "Proceed?",
            abort=True,
        )

    dst.mkdir(parents=True, exist_ok=True)
    n_copied = 0
    for md in md_files:
        rel = md.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(md, target)
        n_copied += 1
    console.print(f"[green]✓[/green] copied {n_copied} files → {dst}")

    existing_cfg = _resolve_config_path()
    if existing_cfg.is_file():
        shutil.copy2(existing_cfg, snapshot)
        console.print(f"[green]✓[/green] config snapshot → {snapshot} (use --rollback to restore)")
    # Preserve single_db: rewriting the config without it silently reverts a
    # consolidated install to sidecar mode, orphaning the folded-in stores.
    if into_vault:
        cfg_path = write_config_file(
            data_dir=cfg.data_dir,
            vault_path=chosen_vault,
            memories_in_vault=True,
            single_db=cfg.single_db,
        )
    else:
        cfg_path = write_config_file(
            data_dir=dst, vault_path=chosen_vault, single_db=cfg.single_db
        )
    console.print(f"[green]✓[/green] config: {cfg_path}")

    new_cfg = Config.from_env()
    mem = Memory(new_cfg)
    try:
        counts = mem.reindex()
    finally:
        mem.close()
    console.print(
        f"[green]✓[/green] reindex: checked {counts['checked']}  "
        f"added {counts['added']}  reindexed {counts['reindexed']}  "
        f"skipped {counts['skipped']}"
    )
    console.print(
        f"\n[dim]Source files at {src} were left untouched. "
        "After verifying the migration with `memo search`, you can rm them.[/dim]"
    )
