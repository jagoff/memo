from __future__ import annotations

import re
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


def _legacy_episode_vec_dims(conn: sqlite3.Connection) -> int | None:
    """FLOAT[N] width of the attached legacy `episode_vec` table, or None."""
    row = conn.execute(
        "SELECT sql FROM legacy.sqlite_master WHERE type='table' AND name='episode_vec'"
    ).fetchone()
    if row is None or not row[0]:
        return None
    match = re.search(r"FLOAT\[(\d+)\]", str(row[0]), re.IGNORECASE)
    return int(match.group(1)) if match else None


def _consolidate_sidecar_dbs() -> None:
    from memo.contradict import ContradictionStore
    from memo.crossref import CrossReferenceIndex
    from memo.embedder_select import active_embedder_identity
    from memo.graph import GraphStore
    from memo.history import HistoryStore
    from memo.sqlite_compat import import_sqlite_vec
    from memo.store.episode_store import EpisodeStore
    from memo.store.fact_edge_store import FactEdgeStore
    from memo.store.turn_store import TurnStore

    cfg = Config.from_env()
    if cfg.single_db:
        console.print("[yellow]![/yellow] already in single_db mode — nothing to merge")
    main_db = cfg.db_path

    # Every sidecar whose cfg.*_db property collapses onto db_path under
    # single_db=1 must be merged here, or its data is silently orphaned.
    legacy_tables: dict[Path, list[str]] = {
        cfg.state_dir / "history.db": ["events", "sync_state"],
        cfg.state_dir / "graph.db": [
            "entities",
            "entity_memory",
            "co_recall",
            "entity_edges",
            "entity_aliases",
            "semantic_relations",
        ],
        cfg.state_dir / "contradictions.db": ["pairs"],
        cfg.state_dir / "crossref.db": ["backlinks"],
        cfg.state_dir / "episodes.db": ["episode_meta", "episode_schema_meta", "episode_vec"],
        cfg.state_dir / "fact_edges.db": ["fact_edges"],
        cfg.state_dir / "verbatim.db": ["turns", "turns_fts"],
    }

    HistoryStore(main_db, device_id=cfg.device_id).close()
    GraphStore(main_db).close()
    ContradictionStore(main_db).close()
    CrossReferenceIndex(main_db).close()
    EpisodeStore(main_db, cfg.embedder_dims, embedder_model=active_embedder_identity(cfg)).close()
    FactEdgeStore(main_db).close()
    TurnStore(main_db).close()

    merged_any = False
    conn = sqlite3.connect(str(main_db), timeout=10.0)
    try:
        conn.execute("PRAGMA busy_timeout = 10000")
        # episode_vec is a vec0 virtual table — the extension must be loaded on
        # this connection for both the legacy SELECT and the main INSERT.
        conn.enable_load_extension(True)
        import_sqlite_vec().load(conn)
        conn.enable_load_extension(False)
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
                if "episode_vec" in tables:
                    legacy_dims = _legacy_episode_vec_dims(conn)
                    if legacy_dims is not None and legacy_dims != cfg.embedder_dims:
                        # Stale-width vectors; EpisodeStore itself would clear
                        # them on open. Episodes are rebuildable, so skip the
                        # whole file instead of merging poisoned rows.
                        tables = []
                        console.print(
                            f"[yellow]![/yellow] {legacy.name} has {legacy_dims}-dim "
                            f"vectors (current: {cfg.embedder_dims}) — skipped; "
                            "rebuild with `memo episodes index --rebuild`"
                        )
                for tbl in tables:
                    if tbl not in present:
                        continue
                    if tbl == "episode_vec":
                        # vec0 does not honor OR IGNORE — filter dups manually.
                        conn.execute(
                            "INSERT INTO main.episode_vec "
                            "SELECT * FROM legacy.episode_vec "
                            "WHERE id NOT IN (SELECT id FROM main.episode_vec)"
                        )
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


def run_backfill_valid_time(cfg: Config) -> int:
    """One-time, idempotent backfill of the world-validity start time.

    Sets ``valid_at = created`` for every record whose ``valid_at`` is still
    NULL (rows written before the bi-temporal columns existed). ``invalid_at``
    is left untouched. Non-destructive — a plain
    ``UPDATE meta SET valid_at = created WHERE valid_at IS NULL`` on the index,
    plus a frontmatter mirror onto each affected record's markdown so disk (the
    source of truth) and index agree — otherwise a later `reindex --rebuild`
    from disk would re-null them. Reference-tier chunks and legacy vault-only
    rows (whose paths don't resolve under ``memory_dir``) are backfilled in the
    index only. Returns the number of records changed.
    """
    import frontmatter

    from memo.errors import StorageError
    from memo.memory import Memory

    mem = Memory(cfg)
    try:
        store = mem.store
        # Snapshot the rows to backfill BEFORE the update, so we know which
        # markdown files to mirror. `created` is NOT NULL in the schema, so the
        # index UPDATE below can never re-set valid_at to NULL — the second run
        # matches nothing (idempotent).
        rows = store._conn.execute(
            "SELECT path, created FROM meta WHERE valid_at IS NULL"
        ).fetchall()
        if not rows:
            return 0
        with store._tx() as cx:
            cx.execute("UPDATE meta SET valid_at = created WHERE valid_at IS NULL")
        # Mirror the value onto the canonical markdown. Only new-layout files
        # that exist under memory_dir are touched; anything else is index-only.
        for row in rows:
            rel_path = str(row["path"])
            created = row["created"]
            try:
                target = mem._safe_path_under(cfg.memory_dir, rel_path)
            except StorageError:
                continue  # path escapes memory_dir (reference/legacy) — index-only
            if not target.is_file():
                continue
            post = frontmatter.loads(target.read_text(encoding="utf-8"))
            if post.metadata.get("valid_at"):
                continue  # disk already carries a value — don't clobber it
            post["valid_at"] = created
            mem._atomic_write_text(rel_path, frontmatter.dumps(post))
        return len(rows)
    finally:
        mem.close()


def _coerce_frontmatter_tags(raw_tags: object) -> list[str] | None:
    if isinstance(raw_tags, str):
        return [tag.strip() for tag in raw_tags.split(",") if tag.strip()]
    if isinstance(raw_tags, list):
        return [str(tag) for tag in raw_tags if str(tag).strip()]
    return None


def _project_tag_slugs(tags: list[str]) -> tuple[list[str], int]:
    from memo.project import slugify_project

    slugs: list[str] = []
    invalid = 0
    for tag in tags:
        folded = tag.strip().casefold()
        if not folded.startswith("project:"):
            continue
        slug = slugify_project(folded.split(":", 1)[1])
        invalid += int(not slug)
        if slug:
            slugs.append(slug)
    return list(dict.fromkeys(slugs)), invalid


def _rewrite_project_tags(tags: list[str], primary: str) -> tuple[list[str], int]:
    from memo.project import slugify_project

    rewritten: list[str] = []
    converted = 0
    primary_written = False
    for tag in tags:
        folded = tag.strip().casefold()
        if not folded.startswith("project:"):
            candidate = tag
        else:
            slug = slugify_project(folded.split(":", 1)[1])
            if slug and slug == primary and not primary_written:
                candidate = f"project:{slug}"
                primary_written = True
            else:
                candidate = f"related-project:{slug or 'unspecified'}"
                converted += 1
        if candidate not in rewritten:
            rewritten.append(candidate)
    return rewritten, converted


def run_normalize_project_tags(cfg: Config, *, dry_run: bool = False) -> dict[str, int]:
    """Resolve historical multi-project tags without losing associations.

    Canonical identity permits one ``project:`` namespace. For an already
    bucketed memory, its folder wins; otherwise the first historical project
    tag wins. Remaining project tags become ``related-project:`` tags. The
    rewrite is atomic, idempotent, and limited to parseable regular Markdown
    files under the configured memory directory.
    """
    import frontmatter

    from memo.atomic_io import atomic_write_text, authority_write_lock
    from memo.project import slugify_project

    root = cfg.memory_dir.resolve()
    changed = converted = invalid = parse_errors = 0
    with authority_write_lock(root):
        for path in sorted(root.rglob("*.md")):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                post = frontmatter.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, ValueError):
                parse_errors += 1
                continue
            tags = _coerce_frontmatter_tags(post.metadata.get("tags") or [])
            if tags is None:
                continue

            distinct, invalid_count = _project_tag_slugs(tags)
            invalid += invalid_count
            if len(distinct) <= 1 and invalid_count == 0:
                continue

            rel = path.relative_to(root)
            folder_slug = (
                slugify_project(rel.parts[0].removeprefix("_")) if len(rel.parts) > 1 else ""
            )
            primary = folder_slug if folder_slug in distinct else (distinct[0] if distinct else "")
            rewritten, converted_count = _rewrite_project_tags(tags, primary)
            converted += converted_count
            if rewritten == tags:
                continue
            changed += 1
            if not dry_run:
                post["tags"] = rewritten
                atomic_write_text(path, frontmatter.dumps(post))
    return {
        "files_changed": changed,
        "tags_converted": converted,
        "invalid_project_tags": invalid,
        "parse_errors": parse_errors,
    }


def _sanitize_privacy_text(raw: str) -> tuple[str | None, bool, bool, str]:
    import frontmatter

    from memo.redact import sanitize_persisted_text, scan_secrets

    has_secret = bool(scan_secrets(raw, entropy=False))
    has_private = "<private>" in raw.casefold()
    if not has_secret and not has_private:
        return None, False, False, "clean"
    sanitized = sanitize_persisted_text(raw, entropy=False).text
    try:
        post = frontmatter.loads(sanitized)
    except (TypeError, ValueError):
        return None, has_secret, has_private, "parse_error"
    if not post.content.strip():
        return None, has_secret, has_private, "emptied"
    tags = _coerce_frontmatter_tags(post.metadata.get("tags") or []) or []
    if not any(tag.casefold() == "_redacted" for tag in tags):
        tags.append("_redacted")
    post["tags"] = tags
    rendered = frontmatter.dumps(post)
    if scan_secrets(rendered, entropy=False) or "<private>" in rendered.casefold():
        return None, has_secret, has_private, "parse_error"
    return rendered, has_secret, has_private, "changed"


def run_sanitize_privacy(cfg: Config, *, dry_run: bool = False) -> dict[str, int]:
    """Remove private spans and mask recognized secrets in canonical Markdown.

    Every changed document is parsed again after sanitization and receives the
    ``_redacted`` audit tag. Files that would become unparsable or body-empty
    are left untouched and reported so the CLI can fail closed.
    """
    import frontmatter

    from memo.atomic_io import atomic_write_text, authority_write_lock

    root = cfg.memory_dir.resolve()
    changed = secret_files = private_files = parse_errors = emptied = 0
    with authority_write_lock(root):
        for path in sorted(root.rglob("*.md")):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                raw = path.read_text(encoding="utf-8")
                frontmatter.loads(raw)
            except (OSError, UnicodeError, ValueError):
                parse_errors += 1
                continue
            rendered, has_secret, has_private, status = _sanitize_privacy_text(raw)
            if status == "clean":
                continue
            secret_files += int(has_secret)
            private_files += int(has_private)
            if status == "parse_error":
                parse_errors += 1
                continue
            if status == "emptied":
                emptied += 1
                continue
            assert rendered is not None
            changed += 1
            if not dry_run:
                atomic_write_text(path, rendered)
    return {
        "files_changed": changed,
        "secret_files": secret_files,
        "private_files": private_files,
        "parse_errors": parse_errors,
        "emptied_files": emptied,
    }


def _reindex_migration(cfg: Config, *, rebuild: bool = False) -> dict[str, int]:
    from memo.memory import Memory

    memory = Memory(cfg)
    try:
        return memory.reindex(rebuild=rebuild)
    finally:
        memory.close()


def _run_normalize_project_tags_migration(*, dry_run: bool) -> None:
    cfg = Config.from_env()
    report = run_normalize_project_tags(cfg, dry_run=dry_run)
    action = "would normalize" if dry_run else "normalized"
    console.print(
        f"[green]✓[/green] {action} {report['files_changed']} memory file(s); "
        f"converted {report['tags_converted']} secondary project tag(s); "
        f"invalid={report['invalid_project_tags']} parse_errors={report['parse_errors']}"
    )
    if report["parse_errors"]:
        raise click.ClickException(
            "some Markdown files could not be parsed; no index rebuild was attempted"
        )
    if not dry_run and report["files_changed"]:
        _reindex_migration(cfg, rebuild=True)
        console.print("[dim]rebuilt the index from normalized Markdown[/dim]")


def _run_sanitize_privacy_migration(*, dry_run: bool) -> None:
    cfg = Config.from_env()
    report = run_sanitize_privacy(cfg, dry_run=dry_run)
    action = "would sanitize" if dry_run else "sanitized"
    console.print(
        f"[green]✓[/green] {action} {report['files_changed']} memory file(s); "
        f"secret_files={report['secret_files']} private_files={report['private_files']} "
        f"parse_errors={report['parse_errors']} emptied={report['emptied_files']}"
    )
    if report["parse_errors"] or report["emptied_files"]:
        raise click.ClickException(
            "some Markdown could not be sanitized safely; no index rebuild was attempted"
        )
    if not dry_run and report["files_changed"]:
        _reindex_migration(cfg, rebuild=True)
        console.print("[dim]rebuilt the index from privacy-sanitized Markdown[/dim]")


def _run_bucket_by_project_migration() -> None:
    cfg = Config.from_env()
    moved = _bucket_by_project(cfg)
    console.print(f"[green]✓[/green] bucketed {moved} memory file(s) by project")
    _reindex_migration(cfg)
    console.print(
        "[dim]reindexed (paths updated). [[id]] wikilinks are unaffected; "
        "Obsidian path-links to moved files would change.[/dim]"
    )


def _run_special_migration(
    *,
    snapshot: Path,
    rollback: bool,
    consolidate_db: bool,
    backfill_valid_time: bool,
    normalize_project_tags: bool,
    sanitize_privacy: bool,
    bucket_by_project: bool,
    dry_run: bool,
) -> bool:
    from memo.setup.config_io import _resolve_config_path

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
        return True
    if consolidate_db:
        _consolidate_sidecar_dbs()
        return True
    if backfill_valid_time:
        changed = run_backfill_valid_time(Config.from_env())
        console.print(
            f"[green]✓[/green] backfilled valid_at = created on {changed} record(s) "
            "(idempotent; invalid_at untouched)"
        )
        return True
    if normalize_project_tags:
        _run_normalize_project_tags_migration(dry_run=dry_run)
        return True
    if sanitize_privacy:
        _run_sanitize_privacy_migration(dry_run=dry_run)
        return True
    if bucket_by_project:
        _run_bucket_by_project_migration()
        return True
    return False


def _resolve_migration_destination(
    cfg: Config,
    *,
    into_vault: bool,
    new_data_dir: str | None,
) -> tuple[Path, Path | None]:
    from memo.config import AI_SUBDIR

    if into_vault:
        chosen_vault = cfg.vault_path
        if chosen_vault is None:
            console.print(
                "[red]✗[/red] --into-vault needs a vault: set MEMO_VAULT_PATH or run "
                "`memo init` and pick an Obsidian vault first."
            )
            sys.exit(1)
        return (chosen_vault / AI_SUBDIR / "memory").resolve(), chosen_vault
    if new_data_dir:
        return Path(new_data_dir).resolve(), cfg.vault_path
    try:
        result = run_picker()
    except KeyboardInterrupt:
        console.print("[yellow]aborted[/yellow]")
        sys.exit(130)
    return result.data_dir, result.vault_path


def _copy_migration_files(
    src: Path,
    dst: Path,
    *,
    force: bool,
    yes: bool,
) -> int:
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
    for markdown in md_files:
        target = dst / markdown.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(markdown, target)
    return len(md_files)


def _write_migration_config(
    cfg: Config,
    *,
    dst: Path,
    chosen_vault: Path | None,
    into_vault: bool,
    snapshot: Path,
) -> Path:
    from memo.setup.config_io import _resolve_config_path

    existing_cfg = _resolve_config_path()
    if existing_cfg.is_file():
        shutil.copy2(existing_cfg, snapshot)
        console.print(f"[green]✓[/green] config snapshot → {snapshot} (use --rollback to restore)")
    # Preserve single_db: rewriting the config without it silently reverts a
    # consolidated install to sidecar mode, orphaning the folded-in stores.
    if into_vault:
        return write_config_file(
            data_dir=cfg.data_dir,
            vault_path=chosen_vault,
            memories_in_vault=True,
            single_db=cfg.single_db,
        )
    return write_config_file(data_dir=dst, vault_path=chosen_vault, single_db=cfg.single_db)


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
    help="Merge the sidecar DBs (history/graph/contradictions/crossref/episodes/fact_edges/verbatim) into the main memvec.db, set MEMO_SINGLE_DB=1 in config, and rename the legacy files to *.db.bak (reversible). Idempotent. Does not move any .md files.",
)
@click.option(
    "--bucket-by-project",
    is_flag=True,
    help="Move existing flat .md files into per-project folders "
    "(memory_dir/<project>/, _global/ when untagged) by their project: tag, "
    "then reindex. Non-destructive (moves only), idempotent.",
)
@click.option(
    "--backfill-valid-time",
    is_flag=True,
    help="One-time backfill: set valid_at = created for every record whose "
    "valid_at is still NULL (rows from before bi-temporal validity), mirroring "
    "the value into markdown frontmatter. Non-destructive, idempotent. Does not "
    "move any .md files or touch invalid_at.",
)
@click.option(
    "--normalize-project-tags",
    is_flag=True,
    help="Resolve historical memories with multiple project: tags: keep the "
    "folder project (or first tag for flat files) as canonical and preserve "
    "the rest as related-project: tags, then rebuild the index.",
)
@click.option(
    "--sanitize-privacy",
    is_flag=True,
    help="Mask recognized secrets and remove <private> spans from canonical "
    "Markdown, add an _redacted audit tag, then rebuild the index.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="With a normalization/sanitization migration: report without modifying files.",
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
    backfill_valid_time: bool,
    normalize_project_tags: bool,
    sanitize_privacy: bool,
    dry_run: bool,
    force: bool,
    yes: bool,
) -> None:
    from memo.setup.config_io import _resolve_config_path

    snapshot = _resolve_config_path().with_suffix(".toml.pre-migrate.bak")
    if _run_special_migration(
        snapshot=snapshot,
        rollback=rollback,
        consolidate_db=consolidate_db,
        backfill_valid_time=backfill_valid_time,
        normalize_project_tags=normalize_project_tags,
        sanitize_privacy=sanitize_privacy,
        bucket_by_project=bucket_by_project,
        dry_run=dry_run,
    ):
        return

    cfg = Config.from_env()
    src = Path(from_dir).resolve() if from_dir else cfg.memory_dir
    if not src.is_dir():
        console.print(f"[red]✗[/red] source dir does not exist: {src}")
        sys.exit(1)

    dst, chosen_vault = _resolve_migration_destination(
        cfg,
        into_vault=into_vault,
        new_data_dir=new_data_dir,
    )

    if dst == src:
        console.print(f"[red]✗[/red] source and destination are the same: {src}")
        sys.exit(1)

    n_copied = _copy_migration_files(src, dst, force=force, yes=yes)
    console.print(f"[green]✓[/green] copied {n_copied} files → {dst}")

    cfg_path = _write_migration_config(
        cfg,
        dst=dst,
        chosen_vault=chosen_vault,
        into_vault=into_vault,
        snapshot=snapshot,
    )
    console.print(f"[green]✓[/green] config: {cfg_path}")

    counts = _reindex_migration(Config.from_env())
    console.print(
        f"[green]✓[/green] reindex: checked {counts['checked']}  "
        f"added {counts['added']}  reindexed {counts['reindexed']}  "
        f"skipped {counts['skipped']}"
    )
    console.print(
        f"\n[dim]Source files at {src} were left untouched. "
        "After verifying the migration with `memo search`, you can rm them.[/dim]"
    )
