"""`memo ops` command group — maintenance jobs adopted from synapse (2026-07-30).

Hosts the nightly GC jobs (`gc-vault-orphans`, `gc-memo-duplicates`), the vault
re-ingestion runner (`vault-ingest`, driven by the `com.memo.vault-ingest`
launchd agent), and the ingest tombstone management (`exclude`). JSON output
shapes stay wire-compatible with the synapse originals so logs and tooling
keep parsing.

Registered onto the root group in cli.py via `cli.add_command(ops_group)`.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

import click

from memo.cli_common import console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config
from memo.contracts import ActorIdentity
from memo.ingest_exclude import IngestExcludeStore
from memo.ops_gc import find_exact_duplicates, find_vault_orphans

_LIST_ALL = 999999
_ACTOR = ActorIdentity(actor_id="memo-ops", actor_kind="system")


@click.group(name="ops")
def ops_group() -> None:
    """Maintenance jobs (GC, vault re-ingest, ingest tombstones)."""


def _all_records() -> tuple[Any, list[dict]]:
    mem = _get_memory(Config.from_env())
    return mem, [r.to_dict() for r in mem.list(limit=_LIST_ALL)]


def _delete_records(mem: Any, records: list[dict]) -> int:
    deleted = 0
    for r in records:
        with contextlib.suppress(ValueError):
            if mem.delete(r["id"], actor=_ACTOR):
                deleted += 1
    return deleted


@ops_group.command(name="gc-vault-orphans")
@click.option("--dry-run", is_flag=True, help="List orphans without deleting.")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON.")
def gc_vault_orphans_cmd(dry_run: bool, as_json: bool) -> None:
    """Delete memo records whose vault source file no longer exists on disk."""
    mem, records = _all_records()
    orphans = find_vault_orphans(records)
    deleted = 0 if dry_run else _delete_records(mem, orphans)
    result = {
        "scanned": len(records),
        "orphans": len(orphans),
        "deleted": deleted,
        "dry_run": dry_run,
    }
    if as_json:
        click.echo(json.dumps(result))
        return
    console.print(
        f"scanned: [cyan]{result['scanned']}[/cyan]  "
        f"orphans: [yellow]{result['orphans']}[/yellow]  "
        f"deleted: [green]{result['deleted']}[/green]"
        f"{'  [dim](dry-run)[/dim]' if dry_run else ''}"
    )


@ops_group.command(name="gc-memo-duplicates")
@click.option("--dry-run", is_flag=True, help="Count would-be deletions without deleting.")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON.")
def gc_memo_duplicates_cmd(dry_run: bool, as_json: bool) -> None:
    """Delete records whose body is an exact duplicate of a newer record."""
    mem, records = _all_records()
    stale = find_exact_duplicates(records)
    dup_groups = len(_dup_group_iter(stale))
    deleted = len(stale) if dry_run else _delete_records(mem, stale)
    result = {
        "scanned": len(records),
        "dup_groups": dup_groups,
        "deleted": deleted,
        "dry_run": dry_run,
    }
    if as_json:
        click.echo(json.dumps(result))
        return
    console.print(
        f"scanned: [cyan]{result['scanned']}[/cyan]  "
        f"dup_groups: [yellow]{result['dup_groups']}[/yellow]  "
        f"deleted: [green]{result['deleted']}[/green]"
        f"{'  [dim](dry-run)[/dim]' if dry_run else ''}"
    )


def _dup_group_iter(stale: list[dict]) -> list[str]:
    """Distinct duplicate groups represented in *stale* (by body hash)."""
    import hashlib

    return sorted({hashlib.sha256(str(s.get("body") or "").encode()).hexdigest() for s in stale})


@ops_group.command(name="vault-ingest")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON.")
def vault_ingest_cmd(as_json: bool) -> None:
    """Re-ingest each Obsidian vault (`git pull` + `memo ingest --prune` + tombstones)."""
    from memo.vault_ingest import run_vault_ingest

    result = run_vault_ingest()
    if as_json:
        click.echo(json.dumps(result))
    else:
        for v in result["vaults"]:
            status = (
                "[green]ok[/green]"
                if v["returncode"] == 0
                else f"[red]exit {v['returncode']}[/red]"
            )
            console.print(f"{v['vault']}: {status}  [dim]{v['path']}[/dim]")
    if not result["ok"]:
        raise SystemExit(1)


@ops_group.group(name="exclude")
def exclude_group() -> None:
    """Ingest tombstones: vault-relative globs skipped on every re-ingest."""


@exclude_group.command(name="list")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON.")
def exclude_list_cmd(as_json: bool) -> None:
    """List tombstoned paths per vault label."""
    store = IngestExcludeStore()
    data = {label: globs for label in store.all_labels() if (globs := store.globs(label))}
    if as_json:
        click.echo(json.dumps(data))
        return
    if not data:
        console.print("[dim]no tombstones[/dim]")
        return
    for label, globs in data.items():
        console.print(f"[bold]{label}[/bold]")
        for g in globs:
            console.print(f"  {g}")


@exclude_group.command(name="add")
@click.argument("vault_label")
@click.argument("rel_path")
def exclude_add_cmd(vault_label: str, rel_path: str) -> None:
    """Tombstone REL_PATH (vault-relative) for VAULT_LABEL."""
    added = IngestExcludeStore().add(vault_label=vault_label, rel_path=rel_path)
    console.print("[green]✓ added[/green]" if added else "[dim]already present[/dim]")


@exclude_group.command(name="remove")
@click.argument("vault_label")
@click.argument("rel_path")
def exclude_remove_cmd(vault_label: str, rel_path: str) -> None:
    """Drop REL_PATH from VAULT_LABEL's tombstones."""
    removed = IngestExcludeStore().remove(vault_label=vault_label, rel_path=rel_path)
    console.print("[green]✓ removed[/green]" if removed else "[dim]not found[/dim]")


@ops_group.command(name="install")
@click.argument("service", type=click.Choice(["chat"]))
@click.option("--port", default=8765, show_default=True, type=int)
@click.option(
    "--dist",
    default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directorio dist de la SPA (opcional).",
)
def ops_install(service: str, port: int, dist: Path | None) -> None:
    """Install a memo launchd agent (currently: chat)."""
    import shutil

    from memo.ops_launchd import install_chat

    memo_bin = shutil.which("memo")
    if not memo_bin:
        raise click.ClickException("no encuentro el binario `memo` en PATH")
    resolved_dist = str(dist.expanduser().resolve()) if dist else None
    try:
        path = install_chat(memo_bin, Path.home(), port=port, dist=resolved_dist)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"installed {path}")


@ops_group.command(name="uninstall")
@click.argument("service", type=click.Choice(["chat"]))
def ops_uninstall(service: str) -> None:
    """Uninstall a memo launchd agent."""
    from memo.ops_launchd import uninstall_chat

    click.echo("removed" if uninstall_chat(Path.home()) else "not installed")


@ops_group.command(name="status")
def ops_status() -> None:
    """Show all com.memo.* launchd agents."""
    import subprocess

    from memo.ops_launchd import parse_launchctl_list

    out = subprocess.run(["launchctl", "list"], capture_output=True, text=True, check=True).stdout
    for row in parse_launchctl_list(out):
        state = f"pid {row['pid']}" if row["pid"] else f"exit {row['last_exit']}"
        click.echo(f"{row['label']}\t{state}")
