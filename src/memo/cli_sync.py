"""`memo sync` command group — multi-machine sync via audit-log replay.

Extracted from cli.py (3a god-module decomposition). Registered onto the
root group in cli.py via `cli.add_command(sync_group)`.

The sync model is pull-only: a machine replays the events missing from its
local store that exist in a remote `history.db`. There is no file diff and no
push (the remote machine pulls instead).
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from memo.cli_common import console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config


def _resolve_remote_history_db(remote: str | None) -> Path | None:
    """Map a ``--remote`` arg to the remote machine's ``history.db``.

    Accepts either a direct path to a ``.db`` file or a memo state dir that
    contains ``history.db``.
    """
    if not remote:
        return None
    p = Path(remote)
    return p if p.suffix == ".db" else p / "history.db"


@click.group(name="sync")
def sync_group() -> None:
    """Multi-machine sync — replay a remote machine's audit log locally."""
    pass


@sync_group.command(name="export-signal")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def sync_export_signal(as_json: bool) -> None:
    """Dump local signal (access/health/feedback) to `signal/*.json` for git.

    The `.md` memorias sync via git; this snapshots the signal tables that
    live only in the rebuildable `memvec.db` so a peer can restore ranking.
    """
    from memo.sync_signal import export_signal, signal_dir_for

    cfg = Config.from_env()
    mem = _get_memory(cfg)
    sig = signal_dir_for(cfg)
    counts = export_signal(mem.store, sig)

    if as_json:
        click.echo(json.dumps({"signal_dir": str(sig), "counts": counts}, indent=2))
        return
    console.print(f"[bold]Exported signal[/bold] → {sig}")
    for table, n in counts.items():
        console.print(f"  {table}: {n}")


@sync_group.command(name="import-signal")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def sync_import_signal(as_json: bool) -> None:
    """Merge a peer's `signal/*.json` snapshot into the local store.

    Idempotent: access = max, health = newer wins, feedback = union by id.
    """
    from memo.sync_signal import import_signal, signal_dir_for

    cfg = Config.from_env()
    mem = _get_memory(cfg)
    sig = signal_dir_for(cfg)
    counts = import_signal(mem.store, sig)

    if as_json:
        click.echo(json.dumps({"signal_dir": str(sig), "merged": counts}, indent=2))
        return
    console.print(f"[bold]Imported signal[/bold] ← {sig}")
    for table, n in counts.items():
        console.print(f"  {table}: {n}")


@sync_group.command(name="diff")
@click.option("--remote", help="Path to remote memo state dir")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def sync_diff(remote: str | None, as_json: bool) -> None:
    """Not supported in the replay sync model (no precomputed diff).

    Use `memo sync pull` to apply missing remote events.
    """
    msg = "replay sync model has no precomputed diff; use `memo sync pull`"
    if as_json:
        click.echo(json.dumps({"error": msg}, indent=2))
        return
    console.print(f"[yellow]{msg}[/yellow]")


@sync_group.command(name="push")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
@click.option("--quiet", is_flag=True, help="Soft-fail (exit 0) if not a git clone — for hooks")
def sync_push(as_json: bool, quiet: bool) -> None:
    """Export signal + commit + git push the memo-sync repo (Stop hook).

    Requires the memorias dir to live inside a git clone (see `memo sync clone`).
    """
    from memo.sync_git import SyncGitError
    from memo.sync_git import sync_push as _git_push

    cfg = Config.from_env()
    mem = _get_memory(cfg)
    try:
        out = _git_push(cfg, mem.store)
    except SyncGitError as e:
        if quiet:
            console.print(f"[dim]sync push skipped: {e}[/dim]")
            return
        raise

    if as_json:
        click.echo(json.dumps(out, indent=2))
        return
    if out.get("pushed"):
        console.print(f"[bold green]Pushed[/bold green] {out['committed_files']} files → {out['branch']}")
    else:
        console.print(f"[dim]Nothing to push ({out.get('reason')})[/dim]")


@sync_group.command(name="pull")
@click.option("--remote", help="Legacy: path to a remote memo state dir (audit-log replay)")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
@click.option("--quiet", is_flag=True, help="Soft-fail (exit 0) if not a git clone — for hooks")
def sync_pull(remote: str | None, as_json: bool, quiet: bool) -> None:
    """Git pull the memo-sync repo + merge remote signal + reindex (SessionStart hook).

    With --remote <path>, falls back to the legacy audit-log replay model.

    Example: memo sync pull
             memo sync pull --remote /path/to/remote/memo   # legacy replay
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    if remote:  # legacy audit-log replay
        remote_db = _resolve_remote_history_db(remote)
        assert remote_db is not None
        diff = mem.sync.sync_from_remote(remote_db)
        console.print("[bold]Pull Sync (replay)[/bold]")
        console.print(f"Applied: {diff.applied}")
        console.print(f"Conflicts: {diff.conflicts}")
        console.print(f"Errors: {diff.errors}")
        return

    from memo.sync_git import SyncGitError
    from memo.sync_git import sync_pull as _git_pull

    try:
        out = _git_pull(cfg, mem.store, mem)
    except SyncGitError as e:
        if quiet:
            console.print(f"[dim]sync pull skipped: {e}[/dim]")
            return
        raise

    if as_json:
        click.echo(json.dumps(out, indent=2))
        return
    console.print(f"[bold green]Pulled[/bold green] {out['branch']}")
    console.print(f"Reindexed: {out['reindexed']}")


@sync_group.command(name="clone")
@click.argument("url")
@click.option("--dest", default=None, help="Clone destination (default ~/repos/memo-sync)")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def sync_clone(url: str, dest: str | None, as_json: bool) -> None:
    """Bootstrap a new machine: clone the memo-sync repo (F6).

    Prints the next steps (point MEMO_DATA_DIR at the cloned memorias dir, then
    `memo reindex --rebuild && memo sync import-signal`). Does not touch config.
    """
    from pathlib import Path

    from memo.sync_git import clone_bootstrap

    dest_path = Path(dest).expanduser() if dest else Path.home() / "repos" / "memo-sync"
    out = clone_bootstrap(url, dest_path)

    if as_json:
        click.echo(json.dumps(out, indent=2))
        return
    console.print(f"[bold green]Cloned[/bold green] → {out['cloned']} ({out['memorias']} memorias)")
    console.print("\n[bold]Next steps on this machine:[/bold]")
    console.print(f"  1. Set [cyan]MEMO_DATA_DIR={out['memorias_dir']}[/cyan] in your config / MCP env")
    console.print("  2. [cyan]memo reindex --rebuild[/cyan]   # build the index from the .md")
    console.print("  3. [cyan]memo sync import-signal[/cyan]  # restore access/health/feedback")


@sync_group.command(name="bootstrap")
@click.argument("url")
@click.option("--dest", default=None, help="Clone destination (default ~/repos/memo-sync)")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def sync_bootstrap(url: str, dest: str | None, as_json: bool) -> None:
    """One-shot new-machine setup: clone the memo-sync repo, point config at it,
    rebuild the index, and import signal. Idempotent (re-run reuses the clone).

    Unlike `clone`, this wires everything end-to-end — after it, memo reads the
    git-synced corpus and the SessionStart/Stop hooks keep it in sync.

    Example: memo sync bootstrap https://github.com/jagoff/memo-sync.git
    """
    from memo.sync_git import bootstrap_clone
    from memo.sync_signal import import_signal, signal_dir_for

    dest_path = Path(dest).expanduser() if dest else Path.home() / "repos" / "memo-sync"
    out = bootstrap_clone(url, dest_path)

    # Reindex + import signal against the freshly-pointed data_dir.
    cfg = Config.from_env(data_dir=Path(out["memorias_dir"]))
    mem = _get_memory(cfg)
    out["reindexed"] = mem.reindex(rebuild=True)
    out["signal"] = import_signal(mem.store, signal_dir_for(cfg))

    if as_json:
        click.echo(json.dumps(out, indent=2))
        return
    verb = "Reused" if out.get("reused") else "Cloned"
    console.print(f"[bold green]{verb}[/bold green] → {out['cloned']} ({out['memorias']} memorias)")
    console.print(f"config → [cyan]{out['config']}[/cyan] (data_dir = {out['memorias_dir']})")
    console.print(f"reindexed: {out['reindexed']}")
    console.print(f"signal merged: {out['signal']}")
    console.print("[bold green]Ready.[/bold green] memo now reads the git-synced corpus.")


@sync_group.command(name="both")
@click.option("--remote", required=True, help="Path to remote memo state dir")
def sync_both(remote: str) -> None:
    """Sync from a remote machine (replay model alias for pull).

    In the replay model "both directions" is achieved by each machine pulling
    the other's audit log; from this side that is a pull.

    Example: memo sync both --remote /path/to/remote/memo
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    remote_db = _resolve_remote_history_db(remote)
    assert remote_db is not None  # --remote is required
    diff = mem.sync.sync_from_remote(remote_db)

    console.print("[bold]Sync (replay)[/bold]")
    console.print(f"Applied: {diff.applied}")
    console.print(f"Conflicts: {diff.conflicts}")
    console.print(f"Errors: {diff.errors}")
