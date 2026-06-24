"""`memo sync` command group — multi-machine sync.

Extracted from cli.py (3a god-module decomposition). Registered onto the
root group in cli.py via `cli.add_command(sync_group)`.

Two tiers (see CLAUDE.md):
- Git remote (`memo sync push` / `pull`, via `memo.sync_git`): the cross-machine
  channel — pull-rebase-before-push under a per-machine flock.
- Legacy audit-log replay (`export-signal` / `import-signal` / `diff`): the
  `--remote <path>` fallback that replays events missing from the local store.
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

    The `.md` memories sync via git; this snapshots the signal tables that
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
    rebuilt = mem.store.rebuild_feedback_vecs(mem.embedder.embed_query)

    if as_json:
        click.echo(json.dumps({"signal_dir": str(sig), "merged": counts, "feedback_vecs_rebuilt": rebuilt}, indent=2))
        return
    console.print(f"[bold]Imported signal[/bold] ← {sig}")
    for table, n in counts.items():
        console.print(f"  {table}: {n}")
    if rebuilt:
        console.print(f"  source_feedback_vec: {rebuilt}")


@sync_group.command(name="diff", hidden=True)
@click.option("--remote", help="Path to remote memo state dir")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def sync_diff(remote: str | None, as_json: bool) -> None:
    """Deprecated: replay sync model has no precomputed diff.

    Use `memo sync pull` to apply missing remote events. Will be removed in a
    future release.
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

    Requires the memories dir to live inside a git clone (see `memo sync clone`).
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
        if remote_db is None:
            raise click.ClickException(
                f"Remote history DB not found at: {remote}"
            )
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

    Prints the next steps (point MEMO_DATA_DIR at the cloned memories dir, then
    `memo reindex --rebuild && memo sync import-signal`). Does not touch config.
    """
    from pathlib import Path

    from memo.sync_git import clone_bootstrap

    dest_path = Path(dest).expanduser() if dest else Path.home() / "repos" / "memo-sync"
    out = clone_bootstrap(url, dest_path)

    if as_json:
        click.echo(json.dumps(out, indent=2))
        return
    console.print(f"[bold green]Cloned[/bold green] → {out['cloned']} ({out['memorias']} memories)")
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
    out["feedback_vecs_rebuilt"] = mem.store.rebuild_feedback_vecs(mem.embedder.embed_query)

    if as_json:
        click.echo(json.dumps(out, indent=2))
        return
    verb = "Reused" if out.get("reused") else "Cloned"
    console.print(f"[bold green]{verb}[/bold green] → {out['cloned']} ({out['memorias']} memories)")
    console.print(f"config → [cyan]{out['config']}[/cyan] (data_dir = {out['memorias_dir']})")
    console.print(f"reindexed: {out['reindexed']}")
    console.print(f"signal merged: {out['signal']}")
    console.print("[bold green]Ready.[/bold green] memo now reads the git-synced corpus.")


@sync_group.command(name="status")
@click.option("--check-remote", is_flag=True, help="Probe the remote (network) for reachability.")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def sync_status(check_remote: bool, as_json: bool) -> None:
    """Is the GitHub sync healthy? Shows clone / ahead-behind / dirty / stranded.

    Kills the silent no-op: if the memories dir isn't a git clone, the
    SessionStart/Stop hooks soft-fail and nothing syncs — this says so plainly.
    """
    from memo.sync_git import sync_status as _status

    cfg = Config.from_env()
    st = _status(cfg, check_remote=check_remote)

    if as_json:
        click.echo(json.dumps(st, indent=2))
        return
    if not st.get("is_git_clone"):
        console.print("[bold red]NOT syncing[/bold red] — data_dir is not a git clone.")
        console.print(f"  [dim]{st.get('reason', '')}[/dim]")
        console.print("  Fix: [cyan]memo sync bootstrap <url>[/cyan]")
        return
    ahead, behind, dirty = st["ahead"], st["behind"], st["dirty_files"]
    if st["pending"]:
        verdict, color = "STRANDED — push failed, will retry", "red"
    elif ahead or dirty:
        verdict, color = f"behind remote by {ahead} commit(s) unpushed", "yellow"
    elif behind:
        verdict, color = f"{behind} remote commit(s) to pull", "yellow"
    else:
        verdict, color = "up to date with GitHub", "green"
    console.print(f"[bold {color}]{verdict}[/bold {color}]")
    console.print(f"  repo:   {st['root']}  ({st['branch']})")
    console.print(f"  remote: {st['remote'] or '—'}")
    console.print(f"  ahead {ahead} · behind {behind} · dirty {dirty} · last {st['last_commit'] or '—'}")
    if check_remote:
        console.print(f"  remote reachable: {st['remote_reachable']}")


@sync_group.command(name="once")
@click.option("--quiet", is_flag=True, help="Soft-fail (exit 0) if not a git clone — for hooks")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def sync_once_cmd(quiet: bool, as_json: bool) -> None:
    """One lock-guarded machine sync (pull-rebase-before-push). Stop-hook flush.

    Unlike `sync auto` (debounced), this always attempts the git step — used at
    session end to flush this session's captures. The machine lock still ensures
    only one process does git at a time.
    """
    from memo.sync_git import sync_once, sync_tier

    cfg = Config.from_env()
    if sync_tier(cfg) != "remote":
        if not quiet:
            console.print("[dim]sync once skipped: local tier (no remote / not a clone)[/dim]")
        if as_json:
            click.echo(json.dumps({"tier": "local", "skipped": "no remote"}))
        return
    mem = _get_memory(cfg)
    out = sync_once(cfg, mem.store, mem)
    if as_json:
        click.echo(json.dumps(out))
        return
    bits = []
    if out.get("pulled"):
        bits.append("pulled")
    if out.get("pushed"):
        bits.append("pushed")
    if out.get("skipped"):
        bits.append(f"skipped({out['skipped']})")
    console.print(f"[green]sync once[/green] → {', '.join(bits) or 'nothing to do'}")


@sync_group.command(name="auto")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def sync_auto(as_json: bool) -> None:
    """Debounced, self-throttled sync — safe to fire every prompt (async hook).

    Makes durable knowledge propagate WITHIN a session instead of only at Stop:
    pulls if ``MEMO_SYNC_PULL_INTERVAL_S`` elapsed, pushes if there's something
    to share (dirty/ahead/stranded) and ``MEMO_SYNC_PUSH_DEBOUNCE_S`` elapsed.
    Cheap no-op (timestamp check, no git) when neither is due; soft-fails when
    the data_dir isn't a git clone. Disable with ``MEMO_SYNC_AUTO=0``.
    """
    import time

    from memo.flags import flag_bool, flag_int

    cfg = Config.from_env()
    did: dict = {"pulled": False, "pushed": False, "skipped": None}

    if not flag_bool("MEMO_SYNC_AUTO"):  # registry default True; MEMO_SYNC_AUTO=0 opts out
        did["skipped"] = "disabled"
        if as_json:
            click.echo(json.dumps(did))
        return

    ts_file = cfg.state_dir / ".sync_auto_ts"
    try:
        ts = json.loads(ts_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        ts = {}
    now = time.time()
    push_debounce = flag_int("MEMO_SYNC_PUSH_DEBOUNCE_S") or 120
    pull_interval = flag_int("MEMO_SYNC_PULL_INTERVAL_S") or 300
    pull_due = now - float(ts.get("last_pull", 0)) >= pull_interval
    push_due = now - float(ts.get("last_push", 0)) >= push_debounce
    if not pull_due and not push_due:
        did["skipped"] = "not due"
        if as_json:
            click.echo(json.dumps(did))
        return

    from memo.sync_git import sync_tier

    if sync_tier(cfg) != "remote":
        did["skipped"] = "local tier (no remote / not a clone)"
        if as_json:
            click.echo(json.dumps(did))
        return

    # Route through the single, lock-guarded machine coordinator: it owns the
    # pull-rebase-before-push ordering and the same-machine lock, so concurrent
    # sessions don't race and an advanced remote rebases cleanly.
    from memo.sync_git import sync_once

    mem = _get_memory(cfg)
    out = sync_once(cfg, mem.store, mem, do_pull=pull_due, do_push=push_due)
    did["pulled"] = bool(out.get("pulled"))
    did["pushed"] = bool(out.get("pushed"))
    if out.get("skipped"):
        did["skipped"] = out["skipped"]
    if pull_due:
        ts["last_pull"] = now
    if push_due:
        ts["last_push"] = now

    import contextlib

    with contextlib.suppress(OSError):
        ts_file.write_text(json.dumps(ts), encoding="utf-8")
    if as_json:
        click.echo(json.dumps(did))


@sync_group.command(name="both", hidden=True)
@click.option("--remote", required=True, help="Path to remote memo state dir")
def sync_both(remote: str) -> None:
    """Deprecated alias for ``sync pull``.

    In the replay model "both directions" is achieved by each machine pulling
    the other's audit log; from this side that is a pull. Will be removed in a
    future release.

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
