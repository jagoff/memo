"""`memo ingest-daemon` command group — batch-ingest worker lifecycle.

Mirrors `cli_recall_daemon`. The ingest daemon runs heavy batch jobs (repo
indexing) in its own process with a single serialized writer, so they never
block the MCP request path or contend on the recall embedder lock. Opt-in:
set `MEMO_INGEST_VIA_DAEMON=1` so `Memory.repo_index` routes here; otherwise
indexing runs in-process as before.

Registered onto the root group in cli.py via
`cli.add_command(ingest_daemon_group)`.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import click

from memo.cli_common import console
from memo.config import Config


@click.group(name="ingest-daemon")
def ingest_daemon_group() -> None:
    """Manage the batch-ingest worker daemon.

    Subcommands: start, stop, status, (internal) _serve.
    """


@ingest_daemon_group.command(name="start")
def ingest_daemon_start() -> None:
    """Start the ingest daemon in the background."""
    import subprocess as _subprocess

    from memo import ingest_client
    from memo.ingest_server import _is_pid_alive, _read_pid

    cfg = Config.from_env()
    pid = _read_pid(cfg.state_dir)
    if pid is not None and _is_pid_alive(pid):
        console.print(f"[dim]ingest daemon already running (pid={pid})[/dim]")
        return

    # No parent-side socket unlink: a concurrent daemon that has already bound
    # its socket but not yet written its pid would look "not running" here and
    # get its live socket unlinked (orphaned). The child unlinks any stale
    # socket under its own start flock; the connect-based probe below is not
    # fooled by a leftover file (a stale socket answers no ping).
    log_dir = Path.home() / "Library" / "Logs" / "memo"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "ingest-daemon.log"

    env = os.environ.copy()
    env["MEMO_NONINTERACTIVE"] = "1"

    with open(log_file, "a") as lf:
        proc = _subprocess.Popen(
            [sys.executable, "-m", "memo.cli", "ingest-daemon", "_serve"],
            stdout=lf,
            stderr=lf,
            stdin=_subprocess.DEVNULL,
            start_new_session=True,
            env=env,
        )

    # Readiness = the child answers a ping on the freshly bound socket
    # (a bare exists() check would pass on a stale file).
    ready = False
    for _ in range(20):
        time.sleep(0.1)
        if proc.poll() is not None:
            break  # child already exited — it can never become ready
        if ingest_client.ping(state_dir=cfg.state_dir) is not None:
            ready = True
            break

    if not ready:
        click.echo("ingest daemon failed to start — check logs", err=True)
        sys.exit(1)

    click.echo(f"ingest daemon started (pid={proc.pid})", err=True)


@ingest_daemon_group.command(name="stop")
def ingest_daemon_stop() -> None:
    """Stop the ingest daemon (sends SIGTERM)."""
    import signal as _signal

    from memo.ingest_server import _cleanup, _is_pid_alive, _read_pid

    cfg = Config.from_env()
    pid = _read_pid(cfg.state_dir)
    if pid is None:
        console.print("[dim]ingest daemon not running (no PID file)[/dim]")
        return
    if not _is_pid_alive(pid):
        console.print("[dim]ingest daemon not running (stale PID file)[/dim]")
        _cleanup(cfg.state_dir)
        return
    try:
        os.kill(pid, _signal.SIGTERM)
        console.print(f"ingest daemon stopped (pid={pid})")
    except ProcessLookupError:
        console.print("[dim]ingest daemon already gone[/dim]")
    _cleanup(cfg.state_dir)


@ingest_daemon_group.command(name="status")
@click.option("--job", "job_id", default=None, help="Poll one queued job id.")
@click.option("--json", "as_json", is_flag=True)
def ingest_daemon_status(job_id: str | None, as_json: bool) -> None:
    """Print daemon health or poll one queued job."""
    import json

    from memo import ingest_client
    from memo.ingest_ledger import IngestFailureLedger
    from memo.ingest_server import _is_pid_alive, _read_pid, _socket_path

    cfg = Config.from_env()
    pid = _read_pid(cfg.state_dir)
    sock = _socket_path(cfg.state_dir)
    running = pid is not None and _is_pid_alive(pid)
    if job_id:
        payload = ingest_client.status(job_id, state_dir=cfg.state_dir) if running else None
        if payload is None:
            payload = {"error": "ingest daemon unreachable", "job_id": job_id}
        if as_json:
            click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        elif "error" in payload:
            console.print(f"[red]{payload['error']}[/red]")
        else:
            console.print(
                f"job={job_id} state={payload.get('state')} error={payload.get('error') or '-'}"
            )
        return

    ping = ingest_client.ping(state_dir=cfg.state_dir) if running else None
    payload = {
        "running": running,
        "pid": pid if running else None,
        "socket": str(sock),
        "daemon": ping,
        "ledger": IngestFailureLedger(cfg.state_dir / "ingest-jobs.jsonl").health(),
    }
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    elif running:
        health = ping.get("health") if isinstance(ping, dict) else {}
        console.print(
            f"[green]running[/green] pid={pid} socket={sock} "
            f"jobs={ping.get('jobs', '?') if isinstance(ping, dict) else '?'} "
            f"worker_alive={health.get('worker_alive', '?') if isinstance(health, dict) else '?'}"
        )
    else:
        console.print(f"[red]stopped[/red]  socket={sock}")


@ingest_daemon_group.command(name="_serve", hidden=True)
def ingest_daemon_serve() -> None:
    """Internal: run the daemon in the foreground (called by 'start')."""
    from memo.ingest_server import run_server

    run_server()
