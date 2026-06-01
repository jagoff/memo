"""`memo maint-daemon` command group — maintenance (synthesis-LLM) lifecycle.

Mirrors `cli_ingest_daemon`. Hosts the consolidation synthesis LLM in its
own process so it stays out of memo-mcp's resident set. Opt-in: set
`MEMO_MAINT_VIA_DAEMON=1` so `Memory.consolidate` routes here; otherwise
consolidation runs in-process as before.

Registered onto the root group in cli.py via
`cli.add_command(maint_daemon_group)`.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import click

from memo.cli_common import console
from memo.config import Config


@click.group(name="maint-daemon")
def maint_daemon_group() -> None:
    """Manage the maintenance (consolidation synthesis-LLM) daemon.

    Subcommands: start, stop, status, (internal) _serve.
    """


@maint_daemon_group.command(name="start")
def maint_daemon_start() -> None:
    """Start the maintenance daemon in the background."""
    import subprocess as _subprocess

    from memo.maint_server import _is_pid_alive, _read_pid, _socket_path

    cfg = Config.from_env()
    pid = _read_pid(cfg.state_dir)
    if pid is not None and _is_pid_alive(pid):
        console.print(f"[dim]maint daemon already running (pid={pid})[/dim]")
        return

    log_dir = Path.home() / "Library" / "Logs" / "memo"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "maint-daemon.log"

    env = os.environ.copy()
    env["MEMO_NONINTERACTIVE"] = "1"

    with open(log_file, "a") as lf:
        proc = _subprocess.Popen(
            [sys.executable, "-m", "memo.cli", "maint-daemon", "_serve"],
            stdout=lf,
            stderr=lf,
            stdin=_subprocess.DEVNULL,
            start_new_session=True,
            env=env,
        )

    sock_path = _socket_path(cfg.state_dir)
    for _ in range(20):
        time.sleep(0.1)
        if sock_path.exists():
            break

    click.echo(f"maint daemon started (pid={proc.pid})", err=True)


@maint_daemon_group.command(name="stop")
def maint_daemon_stop() -> None:
    """Stop the maintenance daemon (sends SIGTERM)."""
    import signal as _signal

    from memo.maint_server import _cleanup, _is_pid_alive, _read_pid

    cfg = Config.from_env()
    pid = _read_pid(cfg.state_dir)
    if pid is None:
        console.print("[dim]maint daemon not running (no PID file)[/dim]")
        return
    if not _is_pid_alive(pid):
        console.print("[dim]maint daemon not running (stale PID file)[/dim]")
        _cleanup(cfg.state_dir)
        return
    try:
        os.kill(pid, _signal.SIGTERM)
        console.print(f"maint daemon stopped (pid={pid})")
    except ProcessLookupError:
        console.print("[dim]maint daemon already gone[/dim]")
    _cleanup(cfg.state_dir)


@maint_daemon_group.command(name="status")
def maint_daemon_status() -> None:
    """Print whether the maintenance daemon is running."""
    from memo.maint_server import _is_pid_alive, _read_pid, _socket_path

    cfg = Config.from_env()
    pid = _read_pid(cfg.state_dir)
    sock = _socket_path(cfg.state_dir)

    if pid is not None and _is_pid_alive(pid):
        console.print(f"[green]running[/green] pid={pid}  socket={sock}")
    else:
        console.print(f"[red]stopped[/red]  socket={sock}")


@maint_daemon_group.command(name="_serve", hidden=True)
def maint_daemon_serve() -> None:
    """Internal: run the daemon in the foreground (called by 'start')."""
    from memo.maint_server import run_server

    run_server()
