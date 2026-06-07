"""`memo recall-daemon` command group — recall hook daemon lifecycle.

Extracted from cli.py (3a god-module decomposition). Registered onto the
root group in cli.py via `cli.add_command(recall_daemon_group)`.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import click

from memo.cli_common import console
from memo.config import Config


@click.group(name="recall-daemon")
def recall_daemon_group() -> None:
    """Manage the persistent recall daemon (fast socket-based recall).

    The daemon keeps the MLX embedder in RAM so recall-hook answers in
    <200 ms instead of 1-2 s per prompt (cold Python + MLX load).

    Subcommands: start, stop, status, (internal) _serve.
    """


@recall_daemon_group.command(name="start")
def recall_daemon_start() -> None:
    """Start the recall daemon in the background."""
    import subprocess as _subprocess

    from memo.recall_server import _is_pid_alive, _read_pid, _socket_path

    cfg = Config.from_env()
    pid = _read_pid(cfg.state_dir)
    if pid is not None and _is_pid_alive(pid):
        console.print(f"[dim]recall daemon already running (pid={pid})[/dim]")
        return

    log_dir = Path.home() / "Library" / "Logs" / "memo"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "recall-daemon.log"

    env = os.environ.copy()
    env["MEMO_NONINTERACTIVE"] = "1"

    with open(log_file, "a") as lf:
        proc = _subprocess.Popen(
            [sys.executable, "-m", "memo.cli", "recall-daemon", "_serve"],
            stdout=lf,
            stderr=lf,
            stdin=_subprocess.DEVNULL,  # don't inherit hook stdin pipe
            start_new_session=True,
            env=env,
        )

    # Wait briefly for the socket to appear
    sock_path = _socket_path(cfg.state_dir)
    for _ in range(20):
        time.sleep(0.1)
        if sock_path.exists():
            break

    click.echo(f"recall daemon started (pid={proc.pid})", err=True)


@recall_daemon_group.command(name="stop")
def recall_daemon_stop() -> None:
    """Stop the recall daemon (sends SIGTERM)."""
    import signal as _signal

    from memo.recall_server import _cleanup, _is_pid_alive, _read_pid

    cfg = Config.from_env()
    pid = _read_pid(cfg.state_dir)
    if pid is None:
        console.print("[dim]recall daemon not running (no PID file)[/dim]")
        return
    if not _is_pid_alive(pid):
        console.print("[dim]recall daemon not running (stale PID file)[/dim]")
        _cleanup(cfg.state_dir)
        return
    try:
        os.kill(pid, _signal.SIGTERM)
        console.print(f"recall daemon stopped (pid={pid})")
    except ProcessLookupError:
        console.print("[dim]recall daemon already gone[/dim]")
    _cleanup(cfg.state_dir)


@recall_daemon_group.command(name="status")
def recall_daemon_status() -> None:
    """Print whether the recall daemon is running."""
    from memo.recall_server import _is_pid_alive, _read_pid, _socket_path

    cfg = Config.from_env()
    pid = _read_pid(cfg.state_dir)
    sock = _socket_path(cfg.state_dir)

    if pid is not None and _is_pid_alive(pid):
        console.print(f"[green]running[/green] pid={pid}  socket={sock}")
    else:
        console.print(f"[red]stopped[/red]  socket={sock}")


@recall_daemon_group.command(name="_serve", hidden=True)
def recall_daemon_serve() -> None:
    """Internal: run the daemon in the foreground (called by 'start')."""
    from memo.recall_server import run_server

    run_server()


# ── Embed daemon — observability over the shared embedder sidecar ────────────
