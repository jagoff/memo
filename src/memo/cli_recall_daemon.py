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

    Subcommands: start, stop, restart, status, (internal) _serve.
    """


@recall_daemon_group.command(name="start")
def recall_daemon_start() -> None:
    """Start the recall daemon in the background."""
    import subprocess as _subprocess

    from memo.recall_server import _is_pid_alive, _read_pid, connect_and_send

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

    # Readiness = the child answers a ping on the freshly bound socket. A bare
    # exists() check would report success for a stale socket file left by a
    # crashed daemon; the child unlinks + rebinds that file under its own start
    # flock, so the parent probes by connecting instead of racing an unlink.
    ready = False
    for _ in range(20):
        time.sleep(0.1)
        if proc.poll() is not None:
            break  # child already exited — it can never become ready
        if connect_and_send(cfg.state_dir, {"op": "ping"}, timeout=0.5) is not None:
            ready = True
            break

    if not ready:
        click.echo("recall daemon failed to start — check logs", err=True)
        sys.exit(1)

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


@recall_daemon_group.command(name="restart")
@click.pass_context
def recall_daemon_restart(ctx: click.Context) -> None:
    """Restart the recall daemon (stop, then start).

    Use after upgrading the runtime so the daemon reloads new code. If the
    daemon is launchd-managed (KeepAlive), the SIGTERM from stop already
    triggers a respawn — this waits for that new process and only starts a
    fresh one if launchd doesn't bring it back, avoiding a double daemon.
    """
    from memo.recall_server import _is_pid_alive, _read_pid

    cfg = Config.from_env()
    old_pid = _read_pid(cfg.state_dir)

    ctx.invoke(recall_daemon_stop)

    # Old process is dead + PID file cleaned. Give a launchd KeepAlive up to
    # ~5s to respawn under its own management; if a new live PID appears,
    # launchd handled the restart and we must NOT start a competing process.
    for _ in range(50):
        time.sleep(0.1)
        pid = _read_pid(cfg.state_dir)
        if pid is not None and pid != old_pid and _is_pid_alive(pid):
            console.print(f"[green]restarted[/green] (respawned by launchd, pid={pid})")
            return

    # Not launchd-managed (or respawn disabled) — start a fresh daemon.
    ctx.invoke(recall_daemon_start)


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
