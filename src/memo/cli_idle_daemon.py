"""`memo idle-daemon` command group — background idle capture daemon.

Runs periodic idle capture every MEMO_SESSION_IDLE_CAPTURE_SECS (default 10s)
to automatically save insights from the current session without needing Claude Code hooks.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click

from memo.cli_common import console
from memo.config import Config


@click.group(name="idle-daemon")
def idle_daemon_group() -> None:
    """Manage the persistent idle capture daemon.

    Runs memo_idle_capture periodically in background to automatically
    save insights from sessions without Claude Code hooks. Useful for opencode,
    Windsurf, and other agents that use MCP instead of hooks.

    Subcommands: start, stop, status, (internal) _serve.
    """


@idle_daemon_group.command(name="start")
def idle_daemon_start() -> None:
    """Start the idle capture daemon in the background."""
    import subprocess as _subprocess

    from memo.daemon_common import is_pid_alive as _is_pid_alive
    from memo.daemon_common import read_pid as _read_pid

    cfg = Config.from_env()
    pid_file = cfg.state_dir / "idle-daemon.pid"
    pid = _read_pid(pid_file) if pid_file.exists() else None
    if pid is not None and _is_pid_alive(pid):
        console.print(f"[dim]idle daemon already running (pid={pid})[/dim]")
        return

    log_dir = Path.home() / "Library" / "Logs" / "memo"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "idle-daemon.log"

    env = os.environ.copy()
    env["MEMO_NONINTERACTIVE"] = "1"

    with open(log_file, "a") as log_fh:
        proc = _subprocess.Popen(
            [sys.executable, "-m", "memo.cli", "idle-daemon", "_serve"],
            stdout=log_fh,
            stderr=_subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )

    pid_file.write_text(str(proc.pid))
    click.echo(f"idle daemon started (pid={proc.pid})", err=True)


@idle_daemon_group.command(name="stop")
def idle_daemon_stop() -> None:
    """Stop the idle capture daemon."""
    import signal as _signal

    from memo.daemon_common import is_pid_alive as _is_pid_alive
    from memo.daemon_common import read_pid as _read_pid

    cfg = Config.from_env()
    pid_file = cfg.state_dir / "idle-daemon.pid"
    pid = _read_pid(pid_file) if pid_file.exists() else None

    if not pid:
        console.print("[dim]idle daemon not running (no PID file)[/dim]")
        return

    if not _is_pid_alive(pid):
        console.print("[dim]idle daemon not running (stale PID file)[/dim]")
        pid_file.unlink(missing_ok=True)
        return

    try:
        os.kill(pid, _signal.SIGTERM)
        console.print(f"idle daemon stopped (pid={pid})")
    except ProcessLookupError:
        console.print("[dim]idle daemon already gone[/dim]")

    pid_file.unlink(missing_ok=True)


@idle_daemon_group.command(name="status")
def idle_daemon_status() -> None:
    """Print whether the idle capture daemon is running."""
    from memo.daemon_common import is_pid_alive as _is_pid_alive
    from memo.daemon_common import read_pid as _read_pid

    cfg = Config.from_env()
    pid_file = cfg.state_dir / "idle-daemon.pid"
    pid = _read_pid(pid_file) if pid_file.exists() else None

    if not pid:
        console.print("[dim]idle daemon: not running[/dim]")
        return

    if _is_pid_alive(pid):
        console.print(f"[dim]idle daemon: running (pid={pid})[/dim]")
    else:
        console.print(f"[dim]idle daemon: not running (stale pid={pid})[/dim]")


@idle_daemon_group.command(name="_serve", hidden=True)
def idle_daemon_serve() -> None:
    """Internal: run the daemon in the foreground (called by 'start')."""
    from memo.server_idle_capture import run_idle_capture_loop

    run_idle_capture_loop()