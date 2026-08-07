"""`memo daemons` — unified supervisor across every background daemon.

Each daemon (recall/ingest/maint/idle/embed) keeps its own lifecycle module
(`cli_recall_daemon.py`, …). This group is a thin *management* surface that
calls them all in one shot, so a user can `memo daemons status` / `start` /
`stop` everything without five separate invocations — while per-daemon CLI
groups stay available for targeted control.

The heavy logic lives in the per-daemon modules, not here.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

import click

from memo.cli_common import console

# (display_name, module, start_fn, stop_fn, status_fn)
_DAEMONS: tuple[tuple[str, str, str, str, str], ...] = (
    (
        "recall",
        "memo.cli_recall_daemon",
        "recall_daemon_start",
        "recall_daemon_stop",
        "recall_daemon_status",
    ),
    (
        "ingest",
        "memo.cli_ingest_daemon",
        "ingest_daemon_start",
        "ingest_daemon_stop",
        "ingest_daemon_status",
    ),
    (
        "maint",
        "memo.cli_maint_daemon",
        "maint_daemon_start",
        "maint_daemon_stop",
        "maint_daemon_status",
    ),
    ("idle", "memo.cli_idle_daemon", "idle_daemon_start", "idle_daemon_stop", "idle_daemon_status"),
)


def _each(kind: str, *, only: tuple[str, ...] | None = None) -> None:
    """Run the `kind` lifecycle op (start/stop/status) on every daemon."""
    for name, module, start, stop, status in _DAEMONS:
        if only and name not in only:
            continue
        fn_name = {"start": start, "stop": stop, "status": status}[kind]
        try:
            mod = importlib.import_module(module)
            fn: Callable[[], None] = getattr(mod, fn_name)
            console.print(f"[bold]{name}:[/bold]")
            fn()
        except Exception as exc:  # pragma: no cover - dependent failure paths
            console.print(f"[yellow]{name}: {type(exc).__name__}: {exc}[/yellow]")


@click.group("daemons")
def daemons_group() -> None:
    """Manage all background daemons in one place."""


@daemons_group.command("status")
def daemons_status() -> None:
    """Show running state of every daemon."""
    _each("status")


@daemons_group.command("start")
@click.option("--only", "only", multiple=True, help="Start only this daemon (repeatable).")
def daemons_start(only: tuple[str, ...]) -> None:
    """Start every background daemon (or only the named ones)."""
    _each("start", only=tuple(only) or None)


@daemons_group.command("stop")
@click.option("--only", "only", multiple=True, help="Stop only this daemon (repeatable).")
def daemons_stop(only: tuple[str, ...]) -> None:
    """Stop every background daemon (or only the named ones)."""
    _each("stop", only=tuple(only) or None)
