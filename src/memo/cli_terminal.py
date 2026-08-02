"""`memo terminal` — diagnostics for disabled legacy terminal input."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import click

from memo.config import Config
from memo.errors import MemoError
from memo.terminal_live import TerminalBridge


def _bridge() -> TerminalBridge:
    return TerminalBridge(Config.from_env())


def _json(value: Any) -> None:
    click.echo(json.dumps(value, indent=2))


def _domain_error(exc: MemoError) -> click.ClickException:
    return click.ClickException(str(exc))


@click.group(name="terminal")
def terminal_group() -> None:
    """Inspect disabled legacy terminal registrations and receipt history."""


@terminal_group.command(name="list")
@click.option("--json", "as_json", is_flag=True, help="Output JSON.")
def terminal_list(as_json: bool) -> None:
    """List deliverable terminals; empty while legacy TTY input is disabled."""
    try:
        registrations = _bridge().list()
    except MemoError as exc:
        raise _domain_error(exc) from exc
    if as_json:
        _json([asdict(item) for item in registrations])
        return
    if not registrations:
        click.echo("No deliverable registered terminals.")
        return
    for item in registrations:
        click.echo(f"{item.id}\t{item.agent}\t{item.tty}\t{item.project}")


@terminal_group.command(name="history")
@click.option("--limit", default=50, type=click.IntRange(min=1, max=500))
@click.option("--json", "as_json", is_flag=True, help="Output JSON.")
def terminal_history(limit: int, as_json: bool) -> None:
    """Show delivery receipts without retaining prompt bodies."""
    receipts = _bridge().history(limit=limit)
    if as_json:
        _json([asdict(item) for item in receipts])
        return
    for item in receipts:
        click.echo(f"{item.created_at}\t{item.status}\t{item.target_id}\t{item.receipt_id}")
