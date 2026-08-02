"""`memo terminal` — immediate coordination with registered local agent TTYs."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
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
    """Chat with or submit input to explicitly registered local agent terminals."""


@terminal_group.command(name="register")
@click.option("--agent", required=True, help="Agent name (for example: codex).")
@click.option("--tty", required=True, type=click.Path(path_type=Path), help="Exact /dev TTY.")
@click.option("--pid", required=True, type=click.IntRange(min=2), help="Agent process PID.")
@click.option("--terminal-app", default="", help="Terminal, iTerm2, or Ghostty.")
@click.option("--project", default="", help="Project directory associated with this agent.")
@click.option("--id-only", is_flag=True, hidden=True)
@click.option("--json", "as_json", is_flag=True, help="Output JSON.")
def terminal_register(
    agent: str,
    tty: Path,
    pid: int,
    terminal_app: str,
    project: str,
    id_only: bool,
    as_json: bool,
) -> None:
    """Register one local agent process and its exact terminal."""
    try:
        registration = _bridge().register(
            agent=agent,
            tty=tty,
            pid=pid,
            terminal_app=terminal_app,
            project=project or os.getcwd(),
        )
    except MemoError as exc:
        raise _domain_error(exc) from exc
    if id_only:
        click.echo(registration.id)
    elif as_json:
        _json(asdict(registration))
    else:
        click.echo(f"registered {registration.id} ({registration.agent} {registration.tty})")


@terminal_group.command(name="list")
@click.option("--json", "as_json", is_flag=True, help="Output JSON.")
def terminal_list(as_json: bool) -> None:
    """List live registered agent terminals and prune stale entries."""
    try:
        registrations = _bridge().list()
    except MemoError as exc:
        raise _domain_error(exc) from exc
    if as_json:
        _json([asdict(item) for item in registrations])
        return
    if not registrations:
        click.echo("No live registered terminals.")
        return
    for item in registrations:
        click.echo(f"{item.id}\t{item.agent}\t{item.tty}\t{item.project}")


@terminal_group.command(name="send")
@click.option("--to", "target_id", required=True, help="Exact registered terminal id.")
@click.option("--message", required=True, help="Prompt text to deliver.")
@click.option("--sender", default="", help="Reply-to terminal id.")
@click.option("--message-id", default="", help="Optional idempotency key.")
@click.option("--submit/--no-submit", default=True, help="Press Return after delivery.")
@click.option("--json", "as_json", is_flag=True, help="Output JSON.")
def terminal_send(
    target_id: str,
    message: str,
    sender: str,
    message_id: str,
    submit: bool,
    as_json: bool,
) -> None:
    """Immediately type a prompt into one registered agent terminal."""
    try:
        receipt = _bridge().send(
            target_id,
            message,
            sender=sender or None,
            message_id=message_id or None,
            submit=submit,
        )
    except MemoError as exc:
        raise _domain_error(exc) from exc
    if as_json:
        _json(asdict(receipt))
    else:
        click.echo(f"{receipt.status} {receipt.receipt_id} via {receipt.transport}")


@terminal_group.command(name="enter")
@click.option("--to", "target_id", required=True, help="Exact registered terminal id.")
@click.option("--sender", default="", help="Origin terminal id.")
@click.option("--message-id", default="", help="Optional idempotency key.")
@click.option("--json", "as_json", is_flag=True, help="Output JSON.")
def terminal_enter(target_id: str, sender: str, message_id: str, as_json: bool) -> None:
    """Press Return in one registered foreground agent terminal."""
    try:
        receipt = _bridge().enter(
            target_id,
            sender=sender or None,
            message_id=message_id or None,
        )
    except MemoError as exc:
        raise _domain_error(exc) from exc
    if as_json:
        _json(asdict(receipt))
    else:
        click.echo(f"{receipt.status} {receipt.receipt_id} via {receipt.transport}")


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
