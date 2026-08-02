"""`memo terminal` — diagnostics for disabled legacy terminal input."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from typing import Any

import click

from memo.config import Config
from memo.errors import MemoError
from memo.flags import flag_bool
from memo.terminal_live import TerminalBridge
from memo.terminal_receiver import (
    ReceiverClient,
    ReceiverSession,
    ReceiverSupervisor,
    read_capability_file,
)


def _bridge() -> TerminalBridge:
    return TerminalBridge(Config.from_env())


def _json(value: Any) -> None:
    click.echo(json.dumps(value, indent=2))


def _domain_error(exc: MemoError) -> click.ClickException:
    return click.ClickException(str(exc))


def _receiver_enabled() -> None:
    if not flag_bool("MEMO_TERMINAL_RECEIVER_ENABLED"):
        raise click.ClickException(
            "receiver-bound terminal transport is disabled; set "
            "MEMO_TERMINAL_RECEIVER_ENABLED=1 explicitly"
        )


def _write_capability(path: str, capability: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600)
    try:
        os.write(fd, capability.encode("ascii"))
    finally:
        os.close(fd)


@click.group(name="terminal")
def terminal_group() -> None:
    """Inspect terminal registrations and use the opt-in receiver transport."""


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


@terminal_group.group(name="receiver")
def receiver_group() -> None:
    """Use the experimental receiver-bound PTY transport."""


@receiver_group.command(name="attach")
@click.option(
    "--capability-file",
    type=click.Path(dir_okay=False, path_type=str),
    help="Write the session capability to a new mode-0600 file.",
)
@click.argument("command", nargs=-1, required=True)
def receiver_attach(capability_file: str | None, command: tuple[str, ...]) -> None:
    """Run COMMAND under a Memo-owned PTY receiver."""
    _receiver_enabled()
    try:
        session = ReceiverSession.fork(list(command))
        supervisor = ReceiverSupervisor(Config.from_env().state_dir, session)
        socket_path = supervisor.start()
    except (OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    try:
        metadata: dict[str, Any] = {
            "socket": str(socket_path),
            "pid": session.child_pid,
            "process_started_at": session.child_start,
        }
        if capability_file is not None:
            try:
                _write_capability(capability_file, supervisor.capability)
            except OSError as exc:
                raise click.ClickException(f"cannot write capability file: {exc}") from exc
            metadata["capability_file"] = capability_file
        else:
            metadata["capability"] = supervisor.capability
        _json(metadata)
        while session.alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        supervisor.close()


@receiver_group.command(name="send")
@click.option("--socket", "socket_path", required=True, type=click.Path(path_type=str))
@click.option("--capability-file", required=True, type=click.Path(dir_okay=False, path_type=str))
@click.option("--message-id", required=True)
@click.option("--message", required=True)
@click.option("--submit/--no-submit", default=True)
def receiver_send(
    socket_path: str,
    capability_file: str,
    message_id: str,
    message: str,
    submit: bool,
) -> None:
    """Send sanitized text through an authenticated receiver socket."""
    _receiver_enabled()
    try:
        response = ReceiverClient(socket_path, read_capability_file(capability_file)).send(
            message_id=message_id,
            text=message,
            submit=submit,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    _json(response)


@receiver_group.command(name="enter")
@click.option("--socket", "socket_path", required=True, type=click.Path(path_type=str))
@click.option("--capability-file", required=True, type=click.Path(dir_okay=False, path_type=str))
@click.option("--message-id", required=True)
def receiver_enter(socket_path: str, capability_file: str, message_id: str) -> None:
    """Press Return through an authenticated receiver socket."""
    _receiver_enabled()
    try:
        response = ReceiverClient(socket_path, read_capability_file(capability_file)).enter(
            message_id=message_id
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    _json(response)
