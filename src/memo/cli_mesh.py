"""CLI surface for authenticated Memo mesh communication."""

from __future__ import annotations

import json
from dataclasses import asdict
from functools import wraps
from pathlib import Path
from typing import Any

import click

from memo.config import Config
from memo.errors import MemoError
from memo.operational_mesh import OperationalMesh, mesh_identity


def _memory_from_env() -> Any:
    from memo.memory import Memory

    return Memory(Config.from_env())


def _json(value: Any) -> None:
    click.echo(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _with_mesh(fn: Any) -> Any:
    @click.option(
        "--transport",
        "transport_path",
        required=True,
        type=click.Path(path_type=Path, file_okay=False),
        help="Explicit isolated local Git transport clone.",
    )
    @click.option(
        "--remote",
        default=None,
        help="Optional Git remote used as the cross-device rendezvous.",
    )
    @click.option(
        "--actor-id",
        required=True,
        help="Authenticated local actor id (':' is reserved for terminal principals).",
    )
    @click.option(
        "--session-id",
        default="",
        help="Terminal/session id; forms device:session.",
    )
    @wraps(fn)
    def wrapper(
        *args: Any,
        transport_path: Path,
        remote: str | None,
        actor_id: str,
        session_id: str,
        **kwargs: Any,
    ) -> Any:
        memory = _memory_from_env()
        try:
            identity = mesh_identity(
                device_id=memory.cfg.device_id,
                actor_id=actor_id,
                session_id=session_id,
                source_client="memo-cli",
            )
            mesh = OperationalMesh(
                memory.operational,
                identity=identity,
                transport_path=transport_path,
                remote=remote,
            )
            return fn(mesh, *args, **kwargs)
        except (MemoError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        finally:
            memory.close()

    return wrapper


@click.group(name="mesh")
def mesh_group() -> None:
    """Communicate between terminals through signed Memo operations."""


@mesh_group.command(name="status")
@_with_mesh
def mesh_status(mesh: OperationalMesh) -> None:
    """Show local/transport heads and current mesh state."""

    _json(asdict(mesh.status()))


@mesh_group.group(name="sync")
def mesh_sync_group() -> None:
    """Publish or ingest signed operational bundles."""


@mesh_sync_group.command(name="publish")
@_with_mesh
def mesh_sync_publish(mesh: OperationalMesh) -> None:
    """Publish this device's signed operations."""

    _json(asdict(mesh.publish()))


@mesh_sync_group.command(name="ingest")
@_with_mesh
def mesh_sync_ingest(mesh: OperationalMesh) -> None:
    """Ingest and verify operations from transport peers."""

    _json(asdict(mesh.ingest()))


@mesh_group.group(name="message")
def mesh_message_group() -> None:
    """Send and inspect Memo-native terminal messages."""


@mesh_message_group.command(name="send")
@click.argument("channel")
@click.argument("body")
@click.option(
    "--to",
    "target_ids",
    multiple=True,
    required=True,
    help="Recipient terminal principal in device:session form.",
)
@click.option("--topic", default="")
@click.option("--expects-ack", is_flag=True)
@click.option("--expires-at", default=None)
@click.option("--idempotency-key", required=True)
@_with_mesh
def mesh_message_send(
    mesh: OperationalMesh,
    channel: str,
    body: str,
    target_ids: tuple[str, ...],
    topic: str,
    expects_ack: bool,
    expires_at: str | None,
    idempotency_key: str,
) -> None:
    """Send one directed, idempotent message to a terminal principal."""

    _json(
        asdict(
            mesh.send_message(
                channel=channel,
                body=body,
                target_ids=target_ids,
                topic=topic,
                expects_ack=expects_ack,
                expires_at=expires_at,
                idempotency_key=idempotency_key,
            )
        )
    )


@mesh_message_group.command(name="list")
@click.option("--channel", default=None)
@_with_mesh
def mesh_message_list(mesh: OperationalMesh, channel: str | None) -> None:
    """List locally verified messages, optionally by channel."""

    rows = [asdict(item) for item in mesh.messages(channel=channel)]
    _json({"messages": rows, "count": len(rows)})


@mesh_group.group(name="delivery")
def mesh_delivery_group() -> None:
    """Reserve due delivery work and record recipient ACKs."""


@mesh_delivery_group.command(name="reserve")
@click.option("--limit", default=100, type=click.IntRange(1, 1000))
@_with_mesh
def mesh_delivery_reserve(mesh: OperationalMesh, limit: int) -> None:
    """Reserve due deliveries addressed to this principal."""

    rows = [asdict(item) for item in mesh.reserve_due(limit=limit)]
    _json({"deliveries": rows, "count": len(rows)})


@mesh_delivery_group.command(name="ack")
@click.argument("message_id")
@click.option("--idempotency-key", required=True)
@_with_mesh
def mesh_delivery_ack(
    mesh: OperationalMesh,
    message_id: str,
    idempotency_key: str,
) -> None:
    """Acknowledge a message addressed to this principal."""

    _json(asdict(mesh.acknowledge(message_id=message_id, idempotency_key=idempotency_key)))


@mesh_group.group(name="presence")
def mesh_presence_group() -> None:
    """Publish and inspect terminal presence leases."""


@mesh_presence_group.command(name="announce")
@click.argument("project")
@click.argument("workspace")
@click.argument("topic")
@click.argument("intent")
@click.option("--file", "files", multiple=True)
@click.option("--ttl", "ttl_seconds", default=60, type=click.IntRange(5, 3600))
@click.option("--idempotency-key", required=True)
@_with_mesh
def mesh_presence_announce(
    mesh: OperationalMesh,
    project: str,
    workspace: str,
    topic: str,
    intent: str,
    files: tuple[str, ...],
    ttl_seconds: int,
    idempotency_key: str,
) -> None:
    """Announce this terminal's signed presence lease."""

    _json(
        asdict(
            mesh.announce_presence(
                project=project,
                workspace=workspace,
                topic=topic,
                intent=intent,
                files=files,
                ttl_seconds=ttl_seconds,
                idempotency_key=idempotency_key,
            )
        )
    )


@mesh_presence_group.command(name="list")
@click.option("--project", default=None)
@_with_mesh
def mesh_presence_list(mesh: OperationalMesh, project: str | None) -> None:
    """List active, verified terminal presence leases."""

    rows = [asdict(item) for item in mesh.active_presence(project=project)]
    _json({"presence": rows, "count": len(rows)})


__all__ = ["mesh_group"]
