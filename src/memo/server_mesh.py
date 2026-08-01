"""MCP tools for authenticated Memo mesh communication."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field

from memo.operational_mesh import OperationalMesh, mesh_identity
from memo.server_annotations import NETWORK_WRITE, annotated_tool


def _mesh(
    memory: Any,
    *,
    transport_path: str,
    remote: str | None,
    actor_id: str,
    session_id: str,
) -> OperationalMesh:
    identity = mesh_identity(
        device_id=memory.cfg.device_id,
        actor_id=actor_id,
        session_id=session_id,
        source_client="memo-mcp",
    )
    return OperationalMesh(
        memory.operational,
        identity=identity,
        transport_path=Path(transport_path),
        remote=remote,
    )


def register(server: Any, memory: Any) -> None:
    @annotated_tool(server, **NETWORK_WRITE)
    def memo_mesh_status(
        transport_path: Annotated[
            str,
            Field(description="Explicit isolated local Git transport clone."),
        ],
        actor_id: Annotated[str, Field(description="Authenticated local actor id.")],
        session_id: Annotated[str, Field(description="Terminal/session id.")] = "",
        remote: Annotated[
            str | None,
            Field(description="Optional Git remote used as the cross-device rendezvous."),
        ] = None,
    ) -> dict[str, Any]:
        """Return signed ledger heads and local terminal mesh state."""

        return asdict(
            _mesh(
                memory,
                transport_path=transport_path,
                remote=remote,
                actor_id=actor_id,
                session_id=session_id,
            ).status()
        )

    @annotated_tool(server, **NETWORK_WRITE)
    def memo_mesh_sync_publish(
        transport_path: Annotated[str, Field(description="Git transport directory.")],
        actor_id: Annotated[str, Field(description="Authenticated local actor id.")],
        session_id: Annotated[str, Field(description="Terminal/session id.")] = "",
        remote: Annotated[str | None, Field(description="Optional remote URL.")] = None,
    ) -> dict[str, Any]:
        """Publish this device's immutable signed operational bundle."""

        return asdict(
            _mesh(
                memory,
                transport_path=transport_path,
                remote=remote,
                actor_id=actor_id,
                session_id=session_id,
            ).publish()
        )

    @annotated_tool(server, **NETWORK_WRITE)
    def memo_mesh_sync_ingest(
        transport_path: Annotated[str, Field(description="Git transport directory.")],
        actor_id: Annotated[str, Field(description="Authenticated local actor id.")],
        session_id: Annotated[str, Field(description="Terminal/session id.")] = "",
        remote: Annotated[str | None, Field(description="Optional remote URL.")] = None,
    ) -> dict[str, Any]:
        """Verify and ingest peer bundles into this local Memo state."""

        return asdict(
            _mesh(
                memory,
                transport_path=transport_path,
                remote=remote,
                actor_id=actor_id,
                session_id=session_id,
            ).ingest()
        )

    @annotated_tool(server, **NETWORK_WRITE)
    def memo_mesh_message_send(
        transport_path: Annotated[str, Field(description="Git transport directory.")],
        actor_id: Annotated[str, Field(description="Authenticated local actor id.")],
        channel: Annotated[str, Field(description="Message channel.")],
        body: Annotated[str, Field(description="Message body.")],
        target_ids: Annotated[
            list[str],
            Field(description="Recipient terminal principals in device:session form."),
        ],
        idempotency_key: Annotated[str, Field(description="Stable request key.")],
        topic: Annotated[str, Field(description="Optional topic label.")] = "",
        expects_ack: Annotated[bool, Field(description="Require recipient ACK.")] = False,
        expires_at: Annotated[
            str | None,
            Field(description="Optional timezone-aware ISO expiry."),
        ] = None,
        session_id: Annotated[str, Field(description="Terminal/session id.")] = "",
        remote: Annotated[str | None, Field(description="Optional remote URL.")] = None,
    ) -> dict[str, Any]:
        """Send one directed message through Memo's signed v2 ledger."""

        mesh = _mesh(
            memory,
            transport_path=transport_path,
            remote=remote,
            actor_id=actor_id,
            session_id=session_id,
        )
        return asdict(
            mesh.send_message(
                channel=channel,
                body=body,
                target_ids=tuple(target_ids),
                topic=topic,
                expects_ack=expects_ack,
                expires_at=expires_at,
                idempotency_key=idempotency_key,
            )
        )

    @annotated_tool(server, **NETWORK_WRITE)
    def memo_mesh_message_list(
        transport_path: Annotated[str, Field(description="Git transport directory.")],
        actor_id: Annotated[str, Field(description="Authenticated local actor id.")],
        channel: Annotated[str | None, Field(description="Optional channel filter.")] = None,
        session_id: Annotated[str, Field(description="Terminal/session id.")] = "",
        remote: Annotated[str | None, Field(description="Optional remote URL.")] = None,
    ) -> dict[str, Any]:
        """List locally verified messages from all ingested terminals."""

        rows = [
            asdict(item)
            for item in _mesh(
                memory,
                transport_path=transport_path,
                remote=remote,
                actor_id=actor_id,
                session_id=session_id,
            ).messages(channel=channel)
        ]
        return {"messages": rows, "count": len(rows)}

    @annotated_tool(server, **NETWORK_WRITE)
    def memo_mesh_delivery_reserve(
        transport_path: Annotated[str, Field(description="Git transport directory.")],
        actor_id: Annotated[str, Field(description="Authenticated recipient actor id.")],
        limit: Annotated[int, Field(description="Maximum due deliveries to reserve.")] = 100,
        session_id: Annotated[str, Field(description="Terminal/session id.")] = "",
        remote: Annotated[str | None, Field(description="Optional remote URL.")] = None,
    ) -> dict[str, Any]:
        """Reserve due deliveries addressed to this authenticated principal."""

        rows = [
            asdict(item)
            for item in _mesh(
                memory,
                transport_path=transport_path,
                remote=remote,
                actor_id=actor_id,
                session_id=session_id,
            ).reserve_due(limit=limit)
        ]
        return {"deliveries": rows, "count": len(rows)}

    @annotated_tool(server, **NETWORK_WRITE)
    def memo_mesh_delivery_ack(
        transport_path: Annotated[str, Field(description="Git transport directory.")],
        actor_id: Annotated[str, Field(description="Authenticated recipient actor id.")],
        message_id: Annotated[str, Field(description="Message id to acknowledge.")],
        idempotency_key: Annotated[str, Field(description="Stable ACK request key.")],
        session_id: Annotated[str, Field(description="Terminal/session id.")] = "",
        remote: Annotated[str | None, Field(description="Optional remote URL.")] = None,
    ) -> dict[str, Any]:
        """Record a signed recipient ACK for one addressed message."""

        return asdict(
            _mesh(
                memory,
                transport_path=transport_path,
                remote=remote,
                actor_id=actor_id,
                session_id=session_id,
            ).acknowledge(
                message_id=message_id,
                idempotency_key=idempotency_key,
            )
        )

    @annotated_tool(server, **NETWORK_WRITE)
    def memo_mesh_presence_announce(
        transport_path: Annotated[str, Field(description="Git transport directory.")],
        actor_id: Annotated[str, Field(description="Authenticated local actor id.")],
        project: Annotated[str, Field(description="Project name.")],
        workspace: Annotated[str, Field(description="Workspace path.")],
        topic: Annotated[str, Field(description="Current topic.")],
        intent: Annotated[str, Field(description="Current intent.")],
        idempotency_key: Annotated[str, Field(description="Stable lease request key.")],
        files: Annotated[list[str] | None, Field(description="Workspace-relative files.")] = None,
        ttl_seconds: Annotated[int, Field(description="Lease TTL, 5-3600 seconds.")] = 60,
        session_id: Annotated[str, Field(description="Terminal/session id.")] = "",
        remote: Annotated[str | None, Field(description="Optional remote URL.")] = None,
    ) -> dict[str, Any]:
        """Announce a signed presence lease for this terminal."""

        return asdict(
            _mesh(
                memory,
                transport_path=transport_path,
                remote=remote,
                actor_id=actor_id,
                session_id=session_id,
            ).announce_presence(
                project=project,
                workspace=workspace,
                topic=topic,
                intent=intent,
                files=tuple(files or ()),
                ttl_seconds=ttl_seconds,
                idempotency_key=idempotency_key,
            )
        )

    @annotated_tool(server, **NETWORK_WRITE)
    def memo_mesh_presence_list(
        transport_path: Annotated[str, Field(description="Git transport directory.")],
        actor_id: Annotated[str, Field(description="Authenticated local actor id.")],
        project: Annotated[str | None, Field(description="Optional project filter.")] = None,
        session_id: Annotated[str, Field(description="Terminal/session id.")] = "",
        remote: Annotated[str | None, Field(description="Optional remote URL.")] = None,
    ) -> dict[str, Any]:
        """List active presence leases from every ingested terminal."""

        rows = [
            asdict(item)
            for item in _mesh(
                memory,
                transport_path=transport_path,
                remote=remote,
                actor_id=actor_id,
                session_id=session_id,
            ).active_presence(project=project)
        ]
        return {"presence": rows, "count": len(rows)}


__all__ = ["register"]
