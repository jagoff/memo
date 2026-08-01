"""Product facade for Memo-native terminal-to-terminal coordination."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from memo.errors import OperationalError, OperationalErrorCode
from memo.git_transport import GitTransport
from memo.identity import PrincipalIdentity
from memo.operational import OperationalStore
from memo.operational_coordination import CoordinationService, MessageView
from memo.operational_delivery import DeliveryService, DeliveryView
from memo.operational_presence import PresenceLease, PresenceService
from memo.operational_sync import OperationalSync, OperationalSyncStatus, SyncResult

TransportFactory = Callable[[Path, str | None], GitTransport]


def _invalid(message: str) -> OperationalError:
    return OperationalError(
        OperationalErrorCode.INVALID_EVENT,
        message,
        retryable=False,
    )


def mesh_identity(
    *,
    device_id: str,
    actor_id: str,
    session_id: str = "",
    source_client: str,
) -> PrincipalIdentity:
    """Build an explicit mesh principal that cannot select another device."""

    device = device_id.strip()
    actor = actor_id.strip()
    source = source_client.strip()
    session = session_id.strip() or f"mesh-{actor}"
    if not device or not actor or not source:
        raise ValueError("device_id, actor_id, and source_client must be non-empty")
    if ":" in actor:
        raise ValueError("actor_id cannot contain ':'; terminal principals use device:session")
    return PrincipalIdentity(
        principal_id=f"{device}:{session}",
        actor_id=actor,
        kind="agent",
        device_id=device,
        session_id=session,
        source_client=source,
    )


def _local_transport(root: Path, remote: str | None) -> GitTransport:
    """Open one local clone, optionally bound to its remote rendezvous."""

    return GitTransport(root, remote=remote)


@dataclass(frozen=True)
class MeshStatus:
    backend_version: int
    device_id: str
    principal_id: str
    transport_path: str
    remote: str | None
    sync: OperationalSyncStatus
    messages: int
    deliveries: int
    active_presence: int


class OperationalMesh:
    """One authenticated Memo mesh endpoint bound to the local device."""

    def __init__(
        self,
        store: OperationalStore,
        *,
        identity: PrincipalIdentity,
        transport_path: Path,
        remote: str | None = None,
        transport_factory: TransportFactory = _local_transport,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if store.backend_version != 2:
            raise _invalid("Memo mesh requires operational ledger v2")
        if not isinstance(identity, PrincipalIdentity):
            raise _invalid("Memo mesh requires an explicit PrincipalIdentity")
        root = Path(transport_path).expanduser()
        if not str(root).strip():
            raise ValueError("transport_path must be non-empty")
        # Resolve authority during construction so even read-only mesh surfaces
        # fail closed when v2 was only partially activated.
        store.context_for(identity)
        self.store = store
        self.identity = identity
        self.transport_path = root.resolve()
        self.remote = remote.strip() if remote and remote.strip() else None
        self.clock = clock or (lambda: datetime.now(UTC))
        self.transport = transport_factory(self.transport_path, self.remote)
        self.coordination = CoordinationService(
            store,
            context_factory=store.context_for,
            clock=self.clock,
        )
        self.delivery = DeliveryService(
            store,
            context_factory=store.context_for,
            clock=self.clock,
        )
        self.presence = PresenceService(
            store,
            context_factory=store.context_for,
            clock=self.clock,
        )
        self.sync = OperationalSync(
            store,
            transport=self.transport,
            device_id=identity.device_id,
            context_factory=lambda: store.context_for(identity),
            clock=self.clock,
        )

    def status(self) -> MeshStatus:
        return MeshStatus(
            backend_version=self.store.backend_version,
            device_id=self.identity.device_id,
            principal_id=self.identity.principal_id,
            transport_path=str(self.transport_path),
            remote=self.remote,
            sync=self.sync.status(),
            messages=len(self.coordination.messages()),
            deliveries=len(self.delivery.deliveries()),
            active_presence=len(self.presence.active()),
        )

    def publish(self) -> SyncResult:
        return self.sync.publish()

    def ingest(self) -> SyncResult:
        return self.sync.ingest()

    def send_message(
        self,
        *,
        channel: str,
        body: str,
        target_ids: tuple[str, ...],
        topic: str = "",
        expects_ack: bool = False,
        expires_at: str | None = None,
        idempotency_key: str,
    ) -> MessageView:
        targets = tuple(item.strip() for item in target_ids if item.strip())
        if not targets:
            raise ValueError("target_ids must contain at least one recipient")
        invalid = [item for item in targets if not all(item.partition(":")[::2])]
        if invalid:
            raise ValueError(
                "Memo mesh recipients must be terminal principals in device:session form"
            )
        return self.coordination.send_message(
            identity=self.identity,
            channel=channel,
            body=body,
            target_ids=targets,
            topic=topic,
            expects_ack=expects_ack,
            expires_at=expires_at,
            idempotency_key=idempotency_key,
        )

    def messages(self, *, channel: str | None = None) -> list[MessageView]:
        return self.coordination.messages(channel=channel)

    def reserve_due(self, *, limit: int = 100) -> list[DeliveryView]:
        return self.delivery.reserve_due(
            identity=self.identity,
            now=self.clock(),
            limit=limit,
        )

    def acknowledge(self, *, message_id: str, idempotency_key: str) -> DeliveryView:
        return self.delivery.acknowledge(
            identity=self.identity,
            message_id=message_id,
            idempotency_key=idempotency_key,
        )

    def announce_presence(
        self,
        *,
        project: str,
        workspace: str,
        topic: str,
        intent: str,
        files: tuple[str, ...] = (),
        ttl_seconds: int = 60,
        idempotency_key: str,
    ) -> PresenceLease:
        return self.presence.announce(
            identity=self.identity,
            project=project,
            workspace=workspace,
            topic=topic,
            intent=intent,
            files=files,
            ttl_seconds=ttl_seconds,
            idempotency_key=idempotency_key,
        )

    def active_presence(self, *, project: str | None = None) -> list[PresenceLease]:
        return self.presence.active(project=project)


__all__ = [
    "MeshStatus",
    "OperationalMesh",
    "TransportFactory",
    "mesh_identity",
]
