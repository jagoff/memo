"""Memo-native presence leases, heartbeat renewal, and workspace conflicts."""

from __future__ import annotations

import hashlib
import posixpath
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from memo.contracts import Visibility
from memo.errors import OperationalError, OperationalErrorCode
from memo.identity import PrincipalIdentity
from memo.operational import OperationalStore
from memo.operational_epoch import CommitContext
from memo.operational_event import OperationalCommand
from memo.operational_event_types import (
    PRESENCE_ANNOUNCED,
    PRESENCE_LEASE_EXPIRED,
    PRESENCE_RENEWED,
)

ContextFactory = Callable[[PrincipalIdentity], CommitContext]


def _invalid(message: str) -> OperationalError:
    return OperationalError(
        OperationalErrorCode.INVALID_EVENT,
        message,
        retryable=False,
    )


def _canonical_time(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("presence timestamps must include a timezone")
    return value.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _ttl(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("ttl_seconds must be an integer")
    return max(5, min(value, 3600))


def _normalize_workspace(value: str) -> str:
    normalized = value.strip().replace("\\", "/")
    if not normalized or "\x00" in normalized:
        raise ValueError("workspace must be non-empty and safe")
    return posixpath.normpath(normalized)


def _normalize_file(value: str) -> str:
    normalized = posixpath.normpath(value.strip().replace("\\", "/"))
    if (
        not normalized
        or normalized == "."
        or normalized.startswith("/")
        or normalized == ".."
        or normalized.startswith("../")
        or "\x00" in normalized
    ):
        raise ValueError(f"presence file must be workspace-relative: {value!r}")
    return normalized


@dataclass(frozen=True)
class PresenceLease:
    id: str
    actor_id: str
    device_id: str
    project: str
    workspace: str
    topic: str
    intent: str
    files: tuple[str, ...]
    ttl_seconds: int
    expires_at: str

    def heartbeat_interval(self, configured_heartbeat_seconds: float = 15.0) -> float:
        if configured_heartbeat_seconds <= 0:
            raise ValueError("configured heartbeat must be positive")
        return min(configured_heartbeat_seconds, max(1.0, self.ttl_seconds / 3))


@dataclass(frozen=True)
class WorkspaceConflict:
    project: str
    file: str
    lease_ids: tuple[str, ...]


class PresenceService:
    def __init__(
        self,
        store: OperationalStore,
        *,
        context_factory: ContextFactory,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if store.backend_version != 2:
            raise _invalid("native presence requires operational ledger v2")
        self.store = store
        self.context_factory = context_factory
        self.clock = clock or (lambda: datetime.now(UTC))

    def _commit(
        self,
        identity: PrincipalIdentity,
        *,
        event_type: str,
        target_id: str,
        project: str,
        workspace: str,
        payload: Mapping[str, object],
        idempotency_key: str,
    ) -> None:
        key = idempotency_key.strip()
        if not key:
            raise ValueError("idempotency_key must be non-empty")
        self.store.commit(
            OperationalCommand(
                event_type=event_type,
                actor=identity,
                target_id=target_id,
                project=project,
                workspace=workspace,
                expires_at=str(payload["expires_at"]) if "expires_at" in payload else None,
                visibility=Visibility.SHARED.value,
                idempotency_key=key,
                caused_by=(),
                subject_uri=f"memo://presence/{target_id}",
                trace_id="",
                payload=payload,
            ),
            context=self.context_factory(identity),
        )

    def announce(
        self,
        *,
        identity: PrincipalIdentity,
        project: str,
        workspace: str,
        topic: str,
        intent: str,
        files: tuple[str, ...] = (),
        ttl_seconds: int = 60,
        idempotency_key: str,
    ) -> PresenceLease:
        normalized_workspace = _normalize_workspace(workspace)
        normalized_files = tuple(sorted(set(_normalize_file(item) for item in files)))
        ttl = _ttl(ttl_seconds)
        lease_id = hashlib.sha256(
            (
                f"{identity.principal_id}\0{project}\0{normalized_workspace}\0"
                f"{idempotency_key.strip()}"
            ).encode()
        ).hexdigest()
        existing = self._all().get(lease_id)
        if existing is not None:
            expected = (
                identity.actor_id,
                identity.device_id,
                project,
                normalized_workspace,
                topic,
                intent,
                normalized_files,
                ttl,
            )
            actual = (
                existing.actor_id,
                existing.device_id,
                existing.project,
                existing.workspace,
                existing.topic,
                existing.intent,
                existing.files,
                existing.ttl_seconds,
            )
            if actual != expected:
                raise _invalid("presence idempotency key identifies a different request")
            return existing
        expires_at = _canonical_time(self.clock() + timedelta(seconds=ttl))
        payload: Mapping[str, object] = {
            "id": lease_id,
            "actor_id": identity.actor_id,
            "device_id": identity.device_id,
            "project": project,
            "workspace": normalized_workspace,
            "topic": topic,
            "intent": intent,
            "files": normalized_files,
            "ttl_seconds": ttl,
            "expires_at": expires_at,
        }
        self._commit(
            identity,
            event_type=PRESENCE_ANNOUNCED,
            target_id=lease_id,
            project=project,
            workspace=normalized_workspace,
            payload=payload,
            idempotency_key=idempotency_key,
        )
        return self._all()[lease_id]

    def renew(
        self,
        *,
        identity: PrincipalIdentity,
        lease_id: str,
        ttl_seconds: int | None = None,
        idempotency_key: str,
    ) -> PresenceLease:
        current = self._all().get(lease_id)
        if current is None:
            raise _invalid(f"unknown presence lease: {lease_id}")
        if (current.actor_id, current.device_id) != (
            identity.actor_id,
            identity.device_id,
        ):
            raise _invalid("presence lease owner differs from authenticated actor")
        ttl = _ttl(ttl_seconds if ttl_seconds is not None else current.ttl_seconds)
        expires_at = _canonical_time(self.clock() + timedelta(seconds=ttl))
        payload: Mapping[str, object] = {
            "id": current.id,
            "actor_id": current.actor_id,
            "device_id": current.device_id,
            "project": current.project,
            "workspace": current.workspace,
            "topic": current.topic,
            "intent": current.intent,
            "files": current.files,
            "ttl_seconds": ttl,
            "expires_at": expires_at,
        }
        self._commit(
            identity,
            event_type=PRESENCE_RENEWED,
            target_id=lease_id,
            project=current.project,
            workspace=current.workspace,
            payload=payload,
            idempotency_key=idempotency_key,
        )
        return self._all()[lease_id]

    def expire(
        self,
        *,
        identity: PrincipalIdentity,
        lease_id: str,
        idempotency_key: str,
    ) -> None:
        current = self._all().get(lease_id)
        if current is None:
            return
        if (current.actor_id, current.device_id) != (
            identity.actor_id,
            identity.device_id,
        ):
            raise _invalid("presence lease owner differs from authenticated actor")
        self._commit(
            identity,
            event_type=PRESENCE_LEASE_EXPIRED,
            target_id=lease_id,
            project=current.project,
            workspace=current.workspace,
            payload={"id": lease_id, "expired_at": _canonical_time(self.clock())},
            idempotency_key=idempotency_key,
        )

    def _all(self) -> dict[str, PresenceLease]:
        rows: dict[str, PresenceLease] = {}
        for event in self.store.ledger.validated_events():
            payload = event.payload
            if event.event_type == PRESENCE_ANNOUNCED:
                key = str(payload["id"])
                if (
                    event.target_id != key
                    or str(payload["actor_id"]) != event.actor.actor_id
                    or str(payload["device_id"]) != event.actor.device_id
                ):
                    continue
                rows[key] = PresenceLease(
                    id=key,
                    actor_id=str(payload["actor_id"]),
                    device_id=str(payload["device_id"]),
                    project=str(payload["project"]),
                    workspace=str(payload["workspace"]),
                    topic=str(payload["topic"]),
                    intent=str(payload["intent"]),
                    files=tuple(str(item) for item in payload["files"]),
                    ttl_seconds=int(payload["ttl_seconds"]),
                    expires_at=str(payload["expires_at"]),
                )
            elif event.event_type == PRESENCE_RENEWED:
                key = str(payload["id"])
                current = rows.get(key)
                if (
                    current is None
                    or event.target_id != key
                    or (current.actor_id, current.device_id)
                    != (event.actor.actor_id, event.actor.device_id)
                    or str(payload["actor_id"]) != current.actor_id
                    or str(payload["device_id"]) != current.device_id
                ):
                    continue
                rows[key] = PresenceLease(
                    id=key,
                    actor_id=current.actor_id,
                    device_id=current.device_id,
                    project=str(payload["project"]),
                    workspace=str(payload["workspace"]),
                    topic=str(payload["topic"]),
                    intent=str(payload["intent"]),
                    files=tuple(str(item) for item in payload["files"]),
                    ttl_seconds=int(payload["ttl_seconds"]),
                    expires_at=str(payload["expires_at"]),
                )
            elif event.event_type == PRESENCE_LEASE_EXPIRED:
                key = str(payload["id"])
                current = rows.get(key)
                if (
                    current is not None
                    and event.target_id == key
                    and (current.actor_id, current.device_id)
                    == (event.actor.actor_id, event.actor.device_id)
                ):
                    rows.pop(key)
        return rows

    def active(
        self,
        *,
        project: str | None = None,
        now: datetime | None = None,
    ) -> list[PresenceLease]:
        cutoff = (now or self.clock()).astimezone(UTC)
        rows = [
            row
            for row in self._all().values()
            if _parse_time(row.expires_at) > cutoff and (project is None or row.project == project)
        ]
        return sorted(rows, key=lambda row: (row.project, row.workspace, row.actor_id, row.id))

    def conflicts(
        self,
        *,
        project: str,
        files: tuple[str, ...] = (),
        now: datetime | None = None,
    ) -> list[WorkspaceConflict]:
        requested = set(_normalize_file(item) for item in files) if files else None
        by_file: dict[str, list[PresenceLease]] = {}
        for lease in self.active(project=project, now=now):
            for file in lease.files:
                if requested is not None and file not in requested:
                    continue
                by_file.setdefault(file, []).append(lease)
        conflicts: list[WorkspaceConflict] = []
        for file, leases in sorted(by_file.items()):
            principals = {(lease.actor_id, lease.device_id) for lease in leases}
            if len(principals) < 2:
                continue
            conflicts.append(
                WorkspaceConflict(
                    project=project,
                    file=file,
                    lease_ids=tuple(sorted(lease.id for lease in leases)),
                )
            )
        return conflicts


__all__ = ["PresenceLease", "PresenceService", "WorkspaceConflict"]
