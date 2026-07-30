"""Canonical portable sessions over the operational v2 ledger."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from memo.atomic_io import authority_write_lock
from memo.errors import (
    OperationalError,
    OperationalErrorCode,
    StorageError,
    ValidationError,
)
from memo.identity import PrincipalIdentity
from memo.operational_event import OperationalCommand, canonical_json_bytes
from memo.operational_event_types import (
    SESSION_CHECKPOINTED,
    SESSION_RECOVERABLE,
    SESSION_TERMINATED,
)
from memo.util import utc_now_iso

if TYPE_CHECKING:
    from memo.operation_views import OperationalViewStore
    from memo.operational import OperationalStore
    from memo.operational_epoch import CommitContext


_SESSION_STATUSES = frozenset({"active", "recoverable", "terminated"})
_ROW_FIELDS = frozenset(
    {
        "session_id",
        "principal_id",
        "project",
        "workspace",
        "status",
        "branch",
        "head",
        "summary",
        "checkpointed_at",
        "source_event_id",
        "recoverable_at",
        "terminated_at",
        "recoverable_reason",
        "updated_event_id",
    }
)
_PORTABLE_LEGACY_KEYS = frozenset(
    {
        "session_id",
        "id",
        "principal_id",
        "project",
        "cwd",
        "workspace",
        "directory",
        "status",
        "branch",
        "head",
        "head_commit",
        "summary",
        "checkpointed_at",
        "updated",
        "created",
        "started_at",
        "ended_at",
        "source_event_id",
        "recoverable_at",
        "terminated_at",
        "recoverable_reason",
        "updated_event_id",
    }
)


def _invalid(message: str) -> OperationalError:
    return OperationalError(
        OperationalErrorCode.INVALID_EVENT,
        message,
        retryable=False,
    )


def _required(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_string(value: object, *, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValidationError(f"{field} must be a string")
    return value


def _canonical_timestamp(
    value: str,
    *,
    field: str,
    allow_empty: bool = False,
) -> str:
    if allow_empty and not value:
        return ""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValidationError(f"{field} must be a timezone-aware ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValidationError(f"{field} must be a timezone-aware ISO-8601 timestamp")
    return parsed.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _idempotency_key(value: str) -> str:
    return _required(value, field="idempotency_key")


@dataclass(frozen=True)
class OperationalSession:
    session_id: str
    principal_id: str
    project: str
    workspace: str
    status: str
    branch: str
    head: str
    summary: str
    checkpointed_at: str
    source_event_id: str
    recoverable_at: str
    terminated_at: str
    recoverable_reason: str
    updated_event_id: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MergedLegacySession:
    checkpoint: OperationalSession
    local_artifacts: Mapping[str, object]


def operational_session_from_row(row: Mapping[str, object]) -> OperationalSession:
    if not isinstance(row, Mapping):
        raise ValidationError("operational session row must be a mapping")
    fields = set(row)
    if fields != _ROW_FIELDS:
        missing = sorted(_ROW_FIELDS.difference(fields))
        unknown = sorted(fields.difference(_ROW_FIELDS))
        details: list[str] = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if unknown:
            details.append(f"unknown {', '.join(unknown)}")
        raise ValidationError(f"operational session row fields are invalid: {'; '.join(details)}")
    status = _required(row["status"], field="status")
    if status not in _SESSION_STATUSES:
        raise ValidationError(f"unsupported operational session status: {status}")
    session = OperationalSession(
        session_id=_required(row["session_id"], field="session_id"),
        principal_id=_required(row["principal_id"], field="principal_id"),
        project=_required(row["project"], field="project"),
        workspace=_required(row["workspace"], field="workspace"),
        status=status,
        branch=_optional_string(row["branch"], field="branch"),
        head=_optional_string(row["head"], field="head"),
        summary=_optional_string(row["summary"], field="summary"),
        checkpointed_at=_canonical_timestamp(
            _required(row["checkpointed_at"], field="checkpointed_at"),
            field="checkpointed_at",
        ),
        source_event_id=_required(row["source_event_id"], field="source_event_id"),
        recoverable_at=_canonical_timestamp(
            _optional_string(row["recoverable_at"], field="recoverable_at"),
            field="recoverable_at",
            allow_empty=True,
        ),
        terminated_at=_canonical_timestamp(
            _optional_string(row["terminated_at"], field="terminated_at"),
            field="terminated_at",
            allow_empty=True,
        ),
        recoverable_reason=_optional_string(
            row["recoverable_reason"],
            field="recoverable_reason",
        ),
        updated_event_id=_required(
            row["updated_event_id"],
            field="updated_event_id",
        ),
    )
    if session.status == "active" and (session.recoverable_at or session.terminated_at):
        raise ValidationError("active operational session has terminal timestamps")
    if session.status == "recoverable" and (not session.recoverable_at or session.terminated_at):
        raise ValidationError("recoverable operational session timestamps are invalid")
    if session.status == "terminated" and not session.terminated_at:
        raise ValidationError("terminated operational session has no terminated_at")
    return session


def _resolve_workspace(value: str) -> str:
    raw = _required(value, field="workspace")
    try:
        return str(Path(raw).expanduser().resolve())
    except OSError as exc:
        raise ValidationError(f"workspace cannot be resolved: {raw}") from exc


def _legacy_timestamp(
    values: list[object],
    *,
    fallback: str,
) -> str:
    candidates: list[str] = []
    for value in values:
        if isinstance(value, str) and value.strip():
            candidates.append(_canonical_timestamp(value.strip(), field="legacy session timestamp"))
    return (
        max(candidates)
        if candidates
        else _canonical_timestamp(
            fallback,
            field="legacy session timestamp",
        )
    )


class LegacySessionMigrator:
    """Pure deterministic merge of the two pre-v2 session stores."""

    @staticmethod
    def merge_legacy(
        json_checkpoint: Mapping[str, object],
        sqlite_row: Mapping[str, object],
    ) -> MergedLegacySession:
        if not isinstance(json_checkpoint, Mapping) or not isinstance(
            sqlite_row,
            Mapping,
        ):
            raise ValidationError("legacy session inputs must be mappings")
        json_id = str(json_checkpoint.get("session_id") or json_checkpoint.get("id") or "").strip()
        sqlite_id = str(sqlite_row.get("session_id") or sqlite_row.get("id") or "").strip()
        if json_id and sqlite_id and json_id != sqlite_id:
            raise _invalid("legacy session identifiers are incompatible")
        session_id = _required(json_id or sqlite_id, field="session_id")

        json_project = str(json_checkpoint.get("project") or "").strip()
        sqlite_project = str(sqlite_row.get("project") or "").strip()
        if json_project and sqlite_project and json_project != sqlite_project:
            raise _invalid("legacy session projects are incompatible")

        json_workspace = str(
            json_checkpoint.get("workspace") or json_checkpoint.get("cwd") or ""
        ).strip()
        sqlite_workspace = str(
            sqlite_row.get("workspace") or sqlite_row.get("directory") or ""
        ).strip()
        resolved_json = _resolve_workspace(json_workspace) if json_workspace else ""
        resolved_sqlite = _resolve_workspace(sqlite_workspace) if sqlite_workspace else ""
        if resolved_json and resolved_sqlite and resolved_json != resolved_sqlite:
            raise _invalid("legacy session workspaces are incompatible")
        workspace = _required(
            resolved_json or resolved_sqlite,
            field="workspace",
        )
        project = _required(
            json_project or sqlite_project or Path(workspace).name,
            field="project",
        )

        raw_status = str(
            sqlite_row.get("status") or json_checkpoint.get("status") or "active"
        ).strip()
        if raw_status in {"completed", "terminated"}:
            status = "terminated"
        elif raw_status == "recoverable":
            status = "recoverable"
        elif raw_status == "active":
            status = "active"
        else:
            raise _invalid(f"unsupported legacy session status: {raw_status}")

        source_bytes = canonical_json_bytes(
            {
                "json_checkpoint": dict(json_checkpoint),
                "sqlite_row": dict(sqlite_row),
            }
        )
        source_event_id = str(
            json_checkpoint.get("source_event_id")
            or sqlite_row.get("source_event_id")
            or f"legacy-session/{hashlib.sha256(source_bytes).hexdigest()}"
        )
        fallback = "1970-01-01T00:00:00Z"
        checkpointed_at = _legacy_timestamp(
            [
                json_checkpoint.get("checkpointed_at"),
                json_checkpoint.get("updated"),
                json_checkpoint.get("created"),
                sqlite_row.get("started_at"),
            ],
            fallback=fallback,
        )
        terminated_at = (
            _legacy_timestamp(
                [
                    sqlite_row.get("ended_at"),
                    json_checkpoint.get("terminated_at"),
                    json_checkpoint.get("updated"),
                ],
                fallback=checkpointed_at,
            )
            if status == "terminated"
            else ""
        )
        recoverable_at = (
            _legacy_timestamp(
                [
                    json_checkpoint.get("recoverable_at"),
                    json_checkpoint.get("updated"),
                    sqlite_row.get("started_at"),
                ],
                fallback=checkpointed_at,
            )
            if status == "recoverable"
            else ""
        )
        local_artifacts = {
            str(key): value
            for key, value in json_checkpoint.items()
            if isinstance(key, str) and key not in _PORTABLE_LEGACY_KEYS
        }
        return MergedLegacySession(
            checkpoint=OperationalSession(
                session_id=session_id,
                principal_id=_required(
                    str(
                        json_checkpoint.get("principal_id")
                        or sqlite_row.get("principal_id")
                        or f"legacy-session:{session_id}"
                    ),
                    field="principal_id",
                ),
                project=project,
                workspace=workspace,
                status=status,
                branch=str(json_checkpoint.get("branch") or ""),
                head=str(json_checkpoint.get("head") or json_checkpoint.get("head_commit") or ""),
                summary=str(json_checkpoint.get("summary") or sqlite_row.get("summary") or ""),
                checkpointed_at=checkpointed_at,
                source_event_id=_required(
                    source_event_id,
                    field="source_event_id",
                ),
                recoverable_at=recoverable_at,
                terminated_at=terminated_at,
                recoverable_reason=str(json_checkpoint.get("recoverable_reason") or ""),
                updated_event_id=str(json_checkpoint.get("updated_event_id") or "legacy"),
            ),
            local_artifacts=local_artifacts,
        )


class OperationalSessionService:
    """Authenticated session commands with one monotonic portable authority."""

    def __init__(
        self,
        *,
        operational: OperationalStore,
        views: OperationalViewStore,
        context_factory: Callable[[PrincipalIdentity], CommitContext],
        clock: Callable[[], str] = utc_now_iso,
    ) -> None:
        self.operational = operational
        self.views = views
        self.context_factory = context_factory
        self.clock = clock

    @staticmethod
    def _assert_owner(
        existing: OperationalSession,
        identity: PrincipalIdentity,
    ) -> None:
        if existing.principal_id != identity.principal_id:
            raise _invalid("session identity differs from authenticated principal")

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        root = getattr(self.operational, "transaction_root", None)
        if root is None:
            yield
            return
        with authority_write_lock(Path(root)):
            yield

    def _catch_up(self) -> None:
        ledger = getattr(self.operational, "ledger", None)
        if ledger is not None:
            self.views.catch_up(ledger)

    def _command_timestamp(
        self,
        *,
        value: str | None,
        project: str,
        idempotency_key: str,
        event_type: str,
        field: str,
    ) -> str:
        if value is not None:
            return _canonical_timestamp(value, field=field)
        normalized_key = _idempotency_key(idempotency_key)
        existing = self.views.idempotency(project, normalized_key)
        if existing is None:
            return _canonical_timestamp(self.clock(), field=field)
        ledger = getattr(self.operational, "ledger", None)
        validated_events = getattr(ledger, "validated_events", None)
        if not callable(validated_events):
            raise StorageError("operational ledger cannot resolve an idempotent timestamp")
        event = next(
            (
                candidate
                for candidate in validated_events()
                if candidate.event_id == existing.event_id
            ),
            None,
        )
        if event is None:
            raise StorageError(
                f"idempotency record references a missing ledger event: {existing.event_id}"
            )
        if event.event_type != event_type:
            raise OperationalError(
                OperationalErrorCode.IDEMPOTENCY_CONFLICT,
                (
                    "operational idempotency key identifies a different event type: "
                    f"{project}/{normalized_key}"
                ),
                retryable=False,
            )
        return _canonical_timestamp(
            _required(event.payload.get(field), field=field),
            field=field,
        )

    def _commit(
        self,
        *,
        identity: PrincipalIdentity,
        session_id: str,
        project: str,
        workspace: str,
        event_type: str,
        idempotency_key: str,
        payload: Mapping[str, object],
        precondition: Callable[[], None],
    ) -> OperationalSession:
        command = OperationalCommand(
            event_type=event_type,
            actor=identity,
            target_id=session_id,
            project=project,
            workspace=workspace,
            expires_at=None,
            visibility="owner",
            idempotency_key=_idempotency_key(idempotency_key),
            caused_by=(),
            subject_uri=f"memo://session/{session_id}",
            trace_id="",
            payload=dict(payload),
        )
        request_hash = hashlib.sha256(canonical_json_bytes(asdict(command))).hexdigest()
        with self._transaction():
            self._catch_up()
            existing = self.views.idempotency(
                command.project,
                command.idempotency_key,
            )
            if existing is not None and existing.request_hash != request_hash:
                raise OperationalError(
                    OperationalErrorCode.IDEMPOTENCY_CONFLICT,
                    (
                        "operational idempotency key identifies a different request: "
                        f"{command.project}/{command.idempotency_key}"
                    ),
                    retryable=False,
                )
            if existing is None:
                precondition()
            self.operational.commit(
                command,
                context=self.context_factory(identity),
            )
            current = self.views.session(session_id)
            if current is None:
                raise StorageError(f"operational session commit did not materialize {session_id}")
            return current

    def checkpoint(
        self,
        *,
        identity: PrincipalIdentity,
        session_id: str,
        project: str,
        workspace: str,
        summary: str = "",
        branch: str = "",
        head: str = "",
        source_event_id: str,
        checkpointed_at: str | None = None,
        idempotency_key: str,
    ) -> OperationalSession:
        normalized_id = _required(session_id, field="session_id")
        normalized_project = _required(project, field="project")
        normalized_workspace = _required(workspace, field="workspace")
        if not isinstance(identity, PrincipalIdentity):
            raise ValidationError("identity must be PrincipalIdentity")

        def precondition() -> None:
            existing = self.views.session(normalized_id)
            if existing is None:
                return
            self._assert_owner(existing, identity)
            if existing.project != normalized_project:
                raise _invalid("session project cannot change")
            if existing.workspace != normalized_workspace:
                raise _invalid("session workspace cannot change")
            if existing.status == "terminated":
                raise _invalid(f"session is terminal: {normalized_id}")
            if existing.status == "recoverable":
                raise _invalid(f"recoverable session cannot checkpoint as active: {normalized_id}")

        with self._transaction():
            self._catch_up()
            timestamp = self._command_timestamp(
                value=checkpointed_at,
                project=normalized_project,
                idempotency_key=idempotency_key,
                event_type=SESSION_CHECKPOINTED,
                field="checkpointed_at",
            )
            payload = {
                "session_id": normalized_id,
                "principal_id": _required(
                    identity.principal_id,
                    field="principal_id",
                ),
                "project": normalized_project,
                "workspace": normalized_workspace,
                "status": "active",
                "branch": _optional_string(branch, field="branch"),
                "head": _optional_string(head, field="head"),
                "summary": _optional_string(summary, field="summary"),
                "checkpointed_at": timestamp,
                "source_event_id": _required(
                    source_event_id,
                    field="source_event_id",
                ),
            }
            return self._commit(
                identity=identity,
                session_id=normalized_id,
                project=normalized_project,
                workspace=normalized_workspace,
                event_type=SESSION_CHECKPOINTED,
                idempotency_key=idempotency_key,
                payload=payload,
                precondition=precondition,
            )

    def mark_recoverable(
        self,
        *,
        identity: PrincipalIdentity,
        session_id: str,
        reason: str = "",
        recoverable_at: str | None = None,
        idempotency_key: str,
    ) -> OperationalSession:
        normalized_id = _required(session_id, field="session_id")
        if not isinstance(identity, PrincipalIdentity):
            raise ValidationError("identity must be PrincipalIdentity")
        with self._transaction():
            self._catch_up()
            existing = self.views.session(normalized_id)
            if existing is None:
                raise _invalid(f"session has no checkpoint: {normalized_id}")

            def precondition() -> None:
                current = self.views.session(normalized_id)
                if current is None:
                    raise _invalid(f"session has no checkpoint: {normalized_id}")
                self._assert_owner(current, identity)
                if current.status == "terminated":
                    raise _invalid(f"session is terminal: {normalized_id}")
                if current.status != "active":
                    raise _invalid(
                        f"session cannot become recoverable from {current.status}: {normalized_id}"
                    )

            return self._commit(
                identity=identity,
                session_id=normalized_id,
                project=existing.project,
                workspace=existing.workspace,
                event_type=SESSION_RECOVERABLE,
                idempotency_key=idempotency_key,
                payload={
                    "session_id": normalized_id,
                    "recoverable_at": self._command_timestamp(
                        value=recoverable_at,
                        project=existing.project,
                        idempotency_key=idempotency_key,
                        event_type=SESSION_RECOVERABLE,
                        field="recoverable_at",
                    ),
                    "reason": _optional_string(reason, field="reason"),
                },
                precondition=precondition,
            )

    def terminate(
        self,
        *,
        identity: PrincipalIdentity,
        session_id: str,
        summary: str = "",
        terminated_at: str | None = None,
        idempotency_key: str,
    ) -> OperationalSession:
        normalized_id = _required(session_id, field="session_id")
        if not isinstance(identity, PrincipalIdentity):
            raise ValidationError("identity must be PrincipalIdentity")
        with self._transaction():
            self._catch_up()
            existing = self.views.session(normalized_id)
            if existing is None:
                raise _invalid(f"session has no checkpoint: {normalized_id}")

            def precondition() -> None:
                current = self.views.session(normalized_id)
                if current is None:
                    raise _invalid(f"session has no checkpoint: {normalized_id}")
                self._assert_owner(current, identity)
                if current.status == "terminated":
                    raise _invalid(f"session is terminal: {normalized_id}")

            return self._commit(
                identity=identity,
                session_id=normalized_id,
                project=existing.project,
                workspace=existing.workspace,
                event_type=SESSION_TERMINATED,
                idempotency_key=idempotency_key,
                payload={
                    "session_id": normalized_id,
                    "terminated_at": self._command_timestamp(
                        value=terminated_at,
                        project=existing.project,
                        idempotency_key=idempotency_key,
                        event_type=SESSION_TERMINATED,
                        field="terminated_at",
                    ),
                    "summary": _optional_string(summary, field="summary"),
                },
                precondition=precondition,
            )

    def get(self, session_id: str) -> OperationalSession | None:
        normalized_id = _required(session_id, field="session_id")
        with self._transaction():
            self._catch_up()
            return self.views.session(normalized_id)

    def list(
        self,
        *,
        limit: int = 10,
        project: str | None = None,
        workspace: str | None = None,
        status: str | None = None,
    ) -> list[OperationalSession]:
        with self._transaction():
            self._catch_up()
            return self.views.sessions(
                limit=limit,
                project=project,
                workspace=workspace,
                status=status,
            )

    def latest_recoverable(
        self,
        *,
        project: str | None = None,
        workspace: str | None = None,
    ) -> OperationalSession | None:
        rows = self.list(
            limit=1,
            project=project,
            workspace=workspace,
            status="recoverable",
        )
        return rows[0] if rows else None


__all__ = [
    "LegacySessionMigrator",
    "MergedLegacySession",
    "OperationalSession",
    "OperationalSessionService",
    "operational_session_from_row",
]
