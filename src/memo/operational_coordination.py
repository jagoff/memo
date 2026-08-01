"""Memo-native channels, messages, handoffs, and coordination tasks."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from memo.contracts import Visibility
from memo.errors import OperationalError, OperationalErrorCode
from memo.identity import PrincipalIdentity
from memo.operational import OperationalStore
from memo.operational_epoch import CommitContext
from memo.operational_event import OperationalCommand
from memo.operational_event_types import (
    CHANNEL_OPENED,
    COORD_HANDOFF_CONSUMED,
    COORD_HANDOFF_CREATED,
    MESSAGE_SENT,
    MESSAGE_SUPERSEDED,
    TASK_ASSIGNED,
    TASK_CANCELLED,
    TASK_COMPLETED,
    TASK_CREATED,
    TASK_EXPIRED,
    TOPIC_TERMINATED,
)

ContextFactory = Callable[[PrincipalIdentity], CommitContext]


def _now(clock: Callable[[], datetime]) -> str:
    value = clock()
    if value.tzinfo is None:
        raise ValueError("coordination clock must include a timezone")
    return value.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _stable_id(kind: str, identity: PrincipalIdentity, key: str) -> str:
    digest = hashlib.sha256(f"{kind}\0{identity.principal_id}\0{key}".encode()).hexdigest()
    return f"{kind}-{digest[:24]}"


def _invalid(message: str) -> OperationalError:
    return OperationalError(
        OperationalErrorCode.INVALID_EVENT,
        message,
        retryable=False,
    )


@dataclass(frozen=True)
class MessageView:
    message_id: str
    event_id: str
    channel: str
    body: str
    actor_id: str
    target_ids: tuple[str, ...]
    topic: str
    expects_ack: bool
    expires_at: str | None
    evidence_uris: tuple[str, ...] = ()
    created_at: str = ""
    superseded_by_message_id: str | None = None


@dataclass(frozen=True)
class ChannelView:
    channel: str
    topic: str
    status: str
    superseded_by_message_id: str | None
    terminated_at: str | None


@dataclass(frozen=True)
class HandoffView:
    id: str
    message_id: str
    project: str
    status: str
    summary: str = ""
    from_actor: str = ""
    to_actor: str = ""
    evidence_uris: tuple[str, ...] = ()
    created_at: str = ""
    consumed_at: str = ""


@dataclass(frozen=True)
class TaskView:
    id: str
    project: str
    title: str
    status: str
    assignee_id: str | None
    result: str | None
    expires_at: str | None
    caused_by: str | None
    created_at: str = ""


class CoordinationService:
    def __init__(
        self,
        store: OperationalStore,
        *,
        context_factory: ContextFactory,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if store.backend_version != 2:
            raise _invalid("native coordination requires operational ledger v2")
        self.store = store
        self.context_factory = context_factory
        self.clock = clock or (lambda: datetime.now(UTC))

    def _commit(
        self,
        identity: PrincipalIdentity,
        *,
        event_type: str,
        project: str,
        target_id: str | None,
        subject_uri: str,
        payload: Mapping[str, object],
        idempotency_key: str,
        expires_at: str | None = None,
        caused_by: tuple[str, ...] = (),
    ):
        key = idempotency_key.strip()
        if not key:
            raise ValueError("idempotency_key must be non-empty")
        return self.store.commit(
            OperationalCommand(
                event_type=event_type,
                actor=identity,
                target_id=target_id,
                project=project or "_global",
                workspace="",
                expires_at=expires_at,
                visibility=Visibility.SHARED.value,
                idempotency_key=key,
                caused_by=caused_by,
                subject_uri=subject_uri,
                trace_id="",
                payload=payload,
            ),
            context=self.context_factory(identity),
        )

    def open_channel(
        self,
        *,
        identity: PrincipalIdentity,
        channel: str,
        topic: str = "",
        idempotency_key: str,
    ) -> ChannelView:
        normalized = channel.strip()
        if not normalized:
            raise ValueError("channel must be non-empty")
        self._commit(
            identity,
            event_type=CHANNEL_OPENED,
            project="_global",
            target_id=normalized,
            subject_uri=f"memo://coord/channel/{normalized}",
            payload={"channel": normalized, "topic": topic.strip()},
            idempotency_key=idempotency_key,
        )
        return self.channels()[normalized]

    def send_message(
        self,
        *,
        identity: PrincipalIdentity,
        channel: str,
        body: str,
        target_ids: tuple[str, ...] = (),
        topic: str = "",
        evidence_uris: tuple[str, ...] = (),
        expects_ack: bool = False,
        expires_at: str | None = None,
        idempotency_key: str,
    ) -> MessageView:
        normalized_channel = channel.strip()
        normalized_body = body.strip()
        if not normalized_channel or not normalized_body:
            raise ValueError("channel and body must be non-empty")
        targets = tuple(sorted(set(item.strip() for item in target_ids if item.strip())))
        evidence = tuple(item.strip() for item in evidence_uris if item.strip())
        message_id = _stable_id("message", identity, idempotency_key.strip())
        existing = next(
            (item for item in self.messages() if item.message_id == message_id),
            None,
        )
        if existing is not None:
            expected = (
                normalized_channel,
                normalized_body,
                identity.actor_id,
                targets,
                topic.strip(),
                bool(expects_ack),
                expires_at,
                evidence,
            )
            actual = (
                existing.channel,
                existing.body,
                existing.actor_id,
                existing.target_ids,
                existing.topic,
                existing.expects_ack,
                existing.expires_at,
                existing.evidence_uris,
            )
            if actual != expected:
                raise _invalid("message idempotency key identifies a different request")
            return existing
        created_at = _now(self.clock)
        result = self._commit(
            identity,
            event_type=MESSAGE_SENT,
            project="_global",
            target_id=message_id,
            subject_uri=f"memo://coord/message/{message_id}",
            payload={
                "message_id": message_id,
                "channel": normalized_channel,
                "body": normalized_body,
                "actor_id": identity.actor_id,
                "target_ids": targets,
                "topic": topic.strip(),
                "expects_ack": bool(expects_ack),
                "evidence_uris": evidence,
                "created_at": created_at,
            },
            idempotency_key=idempotency_key,
            expires_at=expires_at,
        )
        return MessageView(
            message_id=message_id,
            event_id=result.event.event_id,
            channel=normalized_channel,
            body=normalized_body,
            actor_id=identity.actor_id,
            target_ids=targets,
            topic=topic.strip(),
            expects_ack=bool(expects_ack),
            expires_at=expires_at,
            evidence_uris=evidence,
            created_at=created_at,
        )

    def messages(
        self,
        *,
        channel: str | None = None,
        now: datetime | None = None,
    ) -> list[MessageView]:
        cutoff = (now or self.clock()).astimezone(UTC)
        rows: dict[str, MessageView] = {}
        for event in self.store.ledger.validated_events():
            payload = event.payload
            if event.event_type == MESSAGE_SENT:
                expires = event.expires_at
                if expires is not None:
                    parsed = datetime.fromisoformat(expires.replace("Z", "+00:00"))
                    if parsed <= cutoff:
                        continue
                message_id = str(payload["message_id"])
                rows[message_id] = MessageView(
                    message_id=message_id,
                    event_id=event.event_id,
                    channel=str(payload["channel"]),
                    body=str(payload["body"]),
                    actor_id=str(payload["actor_id"]),
                    target_ids=tuple(str(item) for item in payload["target_ids"]),
                    topic=str(payload["topic"]),
                    expects_ack=bool(payload["expects_ack"]),
                    expires_at=expires,
                    evidence_uris=tuple(str(item) for item in payload["evidence_uris"]),
                    created_at=str(payload["created_at"]),
                )
            elif event.event_type == MESSAGE_SUPERSEDED:
                old = rows.get(str(payload["message_id"]))
                if old is not None:
                    rows[old.message_id] = replace(
                        old,
                        superseded_by_message_id=str(payload["superseded_by_message_id"]),
                    )
        selected = [row for row in rows.values() if channel is None or row.channel == channel]
        return sorted(selected, key=lambda row: (row.created_at, row.event_id))

    def channels(self) -> dict[str, ChannelView]:
        rows: dict[str, ChannelView] = {}
        for event in self.store.ledger.validated_events():
            payload = event.payload
            if event.event_type == CHANNEL_OPENED:
                name = str(payload["channel"])
                rows[name] = ChannelView(name, str(payload["topic"]), "open", None, None)
            elif event.event_type == MESSAGE_SENT:
                name = str(payload["channel"])
                rows.setdefault(name, ChannelView(name, str(payload["topic"]), "open", None, None))
            elif event.event_type == MESSAGE_SUPERSEDED:
                name = str(payload["channel"])
                current = rows.get(name, ChannelView(name, "", "open", None, None))
                rows[name] = replace(
                    current,
                    superseded_by_message_id=str(payload["superseded_by_message_id"]),
                )
            elif event.event_type == TOPIC_TERMINATED:
                name = str(payload["channel"])
                current = rows.get(
                    name, ChannelView(name, str(payload["topic"]), "open", None, None)
                )
                rows[name] = replace(
                    current,
                    status="terminated",
                    terminated_at=str(payload["terminated_at"]),
                )
        return rows

    def supersede_message(
        self,
        *,
        identity: PrincipalIdentity,
        message_id: str,
        superseded_by_message_id: str,
        idempotency_key: str,
    ) -> MessageView:
        current = next((item for item in self.messages() if item.message_id == message_id), None)
        if current is None:
            raise _invalid(f"unknown message: {message_id}")
        self._commit(
            identity,
            event_type=MESSAGE_SUPERSEDED,
            project="_global",
            target_id=message_id,
            subject_uri=f"memo://coord/message/{message_id}",
            payload={
                "channel": current.channel,
                "message_id": message_id,
                "superseded_by_message_id": superseded_by_message_id,
            },
            idempotency_key=idempotency_key,
            caused_by=(current.event_id,),
        )
        return next(item for item in self.messages() if item.message_id == message_id)

    def terminate_topic(
        self,
        *,
        identity: PrincipalIdentity,
        channel: str,
        topic: str = "",
        idempotency_key: str,
    ) -> ChannelView:
        self._commit(
            identity,
            event_type=TOPIC_TERMINATED,
            project="_global",
            target_id=channel,
            subject_uri=f"memo://coord/channel/{channel}",
            payload={"channel": channel, "topic": topic, "terminated_at": _now(self.clock)},
            idempotency_key=idempotency_key,
        )
        return self.channels()[channel]

    def create_handoff(
        self,
        *,
        identity: PrincipalIdentity,
        message_id: str,
        project: str,
        summary: str,
        to_actor: str = "",
        evidence_uris: tuple[str, ...] = (),
        idempotency_key: str,
    ) -> HandoffView:
        message = next((item for item in self.messages() if item.message_id == message_id), None)
        if message is None:
            raise _invalid(f"unknown message: {message_id}")
        handoff_id = _stable_id("handoff", identity, idempotency_key.strip())
        existing = self.handoffs().get(handoff_id)
        if existing is not None:
            if (
                existing.message_id,
                existing.project,
                existing.summary,
                existing.to_actor,
                existing.evidence_uris,
            ) != (message_id, project, summary, to_actor, tuple(evidence_uris)):
                raise _invalid("handoff idempotency key identifies a different request")
            return existing
        self._commit(
            identity,
            event_type=COORD_HANDOFF_CREATED,
            project=project,
            target_id=handoff_id,
            subject_uri=f"memo://coord/handoff/{handoff_id}",
            payload={
                "id": handoff_id,
                "message_id": message_id,
                "project": project,
                "summary": summary,
                "from_actor": identity.actor_id,
                "to_actor": to_actor,
                "evidence_uris": tuple(evidence_uris),
                "created_at": _now(self.clock),
            },
            idempotency_key=idempotency_key,
            caused_by=(message.event_id,),
        )
        return self.handoffs()[handoff_id]

    def handoffs(self) -> dict[str, HandoffView]:
        rows: dict[str, HandoffView] = {}
        for event in self.store.ledger.validated_events():
            payload = event.payload
            if event.event_type == COORD_HANDOFF_CREATED:
                key = str(payload["id"])
                rows[key] = HandoffView(
                    id=key,
                    message_id=str(payload["message_id"]),
                    project=str(payload["project"]),
                    status="open",
                    summary=str(payload["summary"]),
                    from_actor=str(payload["from_actor"]),
                    to_actor=str(payload["to_actor"]),
                    evidence_uris=tuple(str(item) for item in payload["evidence_uris"]),
                    created_at=str(payload["created_at"]),
                )
            elif event.event_type == COORD_HANDOFF_CONSUMED:
                key = str(payload["id"])
                current = rows.get(key)
                if current is not None:
                    rows[key] = replace(
                        current,
                        status="consumed",
                        consumed_at=str(payload["consumed_at"]),
                    )
        return rows

    def consume_handoff(
        self,
        *,
        identity: PrincipalIdentity,
        handoff_id: str,
        idempotency_key: str,
    ) -> HandoffView:
        current = self.handoffs().get(handoff_id)
        if current is None:
            raise _invalid(f"unknown handoff: {handoff_id}")
        if current.to_actor and current.to_actor != identity.actor_id:
            raise _invalid("handoff target differs from authenticated actor")
        if current.status == "consumed":
            return current
        self._commit(
            identity,
            event_type=COORD_HANDOFF_CONSUMED,
            project=current.project,
            target_id=handoff_id,
            subject_uri=f"memo://coord/handoff/{handoff_id}",
            payload={
                "id": handoff_id,
                "consumed_at": _now(self.clock),
                "actor_id": identity.actor_id,
            },
            idempotency_key=idempotency_key,
        )
        return self.handoffs()[handoff_id]

    def create_task(
        self,
        *,
        identity: PrincipalIdentity,
        project: str,
        title: str,
        assignee_id: str | None = None,
        expires_at: str | None = None,
        caused_by: str | None = None,
        idempotency_key: str,
    ) -> TaskView:
        task_id = _stable_id("task", identity, idempotency_key.strip())
        existing = self.tasks().get(task_id)
        if existing is not None:
            if (
                existing.project,
                existing.title,
                existing.assignee_id,
                existing.expires_at,
                existing.caused_by,
            ) != (project, title, assignee_id, expires_at, caused_by):
                raise _invalid("task idempotency key identifies a different request")
            return existing
        self._commit(
            identity,
            event_type=TASK_CREATED,
            project=project,
            target_id=task_id,
            subject_uri=f"memo://coord/task/{task_id}",
            payload={
                "id": task_id,
                "project": project,
                "title": title,
                "assignee_id": assignee_id or "",
                "created_at": _now(self.clock),
            },
            idempotency_key=idempotency_key,
            expires_at=expires_at,
            caused_by=(caused_by,) if caused_by else (),
        )
        return self.tasks()[task_id]

    def tasks(self) -> dict[str, TaskView]:
        rows: dict[str, TaskView] = {}
        for event in self.store.ledger.validated_events():
            payload = event.payload
            if event.event_type == TASK_CREATED:
                key = str(payload["id"])
                assignee = str(payload["assignee_id"])
                rows[key] = TaskView(
                    id=key,
                    project=str(payload["project"]),
                    title=str(payload["title"]),
                    status="open",
                    assignee_id=assignee or None,
                    result=None,
                    expires_at=event.expires_at,
                    caused_by=event.caused_by[0] if event.caused_by else None,
                    created_at=str(payload["created_at"]),
                )
            elif event.event_type in {TASK_ASSIGNED, TASK_COMPLETED, TASK_CANCELLED, TASK_EXPIRED}:
                key = str(payload["id"])
                current = rows.get(key)
                if current is None:
                    continue
                if event.event_type == TASK_ASSIGNED and current.status == "open":
                    rows[key] = replace(current, assignee_id=str(payload["assignee_id"]))
                elif event.event_type == TASK_COMPLETED and current.status == "open":
                    rows[key] = replace(current, status="completed", result=str(payload["result"]))
                elif event.event_type == TASK_CANCELLED and current.status == "open":
                    rows[key] = replace(current, status="cancelled")
                elif event.event_type == TASK_EXPIRED and current.status == "open":
                    rows[key] = replace(current, status="expired")
        return rows

    def _task_transition(
        self,
        *,
        identity: PrincipalIdentity,
        task_id: str,
        event_type: str,
        idempotency_key: str,
        value: str = "",
    ) -> TaskView:
        current = self.tasks().get(task_id)
        if current is None:
            raise _invalid(f"unknown task: {task_id}")
        if current.status != "open":
            if event_type == TASK_COMPLETED and current.status == "completed":
                return current
            raise _invalid(f"task is terminal: {task_id}")
        timestamp = _now(self.clock)
        if event_type == TASK_ASSIGNED:
            payload = {"id": task_id, "assignee_id": value, "assigned_at": timestamp}
        elif event_type == TASK_COMPLETED:
            if current.assignee_id and current.assignee_id != identity.actor_id:
                raise _invalid("task assignee differs from authenticated actor")
            payload = {"id": task_id, "result": value, "completed_at": timestamp}
        else:
            payload = {"id": task_id, "at": timestamp}
        self._commit(
            identity,
            event_type=event_type,
            project=current.project,
            target_id=task_id,
            subject_uri=f"memo://coord/task/{task_id}",
            payload=payload,
            idempotency_key=idempotency_key,
        )
        return self.tasks()[task_id]

    def assign_task(
        self, *, identity: PrincipalIdentity, task_id: str, assignee_id: str, idempotency_key: str
    ) -> TaskView:
        return self._task_transition(
            identity=identity,
            task_id=task_id,
            event_type=TASK_ASSIGNED,
            value=assignee_id,
            idempotency_key=idempotency_key,
        )

    def complete_task(
        self, *, identity: PrincipalIdentity, task_id: str, result: str, idempotency_key: str
    ) -> TaskView:
        return self._task_transition(
            identity=identity,
            task_id=task_id,
            event_type=TASK_COMPLETED,
            value=result,
            idempotency_key=idempotency_key,
        )

    def cancel_task(
        self, *, identity: PrincipalIdentity, task_id: str, idempotency_key: str
    ) -> TaskView:
        return self._task_transition(
            identity=identity,
            task_id=task_id,
            event_type=TASK_CANCELLED,
            idempotency_key=idempotency_key,
        )

    def expire_task(
        self, *, identity: PrincipalIdentity, task_id: str, idempotency_key: str
    ) -> TaskView:
        return self._task_transition(
            identity=identity,
            task_id=task_id,
            event_type=TASK_EXPIRED,
            idempotency_key=idempotency_key,
        )


__all__ = [
    "ChannelView",
    "CoordinationService",
    "HandoffView",
    "MessageView",
    "TaskView",
]
