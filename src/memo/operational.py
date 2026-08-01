"""Native operational continuity derived from :mod:`memo.operation_ledger`.

This replaces external focus/handoff/attention/receipt services while keeping
the boundary narrow: memo records continuity; it never schedules or executes
agent work.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from memo.atomic_io import atomic_write_text, authority_write_lock
from memo.contracts import (
    MEMO_OPERATIONAL_SCHEMA,
    ActorIdentity,
    Visibility,
    WriteReceipt,
)
from memo.operation_ledger import OperationLedger
from memo.util import utc_now_iso


class _LedgerView:
    """Read-only ledger facade; writes require the authorized store path."""

    def __init__(self, ledger: OperationLedger) -> None:
        self.__ledger = ledger

    def append(self, *args: Any, **kwargs: Any) -> Any:
        raise PermissionError("operational ledger appends require store authorization")

    def validated_events(self) -> Any:
        return self.__ledger.validated_events()

    def iter_events(self) -> Any:
        """Expose read-only iteration for compatibility and diagnostics."""
        return self.__ledger.iter_events()

    def validate_import_events(self, rows: list[dict[str, Any]]) -> None:
        """Validate a bundle without exposing the ledger mutation surface."""
        self.__ledger.validate_import_events(rows)

    def head_hashes(self) -> Any:
        return self.__ledger.head_hashes()

    def verify(self) -> dict[str, Any]:
        """Expose integrity verification without exposing ledger mutation."""
        return self.__ledger.verify()


class _V2LedgerView:
    """Read-only public view over the signed v2 ledger."""

    def __init__(self, ledger: OperationLedgerV2) -> None:
        self.__ledger = ledger

    def append(self, *args: Any, **kwargs: Any) -> Any:
        raise PermissionError("operational ledger appends require store authorization")

    def validated_events(self) -> Any:
        return self.__ledger.validated_events()

    def verify(self) -> dict[str, Any]:
        return asdict(self.__ledger.verify())

    def encode_bundle(self, bundle: Any) -> bytes:
        return self.__ledger.encode_bundle(bundle)

    def decode_bundle(self, encoded: bytes) -> Any:
        return self.__ledger.decode_bundle(encoded)


if TYPE_CHECKING:
    from memo.identity import PrincipalIdentity
    from memo.operation_ledger_v2 import OperationLedgerV2
    from memo.operation_views import OperationalViewStore
    from memo.operational_epoch import CommitContext, EpochFence
    from memo.operational_event import CommandResult, OperationalCommand


@dataclass(frozen=True)
class FocusItem:
    id: str
    project: str
    summary: str
    updated_at: str
    actor_id: str = "memo"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Handoff:
    id: str
    project: str
    summary: str
    from_actor: str
    to_actor: str
    created_at: str
    consumed_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AttentionItem:
    id: str
    project: str
    summary: str
    severity: str
    created_at: str
    acknowledged_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConflictRecord:
    id: str
    topic: str
    summary: str
    lifecycle_state: str
    freeze_write: bool
    created_at: str
    resolved_at: str = ""
    resolution: str = ""
    evidence_uris: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OperationalSignal:
    """Durable watcher marker, fenced by monotonically increasing epoch."""

    marker: str
    epoch: int
    fence: str
    payload: dict[str, Any]
    created_at: str


def _id(prefix: str) -> str:
    return f"{prefix}-{secrets.token_hex(8)}"


ProjectionState = dict[str, Any]
ProjectionHandler = Callable[[ProjectionState, dict[str, Any]], None]


def _apply_focus_set(state: ProjectionState, payload: dict[str, Any]) -> None:
    state["focus"][payload["project"]] = dict(payload)


def _apply_focus_clear(state: ProjectionState, payload: dict[str, Any]) -> None:
    state["focus"].pop(payload["project"], None)


def _apply_handoff_create(state: ProjectionState, payload: dict[str, Any]) -> None:
    state["handoffs"][payload["id"]] = dict(payload)


def _apply_handoff_consume(state: ProjectionState, payload: dict[str, Any]) -> None:
    item = state["handoffs"].get(payload["id"])
    if item is not None:
        item["consumed_at"] = payload["consumed_at"]


def _apply_attention_add(state: ProjectionState, payload: dict[str, Any]) -> None:
    state["attention"][payload["id"]] = dict(payload)


def _apply_attention_ack(state: ProjectionState, payload: dict[str, Any]) -> None:
    item = state["attention"].get(payload["id"])
    if item is not None:
        item["acknowledged_at"] = payload["acknowledged_at"]


def _apply_conflict_open(state: ProjectionState, payload: dict[str, Any]) -> None:
    state["conflicts"][payload["id"]] = dict(payload)


def _apply_conflict_resolve(state: ProjectionState, payload: dict[str, Any]) -> None:
    conflict = state["conflicts"].get(payload["id"])
    if conflict is not None:
        conflict.update(
            {
                "lifecycle_state": "resolved",
                "resolved_at": payload["resolved_at"],
                "resolution": payload["resolution"],
            }
        )


def _apply_outcome_record(state: ProjectionState, payload: dict[str, Any]) -> None:
    state["outcomes"][payload["task_id"]] = dict(payload)


def _apply_signal_remember(state: ProjectionState, payload: dict[str, Any]) -> None:
    marker = str(payload["marker"])
    current = state["signals"].get(marker)
    # Epoch fencing makes delayed watcher writes harmless.
    if current is not None and int(payload["epoch"]) < int(current.get("epoch", 0)):
        return
    state["signals"][marker] = dict(payload)


def _resolved_anomaly_patch(payload: dict[str, Any]) -> dict[str, str]:
    return {
        "lifecycle_state": "resolved",
        "resolved_at": str(payload.get("created_at") or ""),
        "resolution": str(payload.get("status") or "resolved"),
    }


def _detected_anomaly_conflict(
    anomaly_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": anomaly_id,
        "topic": (
            f"{payload.get('memory_id_a', '')} "
            f"{payload.get('memory_id_b', '')} "
            f"{payload.get('relationship', '')}"
        ).strip(),
        "summary": str(payload.get("summary") or "semantic contradiction"),
        "lifecycle_state": "detected",
        "freeze_write": True,
        "created_at": str(payload.get("created_at") or ""),
        "resolved_at": "",
        "resolution": "",
        "evidence_uris": list(payload.get("evidence_uris") or ()),
        "metadata": {
            "kind": "semantic_contradiction",
            "severity": payload.get("severity"),
            "confidence": payload.get("confidence"),
            # Structured subject-memory ids so matching and delete-time GC never
            # have to parse them back out of the prose summary/topic.
            "memory_ids": [
                mid
                for mid in (
                    str(payload.get("memory_id_a") or ""),
                    str(payload.get("memory_id_b") or ""),
                )
                if mid
            ],
        },
    }


def _apply_anomaly_record(state: ProjectionState, payload: dict[str, Any]) -> None:
    if payload.get("kind") != "semantic_contradiction":
        return
    anomaly_id = str(payload["anomaly_id"])
    if payload.get("state") == "resolved":
        conflict = state["conflicts"].get(anomaly_id)
        if conflict is not None:
            conflict.update(_resolved_anomaly_patch(payload))
        return
    state["conflicts"][anomaly_id] = _detected_anomaly_conflict(anomaly_id, payload)


_PROJECTION_HANDLERS: dict[str, ProjectionHandler] = {
    "focus.set": _apply_focus_set,
    "focus.clear": _apply_focus_clear,
    "handoff.create": _apply_handoff_create,
    "handoff.consume": _apply_handoff_consume,
    "attention.add": _apply_attention_add,
    "attention.ack": _apply_attention_ack,
    "conflict.open": _apply_conflict_open,
    "conflict.resolve": _apply_conflict_resolve,
    "outcome.record": _apply_outcome_record,
    "anomaly.record": _apply_anomaly_record,
    "signal.remember": _apply_signal_remember,
}


_MEMORIA_URI_RE = re.compile(r"memo://(?:memoria|memory)/([0-9a-fA-F]{8,})")


def _conflict_member_ids(row: dict[str, Any]) -> list[str]:
    """Subject-memory ids of a conflict, or ``[]`` for topic-scoped conflicts.

    Semantic-contradiction anomalies carry the two conflicting memory ids in
    ``metadata.memory_ids`` (older records only in their ``memo://memoria/``
    evidence uris). Manually-opened topic conflicts carry none.
    """
    meta = row.get("metadata") or {}
    ids = [str(m).strip() for m in (meta.get("memory_ids") or ()) if str(m).strip()]
    if ids:
        return ids
    out: list[str] = []
    for uri in row.get("evidence_uris") or ():
        match = _MEMORIA_URI_RE.search(str(uri))
        if match:
            out.append(match.group(1))
    return out


def _conflict_matches_query(row: dict[str, Any], query_cf: str) -> bool:
    """Whether a write whose topic is ``query_cf`` is subject to ``row``.

    Id-scoped (semantic-contradiction) conflicts match ONLY when the query
    references one of their subject memory ids — never their prose ``summary``,
    so common words like "memo"/"contradiction"/"between" no longer freeze
    unrelated writes. Topic-scoped (manually-opened) conflicts keep matching
    on their ``topic`` (not the prose summary).
    """
    member_ids = _conflict_member_ids(row)
    if member_ids:
        return any(mid.casefold() in query_cf for mid in member_ids)
    topic_cf = str(row.get("topic", "")).casefold()
    if not topic_cf:
        return False
    if query_cf in topic_cf or topic_cf in query_cf:
        return True
    return any(token in topic_cf for token in query_cf.split() if len(token) >= 3)


class OperationalStore:
    ledger: Any
    __ledger: Any
    views: OperationalViewStore
    epoch_fence: EpochFence | None
    transaction_root: Path
    backend_version: int

    def __init__(
        self,
        state_dir: Path,
        *,
        device_id: str,
        context_provider: Callable[[], CommitContext] | None = None,
        epoch_fence: EpochFence | None = None,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.__ledger = OperationLedger(self.state_dir, device_id=device_id)
        self.ledger = _LedgerView(self.__ledger)
        self.snapshot_path = self.state_dir / "operational-state.json"
        self._v2_enabled = False
        self.backend_version = 1
        # Legacy writers are fail-closed unless an authenticated epoch context
        # provider is explicitly composed (tests may inject an in-memory one).
        self._context_provider = context_provider
        self.epoch_fence = epoch_fence

    @classmethod
    def for_v2(
        cls,
        *,
        ledger: OperationLedgerV2,
        views: OperationalViewStore,
        epoch_fence: EpochFence,
        transaction_root: Path,
        context_provider: Callable[[], CommitContext] | None = None,
    ) -> OperationalStore:
        """Construct the dormant v2 service without activating the public facade."""
        instance = cls.__new__(cls)
        instance.state_dir = Path(transaction_root).parent
        instance.__ledger = ledger
        instance.ledger = _V2LedgerView(ledger)
        instance.views = views
        instance.epoch_fence = epoch_fence
        instance.transaction_root = Path(transaction_root)
        instance.snapshot_path = instance.state_dir / "operational-state.json"
        instance._v2_enabled = True
        instance.backend_version = 2
        instance._context_provider = context_provider
        return instance

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "schema": MEMO_OPERATIONAL_SCHEMA,
            "focus": {},
            "handoffs": {},
            "attention": {},
            "conflicts": {},
            "outcomes": {},
            "signals": {},
            "last_event_hash": "",
            "journal_heads": {},
        }

    def _read_snapshot(self) -> dict[str, Any]:
        if self._v2_enabled:
            ledger = cast("OperationLedgerV2", self.__ledger)
            self.views.catch_up(ledger)
            return cast(dict[str, Any], self.views.state())
        events = self.ledger.validated_events()
        journal_heads: dict[str, str] = {}
        for event in events:
            journal_heads[event.device_id] = event.event_hash
        try:
            data = json.loads(self.snapshot_path.read_text(encoding="utf-8"))
            if (
                data.get("schema") == MEMO_OPERATIONAL_SCHEMA
                and data.get("journal_heads") == journal_heads
            ):
                return data
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            pass
        return self.rebuild(events=events)

    def _write_snapshot(self, data: dict[str, Any]) -> None:
        atomic_write_text(
            self.snapshot_path,
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        )

    def commit(
        self,
        command: OperationalCommand,
        *,
        context: CommitContext,
    ) -> CommandResult:
        """Commit one authenticated v2 command through ledger and derived views."""
        from memo.errors import OperationalError, OperationalErrorCode
        from memo.operational_epoch import CommitContext
        from memo.operational_event import (
            CommandResult,
            OperationalCommand,
            canonical_json_bytes,
        )

        if not self._v2_enabled:
            raise OperationalError(
                OperationalErrorCode.INVALID_EVENT,
                "operational v2 is not active for this store",
                retryable=False,
            )
        if not isinstance(command, OperationalCommand):
            raise OperationalError(
                OperationalErrorCode.INVALID_EVENT,
                "operational command is required",
                retryable=False,
            )
        if not isinstance(context, CommitContext):
            raise OperationalError(
                OperationalErrorCode.INVALID_EVENT,
                "authenticated commit context is required",
                retryable=False,
            )
        if not command.idempotency_key:
            raise OperationalError(
                OperationalErrorCode.INVALID_EVENT,
                "operational v2 idempotency key is required",
                retryable=False,
            )
        if command.actor != context.identity:
            raise OperationalError(
                OperationalErrorCode.INVALID_EVENT,
                "operational command actor differs from authenticated principal",
                retryable=False,
            )
        request_hash = hashlib.sha256(canonical_json_bytes(asdict(command))).hexdigest()
        ledger = cast("OperationLedgerV2", self.__ledger)
        # Retain the durable epoch fence through the complete append + view
        # commit.  A one-shot ``verify`` here leaves a race where authority
        # activation can advance the marker between validation and fsync,
        # allowing a stale writer to append under the old epoch.
        # ``verified`` owns the admission/write locks; do not wrap it in a
        # second authority lock (the file lock is intentionally non-reentrant).
        fence = self.epoch_fence
        if fence is None:
            raise OperationalError(
                OperationalErrorCode.INVALID_EVENT,
                "operational epoch fence is unavailable",
                retryable=False,
            )
        if hasattr(fence, "verified"):
            admission: AbstractContextManager[None] = fence.verified(context)
        else:
            # Test doubles and older injected fences expose only verify();
            # retain compatibility while still checking before append.
            fence.verify(context)
            admission = nullcontext()
        with admission:
            self.views.catch_up(ledger)
            if not self.views.supports(command.event_type):
                raise OperationalError(
                    OperationalErrorCode.INVALID_EVENT,
                    (f"operational event type has no active view reducer: {command.event_type}"),
                    retryable=False,
                )
            existing = self.views.idempotency(
                command.project,
                command.idempotency_key,
            )
            if existing is not None:
                if existing.request_hash != request_hash:
                    raise OperationalError(
                        OperationalErrorCode.IDEMPOTENCY_CONFLICT,
                        (
                            "operational idempotency key identifies a different request: "
                            f"{command.project}/{command.idempotency_key}"
                        ),
                        retryable=False,
                    )
                event = next(
                    (
                        candidate
                        for candidate in ledger.validated_events()
                        if candidate.event_id == existing.event_id
                    ),
                    None,
                )
                if event is None:
                    raise OperationalError(
                        OperationalErrorCode.STORAGE_UNAVAILABLE,
                        "idempotency record references a missing ledger event",
                        retryable=False,
                    )
                return CommandResult(
                    event=event,
                    replayed=True,
                    result=existing.result,
                )
            event = ledger.append(command, context=context)
            self.views.catch_up(ledger)
            persisted = self.views.idempotency(
                command.project,
                command.idempotency_key,
            )
            if persisted is None or persisted.event_id != event.event_id:
                raise OperationalError(
                    OperationalErrorCode.STORAGE_UNAVAILABLE,
                    "operational view commit did not persist idempotency result",
                    retryable=False,
                )
            return CommandResult(
                event=event,
                replayed=False,
                result=persisted.result,
            )

    def export_bundles(self, *, origins: tuple[str, ...] | None = None) -> tuple[Any, ...]:
        """Export verified v2 bundles without exposing the ledger writer."""

        if not self._v2_enabled:
            raise RuntimeError("operational bundle export requires ledger v2")
        ledger = cast("OperationLedgerV2", self.__ledger)
        return ledger.export_bundles(origins=origins)

    def import_bundles(
        self,
        bundles: tuple[Any, ...],
        *,
        context: CommitContext,
    ) -> Any:
        """Import through the authenticated ledger and catch derived views up."""

        if not self._v2_enabled:
            raise RuntimeError("operational bundle import requires ledger v2")
        ledger = cast("OperationLedgerV2", self.__ledger)
        report = ledger.import_bundles(bundles, context=context)
        self.views.catch_up(ledger)
        return report

    def context_for(self, identity: PrincipalIdentity) -> CommitContext:
        """Derive a fresh authenticated v2 context for one local principal.

        Product surfaces must not reuse the runtime daemon's fixed identity or
        reach into the epoch fence directly.  They present an explicit
        ``PrincipalIdentity`` here; Memo first obtains its authority-bound
        context, rejects cross-device impersonation, and then asks the fence to
        revalidate the same epoch/control marker for the requested principal.
        """
        from memo.errors import OperationalError, OperationalErrorCode
        from memo.identity import PrincipalIdentity
        from memo.operational_epoch import CommitContext

        if not self._v2_enabled or self.backend_version != 2:
            raise OperationalError(
                OperationalErrorCode.INVALID_EVENT,
                "authenticated operational contexts require ledger v2",
                retryable=False,
            )
        if not isinstance(identity, PrincipalIdentity):
            raise OperationalError(
                OperationalErrorCode.INVALID_EVENT,
                "operational principal identity is required",
                retryable=False,
            )
        provider = self._context_provider
        fence = self.epoch_fence
        if provider is None or fence is None:
            raise OperationalError(
                OperationalErrorCode.INVALID_EVENT,
                "operational v2 authority is unavailable",
                retryable=False,
            )
        authority_context = provider()
        if not isinstance(authority_context, CommitContext):
            raise OperationalError(
                OperationalErrorCode.INVALID_EVENT,
                "operational v2 authority returned an invalid context",
                retryable=False,
            )
        if identity.device_id != authority_context.origin_device:
            raise OperationalError(
                OperationalErrorCode.INVALID_EVENT,
                "operational principal is not bound to the local device",
                retryable=False,
            )
        return fence.context(
            identity,
            request_epoch=authority_context.authority_epoch,
            request_control_oid=authority_context.control_oid,
        )

    def rebuild(self, *, events: list[Any] | None = None) -> dict[str, Any]:
        if self._v2_enabled:
            ledger = cast("OperationLedgerV2", self.__ledger)
            materialized = events if events is not None else ledger.validated_events()
            self.views.rebuild(materialized)
            return cast(dict[str, Any], self.views.state())
        state = self._empty()
        for event in events if events is not None else self.ledger.validated_events():
            self._apply(state, event.op, event.payload)
            state["last_event_hash"] = event.event_hash
            state["journal_heads"][event.device_id] = event.event_hash
        self._write_snapshot(state)
        return state

    @staticmethod
    def _apply(state: dict[str, Any], op: str, payload: dict[str, Any]) -> None:
        handler = _PROJECTION_HANDLERS.get(op)
        if handler is not None:
            handler(state, payload)

    def _commit(
        self,
        op: str,
        payload: dict[str, Any],
        *,
        subject_uri: str,
        actor: ActorIdentity | None = None,
        trace_id: str = "",
        context: CommitContext | None = None,
    ) -> dict[str, Any]:
        if self._v2_enabled:
            return self._commit_v2(
                op,
                payload,
                subject_uri=subject_uri,
                trace_id=trace_id,
                context=context,
            )
        _, admission = self._legacy_admission(
            context,
            purpose="operational writes",
        )
        with admission, authority_write_lock(self.snapshot_path):
            event = self._append_authorized_event(
                op,
                subject_uri=subject_uri,
                actor=actor,
                trace_id=trace_id,
                payload=payload,
            )
            try:
                state = json.loads(self.snapshot_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, OSError, json.JSONDecodeError):
                state = {}
            current_heads = self.__ledger.head_hashes()
            snapshot_heads = state.get("journal_heads")
            expected_heads = dict(current_heads)
            if event.previous_hash:
                expected_heads[event.device_id] = event.previous_hash
            else:
                expected_heads.pop(event.device_id, None)
            event_is_current_head = current_heads.get(event.device_id) == event.event_hash
            if (
                event_is_current_head
                and state.get("schema") == MEMO_OPERATIONAL_SCHEMA
                and snapshot_heads == expected_heads
            ):
                self._apply(state, op, payload)
                state["last_event_hash"] = event.event_hash
                state["journal_heads"] = current_heads
                self._write_snapshot(state)
            else:
                state = self.rebuild()
        return {"event": event.to_dict(), "state": state}

    def _commit_v2(
        self,
        op: str,
        payload: dict[str, Any],
        *,
        subject_uri: str,
        trace_id: str,
        context: CommitContext | None,
    ) -> dict[str, Any]:
        """Translate the stable public facade into one authenticated v2 command."""
        from memo.errors import OperationalError, OperationalErrorCode
        from memo.operational_epoch import CommitContext
        from memo.operational_event import OperationalCommand, canonical_json_bytes
        from memo.operational_event_types import (
            ATTENTION_ACKNOWLEDGED,
            ATTENTION_ADDED,
            CONFLICT_OPENED,
            CONFLICT_RESOLVED,
            FOCUS_CLEARED,
            FOCUS_SET,
            HANDOFF_CONSUMED,
            HANDOFF_CREATED,
            OUTCOME_RECORDED,
        )

        event_types = {
            "focus.set": FOCUS_SET,
            "focus.clear": FOCUS_CLEARED,
            "handoff.create": HANDOFF_CREATED,
            "handoff.consume": HANDOFF_CONSUMED,
            "attention.add": ATTENTION_ADDED,
            "attention.ack": ATTENTION_ACKNOWLEDGED,
            "conflict.open": CONFLICT_OPENED,
            "conflict.resolve": CONFLICT_RESOLVED,
            "outcome.record": OUTCOME_RECORDED,
        }
        event_type = event_types.get(op)
        if event_type is None:
            raise OperationalError(
                OperationalErrorCode.INVALID_EVENT,
                f"operational v2 does not admit legacy operation {op!r}",
                retryable=False,
            )
        authenticated = context or (self._context_provider() if self._context_provider else None)
        if not isinstance(authenticated, CommitContext):
            raise OperationalError(
                OperationalErrorCode.INVALID_EVENT,
                "authenticated epoch context is required for operational v2 writes",
                retryable=False,
            )
        explicit_key = str(payload.get("idempotency_key") or "").strip()
        request_key = explicit_key or trace_id.strip()
        if not request_key:
            request_key = hashlib.sha256(
                canonical_json_bytes(
                    {
                        "op": op,
                        "payload": payload,
                        "subject_uri": subject_uri,
                    }
                )
            ).hexdigest()
        project = str(payload.get("project") or "_global")
        target_id = (
            str(payload.get("id") or payload.get("task_id") or payload.get("project") or "") or None
        )
        command = OperationalCommand(
            event_type=event_type,
            actor=authenticated.identity,
            target_id=target_id,
            project=project,
            workspace="",
            expires_at=None,
            visibility=Visibility.LOCAL_ONLY.value,
            idempotency_key=request_key,
            caused_by=(),
            subject_uri=subject_uri,
            trace_id=trace_id,
            payload=payload,
        )
        result = self.commit(command, context=authenticated)
        event = json.loads(canonical_json_bytes(result.event))
        return {"event": event, "state": self._read_snapshot()}

    def _legacy_admission(
        self,
        context: CommitContext | None,
        *,
        purpose: str,
    ) -> tuple[CommitContext, AbstractContextManager[None]]:
        """Resolve and retain the authenticated fence for one v1 mutation.

        The legacy journal remains migration-compatible, but every mutation is
        admitted by the same durable epoch fence used by v2.  Retaining the
        fence through the durable write closes the verify-then-append race.
        """
        from memo.errors import OperationalError, OperationalErrorCode
        from memo.operational_epoch import CommitContext

        authenticated = context or (self._context_provider() if self._context_provider else None)
        if not isinstance(authenticated, CommitContext):
            raise OperationalError(
                OperationalErrorCode.INVALID_EVENT,
                f"authenticated epoch context is required for {purpose}",
                retryable=False,
            )
        fence = self.epoch_fence
        if fence is None:
            raise OperationalError(
                OperationalErrorCode.INVALID_EVENT,
                f"authenticated epoch fence is required for {purpose}",
                retryable=False,
            )
        if hasattr(fence, "verified"):
            admission: AbstractContextManager[None] = fence.verified(authenticated)
        else:
            fence.verify(authenticated)
            admission = nullcontext()
        return authenticated, admission

    def _append_authorized_event(self, op: str, **kwargs: Any) -> Any:
        """Internal append seam; callers must already hold epoch admission."""
        return self.__ledger.append(op, **kwargs)

    def import_events(
        self,
        rows: list[dict[str, Any]],
        *,
        context: CommitContext | None = None,
    ) -> dict[str, int]:
        """Import a verified legacy bundle through the authorized store path."""
        _, admission = self._legacy_admission(
            context,
            purpose="operational event imports",
        )
        with admission:
            return cast(dict[str, int], self.__ledger.import_events(rows))

    def append_legacy_import(
        self,
        op: str,
        *,
        subject_uri: str,
        trace_id: str,
        payload: dict[str, Any],
        event_id: str,
        context: CommitContext | None = None,
    ) -> Any:
        """Append one migration-only v1 event through epoch authorization."""
        _, admission = self._legacy_admission(
            context,
            purpose="legacy operational migration",
        )
        with admission:
            return self._append_authorized_event(
                op,
                subject_uri=subject_uri,
                trace_id=trace_id,
                payload=payload,
                event_id=event_id,
            )

    def set_focus(
        self,
        *,
        project: str,
        summary: str,
        actor: ActorIdentity | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FocusItem:
        item = FocusItem(
            id=_id("focus"),
            project=project,
            summary=summary,
            updated_at=utc_now_iso(),
            actor_id=(actor.actor_id if actor else "memo"),
            metadata=dict(metadata or {}),
        )
        self._commit(
            "focus.set",
            asdict(item),
            subject_uri=f"memo://focus/{project}",
            actor=actor,
        )
        return item

    def clear_focus(self, project: str, *, actor: ActorIdentity | None = None) -> bool:
        existed = project in self._read_snapshot()["focus"]
        self._commit(
            "focus.clear",
            {"project": project, "cleared_at": utc_now_iso()},
            subject_uri=f"memo://focus/{project}",
            actor=actor,
        )
        return existed

    def create_handoff(
        self,
        *,
        project: str,
        summary: str,
        from_actor: str,
        to_actor: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Handoff:
        item = Handoff(
            id=_id("handoff"),
            project=project,
            summary=summary,
            from_actor=from_actor,
            to_actor=to_actor,
            created_at=utc_now_iso(),
            metadata=dict(metadata or {}),
        )
        self._commit(
            "handoff.create",
            asdict(item),
            subject_uri=f"memo://handoff/{item.id}",
            actor=ActorIdentity(actor_id=from_actor, actor_kind="agent"),
        )
        return item

    def consume_handoff(self, id_: str, *, actor_id: str = "memo") -> bool:
        with authority_write_lock(self.state_dir / "operational-transactions"):
            state = self._read_snapshot()
            item = state["handoffs"].get(id_)
            if not item or item.get("consumed_at"):
                return False
            self._commit(
                "handoff.consume",
                {"id": id_, "consumed_at": utc_now_iso()},
                subject_uri=f"memo://handoff/{id_}",
                actor=ActorIdentity(actor_id=actor_id, actor_kind="agent"),
            )
            return True

    def add_attention(
        self,
        *,
        project: str,
        summary: str,
        severity: str = "medium",
        metadata: dict[str, Any] | None = None,
    ) -> AttentionItem:
        if severity not in {"low", "medium", "high", "critical"}:
            raise ValueError("attention severity must be low|medium|high|critical")
        item = AttentionItem(
            id=_id("attention"),
            project=project,
            summary=summary,
            severity=severity,
            created_at=utc_now_iso(),
            metadata=dict(metadata or {}),
        )
        self._commit(
            "attention.add",
            asdict(item),
            subject_uri=f"memo://attention/{item.id}",
        )
        return item

    def acknowledge_attention(self, id_: str, *, actor_id: str = "memo") -> bool:
        with authority_write_lock(self.state_dir / "operational-transactions"):
            item = self._read_snapshot()["attention"].get(id_)
            if not item or item.get("acknowledged_at"):
                return False
            self._commit(
                "attention.ack",
                {"id": id_, "acknowledged_at": utc_now_iso()},
                subject_uri=f"memo://attention/{id_}",
                actor=ActorIdentity(actor_id=actor_id, actor_kind="agent"),
            )
            return True

    def open_conflict(
        self,
        *,
        topic: str,
        summary: str,
        freeze_write: bool = True,
        evidence_uris: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ConflictRecord:
        item = ConflictRecord(
            id=_id("conflict"),
            topic=topic,
            summary=summary,
            lifecycle_state="detected",
            freeze_write=freeze_write,
            created_at=utc_now_iso(),
            evidence_uris=tuple(evidence_uris or ()),
            metadata=dict(metadata or {}),
        )
        self._commit(
            "conflict.open",
            asdict(item),
            subject_uri=f"memo://conflict/{item.id}",
        )
        return item

    def resolve_conflict(
        self,
        id_: str,
        *,
        resolution: str,
        actor: ActorIdentity,
    ) -> bool:
        if actor.actor_kind != "human":
            raise PermissionError("conflict resolution requires human authority")
        with authority_write_lock(self.state_dir / "operational-transactions"):
            item = self._read_snapshot()["conflicts"].get(id_)
            if not item or item.get("lifecycle_state") == "resolved":
                return False
            self._commit(
                "conflict.resolve",
                {
                    "id": id_,
                    "resolved_at": utc_now_iso(),
                    "resolution": resolution,
                },
                subject_uri=f"memo://conflict/{id_}",
                actor=actor,
            )
            return True

    def gc_conflicts_for_memory(
        self,
        memory_id: str,
        *,
        reason: str = "subject memory deleted",
    ) -> int:
        """Auto-resolve active conflicts whose subject memories include
        ``memory_id``.

        Called on delete so a detected contradiction never orphans into a
        permanent ``freeze_write`` block once one of the two memories it was
        about is gone. Unlike :meth:`resolve_conflict` this is a system-level
        cleanup and does not require human authority. Returns the count
        resolved.
        """
        mid = str(memory_id).strip().casefold()
        if not mid:
            return 0
        with authority_write_lock(self.state_dir / "operational-transactions"):
            conflicts = self._read_snapshot()["conflicts"]
            targets = [
                cid
                for cid, row in conflicts.items()
                if row.get("lifecycle_state") not in {"resolved", "archived"}
                and any(m.casefold() == mid for m in _conflict_member_ids(row))
            ]
            for cid in targets:
                self._commit(
                    "conflict.resolve",
                    {
                        "id": cid,
                        "resolved_at": utc_now_iso(),
                        "resolution": reason,
                    },
                    subject_uri=f"memo://conflict/{cid}",
                    actor=ActorIdentity(actor_id="memo-gc", actor_kind="system"),
                )
            return len(targets)

    def record_outcome(
        self,
        *,
        task_id: str,
        status: str,
        memory_ids: list[str],
        actor_id: str = "memo",
        artifacts: list[str] | None = None,
        environment: dict[str, Any] | None = None,
        idempotency_key: str = "",
    ) -> dict[str, Any]:
        if status not in {"success", "failure", "partial"}:
            raise ValueError("outcome status must be success|failure|partial")
        with authority_write_lock(self.state_dir / "operational-transactions"):
            state = self._read_snapshot()
            existing = state["outcomes"].get(task_id)
            if existing and idempotency_key and existing.get("idempotency_key") == idempotency_key:
                return dict(existing)
            payload = {
                "task_id": task_id,
                "status": status,
                "memory_ids": list(dict.fromkeys(memory_ids)),
                "artifacts": list(artifacts or ()),
                "environment": dict(environment or {}),
                "actor_id": actor_id,
                "idempotency_key": idempotency_key,
                "recorded_at": utc_now_iso(),
            }
            self._commit(
                "outcome.record",
                payload,
                subject_uri=f"memo://outcome/{task_id}",
                actor=ActorIdentity(actor_id=actor_id, actor_kind="agent"),
            )
            return payload

    def record_anomaly(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Record a semantic anomaly through the authenticated Memo boundary.

        Legacy journals retain their historical ``anomaly.record`` wire event.
        The v2 ledger has a single canonical conflict model, so detected and
        resolved anomalies become conflict lifecycle commands instead of
        introducing a parallel event family.
        """
        anomaly_id = str(payload.get("anomaly_id") or "").strip()
        state = str(payload.get("state") or "").strip()
        if not anomaly_id:
            raise ValueError("anomaly_id must not be empty")
        if state not in {"detected", "resolved"}:
            raise ValueError("anomaly state must be detected|resolved")
        if not self._v2_enabled:
            return self._commit(
                "anomaly.record",
                dict(payload),
                subject_uri=f"memo://anomaly/{anomaly_id}",
            )
        if state == "resolved":
            return self._commit(
                "conflict.resolve",
                {
                    "id": anomaly_id,
                    "resolved_at": str(payload.get("created_at") or utc_now_iso()),
                    "resolution": str(payload.get("status") or "resolved"),
                },
                subject_uri=f"memo://conflict/{anomaly_id}",
                actor=ActorIdentity(actor_id="memo-anomaly", actor_kind="system"),
            )
        conflict = _detected_anomaly_conflict(anomaly_id, payload)
        return self._commit(
            "conflict.open",
            conflict,
            subject_uri=f"memo://conflict/{anomaly_id}",
            actor=ActorIdentity(actor_id="memo-anomaly", actor_kind="system"),
        )

    def remember_signal(
        self,
        *,
        marker: str,
        payload: dict[str, Any] | None = None,
        epoch: int = 0,
        fence: str = "",
        actor_id: str = "memo",
    ) -> OperationalSignal:
        """Record an idempotent watcher marker in the native journal.

        A marker is the idempotency key. Writes from an older epoch are rejected;
        callers should rotate ``fence`` when leadership changes.
        """
        marker = str(marker).strip()
        if not marker:
            raise ValueError("signal marker must not be empty")
        if int(epoch) < 0:
            raise ValueError("signal epoch must be non-negative")
        from memo.errors import OperationalError, OperationalErrorCode
        from memo.operational_epoch import CommitContext

        context = self._context_provider() if self._context_provider else None
        if not isinstance(context, CommitContext):
            raise OperationalError(
                OperationalErrorCode.INVALID_EVENT,
                "authenticated epoch context is required for operational signals",
                retryable=False,
            )
        if context.authority_epoch != int(epoch):
            raise ValueError("signal epoch does not match authenticated context")
        if fence and fence != context.control_oid:
            raise ValueError("signal fence does not match authenticated control")
        with authority_write_lock(self.state_dir / "operational-transactions"):
            state = self._read_snapshot()
            existing = state["signals"].get(marker)
            if existing is not None:
                if int(epoch) < int(existing.get("epoch", 0)):
                    raise ValueError("stale signal epoch")
                if int(epoch) == int(existing.get("epoch", 0)):
                    return OperationalSignal(
                        marker=marker,
                        epoch=int(existing["epoch"]),
                        fence=str(existing.get("fence", "")),
                        payload=dict(existing.get("payload") or {}),
                        created_at=str(existing.get("created_at", "")),
                    )
            item = OperationalSignal(
                marker, int(epoch), str(fence), dict(payload or {}), utc_now_iso()
            )
            self._commit(
                "signal.remember",
                asdict(item),
                subject_uri=f"memo://signal/{marker}",
                actor=ActorIdentity(actor_id=actor_id, actor_kind="agent"),
                context=context,
            )
            return item

    def list_signals(
        self, *, marker: str | None = None, min_epoch: int | None = None, limit: int = 100
    ) -> list[OperationalSignal]:
        rows = self._read_snapshot().get("signals", {})
        out: list[OperationalSignal] = []
        for key, row in rows.items():
            if marker and key != marker:
                continue
            if min_epoch is not None and int(row.get("epoch", 0)) < min_epoch:
                continue
            out.append(
                OperationalSignal(
                    key,
                    int(row.get("epoch", 0)),
                    str(row.get("fence", "")),
                    dict(row.get("payload") or {}),
                    str(row.get("created_at", "")),
                )
            )
        return sorted(out, key=lambda item: (item.epoch, item.created_at), reverse=True)[
            : max(1, min(limit, 1000))
        ]

    def receipt(
        self,
        operation: str,
        *,
        subject_uri: str,
        trace_id: str = "",
        actor_id: str = "memo",
        metadata: dict[str, Any] | None = None,
    ) -> WriteReceipt:
        result = self._commit(
            f"receipt.{operation}",
            {"operation": operation, "metadata": dict(metadata or {})},
            subject_uri=subject_uri,
            actor=ActorIdentity(actor_id=actor_id, actor_kind="agent"),
            trace_id=trace_id,
        )
        event = result["event"]
        return WriteReceipt(
            receipt_id=str(event["event_id"]),
            operation=operation,
            subject_uri=subject_uri,
            trace_id=trace_id,
            actor_id=actor_id,
            event_hash=str(event["event_hash"]),
            generated_at=str(event["ts"]),
            metadata=dict(metadata or {}),
        )

    def state(self, *, project: str | None = None) -> dict[str, Any]:
        state = self._read_snapshot()
        if not project:
            return state
        return {
            **state,
            "focus": {key: value for key, value in state["focus"].items() if key == project},
            "handoffs": {
                key: value
                for key, value in state["handoffs"].items()
                if value.get("project") == project
            },
            "attention": {
                key: value
                for key, value in state["attention"].items()
                if value.get("project") == project
            },
        }

    def active_conflicts(self, query: str = "") -> list[dict[str, Any]]:
        query_cf = query.casefold().strip()
        rows = [
            dict(row)
            for row in self._read_snapshot()["conflicts"].values()
            if row.get("lifecycle_state") not in {"resolved", "archived"}
        ]
        if query_cf:
            rows = [row for row in rows if _conflict_matches_query(row, query_cf)]
        rows.sort(
            key=lambda row: (bool(row.get("freeze_write")), str(row.get("created_at") or "")),
            reverse=True,
        )
        return rows


__all__ = [
    "AttentionItem",
    "ConflictRecord",
    "FocusItem",
    "Handoff",
    "OperationalStore",
    "Visibility",
]
