"""Signed operational bundle publication and incremental ingestion."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

from memo.errors import OperationalError, OperationalErrorCode
from memo.git_transport import GitTransport, TransportHead
from memo.operational import OperationalStore
from memo.operational_epoch import CommitContext
from memo.operational_event import OriginBundle, canonical_json_bytes


@dataclass(frozen=True)
class SyncResult:
    published_events: int
    ingested_events: int
    duplicates: int
    gaps: Mapping[str, int]
    pending: int


@dataclass(frozen=True)
class RecoveryResult:
    device_id: str
    requested_sequence: int
    recovered_events: int
    remaining_gap: int | None


@dataclass(frozen=True)
class OperationalSyncStatus:
    local_heads: Mapping[str, int]
    remote_heads: Mapping[str, int]
    last_publish_at: str | None
    last_ingest_at: str | None
    pending: int
    gaps: Mapping[str, int]
    health: str


class OperationalSync:
    """Replicate only immutable signed v2 artifacts; SQLite remains local."""

    def __init__(
        self,
        store: OperationalStore,
        *,
        transport: GitTransport,
        device_id: str,
        context_factory: Callable[[], CommitContext],
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if store.backend_version != 2:
            raise OperationalError(
                OperationalErrorCode.INVALID_EVENT,
                "operational sync requires ledger v2",
                retryable=False,
            )
        self.store = store
        self.transport = transport
        self.device_id = device_id
        self.context_factory = context_factory
        self.clock = clock or (lambda: datetime.now(UTC))
        self._last_publish_at: str | None = None
        self._last_ingest_at: str | None = None
        self._gaps: dict[str, int] = {}

    def publish(self) -> SyncResult:
        bundle = next(
            (
                item
                for item in self.store.export_bundles()
                if item.anchor.origin_device == self.device_id
            ),
            None,
        )
        if bundle is None:
            return SyncResult(0, 0, 0, {}, 0)
        result = self.transport.publish(bundle)
        self._last_publish_at = self._now()
        return SyncResult(
            published_events=result.published_events,
            ingested_events=0,
            duplicates=result.duplicates,
            gaps={},
            pending=0,
        )

    def ingest(self) -> SyncResult:
        local_bundles = {
            item.anchor.origin_device: item for item in self.store.export_bundles()
        }
        bundles: list[OriginBundle] = []
        gaps: dict[str, int] = {}
        for origin in self.transport.origins():
            head = self.transport.read_head(origin)
            assert head is not None
            bundle, gap = self._read_contiguous_bundle(
                head,
                local=local_bundles.get(origin),
            )
            bundles.append(bundle)
            if gap is not None:
                gaps[origin] = gap

        inserted = 0
        duplicates = 0
        if bundles:
            report = self.store.import_bundles(
                tuple(bundles),
                context=self.context_factory(),
            )
            inserted = int(report.events_inserted)
            duplicates = int(report.events_replayed)
        self._last_ingest_at = self._now()
        self._gaps = gaps
        pending = self._pending_count()
        return SyncResult(
            published_events=0,
            ingested_events=inserted,
            duplicates=duplicates,
            gaps=dict(gaps),
            pending=pending,
        )

    def recover_gap(
        self,
        *,
        device_id: str,
        expected_sequence: int,
    ) -> RecoveryResult:
        before = self._local_heads().get(device_id, 0)
        result = self.ingest()
        after = self._local_heads().get(device_id, 0)
        return RecoveryResult(
            device_id=device_id,
            requested_sequence=expected_sequence,
            recovered_events=max(0, after - before),
            remaining_gap=result.gaps.get(device_id),
        )

    def status(self) -> OperationalSyncStatus:
        local = self._local_heads()
        remote = self._remote_heads()
        pending = sum(max(0, sequence - local.get(origin, 0)) for origin, sequence in remote.items())
        return OperationalSyncStatus(
            local_heads=local,
            remote_heads=remote,
            last_publish_at=self._last_publish_at,
            last_ingest_at=self._last_ingest_at,
            pending=pending,
            gaps=dict(self._gaps),
            health="degraded" if self._gaps else "healthy",
        )

    def _read_contiguous_bundle(
        self,
        head: TransportHead,
        *,
        local: OriginBundle | None,
    ) -> tuple[OriginBundle, int | None]:
        try:
            anchor_bytes = self.transport.read_anchor(head)
            checkpoint = self.transport.read_checkpoint(head)
        except FileNotFoundError as exc:
            raise OperationalError(
                OperationalErrorCode.NOT_FOUND,
                f"operational transport authority artifact is missing: {head.origin_device}",
                retryable=True,
            ) from exc
        try:
            anchor_value = json.loads(anchor_bytes.decode("utf-8"))
            if not isinstance(anchor_value, dict) or canonical_json_bytes(anchor_value) != anchor_bytes:
                raise ValueError
            base_sequence = int(str(anchor_value["base_sequence"]))
            base_hash = str(anchor_value["base_event_hash"])
        except (TypeError, ValueError, KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OperationalError(
                OperationalErrorCode.ANCHOR_CONFLICT,
                f"invalid operational transport anchor: {head.origin_device}",
                retryable=False,
            ) from exc
        local_events = (
            {event.origin_sequence: event for event in local.events}
            if local is not None and local.anchor.anchor_hash == head.anchor_hash
            else {}
        )
        event_values: list[dict[str, object]] = []
        gap: int | None = None
        for sequence in range(base_sequence + 1, head.sequence + 1):
            encoded = self.transport.read_segment(head, sequence)
            if encoded is None:
                cached = local_events.get(sequence)
                if cached is None:
                    gap = sequence
                    break
                encoded = canonical_json_bytes(cached)
            try:
                value = json.loads(encoded.decode("utf-8"))
                if not isinstance(value, dict) or canonical_json_bytes(value) != encoded:
                    raise ValueError
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise OperationalError(
                    OperationalErrorCode.INVALID_EVENT,
                    f"invalid operational transport event: {head.origin_device}/{sequence}",
                    retryable=False,
                ) from exc
            event_values.append(value)
        if event_values:
            final_sequence = int(str(event_values[-1]["origin_sequence"]))
            final_hash = str(event_values[-1]["event_hash"])
            final_key_id = str(event_values[-1]["key_id"])
            final_signature = str(event_values[-1]["signature"])
        else:
            final_sequence = base_sequence
            final_hash = base_hash
            final_key_id = str(anchor_value["key_id"])
            final_signature = str(anchor_value["signature"])
        if gap is None and (
            final_sequence != head.sequence
            or final_hash != head.event_hash
            or final_key_id != head.key_id
            or final_signature != head.signature
        ):
            raise OperationalError(
                OperationalErrorCode.ANCHOR_CONFLICT,
                f"signed operational transport head does not match chain: {head.origin_device}",
                retryable=False,
            )
        encoded_bundle = canonical_json_bytes(
            {
                "anchor": anchor_value,
                "checkpoint": base64.urlsafe_b64encode(checkpoint)
                .rstrip(b"=")
                .decode("ascii"),
                "events": event_values,
                "head_sequence": final_sequence,
                "head_hash": final_hash,
            }
        )
        bundle = self.store.ledger.decode_bundle(encoded_bundle)
        return bundle, gap

    def _local_heads(self) -> dict[str, int]:
        return {
            bundle.anchor.origin_device: bundle.head_sequence
            for bundle in self.store.export_bundles()
        }

    def _remote_heads(self) -> dict[str, int]:
        heads: dict[str, int] = {}
        for origin in self.transport.origins():
            head = self.transport.read_head(origin)
            if head is not None:
                heads[origin] = head.sequence
        return heads

    def _pending_count(self) -> int:
        local = self._local_heads()
        return sum(
            max(0, sequence - local.get(origin, 0))
            for origin, sequence in self._remote_heads().items()
        )

    def _now(self) -> str:
        value = self.clock()
        if value.tzinfo is None:
            raise ValueError("operational sync clock must include a timezone")
        return value.astimezone(UTC).isoformat(timespec="milliseconds").replace(
            "+00:00", "Z"
        )


__all__ = [
    "OperationalSync",
    "OperationalSyncStatus",
    "RecoveryResult",
    "SyncResult",
]
