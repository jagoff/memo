"""Memo-native append-only operational ledger.

The JSONL journal is the authority for operational state and sync receipts.
Each device owns one hash chain.  A compact head file makes appends O(1), while
``verify`` and ``iter_events`` can rebuild state from the journal alone.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from memo.atomic_io import atomic_write_text, authority_write_lock
from memo.contracts import ActorIdentity, MemoEvent, new_event_id

_log = logging.getLogger(__name__)
_DEVICE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_TIMESTAMP_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})\Z"
)
_ACTOR_KINDS = {"human", "agent", "tool", "system", "device"}


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class LedgerIntegrityError(RuntimeError):
    """The native journal cannot be trusted or appended safely."""


def _timestamp_parts(ts: str) -> tuple[str, datetime]:
    """Validate an event timestamp and return its filesystem-safe UTC day."""
    if not isinstance(ts, str) or not _TIMESTAMP_RE.fullmatch(ts):
        raise LedgerIntegrityError(f"invalid journal timestamp: {ts!r}")
    try:
        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LedgerIntegrityError(f"invalid journal timestamp: {ts!r}") from exc
    if parsed.tzinfo is None:
        raise LedgerIntegrityError(f"journal timestamp must include a timezone: {ts!r}")
    normalized = parsed.astimezone(UTC)
    return normalized.date().isoformat(), normalized


class OperationLedger:
    def __init__(self, state_dir: Path, *, device_id: str) -> None:
        self.root = Path(state_dir) / "journal"
        self.device_id = str(device_id)
        if not _DEVICE_ID_RE.fullmatch(self.device_id):
            raise LedgerIntegrityError(f"unsafe journal device id: {self.device_id!r}")

    @property
    def _head_path(self) -> Path:
        return self.root / "heads" / f"{self.device_id}.json"

    def _segment_path(self, ts: str) -> Path:
        day, _ = _timestamp_parts(ts)
        return self.root / "events" / self.device_id / f"{day}.jsonl"

    def _assert_no_symlink_components(self, path: Path) -> None:
        """Reject symlinks inside Memo's journal authority boundary."""
        root = self.root.absolute()
        target = Path(path).absolute()
        try:
            relative = target.relative_to(root)
        except ValueError as exc:
            raise LedgerIntegrityError(f"journal path escapes authority root: {target}") from exc
        components = [root]
        current = root
        for part in relative.parts:
            current /= part
            components.append(current)
        for component in components:
            if component.is_symlink():
                raise LedgerIntegrityError(f"unsafe journal symlink: {component}")

    def _read_head(self) -> tuple[int, str]:
        try:
            raw = json.loads(self._head_path.read_text(encoding="utf-8"))
            return int(raw.get("sequence") or 0), str(raw.get("event_hash") or "")
        except FileNotFoundError:
            return 0, ""
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise LedgerIntegrityError(f"invalid journal head: {self._head_path}") from exc

    def _write_head(self, device: str, event: MemoEvent | None) -> None:
        path = self.root / "heads" / f"{device}.json"
        payload = {
            "sequence": event.sequence if event else 0,
            "event_hash": event.event_hash if event else "",
        }
        atomic_write_text(path, json.dumps(payload, sort_keys=True))

    @staticmethod
    def _last_complete_line(path: Path) -> str:
        """Read the final JSONL row without loading an entire daily segment."""
        try:
            with path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                end = handle.tell()
                if end == 0:
                    return ""
                position = end
                chunks: list[bytes] = []
                while position > 0:
                    size = min(8192, position)
                    position -= size
                    handle.seek(position)
                    chunks.append(handle.read(size))
                    data = b"".join(reversed(chunks)).rstrip(b"\r\n")
                    if b"\n" in data or position == 0:
                        return data.rsplit(b"\n", 1)[-1].decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise LedgerIntegrityError(f"cannot read journal segment tail: {path}") from exc
        return ""

    def _last_event(self, device: str) -> MemoEvent | None:
        event_dir = self.root / "events" / device
        if not event_dir.is_dir():
            return None
        paths = sorted(event_dir.glob("*.jsonl"))
        for path in reversed(paths):
            if path.is_symlink():
                raise LedgerIntegrityError(f"unsafe journal segment: {path}")
            line = self._last_complete_line(path)
            if not line:
                continue
            try:
                event = self._decode_event(json.loads(line))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise LedgerIntegrityError(f"malformed journal tail: {path}") from exc
            if event.device_id != device:
                raise LedgerIntegrityError(f"journal device mismatch in {path}")
            day, _ = _timestamp_parts(event.ts)
            if path.stem != day:
                raise LedgerIntegrityError(f"journal timestamp/path mismatch in {path}")
            calculated = replace(event, event_hash="").with_hash().event_hash
            if calculated != event.event_hash:
                raise LedgerIntegrityError(f"journal tail hash mismatch in {path}")
            return event
        return None

    def _append_position(self) -> tuple[int, str, str]:
        """Return the journal-authoritative tail, repairing only its head cache."""
        cached_sequence, cached_hash = self._read_head()
        tail = self._last_event(self.device_id)
        if tail is None:
            if cached_sequence or cached_hash:
                raise LedgerIntegrityError("journal head exists without an event chain")
            return 0, "", ""
        if (cached_sequence, cached_hash) != (tail.sequence, tail.event_hash):
            # A crash may durably append the event before atomically replacing
            # the advisory head. Validate the full chain before blessing it.
            validated = self._read_device_events_strict(self.device_id)
            tail = validated[-1]
            self._write_head(self.device_id, tail)
        return tail.sequence, tail.event_hash, tail.ts

    def append(
        self,
        op: str,
        *,
        subject_uri: str,
        actor: ActorIdentity | None = None,
        trace_id: str = "",
        payload: dict[str, Any] | None = None,
        content_hash: str = "",
        event_id: str | None = None,
        ts: str | None = None,
    ) -> MemoEvent:
        if not op.strip():
            raise ValueError("journal op cannot be empty")
        if not subject_uri.strip():
            raise ValueError("journal subject_uri cannot be empty")
        stamp = ts or _now_iso()
        _, stamp_dt = _timestamp_parts(stamp)
        segment = self._segment_path(stamp)
        self._assert_no_symlink_components(segment)
        self._assert_no_symlink_components(self._head_path)
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        with authority_write_lock(self.root):
            self._assert_no_symlink_components(segment)
            self._assert_no_symlink_components(self._head_path)
            sequence, previous_hash, previous_ts = self._append_position()
            if previous_ts:
                _, previous_dt = _timestamp_parts(previous_ts)
                if stamp_dt < previous_dt:
                    raise LedgerIntegrityError("journal timestamps must be nondecreasing")
            event = MemoEvent(
                event_id=event_id or new_event_id(),
                ts=stamp,
                device_id=self.device_id,
                sequence=sequence + 1,
                op=op.strip(),
                subject_uri=subject_uri.strip(),
                actor=actor or ActorIdentity(),
                trace_id=trace_id,
                payload=dict(payload or {}),
                content_hash=content_hash,
                previous_hash=previous_hash,
            ).with_hash()
            event_dir = self.root / "events" / self.device_id
            event_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            self._head_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            self._assert_no_symlink_components(segment)
            self._assert_no_symlink_components(self._head_path)
            with segment.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._write_head(self.device_id, event)
            return event

    def iter_events(
        self,
        *,
        device_id: str | None = None,
        limit: int | None = None,
    ) -> list[MemoEvent]:
        devices = [device_id] if device_id else self._device_ids()
        rows: list[MemoEvent] = []
        for device in devices:
            rows.extend(self._read_device_events_strict(str(device)))
        rows.sort(key=lambda event: (event.ts, event.device_id, event.sequence))
        return rows[-limit:] if limit is not None else rows

    def _read_device_events_strict(self, device: str) -> list[MemoEvent]:
        rows: list[MemoEvent] = []
        expected_previous = ""
        expected_sequence = 1
        previous_dt: datetime | None = None
        for path in self._device_segment_paths(device):
            for line_number, line in self._segment_lines(path):
                if not line.strip():
                    continue
                event = self._decode_journal_row(path, line_number, line)
                event_dt = self._validate_device_event(
                    event,
                    device=device,
                    path=path,
                    line_number=line_number,
                    expected_sequence=expected_sequence,
                    expected_previous=expected_previous,
                    previous_dt=previous_dt,
                )
                rows.append(event)
                expected_sequence += 1
                expected_previous = event.event_hash
                previous_dt = event_dt
        return rows

    def _device_segment_paths(self, device: str) -> list[Path]:
        if not _DEVICE_ID_RE.fullmatch(device):
            raise LedgerIntegrityError(f"unsafe journal device id: {device!r}")
        event_dir = self.root / "events" / device
        if event_dir.is_symlink():
            raise LedgerIntegrityError(f"unsafe journal device path: {event_dir}")
        if not event_dir.is_dir():
            return []
        paths = sorted(event_dir.glob("*.jsonl"))
        for path in paths:
            if path.is_symlink():
                raise LedgerIntegrityError(f"unsafe journal segment: {path}")
        return paths

    @staticmethod
    def _segment_lines(path: Path) -> list[tuple[int, str]]:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise LedgerIntegrityError(f"cannot read journal segment: {path}") from exc
        return list(enumerate(lines, start=1))

    @classmethod
    def _decode_journal_row(cls, path: Path, line_number: int, line: str) -> MemoEvent:
        try:
            return cls._decode_event(json.loads(line))
        except (
            LedgerIntegrityError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            raise LedgerIntegrityError(f"malformed journal row in {path}:{line_number}") from exc

    @staticmethod
    def _validate_device_event(
        event: MemoEvent,
        *,
        device: str,
        path: Path,
        line_number: int,
        expected_sequence: int,
        expected_previous: str,
        previous_dt: datetime | None,
    ) -> datetime:
        location = f"{path}:{line_number}"
        if event.device_id != device:
            raise LedgerIntegrityError(f"journal device mismatch in {location}")
        day, event_dt = _timestamp_parts(event.ts)
        if path.stem != day:
            raise LedgerIntegrityError(f"journal timestamp/path mismatch in {location}")
        if event.sequence != expected_sequence:
            raise LedgerIntegrityError(f"sequence {event.sequence} != expected {expected_sequence}")
        if event.previous_hash != expected_previous:
            raise LedgerIntegrityError(f"previous_hash mismatch at sequence {event.sequence}")
        calculated = replace(event, event_hash="").with_hash().event_hash
        if calculated != event.event_hash:
            raise LedgerIntegrityError(f"event_hash mismatch at sequence {event.sequence}")
        if previous_dt is not None and event_dt < previous_dt:
            raise LedgerIntegrityError("journal timestamps must be nondecreasing")
        return event_dt

    def validated_events(self) -> list[MemoEvent]:
        """Return all events only when every device chain validates."""
        rows: list[MemoEvent] = []
        for device in self._device_ids():
            rows.extend(self._read_device_events_strict(device))
        rows.sort(key=lambda event: (event.ts, event.device_id, event.sequence))
        return rows

    def _device_ids(self) -> list[str]:
        devices = self._event_device_ids()
        devices.update(self._head_device_ids())
        return sorted(devices)

    def _event_device_ids(self) -> set[str]:
        devices: set[str] = set()
        event_root = self.root / "events"
        if event_root.is_symlink():
            raise LedgerIntegrityError(f"unsafe journal event root: {event_root}")
        if not event_root.is_dir():
            return devices
        for path in event_root.iterdir():
            if path.is_symlink():
                raise LedgerIntegrityError(f"unsafe journal device path: {path}")
            if not path.is_dir():
                continue
            self._validate_device_id(path.name)
            devices.add(path.name)
        return devices

    def _head_device_ids(self) -> set[str]:
        devices: set[str] = set()
        head_root = self.root / "heads"
        if head_root.is_symlink():
            raise LedgerIntegrityError(f"unsafe journal head root: {head_root}")
        if not head_root.is_dir():
            return devices
        for path in head_root.iterdir():
            if path.is_symlink():
                raise LedgerIntegrityError(f"unsafe journal head: {path}")
            if not path.is_file() or path.suffix != ".json":
                continue
            self._validate_device_id(path.stem)
            devices.add(path.stem)
        return devices

    @staticmethod
    def _validate_device_id(device: str) -> None:
        if not _DEVICE_ID_RE.fullmatch(device):
            raise LedgerIntegrityError(f"unsafe journal device id: {device!r}")

    @staticmethod
    def _decode_event(raw: object) -> MemoEvent:
        if not isinstance(raw, dict):
            raise LedgerIntegrityError("journal event must be a JSON object")
        actor_value = raw.get("actor")
        if actor_value is None:
            actor_raw: dict[str, Any] = {}
        elif isinstance(actor_value, dict):
            actor_raw = actor_value
        else:
            raise LedgerIntegrityError("journal actor must be a JSON object")
        actor_kind = str(actor_raw.get("actor_kind") or "system")
        if actor_kind not in _ACTOR_KINDS:
            raise ValueError(f"invalid journal actor kind: {actor_kind!r}")
        actor = ActorIdentity(
            actor_id=str(actor_raw.get("actor_id") or "memo"),
            actor_kind=actor_kind,  # type: ignore[arg-type]
            signature=str(actor_raw.get("signature") or ""),
            source_client=str(actor_raw.get("source_client") or ""),
        )
        return MemoEvent(
            event_id=str(raw["event_id"]),
            ts=str(raw["ts"]),
            device_id=str(raw["device_id"]),
            sequence=int(raw["sequence"]),
            op=str(raw["op"]),
            subject_uri=str(raw["subject_uri"]),
            actor=actor,
            trace_id=str(raw.get("trace_id") or ""),
            payload=dict(raw.get("payload") or {}),
            content_hash=str(raw.get("content_hash") or ""),
            previous_hash=str(raw.get("previous_hash") or ""),
            event_hash=str(raw.get("event_hash") or ""),
            schema=str(raw.get("schema") or "memo.event.v1"),
        )

    def verify(self) -> dict[str, Any]:
        devices: dict[str, dict[str, Any]] = {}
        ok = True
        try:
            device_ids = self._device_ids()
        except LedgerIntegrityError as exc:
            return {"ok": False, "devices": {}, "events": 0, "errors": [str(exc)]}
        for device in device_ids:
            errors: list[str] = []
            try:
                events = self._read_device_events_strict(device)
            except LedgerIntegrityError as exc:
                events = []
                errors.append(str(exc))
            count = len(events)
            try:
                head_sequence, head_hash = self._read_device_head(device, strict=True)
            except LedgerIntegrityError as exc:
                head_sequence, head_hash = 0, ""
                errors.append(str(exc))
            expected_head = events[-1] if events else None
            if (head_sequence, head_hash) != (
                expected_head.sequence if expected_head else 0,
                expected_head.event_hash if expected_head else "",
            ):
                errors.append("head does not match final event")
            if errors:
                ok = False
            devices[device] = {
                "ok": not errors,
                "events": count,
                "head_sequence": head_sequence,
                "head_hash": head_hash,
                "errors": errors,
            }
        return {"ok": ok, "devices": devices, "events": sum(v["events"] for v in devices.values())}

    def import_events(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Import complete foreign device chains without rewriting their hashes.

        Each supplied device chain must start at sequence one. Existing chains
        may be extended only when the common prefix is byte-for-byte identical;
        divergent histories are rejected instead of silently choosing a winner.
        """
        grouped = self._validated_import_groups(rows)

        imported = 0
        unchanged = 0
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        with authority_write_lock(self.root):
            for device, incoming in grouped.items():
                new_count, unchanged_count = self._merge_import_chain(device, incoming)
                imported += new_count
                unchanged += unchanged_count
        return {
            "devices": len(grouped),
            "imported": imported,
            "unchanged": unchanged,
        }

    def validate_import_events(self, rows: list[dict[str, Any]]) -> None:
        """Validate an import against the current journal without writing it."""
        grouped = self._validated_import_groups(rows)
        with authority_write_lock(self.root):
            for device, incoming in grouped.items():
                existing = self.iter_events(device_id=device)
                self._assert_common_prefix(device, existing, incoming)

    def _validated_import_groups(
        self,
        rows: list[dict[str, Any]],
    ) -> dict[str, list[MemoEvent]]:
        grouped = self._group_import_events(rows)
        for device, events in grouped.items():
            self._validate_import_chain(device, events)
        return grouped

    def _group_import_events(self, rows: list[dict[str, Any]]) -> dict[str, list[MemoEvent]]:
        grouped: dict[str, list[MemoEvent]] = {}
        for raw in rows:
            event = self._decode_event(dict(raw))
            if not _DEVICE_ID_RE.fullmatch(event.device_id):
                raise LedgerIntegrityError(f"unsafe journal device id: {event.device_id!r}")
            grouped.setdefault(event.device_id, []).append(event)
        return grouped

    @staticmethod
    def _validate_import_chain(device: str, events: list[MemoEvent]) -> None:
        events.sort(key=lambda event: event.sequence)
        previous_hash = ""
        previous_dt: datetime | None = None
        for sequence, event in enumerate(events, start=1):
            if event.device_id != device:
                raise LedgerIntegrityError(f"foreign chain device mismatch: {event.device_id}")
            _, event_dt = _timestamp_parts(event.ts)
            if event.sequence != sequence:
                raise LedgerIntegrityError(
                    f"foreign chain {device} sequence {event.sequence} != {sequence}"
                )
            if event.previous_hash != previous_hash:
                raise LedgerIntegrityError(
                    f"foreign chain {device} previous hash mismatch at {sequence}"
                )
            calculated = replace(event, event_hash="").with_hash().event_hash
            if event.event_hash != calculated:
                raise LedgerIntegrityError(
                    f"foreign chain {device} event hash mismatch at {sequence}"
                )
            if previous_dt is not None and event_dt < previous_dt:
                raise LedgerIntegrityError(
                    f"foreign chain {device} timestamps decrease at {sequence}"
                )
            previous_hash = event.event_hash
            previous_dt = event_dt

    @staticmethod
    def _assert_common_prefix(
        device: str,
        existing: list[MemoEvent],
        incoming: list[MemoEvent],
    ) -> None:
        for index in range(min(len(existing), len(incoming))):
            if existing[index].event_hash != incoming[index].event_hash:
                raise LedgerIntegrityError(
                    f"foreign chain {device} diverges at sequence {index + 1}"
                )

    def _merge_import_chain(
        self,
        device: str,
        incoming: list[MemoEvent],
    ) -> tuple[int, int]:
        existing = self.iter_events(device_id=device)
        self._assert_common_prefix(device, existing, incoming)
        if len(existing) >= len(incoming):
            return 0, len(incoming)
        merged = [*existing, *incoming[len(existing) :]]
        self._write_import_chain(device, merged)
        return len(incoming) - len(existing), len(existing)

    def _write_import_chain(self, device: str, events: list[MemoEvent]) -> None:
        if not _DEVICE_ID_RE.fullmatch(device):
            raise LedgerIntegrityError(f"unsafe journal device id: {device!r}")
        by_day: dict[str, list[MemoEvent]] = {}
        for event in events:
            day, _ = _timestamp_parts(event.ts)
            by_day.setdefault(day, []).append(event)
        event_dir = self.root / "events" / device
        if event_dir.is_symlink():
            raise LedgerIntegrityError(f"unsafe journal device path: {event_dir}")
        event_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        for day, day_events in by_day.items():
            path = event_dir / f"{day}.jsonl"
            if path.is_symlink():
                raise LedgerIntegrityError(f"unsafe journal segment: {path}")
            atomic_write_text(
                path,
                "".join(
                    json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
                    for event in day_events
                ),
            )
        self._write_head(device, events[-1])

    def head_hashes(self) -> dict[str, str]:
        """Return strictly decoded advisory heads for every known device."""
        return {
            device: self._read_device_head(device, strict=True)[1] for device in self._device_ids()
        }

    def _read_device_head(self, device: str, *, strict: bool = False) -> tuple[int, str]:
        path = self.root / "heads" / f"{device}.json"
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return int(raw.get("sequence") or 0), str(raw.get("event_hash") or "")
        except FileNotFoundError:
            return 0, ""
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            if strict:
                raise LedgerIntegrityError(f"invalid journal head: {path}") from exc
            return 0, ""


__all__ = ["LedgerIntegrityError", "OperationLedger"]
