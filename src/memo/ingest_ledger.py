"""Append-only, hash-chained ingest job ledger."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

from memo.util import utc_now_iso


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


class IngestFailureLedger:
    """Persistent operational telemetry without retaining job payloads."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        events, _errors = self.read()
        self._sequence = len(events)
        self._last_hash = str(events[-1]["event_hash"]) if events else ""

    def append(self, event: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            body = {
                "schema": "memo.ingest_job_event.v1",
                "sequence": self._sequence + 1,
                "recorded_at": utc_now_iso(),
                "previous_hash": self._last_hash,
                **event,
            }
            event_hash = hashlib.sha256(_canonical(body)).hexdigest()
            row = {**body, "event_hash": event_hash}
            encoded = _canonical(row) + b"\n"
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(
                self.path,
                os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                0o600,
            )
            try:
                os.write(descriptor, encoded)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self._sequence += 1
            self._last_hash = event_hash
            return row

    def read(self) -> tuple[list[dict[str, Any]], list[str]]:
        if not self.path.exists():
            return [], []
        events: list[dict[str, Any]] = []
        errors: list[str] = []
        previous = ""
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            return [], [f"{type(exc).__name__}: {exc}"]
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON ({exc})")
                continue
            if not isinstance(event, dict):
                errors.append(f"line {line_number}: event is not an object")
                continue
            actual_hash = str(event.get("event_hash") or "")
            body = {key: value for key, value in event.items() if key != "event_hash"}
            expected_hash = hashlib.sha256(_canonical(body)).hexdigest()
            if actual_hash != expected_hash:
                errors.append(f"line {line_number}: event hash mismatch")
                continue
            if str(event.get("previous_hash") or "") != previous:
                errors.append(f"line {line_number}: previous hash mismatch")
                continue
            events.append(event)
            previous = actual_hash
        return events, errors

    def fatal_failure_counts(self) -> dict[str, int]:
        events, _errors = self.read()
        counts: dict[str, int] = {}
        for event in events:
            if event.get("event") != "error" or not event.get("fatal"):
                continue
            fingerprint = str(event.get("fingerprint") or "")
            if fingerprint:
                counts[fingerprint] = counts.get(fingerprint, 0) + 1
        return counts

    def health(self) -> dict[str, Any]:
        events, errors = self.read()
        return {
            "schema": "memo.ingest_ledger_health.v1",
            "path": str(self.path),
            "events": len(events),
            "chain_valid": not errors,
            "errors": errors,
        }


__all__ = ["IngestFailureLedger"]
