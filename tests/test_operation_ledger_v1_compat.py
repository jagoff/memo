from __future__ import annotations

from memo.contracts import ActorIdentity
from memo.operation_ledger import OperationLedger
from memo.operation_ledger_v1 import LegacyOperationLedger


def test_frozen_v1_append_is_byte_compatible(tmp_path) -> None:
    kwargs = {
        "subject_uri": "memo://focus/demo",
        "actor": ActorIdentity(
            actor_id="agent-a",
            actor_kind="agent",
            signature="sig",
            source_client="codex",
        ),
        "trace_id": "trace-1",
        "payload": {"summary": "café", "project": "demo"},
        "content_hash": "content",
        "event_id": "event-1",
        "ts": "2026-07-29T12:00:00+00:00",
    }
    active_root = tmp_path / "active"
    frozen_root = tmp_path / "frozen"

    OperationLedger(active_root, device_id="device-a").append("focus.set", **kwargs)
    LegacyOperationLedger(frozen_root, device_id="device-a").append("focus.set", **kwargs)

    active_segment = active_root / "journal/events/device-a/2026-07-29.jsonl"
    frozen_segment = frozen_root / "journal/events/device-a/2026-07-29.jsonl"
    assert frozen_segment.read_bytes() == active_segment.read_bytes()
    assert (
        frozen_root / "journal/heads/device-a.json"
    ).read_bytes() == (active_root / "journal/heads/device-a.json").read_bytes()


def test_v1_reader_preserves_bytes_and_head(tmp_path) -> None:
    ledger = OperationLedger(tmp_path, device_id="device-a")
    ledger.append(
        "focus.set",
        subject_uri="memo://focus/demo",
        event_id="event-1",
        ts="2026-07-29T12:00:00Z",
        payload={"project": "demo"},
    )
    path = tmp_path / "journal/events/device-a/2026-07-29.jsonl"
    before = path.read_bytes()

    frozen = LegacyOperationLedger(tmp_path, device_id="device-a")

    assert frozen.verify()["ok"] is True
    assert frozen.head_hashes()["device-a"]
    assert path.read_bytes() == before
