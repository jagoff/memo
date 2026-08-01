from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from memo.cli import cli
from memo.event_surface import (
    MAX_EVENT_SCAN_BYTES,
    ingest_event,
    list_event_page,
    list_events,
)


def _event_path(state_dir: Path) -> Path:
    return state_dir / "events" / "terminal-conversation.jsonl"


def _write_events(state_dir: Path, rows: list[object]) -> Path:
    path = _event_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"".join(json.dumps(row, sort_keys=True).encode("utf-8") + b"\n" for row in rows)
    )
    return path


def _env(state_dir: Path) -> dict[str, str]:
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(state_dir / "vault"),
        "MEMO_STATE_DIR": str(state_dir),
    }


def _cursor_offset(cursor: str) -> int:
    """Validate the public opaque-cursor envelope and return its byte offset."""
    version, raw_offset, digest, continuation = cursor.split(":")
    assert version == "v1"
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")
    assert continuation in {"0", "1"}
    return int(raw_offset)


def test_legacy_list_output_and_limit_zero_contract_are_unchanged(tmp_path: Path) -> None:
    rows = [
        {"event_id": "one", "kind": "terminal"},
        {"event_id": "two", "kind": "conversation"},
        {"event_id": "three", "kind": "terminal"},
    ]
    _write_events(tmp_path, rows)

    assert list_events(state_dir=tmp_path, limit=2) == rows[-2:]
    assert list_events(state_dir=tmp_path, limit=0) == rows

    result = CliRunner().invoke(
        cli,
        ["events", "list", "--limit", "0"],
        env=_env(tmp_path),
    )

    assert result.exit_code == 0
    assert result.output == json.dumps(rows, sort_keys=True) + "\n"


def test_cursor_enables_exact_paginated_envelope_and_physical_order(
    tmp_path: Path,
) -> None:
    rows = [
        {"event_id": "one", "kind": "terminal"},
        {"event_id": "skip", "kind": "conversation"},
        {"event_id": "two", "kind": "terminal"},
        {"event_id": "three", "kind": "terminal"},
    ]
    path = _write_events(tmp_path, rows)

    first = list_event_page(
        state_dir=tmp_path,
        cursor="",
        kind="terminal",
        limit=2,
    )
    second = list_event_page(
        state_dir=tmp_path,
        cursor=first["next_cursor"],
        kind="terminal",
        limit=2,
    )

    assert set(first) == {"events", "next_cursor", "has_more"}
    assert first["events"] == [rows[0], rows[2]]
    assert first["has_more"] is True
    assert second["events"] == [rows[3]]
    assert second["has_more"] is False
    assert _cursor_offset(second["next_cursor"]) == path.stat().st_size


def test_eof_cursor_resumes_only_new_appends_without_rereading_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = {"event_id": "old", "kind": "terminal"}
    path = _write_events(tmp_path, [original])
    at_eof = list_event_page(state_dir=tmp_path, cursor="0")
    assert _cursor_offset(at_eof["next_cursor"]) == path.stat().st_size

    appended = {"event_id": "new", "kind": "terminal"}
    with path.open("ab") as fh:
        fh.write(json.dumps(appended).encode("utf-8") + b"\n")

    real_loads = json.loads

    def reject_historical_prefix(value: str | bytes | bytearray, *args: Any, **kwargs: Any) -> Any:
        text = bytes(value).decode("utf-8") if isinstance(value, (bytes, bytearray)) else value
        if '"old"' in text:
            raise AssertionError("the historical prefix was read again")
        return real_loads(value, *args, **kwargs)

    monkeypatch.setattr("memo.event_surface.json.loads", reject_historical_prefix)
    resumed = list_event_page(
        state_dir=tmp_path,
        cursor=at_eof["next_cursor"],
    )

    assert resumed["events"] == [appended]
    assert resumed["has_more"] is False
    assert _cursor_offset(resumed["next_cursor"]) == path.stat().st_size


def test_page_snapshot_defers_concurrent_append_to_next_poll(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initial = {"event_id": "initial", "kind": "terminal"}
    appended = {"event_id": "concurrent", "kind": "terminal"}
    path = _write_events(tmp_path, [initial])
    initial_size = path.stat().st_size
    real_loads = json.loads
    appended_once = False

    def append_during_decode(value: str | bytes | bytearray, *args: Any, **kwargs: Any) -> Any:
        nonlocal appended_once
        if not appended_once:
            appended_once = True
            with path.open("ab") as fh:
                fh.write(json.dumps(appended).encode("utf-8") + b"\n")
        return real_loads(value, *args, **kwargs)

    monkeypatch.setattr("memo.event_surface.json.loads", append_during_decode)
    first = list_event_page(state_dir=tmp_path, cursor="0")
    second = list_event_page(state_dir=tmp_path, cursor=first["next_cursor"])

    assert first["events"] == [initial]
    assert first["has_more"] is False
    assert _cursor_offset(first["next_cursor"]) == initial_size
    assert second["events"] == [appended]
    assert second["has_more"] is False


def test_partial_line_at_snapshot_is_replayed_after_writer_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = {"event_id": "first", "kind": "terminal"}
    second = {"event_id": "second", "kind": "terminal", "payload": "completed later"}
    first_line = json.dumps(first).encode("utf-8") + b"\n"
    second_line = json.dumps(second).encode("utf-8") + b"\n"
    split = len(second_line) // 2
    path = _event_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(first_line + second_line[:split])
    real_loads = json.loads
    completed = False

    def complete_during_decode(value: str | bytes | bytearray, *args: Any, **kwargs: Any) -> Any:
        nonlocal completed
        if not completed:
            completed = True
            with path.open("ab") as fh:
                fh.write(second_line[split:])
        return real_loads(value, *args, **kwargs)

    monkeypatch.setattr("memo.event_surface.json.loads", complete_during_decode)
    page_one = list_event_page(state_dir=tmp_path, cursor="0")
    page_two = list_event_page(
        state_dir=tmp_path,
        cursor=page_one["next_cursor"],
    )

    assert page_one["events"] == [first]
    assert page_one["has_more"] is False
    assert _cursor_offset(page_one["next_cursor"]) == len(first_line)
    assert page_two["events"] == [second]
    assert page_two["has_more"] is False
    assert _cursor_offset(page_two["next_cursor"]) == path.stat().st_size


def test_malformed_lines_advance_cursor_and_since_is_fail_safe(tmp_path: Path) -> None:
    rows = [
        {"event_id": "old", "kind": "terminal", "timestamp": "2026-07-01T00:00:00Z"},
        {"event_id": "new", "kind": "terminal", "timestamp": "2026-07-30T00:00:00Z"},
        {"event_id": "unknown", "kind": "terminal", "timestamp": "not-a-date"},
    ]
    path = _event_path(tmp_path)
    path.parent.mkdir(parents=True)
    with path.open("wb") as fh:
        fh.write(json.dumps(rows[0]).encode("utf-8") + b"\n")
        fh.write(b"{malformed json\n")
        fh.write(b"\xff\xfe invalid utf8\n")
        for row in rows[1:]:
            fh.write(json.dumps(row).encode("utf-8") + b"\n")

    page = list_event_page(
        state_dir=tmp_path,
        cursor="0",
        since="2026-07-15T00:00:00+00:00",
    )
    eof = list_event_page(state_dir=tmp_path, cursor=page["next_cursor"])

    assert page["events"] == rows[1:]
    assert page["has_more"] is False
    assert _cursor_offset(page["next_cursor"]) == path.stat().st_size
    assert eof["events"] == []
    assert eof["has_more"] is False
    assert _cursor_offset(eof["next_cursor"]) == path.stat().st_size


def test_cursor_past_eof_recovers_after_truncation(tmp_path: Path) -> None:
    replacement = {"event_id": "replacement", "kind": "terminal"}
    path = _write_events(
        tmp_path,
        [{"event_id": f"old-{index}", "kind": "terminal"} for index in range(20)],
    )
    old_eof = path.stat().st_size
    _write_events(tmp_path, [replacement])
    assert old_eof > path.stat().st_size

    page = list_event_page(state_dir=tmp_path, cursor=str(old_eof))

    assert page["events"] == [replacement]
    assert page["has_more"] is False
    assert _cursor_offset(page["next_cursor"]) == path.stat().st_size


def test_cursor_inside_rotated_regrown_file_restarts_from_zero(tmp_path: Path) -> None:
    path = _write_events(
        tmp_path,
        [{"event_id": f"old-{index}", "kind": "terminal"} for index in range(10)],
    )
    old_page = list_event_page(
        state_dir=tmp_path,
        cursor="0",
        limit=3,
    )
    old_cursor = _cursor_offset(old_page["next_cursor"])
    replacement = {
        "event_id": "replacement",
        "kind": "terminal",
        "payload": "x" * (old_cursor + 100),
    }
    _write_events(tmp_path, [replacement])
    assert old_cursor < path.stat().st_size
    assert path.read_bytes()[old_cursor - 1 : old_cursor] != b"\n"

    recovered = list_event_page(
        state_dir=tmp_path,
        cursor=str(old_cursor),
    )

    assert recovered["events"] == [replacement]
    assert recovered["has_more"] is False
    assert _cursor_offset(recovered["next_cursor"]) == path.stat().st_size


def test_opaque_cursor_fingerprint_detects_same_boundary_replacement(tmp_path: Path) -> None:
    path = _write_events(
        tmp_path,
        [{"event_id": f"old-{index}", "kind": "terminal"} for index in range(10)],
    )
    old_page = list_event_page(state_dir=tmp_path, cursor="0", limit=3)
    old_cursor = old_page["next_cursor"]
    old_offset = _cursor_offset(old_cursor)

    replacement = {"event_id": "replacement-prefix", "kind": "terminal", "payload": ""}
    empty_line = json.dumps(replacement, sort_keys=True).encode("utf-8") + b"\n"
    replacement["payload"] = "x" * (old_offset - len(empty_line))
    final = {"event_id": "replacement-final", "kind": "terminal"}
    _write_events(tmp_path, [replacement, final])
    assert path.read_bytes()[old_offset - 1 : old_offset] == b"\n"

    recovered = list_event_page(state_dir=tmp_path, cursor=old_cursor)

    assert recovered["events"] == [replacement, final]
    assert recovered["has_more"] is False
    assert _cursor_offset(recovered["next_cursor"]) == path.stat().st_size


def test_more_than_one_hundred_thousand_events_drain_in_bounded_pages(
    tmp_path: Path,
) -> None:
    total = 100_003
    path = _event_path(tmp_path)
    path.parent.mkdir(parents=True)
    with path.open("wb") as fh:
        for index in range(total):
            fh.write(
                json.dumps(
                    {"event_id": f"event-{index:06d}", "kind": "terminal"},
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )

    cursor = "0"
    count = 0
    pages = 0
    first_id = ""
    last_id = ""
    while True:
        page = list_event_page(
            state_dir=tmp_path,
            cursor=cursor,
            kind="terminal",
            limit=1_000,
        )
        events = page["events"]
        assert len(events) <= 1_000
        if events:
            first_id = first_id or str(events[0]["event_id"])
            last_id = str(events[-1]["event_id"])
        count += len(events)
        pages += 1
        previous = cursor
        cursor = page["next_cursor"]
        if not page["has_more"]:
            break
        assert cursor != previous

    assert count == total
    assert pages == 101
    assert first_id == "event-000000"
    assert last_id == "event-100002"
    assert _cursor_offset(cursor) == path.stat().st_size


def test_filtered_empty_pages_advance_with_bounded_physical_scan(
    tmp_path: Path,
) -> None:
    rows = [{"event_id": f"terminal-{index}", "kind": "terminal"} for index in range(1_500)]
    path = _write_events(tmp_path, rows)

    first = list_event_page(
        state_dir=tmp_path,
        cursor="0",
        kind="signal",
        limit=1_000,
    )
    second = list_event_page(
        state_dir=tmp_path,
        cursor=first["next_cursor"],
        kind="signal",
        limit=1_000,
    )

    assert first["events"] == []
    assert first["has_more"] is True
    assert first["next_cursor"] != "0"
    assert second["events"] == []
    assert second["has_more"] is False
    assert _cursor_offset(second["next_cursor"]) == path.stat().st_size


def test_filtered_hundred_thousand_line_backlog_advances_in_empty_pages(
    tmp_path: Path,
) -> None:
    total = 100_003
    path = _event_path(tmp_path)
    path.parent.mkdir(parents=True)
    with path.open("wb") as fh:
        for index in range(total):
            fh.write(
                json.dumps(
                    {
                        "event_id": f"conversation-{index:06d}",
                        "kind": "conversation",
                        "timestamp": "2026-01-01T00:00:00Z",
                    },
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )

    for query in (
        {"kind": "terminal"},
        {"kind": "conversation", "since": "2026-07-01T00:00:00Z"},
    ):
        cursor = "0"
        pages = 0
        while True:
            page = list_event_page(
                state_dir=tmp_path,
                cursor=cursor,
                limit=1_000,
                **query,
            )
            assert page["events"] == []
            pages += 1
            previous = cursor
            cursor = page["next_cursor"]
            if not page["has_more"]:
                break
            assert cursor != previous

        assert pages == 101
        assert _cursor_offset(cursor) == path.stat().st_size


def test_page_byte_budget_stops_after_current_physical_line(tmp_path: Path) -> None:
    payload = "x" * (1024 * 1024)
    path = _write_events(
        tmp_path,
        [
            {"event_id": f"large-{index}", "kind": "conversation", "payload": payload}
            for index in range(6)
        ],
    )

    page = list_event_page(
        state_dir=tmp_path,
        cursor="0",
        kind="terminal",
        limit=1_000,
    )

    assert page["events"] == []
    assert page["has_more"] is True
    next_offset = _cursor_offset(page["next_cursor"])
    assert 0 < next_offset < path.stat().st_size
    assert next_offset <= MAX_EVENT_SCAN_BYTES + len(payload) + 200


@pytest.mark.parametrize("limit", [0, 1_001])
def test_paginated_limit_is_bounded(tmp_path: Path, limit: int) -> None:
    with pytest.raises(ValueError, match="limit must be between 1 and 1000"):
        list_event_page(state_dir=tmp_path, cursor="0", limit=limit)


def test_cursor_rejects_impossible_continuation_at_origin(tmp_path: Path) -> None:
    digest = "0" * 64

    with pytest.raises(ValueError, match="cursor is invalid"):
        list_event_page(state_dir=tmp_path, cursor=f"v1:0:{digest}:1")


def test_cli_cursor_accepts_empty_origin_and_since_requires_cursor(tmp_path: Path) -> None:
    row = {"event_id": "one", "kind": "terminal", "created_at": "2026-07-30T00:00:00Z"}
    _write_events(tmp_path, [row])
    runner = CliRunner()

    paginated = runner.invoke(
        cli,
        ["events", "list", "--cursor", "", "--since", "2026-07-01T00:00:00Z"],
        env=_env(tmp_path),
    )
    unpaginated_since = runner.invoke(
        cli,
        ["events", "list", "--since", "2026-07-01T00:00:00Z"],
        env=_env(tmp_path),
    )

    assert paginated.exit_code == 0
    assert json.loads(paginated.output)["events"] == [row]
    assert unpaginated_since.exit_code == 2
    assert "--since requires --cursor" in unpaginated_since.output


def test_ingest_reconciles_legacy_prefix_once_then_reads_only_delta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    total = 100_001
    path = _event_path(tmp_path)
    path.parent.mkdir(parents=True)
    with path.open("wb") as fh:
        for index in range(total):
            fh.write(
                json.dumps(
                    {
                        "event_id": f"legacy-{index:06d}",
                        "kind": "terminal",
                        "schema": "memo.terminal_event.v1",
                    },
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )

    first = ingest_event(
        {"event_id": "indexed-new", "kind": "terminal"},
        state_dir=tmp_path,
    )
    assert first["accepted"] is True

    real_loads = json.loads

    def reject_legacy_prefix(value: str | bytes | bytearray, *args: Any, **kwargs: Any) -> Any:
        text = bytes(value).decode("utf-8") if isinstance(value, (bytes, bytearray)) else value
        if '"legacy-' in text:
            raise AssertionError("steady-state ingest reread the legacy prefix")
        return real_loads(value, *args, **kwargs)

    monkeypatch.setattr("memo.event_surface.json.loads", reject_legacy_prefix)
    second = ingest_event(
        {"event_id": "steady-new", "kind": "terminal"},
        state_dir=tmp_path,
    )

    assert second["accepted"] is True


def test_concurrent_same_id_ingest_appends_exactly_once(tmp_path: Path) -> None:
    event = {"event_id": "same-id", "kind": "signal", "payload": {"epoch": 7}}

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(
                lambda _: ingest_event(event, state_dir=tmp_path),
                range(16),
            )
        )

    assert sum(result["accepted"] is True for result in results) == 1
    assert sum(result["duplicate"] is True for result in results) == 15
    rows = [
        json.loads(line) for line in _event_path(tmp_path).read_text(encoding="utf-8").splitlines()
    ]
    assert [row["event_id"] for row in rows] == ["same-id"]


def test_retry_repairs_append_when_receipt_commit_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import memo.event_surface as surface

    event = {"event_id": "crash-window", "kind": "signal", "payload": {"epoch": 8}}
    real_record_receipt = surface._record_receipt
    failed = False

    def fail_once(*args: Any, **kwargs: Any) -> None:
        nonlocal failed
        record = args[1]
        if record["event_id"] == event["event_id"] and not failed:
            failed = True
            raise OSError("simulated receipt failure after append")
        real_record_receipt(*args, **kwargs)

    monkeypatch.setattr(surface, "_record_receipt", fail_once)
    with pytest.raises(OSError, match="simulated receipt failure"):
        ingest_event(event, state_dir=tmp_path)

    monkeypatch.setattr(surface, "_record_receipt", real_record_receipt)
    retried = ingest_event(event, state_dir=tmp_path)
    rows = [
        json.loads(line) for line in _event_path(tmp_path).read_text(encoding="utf-8").splitlines()
    ]

    assert retried["accepted"] is False
    assert retried["duplicate"] is True
    assert [row["event_id"] for row in rows] == ["crash-window"]


def test_index_connection_closes_when_schema_initialization_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import memo.event_surface as surface

    class BrokenConnection:
        closed = False

        def execute(self, _sql: str) -> None:
            raise sqlite3.DatabaseError("simulated schema failure")

        def close(self) -> None:
            self.closed = True

    conn = BrokenConnection()
    monkeypatch.setattr(surface.sqlite3, "connect", lambda *_args, **_kwargs: conn)
    data_path = _event_path(tmp_path)
    data_path.parent.mkdir(parents=True)

    with pytest.raises(sqlite3.DatabaseError, match="simulated schema failure"):
        surface._open_index(data_path)

    assert conn.closed is True


def test_ingest_refuses_symlinked_journal(tmp_path: Path) -> None:
    target = tmp_path / "must-not-change.jsonl"
    target.write_text("sentinel\n", encoding="utf-8")
    journal = _event_path(tmp_path)
    journal.parent.mkdir(parents=True)
    journal.symlink_to(target)

    with pytest.raises(ValueError, match="unsafe event-journal path"):
        ingest_event(
            {"event_id": "unsafe-link", "kind": "terminal"},
            state_dir=tmp_path,
        )

    assert target.read_text(encoding="utf-8") == "sentinel\n"


def test_event_readers_refuse_symlinked_journal(tmp_path: Path) -> None:
    target = tmp_path / "private.jsonl"
    target.write_text('{"event_id":"secret","kind":"terminal"}\n', encoding="utf-8")
    journal = _event_path(tmp_path)
    journal.parent.mkdir(parents=True)
    journal.symlink_to(target)

    with pytest.raises(ValueError, match="unsafe event-journal path"):
        list_events(state_dir=tmp_path)
    with pytest.raises(ValueError, match="unsafe event-journal path"):
        list_event_page(state_dir=tmp_path, cursor="0")


def test_ingest_refuses_symlinked_lock_file(tmp_path: Path) -> None:
    target = tmp_path / "must-not-lock.txt"
    target.write_text("sentinel\n", encoding="utf-8")
    lock_path = tmp_path / "events" / "terminal-conversation.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.symlink_to(target)

    with pytest.raises(ValueError, match="unsafe event-lock path"):
        ingest_event(
            {"event_id": "unsafe-lock", "kind": "terminal"},
            state_dir=tmp_path,
        )

    assert target.read_text(encoding="utf-8") == "sentinel\n"


@pytest.mark.parametrize("payload", ["not-json", "[]", '{"epoch":"zero"}'])
def test_ingest_fails_closed_on_invalid_context(tmp_path: Path, payload: str) -> None:
    context_path = tmp_path / "events" / "context.json"
    context_path.parent.mkdir(parents=True)
    context_path.write_text(payload, encoding="utf-8")

    with pytest.raises(RuntimeError, match="invalid event context"):
        ingest_event(
            {"event_id": "invalid-context", "kind": "terminal"},
            state_dir=tmp_path,
            expected_epoch=0,
        )

    assert not _event_path(tmp_path).exists()


def test_ingest_rejects_non_string_kind(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="kind must be"):
        ingest_event(
            {"event_id": "bad-kind", "kind": {"nested": True}},
            state_dir=tmp_path,
        )


def test_index_scan_discards_oversized_legacy_line_in_bounded_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import memo.event_surface as surface

    monkeypatch.setattr(surface, "MAX_EVENT_RECORD_BYTES", 256)
    valid = {"event_id": "legacy-valid", "kind": "terminal"}
    path = _event_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x" * 300 + b"\n" + json.dumps(valid).encode() + b"\n")

    result = ingest_event(
        {"event_id": "new-valid", "kind": "terminal"},
        state_dir=tmp_path,
    )

    assert result["accepted"] is True
    index_path = path.with_name("terminal-conversation-index.sqlite3")
    with closing(sqlite3.connect(index_path)) as conn:
        event_ids = {
            str(row[0]) for row in conn.execute("SELECT event_id FROM event_receipts").fetchall()
        }
    assert event_ids == {"legacy-valid", "new-valid"}


def test_index_scan_truncates_oversized_incomplete_tail_before_append(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import memo.event_surface as surface

    monkeypatch.setattr(surface, "MAX_EVENT_RECORD_BYTES", 256)
    path = _event_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x" * 300)

    ingest_event(
        {"event_id": "after-tail", "kind": "terminal"},
        state_dir=tmp_path,
    )

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["event_id"] for row in rows] == ["after-tail"]
