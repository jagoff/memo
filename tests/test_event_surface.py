from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from memo.errors import StorageError, ValidationError
from memo.event_surface import ingest_event, list_events


def _event(event_id: str, *, body: str = "hello") -> dict[str, str]:
    return {"event_id": event_id, "kind": "terminal", "body": body}


def test_concurrent_duplicate_terminal_event_is_appended_exactly_once(tmp_path: Path) -> None:
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(
                lambda _: ingest_event(_event("terminal-a-1"), state_dir=tmp_path),
                range(24),
            )
        )

    assert sum(result["accepted"] is True for result in results) == 1
    assert sum(result["duplicate"] is True for result in results) == 23
    assert [row["event_id"] for row in list_events(state_dir=tmp_path)] == ["terminal-a-1"]


def test_concurrent_distinct_terminal_events_all_survive(tmp_path: Path) -> None:
    event_ids = [f"terminal-{index}" for index in range(24)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(
            pool.map(lambda event_id: ingest_event(_event(event_id), state_dir=tmp_path), event_ids)
        )

    assert {row["event_id"] for row in list_events(state_dir=tmp_path)} == set(event_ids)


def test_terminal_event_idempotency_conflict_is_rejected(tmp_path: Path) -> None:
    ingest_event(_event("terminal-a-1"), state_dir=tmp_path)

    with pytest.raises(ValidationError, match="different payload"):
        ingest_event(_event("terminal-a-1", body="changed"), state_dir=tmp_path)


def test_terminal_event_corruption_fails_closed(tmp_path: Path) -> None:
    event_dir = tmp_path / "events"
    event_dir.mkdir()
    (event_dir / "terminal-conversation.jsonl").write_text("not-json\n", encoding="utf-8")

    with pytest.raises(StorageError, match="invalid JSON"):
        ingest_event(_event("terminal-a-1"), state_dir=tmp_path)


@pytest.mark.parametrize("encoded", ("not-json", "[]"))
def test_terminal_event_context_corruption_fails_closed(
    tmp_path: Path,
    encoded: str,
) -> None:
    event_dir = tmp_path / "events"
    event_dir.mkdir()
    (event_dir / "context.json").write_text(encoded, encoding="utf-8")

    with pytest.raises(StorageError, match="context"):
        ingest_event(
            _event("terminal-a-1"),
            state_dir=tmp_path,
            expected_epoch=0,
        )
