"""Memo-native backend replay contracts."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from memo.errors import AmbiguousIdError
from memo.memory.record import MemoryRecord
from memo.memory.replay_ops import _parse_replay_uri, _ReplayOpsMixin


class _Harness(_ReplayOpsMixin):
    pass


def _record(id_: str = "a" * 32) -> MemoryRecord:
    return MemoryRecord(
        id=id_,
        path=f"{id_}.md",
        title="Replay target",
        type="note",
        tags=[],
        created="2026-01-01T00:00:00+00:00",
        updated="2026-01-01T00:00:00+00:00",
        body="durable content",
    )


def _source(**overrides: str) -> dict[str, str]:
    source = {
        "id": "repo-1",
        "name": "memo",
        "url": "https://example.test/memo.git",
        "ref": "main",
        "commit_sha": "abcdef1234567890",
        "indexed_at": "2026-07-01T00:00:00+00:00",
        "status": "ready",
    }
    source.update(overrides)
    return source


@pytest.fixture
def harness() -> _Harness:
    instance = _Harness()
    instance.get = MagicMock(return_value=None)  # type: ignore[method-assign]
    instance.store = MagicMock()
    instance.store.get_repo_source.return_value = None
    instance.store.repo_counts.return_value = {
        "files": 1,
        "lines": 20,
        "chunks": 2,
        "embedded_chunks": 2,
    }
    return instance


def test_native_uri_parser_preserves_repo_url_subpath() -> None:
    parts = _parse_replay_uri("memo://repo/https://example.test/org/memo.git")

    assert parts is not None
    assert parts.resource_type == "repo"
    assert parts.resource_id == "https:"
    assert parts.subpath == "/example.test/org/memo.git"


def test_native_uri_parser_rejects_foreign_uri_and_reports_unknown_type(
    harness: _Harness,
) -> None:
    foreign = harness.backend_native_replay_resolve("https://example.test/not-memo")
    unknown = harness.backend_native_replay_resolve("memo://episode/one")

    assert foreign["status"] == "unsupported"
    assert "only replays" in foreign["detail"]
    assert unknown["status"] == "unsupported"
    assert "episode" in unknown["detail"]


def test_native_replay_resolves_memory_with_complete_envelope(harness: _Harness) -> None:
    record = _record()
    harness.get.return_value = record

    payload = harness.backend_native_replay_resolve(
        f"memo://memoria/{record.id[:8]}",
        trace_id="trace-1",
        backend_version="4.0.0",
    )

    assert payload["status"] == "found"
    assert payload["target"] == {"kind": "memoria", "id": record.id, "path": record.path}
    assert payload["content_hash"]
    assert payload["trace_id"] == "trace-1"
    assert payload["backend_version"] == "4.0.0"
    harness.get.assert_called_once_with(record.id[:8])


@pytest.mark.parametrize(
    ("uri", "expected_detail"),
    [
        ("memo://memoria/", "did not include an id"),
        ("memo://repo/", "did not include a repo"),
        ("memo://repo-index/memo", "must include"),
    ],
)
def test_native_replay_validates_required_identifiers(
    harness: _Harness,
    uri: str,
    expected_detail: str,
) -> None:
    payload = harness.backend_native_replay_resolve(uri)

    assert payload["status"] == "missing"
    assert expected_detail in payload["detail"]
    harness.store.get_repo_source.assert_not_called()


def test_native_replay_reports_missing_and_ambiguous_memory(harness: _Harness) -> None:
    missing = harness.backend_native_replay_resolve("memo://memoria/abcd")
    harness.get.side_effect = AmbiguousIdError("abcd", ["abcd-1", "abcd-2"])
    ambiguous = harness.backend_native_replay_resolve("memo://memoria/abcd")

    assert missing["status"] == "missing"
    assert ambiguous["status"] == "error"
    assert "2 matches" in ambiguous["detail"]


def test_native_replay_resolves_repo_index(harness: _Harness) -> None:
    harness.store.get_repo_source.return_value = _source()

    payload = harness.backend_native_replay_resolve("memo://repo-index/memo/abcdef12")

    assert payload["status"] == "found"
    assert payload["target"]["semantic_status"] == "semantic_ready"
    assert payload["target"]["counts"]["pending_chunks"] == 0
    harness.store.get_repo_source.assert_called_once_with("memo")


def test_native_replay_accepts_unknown_repo_index_commit(harness: _Harness) -> None:
    harness.store.get_repo_source.return_value = _source(commit_sha="")

    payload = harness.backend_native_replay_resolve("memo://repo-index/memo/unknown")

    assert payload["status"] == "found"


def test_native_replay_returns_commit_mismatch_with_current_target(harness: _Harness) -> None:
    harness.store.get_repo_source.return_value = _source()

    payload = harness.backend_native_replay_resolve("memo://repo-index/memo/deadbeef")

    assert payload["status"] == "missing"
    assert payload["target"] == {
        "kind": "repo_index",
        "repo_id": "repo-1",
        "name": "memo",
        "commit_sha": "abcdef1234567890",
    }
    assert "replay URI" in payload["detail"]


def test_native_replay_reports_missing_repo_source(harness: _Harness) -> None:
    payload = harness.backend_native_replay_resolve("memo://repo/missing")

    assert payload["status"] == "missing"
    assert "repo source was not found" in payload["detail"]


def test_native_replay_resolves_repo_by_full_url(harness: _Harness) -> None:
    source = _source()
    harness.store.get_repo_source.return_value = source

    payload = harness.backend_native_replay_resolve("memo://repo/https://example.test/memo.git")

    assert payload["status"] == "found"
    assert payload["target"]["name"] == "memo"
    harness.store.get_repo_source.assert_called_once_with(source["url"])


def test_repo_payload_without_id_does_not_query_counts() -> None:
    harness = _Harness()
    harness.store = MagicMock()

    payload = harness._repo_replay_payload(_source(id="", status="empty"))

    harness.store.repo_counts.assert_not_called()
    assert payload["semantic_status"] == "empty"
    assert payload["counts"] == {
        "files": 0,
        "lines": 0,
        "chunks": 0,
        "embedded_chunks": 0,
        "pending_chunks": 0,
    }


def test_repo_payload_clamps_negative_pending_chunks(harness: _Harness) -> None:
    harness.store.repo_counts.return_value = {
        "files": 1,
        "lines": 20,
        "chunks": 2,
        "embedded_chunks": 3,
    }

    payload = harness._repo_replay_payload(_source())

    assert payload["semantic_status"] == "semantic_ready"
    assert payload["counts"]["pending_chunks"] == 0
