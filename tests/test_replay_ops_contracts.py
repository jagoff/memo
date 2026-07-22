"""Backend-native replay contracts for optional and dependency-free URI paths."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import memo.memory.replay_ops as replay_ops
from memo.errors import AmbiguousIdError
from memo.memory.record import MemoryRecord
from memo.memory.replay_ops import _ReplayOpsMixin


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


def _source(**overrides) -> dict:
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


def _parts(resource_type: str, resource_id: str = "", subpath: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        resource_type=resource_type,
        resource_id=resource_id,
        subpath=subpath,
    )


@pytest.fixture
def helper_harness(monkeypatch: pytest.MonkeyPatch) -> _Harness:
    monkeypatch.setattr(replay_ops, "_HAS_URI_HELPERS", True)
    monkeypatch.setattr(
        replay_ops,
        "is_memo_uri",
        lambda uri: uri.startswith("memo://"),
        raising=False,
    )
    harness = _Harness()
    harness.get = MagicMock(return_value=None)  # type: ignore[method-assign]
    harness.store = MagicMock()
    harness.store.get_repo_source.return_value = None
    harness.store.repo_counts.return_value = {
        "files": 1,
        "lines": 20,
        "chunks": 2,
        "embedded_chunks": 2,
    }
    return harness


def test_shared_uri_helpers_reject_foreign_and_malformed_uris(
    helper_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = MagicMock(return_value=None)
    monkeypatch.setattr(replay_ops, "parse_uri", parser, raising=False)

    foreign = helper_harness.backend_native_replay_resolve("memflow://kernel/focus")
    malformed = helper_harness.backend_native_replay_resolve("memo://bad")

    assert foreign["status"] == "unsupported"
    assert malformed["status"] == "missing"
    parser.assert_called_once_with("memo://bad")


@pytest.mark.parametrize("error_type", [TypeError, ValueError])
def test_shared_uri_parser_failure_is_returned_as_missing_payload(
    helper_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
) -> None:
    def _raise(_uri: str):
        raise error_type("invalid percent escape")

    monkeypatch.setattr(replay_ops, "parse_uri", _raise, raising=False)

    payload = helper_harness.backend_native_replay_resolve("memo://bad/%ZZ")

    assert payload["status"] == "missing"
    assert payload["content_hash"] == ""
    assert payload["resolution_mode"] == "backend_native"


def test_shared_uri_helpers_resolve_memory_with_complete_envelope(
    helper_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _record()
    helper_harness.get.return_value = record
    monkeypatch.setattr(
        replay_ops,
        "parse_uri",
        lambda _uri: _parts("memoria", record.id[:8]),
        raising=False,
    )

    payload = helper_harness.backend_native_replay_resolve(
        f"memo://memoria/{record.id[:8]}",
        trace_id="trace-1",
        backend_version="4.0.0",
    )

    assert payload["status"] == "found"
    assert payload["target"] == {"kind": "memoria", "id": record.id, "path": record.path}
    assert payload["content_hash"]
    assert payload["trace_id"] == "trace-1"
    assert payload["backend_version"] == "4.0.0"


@pytest.mark.parametrize(
    ("parts", "expected_detail"),
    [
        (_parts("memoria"), "did not include an id"),
        (_parts("repo"), "did not include a repo"),
        (_parts("repo-index"), "must include"),
    ],
)
def test_shared_uri_helpers_validate_required_identifiers(
    helper_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
    parts: SimpleNamespace,
    expected_detail: str,
) -> None:
    monkeypatch.setattr(replay_ops, "parse_uri", lambda _uri: parts, raising=False)

    payload = helper_harness.backend_native_replay_resolve("memo://incomplete")

    assert payload["status"] == "missing"
    assert expected_detail in payload["detail"]


def test_shared_uri_helpers_report_ambiguous_memory_prefix(
    helper_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper_harness.get.side_effect = AmbiguousIdError("abcd", ["abcd-1", "abcd-2"])
    monkeypatch.setattr(
        replay_ops,
        "parse_uri",
        lambda _uri: _parts("memoria", "abcd"),
        raising=False,
    )

    payload = helper_harness.backend_native_replay_resolve("memo://memoria/abcd")

    assert payload["status"] == "error"
    assert "2 matches" in payload["detail"]


def test_shared_uri_helpers_resolve_repo_index_from_parser_subpath(
    helper_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper_harness.store.get_repo_source.return_value = _source()
    monkeypatch.setattr(
        replay_ops,
        "parse_uri",
        lambda _uri: _parts("repo-index", subpath="memo/abcdef12"),
        raising=False,
    )

    payload = helper_harness.backend_native_replay_resolve("memo://repo-index/memo/abcdef12")

    assert payload["status"] == "found"
    assert payload["target"]["semantic_status"] == "semantic_ready"
    assert payload["target"]["counts"]["pending_chunks"] == 0
    helper_harness.store.get_repo_source.assert_called_once_with("memo")


def test_shared_uri_helpers_require_repo_index_commit_before_store_query(
    helper_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        replay_ops,
        "parse_uri",
        lambda _uri: _parts("repo-index", resource_id="memo", subpath=""),
        raising=False,
    )

    payload = helper_harness.backend_native_replay_resolve("memo://repo-index/memo")

    assert payload["status"] == "missing"
    assert "<repo-name>/<commit-prefix>" in payload["detail"]
    helper_harness.store.get_repo_source.assert_not_called()


def test_shared_uri_helpers_return_commit_mismatch_with_current_target(
    helper_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper_harness.store.get_repo_source.return_value = _source()
    monkeypatch.setattr(
        replay_ops,
        "parse_uri",
        lambda _uri: _parts("repo-index", "memo", "deadbeef"),
        raising=False,
    )

    payload = helper_harness.backend_native_replay_resolve("memo://repo-index/memo/deadbeef")

    assert payload["status"] == "missing"
    assert payload["target"] == {
        "kind": "repo_index",
        "repo_id": "repo-1",
        "name": "memo",
        "commit_sha": "abcdef1234567890",
    }


def test_shared_uri_helpers_resolve_repo_and_reject_unknown_resource(
    helper_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper_harness.store.get_repo_source.return_value = _source()
    monkeypatch.setattr(
        replay_ops,
        "parse_uri",
        lambda _uri: _parts("repo", "repo-1"),
        raising=False,
    )
    found = helper_harness.backend_native_replay_resolve("memo://repo/repo-1")

    monkeypatch.setattr(
        replay_ops,
        "parse_uri",
        lambda _uri: _parts("episode", "one"),
        raising=False,
    )
    unsupported = helper_harness.backend_native_replay_resolve("memo://episode/one")

    assert found["status"] == "found"
    assert found["target"]["name"] == "memo"
    assert unsupported["status"] == "unsupported"
    assert "episode" in unsupported["detail"]


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("memo://memoria/", "did not include an id"),
        ("memo://repo-index/memo", "must include"),
        ("memo://repo/", "did not include a repo"),
        ("https://example.test/not-memo", "only replays"),
    ],
)
def test_manual_uri_fallback_validates_malformed_inputs(
    monkeypatch: pytest.MonkeyPatch,
    uri: str,
    expected: str,
) -> None:
    monkeypatch.setattr(replay_ops, "_HAS_URI_HELPERS", False)
    harness = _Harness()
    harness.store = MagicMock()
    harness.get = MagicMock(return_value=None)  # type: ignore[method-assign]

    payload = harness.backend_native_replay_resolve(uri)

    assert payload["status"] in {"missing", "unsupported"}
    assert expected in payload["detail"]


def test_manual_uri_fallback_reports_ambiguous_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(replay_ops, "_HAS_URI_HELPERS", False)
    harness = _Harness()
    harness.store = MagicMock()
    harness.get = MagicMock(  # type: ignore[method-assign]
        side_effect=AmbiguousIdError("abcd", ["abcd-1", "abcd-2"])
    )

    payload = harness.backend_native_replay_resolve("memo://memoria/abcd")

    assert payload["status"] == "error"
    assert "ambiguous memory id prefix" in payload["detail"]


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


def test_repo_payload_never_reports_negative_pending_chunks() -> None:
    """Defensive replay output remains coherent if derived counts drift."""

    harness = _Harness()
    harness.store = MagicMock()
    harness.store.repo_counts.return_value = {
        "files": 1,
        "lines": 2,
        "chunks": 1,
        "embedded_chunks": 2,
    }

    payload = harness._repo_replay_payload(_source())

    assert payload["counts"]["pending_chunks"] == 0
    assert payload["semantic_status"] == "semantic_ready"
