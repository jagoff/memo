from __future__ import annotations

import json

from click.testing import CliRunner

from memo.cli import cli


def test_backend_native_capabilities_cli_json(monkeypatch) -> None:
    monkeypatch.setenv("SYNAPSE_TRACE_ID", "synapse://trace/test")

    result = CliRunner().invoke(
        cli,
        ["backend-native", "capabilities", "--json"],
        env={
            "MEMO_NONINTERACTIVE": "1",
            "MEMO_MODEL_PROFILE": "balanced",
            "MEMO_EMBEDDER_DIMS": "",
        },
    )
    payload = json.loads(result.output)

    assert result.exit_code == 0
    assert payload["schema"] == "synapse.backend_native.v1"
    assert payload["protocol_version"] == "backend_native.v1"
    assert payload["backend"] == "memo"
    assert payload["capabilities"]["replay_resolve"] is True
    assert payload["capabilities"]["memory_replay"] is True
    assert payload["capabilities"]["operational_replay"] is True
    assert payload["trace_id"] == "synapse://trace/test"
    assert payload["model_profile"]["schema"] == "memo.profile_status.v1"
    assert payload["model_profile"]["active"]["embedder_dims"] == 1024


def test_backend_native_replay_resolves_memoria(mock_memory) -> None:
    rec = mock_memory.save(content="Native replay content.", title="Native Replay")

    payload = mock_memory.backend_native_replay_resolve(
        f"memo://memoria/{rec.id[:8]}",
        trace_id="synapse://trace/replay",
        backend_version="test",
    )

    assert payload["schema"] == "synapse.backend_native.v1"
    assert payload["protocol_version"] == "backend_native.v1"
    assert payload["backend"] == "memo"
    assert payload["uri"] == f"memo://memoria/{rec.id[:8]}"
    assert payload["status"] == "found"
    assert payload["resolution_mode"] == "backend_native"
    assert payload["trace_id"] == "synapse://trace/replay"
    assert payload["content_hash"]


def test_backend_native_replay_missing_memoria(mock_memory) -> None:
    payload = mock_memory.backend_native_replay_resolve("memo://memoria/does-not-exist")

    assert payload["status"] == "missing"
    assert payload["content_hash"] == ""


def test_backend_native_replay_resolves_repo_index(mock_memory) -> None:
    mock_memory.store.upsert_repo_index(
        source={
            "id": "repo1",
            "name": "sample",
            "url": "https://example.test/sample.git",
            "ref": "HEAD",
            "commit_sha": "abcdef1234567890",
            "clone_path": "/tmp/sample",
            "indexed_at": "2026-05-24T00:00:00Z",
            "status": "semantic_pending",
            "extra": {},
        },
        files=[
            {
                "id": "file1",
                "path": "README.md",
                "language": "markdown",
                "size_bytes": 12,
                "sha256": "sha",
                "line_count": 1,
                "lines": [
                    {"id": "line1", "line_no": 1, "text": "hello", "text_hash": "l1"},
                ],
                "chunks": [
                    {
                        "id": "chunk1",
                        "chunk_seq": 0,
                        "line_start": 1,
                        "line_end": 1,
                        "text_hash": "c1",
                        "body_text": "hello",
                    },
                ],
            },
        ],
    )

    payload = mock_memory.backend_native_replay_resolve("memo://repo-index/sample/abcdef123456")

    assert payload["status"] == "found"
    assert payload["content_hash"]
    assert payload["target"]["kind"] == "repo"
    assert payload["target"]["id"] == "repo1"
    assert payload["target"]["counts"]["files"] == 1
    assert payload["target"]["counts"]["pending_chunks"] == 1


def test_backend_native_replay_resolves_repo_uri(mock_memory) -> None:
    mock_memory.store.upsert_repo_index(
        source={
            "id": "repo1",
            "name": "sample",
            "url": "https://example.test/sample.git",
            "ref": "HEAD",
            "commit_sha": "abcdef1234567890",
            "clone_path": "/tmp/sample",
            "indexed_at": "2026-05-24T00:00:00Z",
            "status": "ready",
            "extra": {},
        },
        files=[],
    )

    payload = mock_memory.backend_native_replay_resolve("memo://repo/repo1")

    assert payload["status"] == "found"
    assert payload["target"]["name"] == "sample"
    assert payload["target"]["commit_sha"] == "abcdef1234567890"


def test_backend_native_replay_missing_repo_index_commit(mock_memory) -> None:
    mock_memory.store.upsert_repo_index(
        source={
            "id": "repo1",
            "name": "sample",
            "url": "https://example.test/sample.git",
            "ref": "HEAD",
            "commit_sha": "abcdef1234567890",
            "clone_path": "/tmp/sample",
            "indexed_at": "2026-05-24T00:00:00Z",
            "status": "ready",
            "extra": {},
        },
        files=[],
    )

    payload = mock_memory.backend_native_replay_resolve("memo://repo-index/sample/deadbeef")

    assert payload["status"] == "missing"
    assert payload["target"]["commit_sha"] == "abcdef1234567890"


def test_backend_native_replay_rejects_foreign_uri(mock_memory) -> None:
    payload = mock_memory.backend_native_replay_resolve("memflow://kernel/current-focus")

    assert payload["status"] == "unsupported"
    assert payload["backend"] == "memo"
