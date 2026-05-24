from __future__ import annotations

import json

from click.testing import CliRunner

from memo.cli import cli


def test_backend_native_capabilities_cli_json(monkeypatch) -> None:
    monkeypatch.setenv("SYNAPSE_TRACE_ID", "synapse://trace/test")

    result = CliRunner().invoke(
        cli,
        ["backend-native", "capabilities", "--json"],
        env={"MEMO_NONINTERACTIVE": "1"},
    )
    payload = json.loads(result.output)

    assert result.exit_code == 0
    assert payload["schema"] == "synapse.backend_native.v1"
    assert payload["protocol_version"] == "backend_native.v1"
    assert payload["backend"] == "memo"
    assert payload["capabilities"]["replay_resolve"] is True
    assert payload["trace_id"] == "synapse://trace/test"


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


def test_backend_native_replay_rejects_foreign_uri(mock_memory) -> None:
    payload = mock_memory.backend_native_replay_resolve("memflow://kernel/current-focus")

    assert payload["status"] == "unsupported"
    assert payload["backend"] == "memo"
