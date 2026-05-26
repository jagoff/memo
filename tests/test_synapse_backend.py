"""Tests for the Synapse backend adapter + provenance round-trip.

Coverage:

- `Memory.save(extra={synapse_*})` persists provenance into
  `meta.extra_json` AND `history.events.delta_json`.
- `Memory.provenance(id)` returns the current state + per-op trail.
- `Memory.update()` records provenance churn in the delta log.
- `MemoSynapseBackend.{health,collect,remember}` implements the
  contract synapse expects, including the kind→memo_type coercion
  and the "synapse" tag injection.
- MCP `memory_save(extra=...)` + `memory_provenance(id)` round-trip
  via FastMCP's in-process tool harness.
- CLI `memo save --meta KEY=VALUE` and `memo provenance ID --json`
  work end-to-end.

All tests use the shared `mock_memory` / `tmp_cfg` fixtures so no
real MLX or vault is touched.
"""

from __future__ import annotations

import json
import os

import pytest
from click.testing import CliRunner

from memo.cli import cli
from memo.memory import _PROVENANCE_KEYS, Memory
from memo.synapse_backend import MemoSynapseBackend

# -- Memory API: provenance plumbing ---------------------------------------


def _sample_provenance() -> dict[str, str]:
    return {
        "synapse_trace_id": "trace-abc-123",
        "synapse_route_reason": "deep_semantic",
        "synapse_write_policy_schema": "synapse.write_policy.v1",
        "synapse_write_target": "memo",
        "synapse_agent_id": "claude-4-7",
        "synapse_agent_signature": "sig-deadbeef",
    }


def test_save_persists_provenance_in_extra_and_history(mock_memory):
    prov = _sample_provenance()
    extra = {**prov, "non_prov_key": "kept"}
    rec = mock_memory.save(content="hello world", title="t1", extra=extra)

    fetched = mock_memory.get(rec.id)
    assert fetched is not None
    for key, value in prov.items():
        assert fetched.extra[key] == value
    assert fetched.extra["non_prov_key"] == "kept"

    events = mock_memory.history.list_recent(limit=10, record_id=rec.id)
    assert len(events) == 1
    save_event = events[0]
    assert save_event["op"] == "save"
    assert save_event["delta"] is not None
    assert save_event["delta"]["_provenance"] == prov


def test_provenance_returns_current_and_events(mock_memory):
    prov_v1 = _sample_provenance()
    rec = mock_memory.save(content="v1 body", title="t1", extra=prov_v1)

    prov_v2 = dict(prov_v1)
    prov_v2["synapse_trace_id"] = "trace-second"
    mock_memory.update(rec.id, content="v2 body", extra=prov_v2)

    payload = mock_memory.provenance(rec.id)
    assert payload is not None
    assert payload["id"] == rec.id
    assert payload["current"] == prov_v2
    assert len(payload["events"]) == 2
    save_ev, update_ev = payload["events"]
    assert save_ev["op"] == "save"
    assert save_ev["provenance"] == prov_v1
    assert update_ev["op"] == "update"
    assert update_ev["provenance"] == prov_v2


def test_provenance_returns_none_for_unknown_id(mock_memory):
    assert mock_memory.provenance("ffffffff" * 4) is None


def test_save_without_extra_records_empty_provenance(mock_memory):
    rec = mock_memory.save(content="plain", title="plain")
    payload = mock_memory.provenance(rec.id)
    assert payload is not None
    assert payload["current"] == {}
    assert payload["events"][0].get("provenance", {}) == {}


def test_provenance_keys_set_is_stable():
    # Synapse's contract depends on these exact keys. If we add one,
    # synapse-side code must also handle it; if we drop one, synapse
    # users see silent data loss. This test pins the public surface.
    assert frozenset({
        "synapse_trace_id",
        "synapse_route_reason",
        "synapse_write_policy_schema",
        "synapse_write_target",
        "synapse_agent_id",
        "synapse_agent_signature",
    }) == _PROVENANCE_KEYS


# -- MemoSynapseBackend adapter --------------------------------------------


def test_backend_health_ready(mock_memory):
    backend = MemoSynapseBackend(mock_memory)
    health = backend.health()
    assert health["name"] == "memo"
    assert health["available"] is True
    assert health["status"] == "ready"
    assert "total=" in health["detail"]


def test_backend_collect_returns_namespaced_refs(mock_memory):
    prov = _sample_provenance()
    rec = mock_memory.save(content="astor terapia", title="Astor TO", extra=prov)

    backend = MemoSynapseBackend(mock_memory)
    refs = backend.collect("astor", k=5, trace_id="probe-1")
    assert refs, "expected at least one hit"
    hit = next(r for r in refs if r["uri"].endswith(rec.id))
    assert hit["source"] == "memo"
    assert hit["uri"] == f"memo://memoria/{rec.id}"
    assert hit["title"] == "Astor TO"
    assert hit["metadata"]["provenance"] == prov
    assert hit["metadata"]["synapse_trace_id"] == "probe-1"
    assert "synapse_trace_id" not in hit["metadata"]["extra"]


def test_backend_collect_empty_query_returns_empty(mock_memory):
    backend = MemoSynapseBackend(mock_memory)
    assert backend.collect("", k=3) == []
    assert backend.collect("   ", k=3) == []


def test_backend_remember_persists_with_provenance(mock_memory):
    backend = MemoSynapseBackend(mock_memory)
    prov = _sample_provenance()
    receipt = backend.remember({
        "kind": "decision",
        "text": "Switch embedder to Qwen3-4B.",
        "target": "memo",
        "metadata": {**prov, "title": "Embedder switch"},
    })

    assert receipt["schema"] == "synapse.memory_write_receipt.v1"
    assert receipt["backend"] == "memo"
    assert receipt["kind"] == "decision"
    assert receipt["trace_id"] == prov["synapse_trace_id"]
    assert receipt["uri"].startswith("memo://memoria/")
    memoria_id = receipt["metadata"]["memoria_id"]

    fetched = mock_memory.get(memoria_id)
    assert fetched is not None
    assert fetched.title == "Embedder switch"
    assert fetched.type == "decision"
    assert "synapse" in fetched.tags
    for key, value in prov.items():
        assert fetched.extra[key] == value


def test_backend_remember_coerces_unknown_kind(mock_memory):
    # Synapse uses kinds like "task" / "idea" that memo doesn't have in
    # its frozenset. The adapter coerces to `note` and tags `kind:<orig>`
    # so the semantic intent survives.
    backend = MemoSynapseBackend(mock_memory)
    receipt = backend.remember({
        "kind": "task",
        "text": "Schedule eval run nightly.",
        "metadata": {"synapse_trace_id": "t-task"},
    })

    fetched = mock_memory.get(receipt["metadata"]["memoria_id"])
    assert fetched is not None
    assert fetched.type == "note"
    assert "kind:task" in fetched.tags
    assert "synapse" in fetched.tags
    assert receipt["kind"] == "task"
    assert receipt["metadata"]["memo_type"] == "note"


def test_backend_remember_rejects_empty_text(mock_memory):
    backend = MemoSynapseBackend(mock_memory)
    with pytest.raises(ValueError, match="empty text"):
        backend.remember({"kind": "note", "text": "   "})


def test_backend_remember_defaults_write_target(mock_memory):
    backend = MemoSynapseBackend(mock_memory)
    receipt = backend.remember({
        "kind": "note",
        "text": "default-target body",
        "metadata": {"synapse_trace_id": "t-default"},
    })
    fetched = mock_memory.get(receipt["metadata"]["memoria_id"])
    assert fetched is not None
    assert fetched.extra["synapse_write_target"] == "memo"


def test_backend_remember_attaches_evidence_paths(mock_memory):
    backend = MemoSynapseBackend(mock_memory)
    receipt = backend.remember({
        "kind": "fact",
        "text": "fact body",
        "evidence_paths": ["memflow://event/abc", "memo://memoria/xyz"],
        "metadata": {"synapse_trace_id": "t-evidence"},
    })
    fetched = mock_memory.get(receipt["metadata"]["memoria_id"])
    assert fetched is not None
    assert fetched.extra["synapse_evidence_paths"] == [
        "memflow://event/abc",
        "memo://memoria/xyz",
    ]
    assert "## Evidence paths" in fetched.body


# -- CLI surface ------------------------------------------------------------


def _cli_env(tmp_cfg) -> dict[str, str]:
    return {
        **os.environ,
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_cfg.data_dir),
        "MEMO_STATE_DIR": str(tmp_cfg.state_dir),
        "MEMO_VAULT_PATH": str(tmp_cfg.vault_path),
        "MEMO_EMBEDDER_DIMS": str(tmp_cfg.embedder_dims),
        "MEMO_AUTO_PROJECT_TAG": "0",
        "MEMO_CONFIG_FILE": str(tmp_cfg.data_dir.parent / "memo-config.toml"),
    }


def test_cli_save_meta_and_provenance(tmp_cfg, monkeypatch):
    # Stub the embedder so the CLI path doesn't try to load MLX.
    import hashlib

    def _fake(self, inputs):
        out = []
        for text in inputs:
            digest = hashlib.sha256((text or "").encode("utf-8")).digest()
            vals = [
                ((digest[i % len(digest)] / 255.0) * 2.0) - 1.0
                for i in range(tmp_cfg.embedder_dims)
            ]
            norm = sum(v * v for v in vals) ** 0.5
            out.append([v / norm for v in vals])
        return out

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _fake)

    runner = CliRunner()
    env = _cli_env(tmp_cfg)
    result = runner.invoke(
        cli,
        [
            "save", "hello via cli",
            "--title", "cli-meta",
            "--type", "decision",
            "--meta", "synapse_trace_id=cli-trace",
            "--meta", "synapse_agent_id=cli-agent",
            "--json",
        ],
        env=env, catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    rec_dict = json.loads(result.output)
    memoria_id = rec_dict["id"]
    assert rec_dict["extra"]["synapse_trace_id"] == "cli-trace"
    assert rec_dict["extra"]["synapse_agent_id"] == "cli-agent"

    prov_result = runner.invoke(
        cli, ["provenance", memoria_id, "--json"],
        env=env, catch_exceptions=False,
    )
    assert prov_result.exit_code == 0, prov_result.output
    payload = json.loads(prov_result.output)
    assert payload["id"] == memoria_id
    assert payload["current"]["synapse_trace_id"] == "cli-trace"
    assert payload["events"][0]["provenance"]["synapse_agent_id"] == "cli-agent"


def test_cli_save_meta_rejects_bad_pair(tmp_cfg, monkeypatch):
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed", lambda self, inputs: [[0.0] * tmp_cfg.embedder_dims for _ in inputs],
    )
    runner = CliRunner()
    env = _cli_env(tmp_cfg)
    result = runner.invoke(
        cli,
        ["save", "x", "--title", "t", "--meta", "no_equals_here"],
        env=env, catch_exceptions=False,
    )
    assert result.exit_code != 0
    assert "KEY=VALUE" in result.output


# -- MCP surface -----------------------------------------------------------


def test_mcp_save_and_provenance_tools(tmp_cfg):
    """memory_save(extra=...) + memory_provenance(id) round-trip."""
    import asyncio
    import hashlib

    from memo.server import build_server

    mem = Memory(tmp_cfg)

    def _fake(inputs):
        out = []
        for text in inputs:
            digest = hashlib.sha256((text or "").encode("utf-8")).digest()
            vals = [
                ((digest[i % len(digest)] / 255.0) * 2.0) - 1.0
                for i in range(tmp_cfg.embedder_dims)
            ]
            norm = sum(v * v for v in vals) ** 0.5
            out.append([v / norm for v in vals])
        return out

    mem.embedder.embed = _fake
    server = build_server(memory=mem)

    def _tool(name):
        tool = asyncio.run(server.get_tool(name))
        if tool is None:
            raise RuntimeError(f"tool {name!r} not registered")
        return tool.fn

    save_fn = _tool("memory_save")
    prov_fn = _tool("memory_provenance")
    prov = _sample_provenance()

    rec = save_fn(content="mcp body", title="mcp-title", extra=prov)
    memoria_id = rec["id"]
    assert rec["extra"]["synapse_trace_id"] == prov["synapse_trace_id"]

    trail = prov_fn(id=memoria_id)
    assert trail is not None
    assert trail["id"] == memoria_id
    assert trail["current"] == prov
    assert trail["events"][0]["provenance"] == prov

    assert prov_fn(id="00000000" * 4) is None
