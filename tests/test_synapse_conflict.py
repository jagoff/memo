"""Tests for the bidirectional Synapse freeze-write protocol.

Coverage:

- `synapse_client.has_blocking_freeze()` correctly filters
  freeze_write + lifecycle_state.
- `Memory.save(respect_synapse_freeze=True)` raises `WriteRefused`
  when synapse reports a blocking conflict; commits normally when
  synapse reports none.
- The freeze check ONLY fires when `extra.synapse_trace_id` is
  present (anonymous saves bypass it).
- Missing synapse binary → graceful no-op (save proceeds).
- MCP `memory_save(respect_synapse_freeze=True)` returns a
  structured `{"status": "refused", "conflict": {...}}` payload.
- `MemoSynapseBackend.remember()` propagates the WriteRefused as a
  ValueError (synapse caller already gets receipt-or-raise).

All tests stub `synapse_client` so no real `synapse` binary needs
to be on PATH.
"""

from __future__ import annotations

from typing import Any

import pytest

import memo.synapse_client as synapse_client
from memo.memory import WriteRefused
from memo.synapse_backend import MemoSynapseBackend

# -- has_blocking_freeze --------------------------------------------------


def test_has_blocking_freeze_returns_first_match():
    rows = [
        {"conflict_id": "skip-1", "freeze_write": False, "lifecycle_state": "detected"},
        {"conflict_id": "skip-2", "freeze_write": True, "lifecycle_state": "resolved"},
        {"conflict_id": "skip-3", "freeze_write": True, "lifecycle_state": "archived"},
        {"conflict_id": "hit-1",  "freeze_write": True, "lifecycle_state": "acknowledged"},
        {"conflict_id": "hit-2",  "freeze_write": True, "lifecycle_state": "detected"},
    ]
    blocked, conflict = synapse_client.has_blocking_freeze(rows)
    assert blocked is True
    assert conflict is not None
    assert conflict["conflict_id"] == "hit-1"


def test_has_blocking_freeze_empty_input():
    blocked, conflict = synapse_client.has_blocking_freeze([])
    assert blocked is False
    assert conflict is None


def test_has_blocking_freeze_resolved_does_not_block():
    rows = [
        {"conflict_id": "r-1", "freeze_write": True, "lifecycle_state": "resolved"},
        {"conflict_id": "r-2", "freeze_write": True, "lifecycle_state": "archived"},
    ]
    blocked, _ = synapse_client.has_blocking_freeze(rows)
    assert blocked is False


def test_has_blocking_freeze_defaults_to_detected():
    """`lifecycle_state` may be missing on legacy payloads — treat as detected."""
    rows = [{"conflict_id": "legacy", "freeze_write": True}]
    blocked, conflict = synapse_client.has_blocking_freeze(rows)
    assert blocked is True
    assert conflict is not None
    assert conflict["conflict_id"] == "legacy"


# -- Memory.save freeze protocol ------------------------------------------


def _sample_prov() -> dict[str, str]:
    return {
        "synapse_trace_id": "trace-freeze-1",
        "synapse_route_reason": "deep_semantic",
        "synapse_agent_id": "claude-4-7",
    }


def _patch_synapse(monkeypatch, *, available: bool, conflicts: list[dict[str, Any]]):
    monkeypatch.setattr(synapse_client, "is_available", lambda: available)
    monkeypatch.setattr(
        synapse_client, "list_conflicts",
        lambda *args, **kwargs: list(conflicts),
    )


def test_save_refuses_when_synapse_freeze_active(mock_memory, monkeypatch):
    _patch_synapse(
        monkeypatch,
        available=True,
        conflicts=[{
            "conflict_id": "C-42",
            "freeze_write": True,
            "lifecycle_state": "detected",
            "summary": "memo says X but memflow says ¬X",
            "severity": "high",
        }],
    )
    with pytest.raises(WriteRefused) as exc_info:
        mock_memory.save(
            content="some body about X",
            title="X is now true",
            extra=_sample_prov(),
            respect_synapse_freeze=True,
        )
    assert exc_info.value.conflict["conflict_id"] == "C-42"
    assert "C-42" in str(exc_info.value)


def test_save_commits_when_no_blocking_freeze(mock_memory, monkeypatch):
    _patch_synapse(monkeypatch, available=True, conflicts=[])
    rec = mock_memory.save(
        content="no freeze body",
        title="no-freeze",
        extra=_sample_prov(),
        respect_synapse_freeze=True,
    )
    assert rec.id


def test_save_bypasses_freeze_when_anonymous(mock_memory, monkeypatch):
    """Saves without `synapse_trace_id` should NEVER call synapse."""
    called: list[bool] = []
    monkeypatch.setattr(synapse_client, "is_available", lambda: True)
    monkeypatch.setattr(
        synapse_client, "list_conflicts",
        lambda *a, **kw: called.append(True) or [],
    )
    rec = mock_memory.save(
        content="anon body",
        title="anon",
        respect_synapse_freeze=True,  # explicit, but no trace_id
    )
    assert rec.id
    assert called == []


def test_save_gracefully_skips_when_synapse_missing(mock_memory, monkeypatch):
    _patch_synapse(monkeypatch, available=False, conflicts=[])
    rec = mock_memory.save(
        content="no synapse body",
        title="no-synapse",
        extra=_sample_prov(),
        respect_synapse_freeze=True,
    )
    assert rec.id


def test_save_env_knob_enables_freeze_check(mock_memory, monkeypatch):
    """`MEMO_RESPECT_SYNAPSE_FREEZE=1` flips the default on."""
    monkeypatch.setenv("MEMO_RESPECT_SYNAPSE_FREEZE", "1")
    _patch_synapse(
        monkeypatch,
        available=True,
        conflicts=[{
            "conflict_id": "C-env",
            "freeze_write": True,
            "lifecycle_state": "detected",
        }],
    )
    with pytest.raises(WriteRefused):
        mock_memory.save(
            content="env-knob body",
            title="env-knob",
            extra=_sample_prov(),
        )


def test_save_explicit_false_overrides_env(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_RESPECT_SYNAPSE_FREEZE", "1")
    _patch_synapse(
        monkeypatch,
        available=True,
        conflicts=[{
            "conflict_id": "C-block",
            "freeze_write": True,
            "lifecycle_state": "detected",
        }],
    )
    rec = mock_memory.save(
        content="override body",
        title="override",
        extra=_sample_prov(),
        respect_synapse_freeze=False,
    )
    assert rec.id


def test_save_swallows_synapse_subprocess_error(mock_memory, monkeypatch):
    """If `list_conflicts` raises, the save must still proceed."""
    monkeypatch.setattr(synapse_client, "is_available", lambda: True)

    def _boom(*_a, **_kw):
        raise RuntimeError("synapse crashed")

    monkeypatch.setattr(synapse_client, "list_conflicts", _boom)
    rec = mock_memory.save(
        content="crash body",
        title="crash",
        extra=_sample_prov(),
        respect_synapse_freeze=True,
    )
    assert rec.id


# -- MCP surface ---------------------------------------------------------


def test_mcp_save_returns_structured_refused(tmp_cfg, monkeypatch):
    """memory_save MCP tool should NOT raise; returns refused payload."""
    import asyncio
    import hashlib

    from memo.memory import Memory
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
    _patch_synapse(
        monkeypatch,
        available=True,
        conflicts=[{
            "conflict_id": "C-mcp",
            "freeze_write": True,
            "lifecycle_state": "detected",
            "summary": "blocking via MCP",
        }],
    )

    server = build_server(memory=mem)
    save_fn = asyncio.run(server.get_tool("memory_save")).fn
    out = save_fn(
        content="mcp body",
        title="mcp",
        extra=_sample_prov(),
        respect_synapse_freeze=True,
    )
    assert out["status"] == "refused"
    assert out["conflict"]["conflict_id"] == "C-mcp"
    assert "C-mcp" in out["message"]


# -- adapter passes through ---------------------------------------------


def test_backend_remember_respects_freeze_by_default(mock_memory, monkeypatch):
    """Synapse-originated writes opt into freeze check automatically."""
    _patch_synapse(
        monkeypatch,
        available=True,
        conflicts=[{
            "conflict_id": "C-backend",
            "freeze_write": True,
            "lifecycle_state": "detected",
        }],
    )
    backend = MemoSynapseBackend(mock_memory)
    with pytest.raises(WriteRefused):
        backend.remember({
            "kind": "decision",
            "text": "the body",
            "metadata": _sample_prov(),
        })


def test_backend_remember_can_opt_out(mock_memory, monkeypatch):
    _patch_synapse(
        monkeypatch,
        available=True,
        conflicts=[{
            "conflict_id": "C-backend-2",
            "freeze_write": True,
            "lifecycle_state": "detected",
        }],
    )
    backend = MemoSynapseBackend(mock_memory)
    receipt = backend.remember({
        "kind": "decision",
        "text": "the body",
        "metadata": {**_sample_prov(), "respect_synapse_freeze": False},
    })
    assert receipt["uri"].startswith("memo://memoria/")
