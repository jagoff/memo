"""QW6: SYNAPSE_TRACE_ID env propagation into memo writes.

When `memo save` / Memory.save is invoked under a synapse trace
context, the trace_id rides from synapse to memo via the
`SYNAPSE_TRACE_ID` env var (set by synapse before forking the
subprocess). Memo must capture it as `extra.synapse_trace_id` so the
provenance walk via `Memory.provenance(id)` can chain memo writes
back to the originating synapse turn.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from memo.config import Config
from memo.memory import Memory


@pytest.fixture()
def memory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[return]
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("MEMO_NONINTERACTIVE", "1")
    # Receipts default off; pin the real flag. (MEMO_DISABLE_RECEIPTS was a
    # phantom — no such flag — so it was a silent no-op.)
    monkeypatch.setenv("MEMO_EMIT_RECEIPTS", "0")
    mem = Memory(Config.from_env())
    yield mem
    mem.close()


def test_env_trace_id_attaches_to_extra(
    memory: Memory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SYNAPSE_TRACE_ID", "trace-env-001")
    rec = memory.save(content="hello world", defer_embed=True)
    prov = memory.provenance(rec.id)
    assert prov["current"]["synapse_trace_id"] == "trace-env-001"


def test_explicit_extra_trace_id_overrides_env(
    memory: Memory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SYNAPSE_TRACE_ID", "from-env")
    rec = memory.save(
        content="hello",
        extra={"synapse_trace_id": "explicit"},
        defer_embed=True,
    )
    prov = memory.provenance(rec.id)
    assert prov["current"]["synapse_trace_id"] == "explicit"


def test_missing_env_leaves_provenance_empty(
    memory: Memory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SYNAPSE_TRACE_ID", raising=False)
    rec = memory.save(content="hello", defer_embed=True)
    prov = memory.provenance(rec.id)
    assert not prov["current"].get("synapse_trace_id")


def test_empty_env_trace_id_is_ignored(
    memory: Memory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SYNAPSE_TRACE_ID", "   ")
    rec = memory.save(content="hello", defer_embed=True)
    prov = memory.provenance(rec.id)
    assert not prov["current"].get("synapse_trace_id")
