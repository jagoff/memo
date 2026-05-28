"""M2b: memo emits ConsciousnessEvent entries to the unified ledger.

Every save/update/delete must produce one append-only JSONL line in the
shared trinity ledger root. Best-effort — never raises, never affects
the operation outcome.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

cc = pytest.importorskip("consciousness_contracts")


@pytest.fixture
def ledger_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "consciousness-ledger"
    monkeypatch.setenv("CONSCIOUSNESS_LEDGER_ROOT", str(root))
    # Force a fresh writer (the module caches one per process).
    import memo.consciousness_ledger as module

    module._writer = None
    return root


def _read_today(root: Path) -> list[dict]:
    files = list(root.glob("*.jsonl"))
    if not files:
        return []
    out: list[dict] = []
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


def test_save_emits_consciousness_event(mock_memory, ledger_root: Path) -> None:
    rec = mock_memory.save(content="ledger test body", title="Ledger T1")
    events = _read_today(ledger_root)
    assert len(events) == 1
    ev = events[0]
    assert ev["schema"] == "consciousness.event.v1"
    assert ev["source"] == "memo"
    assert ev["op"] == "save"
    assert ev["subject_uri"] == f"memo://memoria/{rec.id}"
    assert ev["payload"]["title"] == "Ledger T1"
    assert ev["payload"]["id"] == rec.id


def test_save_with_synapse_trace_propagates_to_ledger(mock_memory, ledger_root: Path) -> None:
    rec = mock_memory.save(
        content="cross-system save",
        title="cross",
        extra={"synapse_trace_id": "trace-from-synapse", "synapse_agent_id": "claude-test"},
    )
    events = _read_today(ledger_root)
    assert len(events) == 1
    ev = events[0]
    assert ev["trace_id"] == "trace-from-synapse"
    assert ev["actor"] == "claude-test"
    assert ev["subject_uri"] == f"memo://memoria/{rec.id}"


def test_update_emits_event(mock_memory, ledger_root: Path) -> None:
    rec = mock_memory.save(content="v1 body", title="orig")
    mock_memory.update(rec.id, title="renamed")
    events = _read_today(ledger_root)
    ops = [e["op"] for e in events]
    assert ops == ["save", "update"]
    assert events[1]["payload"]["title"] == "renamed"


def test_delete_emits_event(mock_memory, ledger_root: Path) -> None:
    rec = mock_memory.save(content="to delete", title="bye")
    mock_memory.delete(rec.id)
    events = _read_today(ledger_root)
    ops = [e["op"] for e in events]
    assert ops == ["save", "delete"]
    assert events[1]["subject_uri"] == f"memo://memoria/{rec.id}"


def test_ledger_disabled_via_env(
    monkeypatch: pytest.MonkeyPatch,
    mock_memory,
    ledger_root: Path,
) -> None:
    monkeypatch.setenv("MEMO_EMIT_LEDGER", "0")
    mock_memory.save(content="silent", title="quiet")
    assert _read_today(ledger_root) == []


def test_failure_in_ledger_does_not_break_save(
    mock_memory,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Simulate ledger writer failure; save must still return a record."""
    monkeypatch.setenv("CONSCIOUSNESS_LEDGER_ROOT", str(tmp_path / "blocked"))
    # Block the directory: create a file where the parent should be a dir.
    (tmp_path / "blocked").write_text("not a dir", encoding="utf-8")
    import memo.consciousness_ledger as module

    module._writer = None  # fresh writer with blocked root
    rec = mock_memory.save(content="resilient body", title="resilient")
    assert rec is not None
    fetched = mock_memory.get(rec.id)
    assert fetched is not None
    # Note: with the path blocked, no events should land but no exception either.
