"""Tests for the best-effort Memflow receipts emitter.

Verifies:

- `emit_receipt()` is a no-op (skipped=True) when `MEMO_EMIT_RECEIPTS`
  is unset or when caller passes `disabled=True`.
- When enabled, returns `{"ok": False, "skipped": True}` cleanly when
  the memflow binary / project root cannot be found (so save/update/
  delete keep working in test/CI environments).
- When enabled + a fake binary is wired in via `MEMO_MEMFLOW_BIN`,
  the subprocess receives the expected `memflow say <text>
  --channel memo-receipts --author memo --no-sync` shape.
- `Memory.save / update / delete / reindex` invoke the emitter once
  per successful op and the receipt carries operation-specific meta.
- `MemoSynapseBackend.remember()` passes `skip_memflow_receipt=True`
  so synapse-originated writes do NOT double-emit.
"""

from __future__ import annotations

import os
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Any

import pytest

import memo.receipts as receipts
import memo.synapse_client as synapse_client
from memo.synapse_backend import _CONTRACTS_AVAILABLE, MemoSynapseBackend

# -- fake memflow binary ---------------------------------------------------


def _write_fake_memflow(dest: Path, log_file: Path) -> None:
    """Write a tiny shell script that records argv to a log and prints a path."""
    script = (
        "#!/bin/sh\n"
        f'echo "$@" >> "{log_file}"\n'
        f'echo "memflow://fake/{int.from_bytes(os.urandom(4), "big")}"\n'
    )
    dest.write_text(script)
    dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def fake_memflow(monkeypatch) -> Path:
    """Provide MEMO_MEMFLOW_BIN + MEMFLOW_PROJECT_ROOT pointed at a tmp tree."""
    workdir = Path(tempfile.mkdtemp(prefix="memo-test-receipts-", dir="/tmp"))
    (workdir / ".memflow").mkdir()
    bin_path = workdir / "fake-memflow"
    log_path = workdir / "calls.log"
    _write_fake_memflow(bin_path, log_path)
    monkeypatch.setenv("MEMO_MEMFLOW_BIN", str(bin_path))
    monkeypatch.setenv("MEMFLOW_PROJECT_ROOT", str(workdir))
    yield workdir
    shutil.rmtree(workdir, ignore_errors=True)


def _read_calls(workdir: Path) -> list[str]:
    log = workdir / "calls.log"
    if not log.is_file():
        return []
    return [line for line in log.read_text().splitlines() if line]


# -- emit_receipt unit ------------------------------------------------------


def test_emit_receipt_no_op_when_env_unset(monkeypatch):
    monkeypatch.delenv("MEMO_EMIT_RECEIPTS", raising=False)
    out = receipts.emit_receipt("save", text="x", meta={"id": "abc"})
    assert out == {"ok": False, "skipped": True, "reason": "MEMO_EMIT_RECEIPTS not set"}


def test_emit_receipt_disabled_kwarg_wins(monkeypatch, fake_memflow):
    monkeypatch.setenv("MEMO_EMIT_RECEIPTS", "1")
    out = receipts.emit_receipt("save", text="x", meta={}, disabled=True)
    assert out["ok"] is False
    assert out["skipped"] is True
    assert _read_calls(fake_memflow) == []  # binary never invoked


def test_emit_receipt_skipped_when_binary_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMO_EMIT_RECEIPTS", "1")
    monkeypatch.delenv("MEMO_MEMFLOW_BIN", raising=False)
    # Keep the project-root check passing (CI has no memflow checkout) so the
    # binary-missing branch is the one that fires, not project-root-missing.
    monkeypatch.setattr(receipts, "_project_root", lambda: tmp_path)
    # Point shutil.which away from any real memflow on PATH.
    monkeypatch.setattr(receipts, "_binary", lambda: None)
    out = receipts.emit_receipt("save", text="x", meta={"id": "abc"})
    assert out == {"ok": False, "skipped": True, "reason": "memflow binary not found"}


def test_emit_receipt_skipped_when_project_root_missing(monkeypatch, fake_memflow):
    monkeypatch.setenv("MEMO_EMIT_RECEIPTS", "1")
    # Override project_root to be missing while keeping the binary present.
    monkeypatch.setattr(receipts, "_project_root", lambda: None)
    out = receipts.emit_receipt("save", text="x", meta={"id": "abc"})
    assert out == {"ok": False, "skipped": True, "reason": "memflow project root not found"}


def test_emit_receipt_invokes_memflow_channel_receipt(monkeypatch, fake_memflow):
    monkeypatch.setenv("MEMO_EMIT_RECEIPTS", "1")
    out = receipts.emit_receipt(
        "save",
        text="hello body",
        meta={"id": "abc123", "type": "note", "tags": "alpha,beta"},
    )
    assert out["ok"] is True
    assert out["path"].startswith("memflow://fake/")
    calls = _read_calls(fake_memflow)
    assert len(calls) == 1
    line = calls[0]
    assert "say hello body" in line
    assert "--channel memo-receipts" in line
    assert "--author memo" in line
    assert "--no-sync" in line
    assert "write fact" not in line


# -- Memory.save / update / delete / reindex wiring ------------------------


def test_save_emits_receipt(monkeypatch, mock_memory, fake_memflow):
    monkeypatch.setenv("MEMO_EMIT_RECEIPTS", "1")
    mock_memory.save(content="body one", title="receipt-save")
    calls = _read_calls(fake_memflow)
    assert len(calls) == 1
    assert "memo-save" in calls[0]
    assert "receipt-save" in calls[0]


def test_save_skip_kwarg_suppresses_receipt(monkeypatch, mock_memory, fake_memflow):
    monkeypatch.setenv("MEMO_EMIT_RECEIPTS", "1")
    mock_memory.save(
        content="body two",
        title="skipped-save",
        skip_memflow_receipt=True,
    )
    assert _read_calls(fake_memflow) == []


def test_save_without_env_knob_emits_nothing(mock_memory, fake_memflow, monkeypatch):
    monkeypatch.delenv("MEMO_EMIT_RECEIPTS", raising=False)
    mock_memory.save(content="quiet body", title="quiet")
    assert _read_calls(fake_memflow) == []


def test_update_emits_receipt_with_delta_keys(monkeypatch, mock_memory, fake_memflow):
    monkeypatch.setenv("MEMO_EMIT_RECEIPTS", "1")
    rec = mock_memory.save(content="initial body", title="will-update")
    # Clear the save receipt before the update.
    (fake_memflow / "calls.log").unlink(missing_ok=True)
    mock_memory.update(rec.id, title="new title", tags=["alpha", "beta"])
    calls = _read_calls(fake_memflow)
    assert len(calls) == 1
    assert "memo-update" in calls[0]
    assert "delta_keys=" in calls[0]
    assert "tags" in calls[0]
    assert "title" in calls[0]


def test_delete_emits_receipt(monkeypatch, mock_memory, fake_memflow):
    monkeypatch.setenv("MEMO_EMIT_RECEIPTS", "1")
    rec = mock_memory.save(content="del body", title="will-delete")
    (fake_memflow / "calls.log").unlink(missing_ok=True)
    mock_memory.delete(rec.id)
    calls = _read_calls(fake_memflow)
    assert len(calls) == 1
    assert "memo-delete" in calls[0]
    assert "will-delete" in calls[0]


def test_delete_no_op_does_not_emit(monkeypatch, mock_memory, fake_memflow):
    monkeypatch.setenv("MEMO_EMIT_RECEIPTS", "1")
    mock_memory.delete("doesnotexist0000")
    assert _read_calls(fake_memflow) == []


# -- synapse adapter opts out ----------------------------------------------


@pytest.mark.skipif(
    not _CONTRACTS_AVAILABLE,
    reason="MemoSynapseBackend needs the optional consciousness-contracts package",
)
def test_synapse_backend_remember_skips_receipt(monkeypatch, mock_memory, fake_memflow):
    """Synapse-originated writes must NOT emit memflow receipts."""
    monkeypatch.setenv("MEMO_EMIT_RECEIPTS", "1")
    monkeypatch.setattr(synapse_client, "is_available", lambda: False)
    backend = MemoSynapseBackend(mock_memory)
    backend.remember(
        {
            "kind": "decision",
            "text": "synapse-originated body",
            "metadata": {
                "synapse_trace_id": "trace-X",
                "synapse_agent_id": "claude-4-7",
            },
        }
    )
    assert _read_calls(fake_memflow) == []


# -- failure modes that must not raise ------------------------------------


def test_emit_receipt_swallows_memflow_nonzero_exit(monkeypatch):
    """If memflow exits non-zero, return an error dict, do NOT raise."""
    monkeypatch.setenv("MEMO_EMIT_RECEIPTS", "1")

    workdir = Path(tempfile.mkdtemp(prefix="memo-test-receipts-fail-", dir="/tmp"))
    try:
        (workdir / ".memflow").mkdir()
        bin_path = workdir / "fake-memflow-fail"
        bin_path.write_text("#!/bin/sh\nexit 7\n")
        bin_path.chmod(bin_path.stat().st_mode | stat.S_IXUSR)
        monkeypatch.setenv("MEMO_MEMFLOW_BIN", str(bin_path))
        monkeypatch.setenv("MEMFLOW_PROJECT_ROOT", str(workdir))

        out = receipts.emit_receipt("save", text="x", meta={"id": "abc"})
        assert out["ok"] is False
        assert "error" in out
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_emit_receipt_swallows_oserror(monkeypatch):
    """If subprocess.run raises (e.g. permission), return error dict gracefully."""
    monkeypatch.setenv("MEMO_EMIT_RECEIPTS", "1")
    monkeypatch.setattr(receipts, "_project_root", lambda: Path("/tmp"))
    monkeypatch.setattr(receipts, "_binary", lambda: "/nonexistent/memflow")

    def _boom(*_a: Any, **_kw: Any):
        raise OSError("boom")

    monkeypatch.setattr(receipts.subprocess, "run", _boom)
    out = receipts.emit_receipt("save", text="x", meta={"id": "abc"})
    assert out["ok"] is False
    assert "error" in out
