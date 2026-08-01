from __future__ import annotations

from pathlib import Path

import pytest

from memo.identity import PrincipalIdentity
from memo.operational import OperationalStore
from memo.operational_epoch import CommitContext


def _context(epoch: int = 3) -> CommitContext:
    return CommitContext(
        identity=PrincipalIdentity(
            principal_id="signal-test",
            actor_id="watcher",
            kind="agent",
            device_id="device-test",
            session_id="session-test",
            source_client="pytest",
        ),
        authority_epoch=epoch,
        control_oid="control-test",
        origin_device="device-test",
    )


class _Fence:
    def verify(self, context: CommitContext) -> None:
        assert context.control_oid == "control-test"


def _store(tmp_path: Path, context: CommitContext) -> OperationalStore:
    return OperationalStore(
        tmp_path,
        device_id="device-test",
        context_provider=lambda: context,
        epoch_fence=_Fence(),
    )


def test_signal_is_fenced_and_idempotent(tmp_path: Path) -> None:
    context = _context()
    store = _store(tmp_path, context)
    first = store.remember_signal(
        marker="watch-1", epoch=3, fence="control-test", payload={"kind": "heartbeat"}
    )
    replay = store.remember_signal(
        marker="watch-1", epoch=3, fence="control-test", payload={"kind": "changed"}
    )
    assert replay == first
    assert [row.marker for row in store.list_signals()] == ["watch-1"]


def test_signal_requires_write_context_but_watcher_epoch_is_independent(tmp_path: Path) -> None:
    context = _context()
    store = _store(tmp_path, context)
    with pytest.raises(Exception, match="authenticated epoch context"):
        OperationalStore(tmp_path / "missing", device_id="device-test").remember_signal(
            marker="watch-1", epoch=3
        )
    remembered = store.remember_signal(marker="watch-2", epoch=2, fence="leader-a")
    assert remembered.epoch == 2
    assert remembered.fence == "leader-a"
    with pytest.raises(ValueError, match="stale signal epoch"):
        store.remember_signal(marker="watch-2", epoch=1, fence="leader-old")
