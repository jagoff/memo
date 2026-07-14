from __future__ import annotations

import threading
from typing import ClassVar

import pytest

from memo import server


class _RecordingThread:
    names: ClassVar[list[str]] = []

    def __init__(self, *, target, name: str, daemon: bool) -> None:
        del target, daemon
        self.name = name

    def start(self) -> None:
        self.names.append(self.name)


def _clear_policy_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "MEMO_UPDATE_CHECK_ENABLED",
        "MEMO_AUTO_UPDATE",
        "MEMO_STATUSLINE_SELFHEAL",
        "MEMO_HOOK_SELFHEAL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_background_tasks_are_silent_by_default(tmp_cfg, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_policy_flags(monkeypatch)
    _RecordingThread.names = []
    monkeypatch.setattr(threading, "Thread", _RecordingThread)

    started = server._start_background_tasks(tmp_cfg)

    assert started == ()
    assert _RecordingThread.names == []


@pytest.mark.parametrize(
    ("flag", "expected"),
    [
        ("MEMO_UPDATE_CHECK_ENABLED", ("memo-update-check",)),
        ("MEMO_AUTO_UPDATE", ("memo-update-check", "memo-auto-update")),
        ("MEMO_STATUSLINE_SELFHEAL", ("memo-statusline-selfheal",)),
        ("MEMO_HOOK_SELFHEAL", ("memo-hook-selfheal",)),
    ],
)
def test_background_tasks_are_individually_opted_in(
    flag: str,
    expected: tuple[str, ...],
    tmp_cfg,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_policy_flags(monkeypatch)
    monkeypatch.setenv(flag, "1")
    _RecordingThread.names = []
    monkeypatch.setattr(threading, "Thread", _RecordingThread)

    started = server._start_background_tasks(tmp_cfg)

    assert started == expected
    assert tuple(_RecordingThread.names) == expected
