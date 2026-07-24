from __future__ import annotations

import threading
from typing import ClassVar

import pytest

from memo import server
from memo.errors import ValidationError


class _RecordingThread:
    names: ClassVar[list[str]] = []

    def __init__(self, *, target, name: str, daemon: bool) -> None:
        del target, daemon
        self.name = name

    def start(self) -> None:
        self.names.append(self.name)


def _clear_policy_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    # MEMO_AUTO_UPDATE defaults ON (memo v4.1.0+); force it off so these wiring
    # tests have a deterministic all-disabled baseline. The default-ON behavior
    # is asserted separately by test_auto_update_runs_by_default.
    for name in (
        "MEMO_UPDATE_CHECK_ENABLED",
        "MEMO_STATUSLINE_SELFHEAL",
        "MEMO_HOOK_SELFHEAL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MEMO_AUTO_UPDATE", "0")


def test_background_tasks_silent_when_all_disabled(
    tmp_cfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_policy_flags(monkeypatch)
    _RecordingThread.names = []
    monkeypatch.setattr(threading, "Thread", _RecordingThread)

    started = server._start_background_tasks(tmp_cfg)

    assert started == ()
    assert _RecordingThread.names == []


def test_auto_update_runs_by_default(tmp_cfg, monkeypatch: pytest.MonkeyPatch) -> None:
    # With no policy flag set at all, memo v4.1.0+ runs the tag check + the
    # background updater by default (the self-heal tasks stay opt-in). The
    # isolated MEMO_CONFIG_DIR means no markdown config leaks into the default.
    for name in (
        "MEMO_UPDATE_CHECK_ENABLED",
        "MEMO_AUTO_UPDATE",
        "MEMO_STATUSLINE_SELFHEAL",
        "MEMO_HOOK_SELFHEAL",
    ):
        monkeypatch.delenv(name, raising=False)
    _RecordingThread.names = []
    monkeypatch.setattr(threading, "Thread", _RecordingThread)

    started = server._start_background_tasks(tmp_cfg)

    assert started == ("memo-update-check", "memo-auto-update")
    assert tuple(_RecordingThread.names) == ("memo-update-check", "memo-auto-update")


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


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("MEMO_MCP_TRANSPORT", "htpp"),
        ("MEMO_MCP_PORT", "0"),
        ("MEMO_MCP_PORT", "70000"),
    ],
)
def test_main_rejects_invalid_mcp_runtime_config_before_build(
    name: str,
    value: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        server,
        "build_server",
        lambda *args, **kwargs: pytest.fail("server built with invalid MCP config"),
    )

    with pytest.raises(ValidationError, match=name):
        server.main()
