"""Live terminal registry and delivery behavior."""

from __future__ import annotations

import os
import pty
import stat
import subprocess
from pathlib import Path

import pytest

from memo.errors import TerminalValidationError
from memo.terminal_live import ProcessSnapshot, TerminalBridge, _is_local_tty_path


def test_local_tty_path_accepts_linux_devpts_and_rejects_non_device_paths() -> None:
    char_mode = stat.S_IFCHR | 0o620

    assert _is_local_tty_path(Path("/dev/pts/7"), char_mode)
    assert _is_local_tty_path(Path("/dev/ttys000"), char_mode)
    assert not _is_local_tty_path(Path("/tmp/pts/7"), char_mode)
    assert not _is_local_tty_path(Path("/dev/pts/7"), stat.S_IFREG | 0o600)


@pytest.fixture
def registered_bridge(tmp_cfg):
    master_fd, slave_fd = pty.openpty()
    tty = Path(os.ttyname(slave_fd))
    state = {"alive": True, "foreground_pgid": 4242}
    payloads: list[tuple[Path, bytes, str]] = []

    def probe(pid: int) -> ProcessSnapshot | None:
        if not state["alive"]:
            return None
        return ProcessSnapshot(
            pid=pid,
            uid=os.getuid(),
            tty=tty,
            started_at="Sat Aug 1 12:00:00 2026",
            pgid=pid,
            foreground_pgid=int(state["foreground_pgid"]),
            command="codex --dangerously-bypass-approvals-and-sandbox",
        )

    def presenter(path: Path, payload: bytes, *, terminal_app: str) -> str:
        payloads.append((path, payload, terminal_app))
        return "test"

    bridge = TerminalBridge(tmp_cfg, process_probe=probe, presenter=presenter)
    registration = bridge.register(
        agent="codex",
        tty=tty,
        pid=4242,
        terminal_app="Ghostty",
        project="/tmp/memo",
    )
    try:
        yield bridge, registration, payloads, state
    finally:
        os.close(slave_fd)
        os.close(master_fd)


def test_register_persists_same_uid_tty_and_prunes_stale_process(tmp_cfg) -> None:
    master_fd, slave_fd = pty.openpty()
    tty = Path(os.ttyname(slave_fd))
    alive = True

    def probe(pid: int) -> ProcessSnapshot | None:
        if not alive:
            return None
        return ProcessSnapshot(
            pid=pid,
            uid=os.getuid(),
            tty=tty,
            started_at="Sat Aug 1 12:00:00 2026",
            pgid=pid,
            foreground_pgid=pid,
            command="codex --dangerously-bypass-approvals-and-sandbox",
        )

    try:
        bridge = TerminalBridge(tmp_cfg, process_probe=probe)
        registration = bridge.register(
            agent="codex",
            tty=tty,
            pid=4242,
            terminal_app="Ghostty",
            project="/tmp/memo",
        )

        assert bridge.list() == [registration]
        assert registration.tty == str(tty)
        assert registration.agent == "codex"

        alive = False
        assert bridge.list() == []
    finally:
        os.close(slave_fd)
        os.close(master_fd)


@pytest.mark.parametrize(
    ("reported", "canonical"),
    [
        pytest.param("Apple_Terminal", "Terminal", id="apple-terminal"),
        pytest.param("iTerm.app", "iTerm2", id="iterm-app"),
    ],
)
def test_register_accepts_term_program_aliases(tmp_cfg, reported: str, canonical: str) -> None:
    master_fd, slave_fd = pty.openpty()
    tty = Path(os.ttyname(slave_fd))

    def probe(pid: int) -> ProcessSnapshot:
        return ProcessSnapshot(
            pid=pid,
            uid=os.getuid(),
            tty=tty,
            started_at="Sat Aug 1 12:00:00 2026",
            pgid=pid,
            foreground_pgid=pid,
            command="codex",
        )

    try:
        registration = TerminalBridge(tmp_cfg, process_probe=probe).register(
            agent="codex",
            tty=tty,
            pid=4242,
            terminal_app=reported,
        )

        assert registration.terminal_app == canonical
    finally:
        os.close(slave_fd)
        os.close(master_fd)


def test_replacing_process_on_same_tty_rotates_registration_id(tmp_cfg) -> None:
    master_fd, slave_fd = pty.openpty()
    tty = Path(os.ttyname(slave_fd))

    def probe(pid: int) -> ProcessSnapshot:
        return ProcessSnapshot(
            pid=pid,
            uid=os.getuid(),
            tty=tty,
            started_at=f"process-start-{pid}",
            pgid=pid,
            foreground_pgid=pid,
            command="codex",
        )

    try:
        bridge = TerminalBridge(tmp_cfg, process_probe=probe)
        original = bridge.register(agent="codex", tty=tty, pid=4242)
        refreshed = bridge.register(agent="codex", tty=tty, pid=4242)
        replacement = bridge.register(agent="codex", tty=tty, pid=4343)

        assert refreshed.id == original.id
        assert replacement.id != original.id
        assert bridge.registration_id(original.id) == ""
        assert bridge.registration_id(replacement.id) == replacement.id
    finally:
        os.close(slave_fd)
        os.close(master_fd)


def test_send_strips_terminal_controls_and_is_idempotent(registered_bridge) -> None:
    bridge, registration, payloads, _state = registered_bridge

    first = bridge.send(
        registration.id,
        "hello\x1b[31m!\x1b[0m\r",
        message_id="msg-1",
    )
    second = bridge.send(
        registration.id,
        "different text",
        message_id="msg-1",
    )

    assert payloads == [(Path(registration.tty), b"hello!\r", "Ghostty")]
    assert first.status == "delivered"
    assert first.transport == "test"
    assert second.receipt_id == first.receipt_id
    assert second.status == "duplicate"


def test_send_refuses_when_agent_is_not_foreground(registered_bridge) -> None:
    bridge, registration, payloads, state = registered_bridge
    state["foreground_pgid"] = 9999

    with pytest.raises(TerminalValidationError, match="foreground"):
        bridge.send(registration.id, "do not deliver")

    assert payloads == []


def test_target_validation_failures_are_receipted(registered_bridge) -> None:
    bridge, registration, payloads, state = registered_bridge

    with pytest.raises(TerminalValidationError, match="not found"):
        bridge.send(
            "term-0123456789abcdef",
            "missing target",
            message_id="missing-target-1",
        )
    state["foreground_pgid"] = 9999
    with pytest.raises(TerminalValidationError, match="foreground"):
        bridge.send(registration.id, "background target", message_id="background-target-1")

    history = bridge.history(limit=2)
    assert [(item.message_id, item.status) for item in history] == [
        ("background-target-1", "failed"),
        ("missing-target-1", "failed"),
    ]
    assert "foreground" in history[0].error
    assert "not found" in history[1].error
    assert "background target" not in repr(history[0])
    assert "missing target" not in repr(history[1])
    assert payloads == []


def test_enter_delivers_only_carriage_return(registered_bridge) -> None:
    bridge, registration, payloads, _state = registered_bridge

    receipt = bridge.enter(registration.id, message_id="enter-1")

    assert payloads == [(Path(registration.tty), b"\r", "Ghostty")]
    assert receipt.kind == "enter"
    assert receipt.status == "delivered"


@pytest.mark.parametrize(
    "message",
    [
        pytest.param("", id="empty"),
        pytest.param("\x1b[31m\r", id="control-only"),
        pytest.param("x" * (16 * 1024 + 1), id="oversized"),
    ],
)
def test_send_rejects_empty_control_only_and_oversized_messages(
    registered_bridge,
    message: str,
) -> None:
    bridge, registration, payloads, _state = registered_bridge

    with pytest.raises(TerminalValidationError):
        bridge.send(registration.id, message)

    assert payloads == []


def test_send_rejects_unbounded_or_malformed_routing_ids(registered_bridge) -> None:
    bridge, registration, payloads, _state = registered_bridge

    with pytest.raises(TerminalValidationError, match="message id"):
        bridge.send(registration.id, "hello", message_id="x" * 129)
    with pytest.raises(TerminalValidationError, match="sender id"):
        bridge.send(registration.id, "hello", sender="not a terminal")

    assert payloads == []


def test_failed_delivery_is_receipted_without_message_body(tmp_cfg) -> None:
    master_fd, slave_fd = pty.openpty()
    tty = Path(os.ttyname(slave_fd))

    def probe(pid: int) -> ProcessSnapshot:
        return ProcessSnapshot(
            pid=pid,
            uid=os.getuid(),
            tty=tty,
            started_at="Sat Aug 1 12:00:00 2026",
            pgid=pid,
            foreground_pgid=pid,
            command="codex",
        )

    def fail(_tty: Path, _payload: bytes, *, terminal_app: str) -> str:
        raise OSError("presenter unavailable")

    try:
        bridge = TerminalBridge(tmp_cfg, process_probe=probe, presenter=fail)
        registration = bridge.register(agent="codex", tty=tty, pid=4242)

        with pytest.raises(TerminalValidationError, match="delivery failed"):
            bridge.send(registration.id, "secret terminal body", message_id="failure-1")

        receipt = bridge.history()[0]
        assert receipt.status == "failed"
        assert receipt.error == "OSError"
        assert "secret terminal body" not in repr(receipt)
    finally:
        os.close(slave_fd)
        os.close(master_fd)


def test_presenter_timeout_cannot_leave_pending_receipt(tmp_cfg) -> None:
    master_fd, slave_fd = pty.openpty()
    tty = Path(os.ttyname(slave_fd))

    def probe(pid: int) -> ProcessSnapshot:
        return ProcessSnapshot(
            pid=pid,
            uid=os.getuid(),
            tty=tty,
            started_at="Sat Aug 1 12:00:00 2026",
            pgid=pid,
            foreground_pgid=pid,
            command="codex",
        )

    def timeout(_tty: Path, _payload: bytes, *, terminal_app: str) -> str:
        raise subprocess.TimeoutExpired(["osascript"], timeout=5)

    try:
        bridge = TerminalBridge(tmp_cfg, process_probe=probe, presenter=timeout)
        registration = bridge.register(agent="codex", tty=tty, pid=4242)

        with pytest.raises(TerminalValidationError, match="delivery failed"):
            bridge.send(registration.id, "secret terminal body", message_id="timeout-1")

        receipt = bridge.history()[0]
        assert receipt.status == "failed"
        assert receipt.error == "TimeoutExpired"
        assert "secret terminal body" not in repr(receipt)
    finally:
        os.close(slave_fd)
        os.close(master_fd)


def test_registration_database_is_private(tmp_cfg) -> None:
    master_fd, slave_fd = pty.openpty()
    tty = Path(os.ttyname(slave_fd))

    def probe(pid: int) -> ProcessSnapshot:
        return ProcessSnapshot(
            pid=pid,
            uid=os.getuid(),
            tty=tty,
            started_at="Sat Aug 1 12:00:00 2026",
            pgid=pid,
            foreground_pgid=pid,
            command="codex",
        )

    try:
        TerminalBridge(tmp_cfg, process_probe=probe, presenter=lambda *_a, **_kw: "test")
        assert (tmp_cfg.state_dir / "terminal_live.db").stat().st_mode & 0o777 == 0o600
    finally:
        os.close(slave_fd)
        os.close(master_fd)
