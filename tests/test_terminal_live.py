"""Live terminal registry and delivery behavior."""

from __future__ import annotations

import os
import pty
from pathlib import Path

import pytest

from memo.errors import TerminalValidationError
from memo.terminal_live import ProcessSnapshot, TerminalBridge


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
