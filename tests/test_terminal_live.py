"""Live terminal registry and delivery behavior."""

from __future__ import annotations

import errno
import os
import pty
import sqlite3
import stat
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import memo.terminal_live as terminal_live
from memo.errors import TerminalDeliveryError, TerminalValidationError
from memo.terminal_live import (
    _PROBE_UNKNOWN,
    ProcessSnapshot,
    TerminalBridge,
    _darwin_process_birth_identity,
    _is_local_tty_path,
    _process_birth_identity,
)


def _allow_transport(_tty: Path, _terminal_app: str) -> bool:
    return True


def _simulate_process_bound_transport() -> bool:
    return True


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
    state = {
        "alive": True,
        "foreground_pgid": 4242,
        "tty": tty,
        "started_at": "Sat Aug 1 12:00:00 2026",
        "probe_unknown": False,
    }
    payloads: list[tuple[Path, bytes, str]] = []

    def probe(pid: int):
        if state["probe_unknown"]:
            return _PROBE_UNKNOWN
        if not state["alive"]:
            return None
        return ProcessSnapshot(
            pid=pid,
            uid=os.getuid(),
            tty=Path(state["tty"]),
            started_at=str(state["started_at"]),
            pgid=pid,
            foreground_pgid=int(state["foreground_pgid"]),
            command="codex --dangerously-bypass-approvals-and-sandbox",
        )

    def presenter(path: Path, payload: bytes, *, terminal_app: str) -> str:
        payloads.append((path, payload, terminal_app))
        return "ghostty-applescript"

    bridge = TerminalBridge(
        tmp_cfg,
        process_probe=probe,
        presenter=presenter,
        transport_probe=_allow_transport,
        process_binding_probe=_simulate_process_bound_transport,
    )
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
        bridge = TerminalBridge(
            tmp_cfg,
            process_probe=probe,
            transport_probe=_allow_transport,
            process_binding_probe=_simulate_process_bound_transport,
        )
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
    ("term_program", "expected"),
    [
        pytest.param("Apple_Terminal", "Terminal", id="apple-terminal"),
        pytest.param("iTerm.app", "iTerm2", id="iterm-app"),
    ],
)
def test_register_normalizes_real_term_program_aliases(
    tmp_cfg,
    term_program: str,
    expected: str,
) -> None:
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
        registration = TerminalBridge(
            tmp_cfg,
            process_probe=probe,
            transport_probe=_allow_transport,
            process_binding_probe=_simulate_process_bound_transport,
        ).register(
            agent="codex",
            tty=tty,
            pid=4242,
            terminal_app=term_program,
        )
        assert registration.terminal_app == expected
    finally:
        os.close(slave_fd)
        os.close(master_fd)


def test_reregister_rotates_id_and_created_at_for_new_process(tmp_cfg, monkeypatch) -> None:
    master_fd, slave_fd = pty.openpty()
    tty = Path(os.ttyname(slave_fd))
    starts = {
        4242: "Sat Aug 1 12:00:00 2026",
        4343: "Sat Aug 1 12:01:00 2026",
    }
    timestamps = iter(
        [
            "2026-08-01T12:00:00+00:00",
            "2026-08-01T12:01:00+00:00",
            "2026-08-01T12:02:00+00:00",
            "2026-08-01T12:03:00+00:00",
            "2026-08-01T12:04:00+00:00",
        ]
    )
    monkeypatch.setattr(terminal_live, "_now", lambda: next(timestamps))

    def probe(pid: int) -> ProcessSnapshot:
        return ProcessSnapshot(
            pid=pid,
            uid=os.getuid(),
            tty=tty,
            started_at=starts[pid],
            pgid=pid,
            foreground_pgid=pid,
            command="codex",
        )

    try:
        bridge = TerminalBridge(
            tmp_cfg,
            process_probe=probe,
            transport_probe=_allow_transport,
            process_binding_probe=_simulate_process_bound_transport,
        )
        first = bridge.register(agent="codex", tty=tty, pid=4242)
        repeated = bridge.register(agent="codex", tty=tty, pid=4242)
        second = bridge.register(agent="codex", tty=tty, pid=4343)
        starts[4343] = "Sat Aug 1 12:02:00 2026"
        restarted = bridge.register(agent="codex", tty=tty, pid=4343)

        assert repeated.id == first.id
        assert repeated.created_at == first.created_at
        assert second.id != first.id
        assert second.created_at != first.created_at
        assert restarted.id not in {first.id, second.id}
        assert restarted.created_at != second.created_at
        listed = bridge.list()
        assert len(listed) == 1
        assert listed[0].id == restarted.id
        assert listed[0].created_at == restarted.created_at
    finally:
        os.close(slave_fd)
        os.close(master_fd)


def test_register_rejects_agent_name_not_present_in_process_command(tmp_cfg) -> None:
    master_fd, slave_fd = pty.openpty()
    tty = Path(os.ttyname(slave_fd))

    def probe(pid: int) -> ProcessSnapshot:
        return ProcessSnapshot(
            pid=pid,
            uid=os.getuid(),
            tty=tty,
            started_at="native-start:4242:1",
            pgid=pid,
            foreground_pgid=pid,
            command="python unrelated_worker.py",
        )

    try:
        bridge = TerminalBridge(
            tmp_cfg,
            process_probe=probe,
            transport_probe=_allow_transport,
            process_binding_probe=_simulate_process_bound_transport,
        )
        with pytest.raises(TerminalValidationError, match="command does not match"):
            bridge.register(agent="codex", tty=tty, pid=4242)
    finally:
        os.close(slave_fd)
        os.close(master_fd)


def test_register_rejects_terminal_without_safe_exact_tty_capability(tmp_cfg) -> None:
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
        bridge = TerminalBridge(
            tmp_cfg,
            process_probe=probe,
            transport_probe=lambda _tty, _app: False,
            process_binding_probe=_simulate_process_bound_transport,
        )
        with pytest.raises(TerminalValidationError, match="no safe exact-TTY"):
            bridge.register(agent="codex", tty=tty, pid=4242)
        assert bridge.list() == []
    finally:
        os.close(slave_fd)
        os.close(master_fd)


def test_default_process_binding_gate_rejects_registration_and_existing_delivery(tmp_cfg) -> None:
    master_fd, slave_fd = pty.openpty()
    tty = Path(os.ttyname(slave_fd))
    payloads: list[bytes] = []

    def probe(pid: int) -> ProcessSnapshot:
        return ProcessSnapshot(
            pid=pid,
            uid=os.getuid(),
            tty=tty,
            started_at="native-start:4242:1",
            pgid=pid,
            foreground_pgid=pid,
            command="codex",
        )

    def present(_tty: Path, payload: bytes, *, terminal_app: str) -> str:
        payloads.append(payload)
        return "ghostty-applescript"

    try:
        enabled = TerminalBridge(
            tmp_cfg,
            process_probe=probe,
            presenter=present,
            transport_probe=_allow_transport,
            process_binding_probe=_simulate_process_bound_transport,
        )
        registration = enabled.register(agent="codex", tty=tty, pid=4242)
        disabled = TerminalBridge(
            tmp_cfg,
            process_probe=probe,
            presenter=present,
            transport_probe=_allow_transport,
        )

        with pytest.raises(TerminalValidationError, match="process-bound"):
            disabled.register(agent="codex", tty=tty, pid=4242)
        with pytest.raises(TerminalValidationError, match="process-bound"):
            disabled.send(
                registration.id,
                "must not reach replacement",
                message_id="process-swap-blocked",
            )

        assert payloads == []
        assert disabled.history() == []
        assert disabled.list() == []
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
        "hello\x1b[31m!\x1b[0m\r",
        message_id="msg-1",
    )

    assert payloads == [(Path(registration.tty), b"hello!\r", "Ghostty")]
    assert first.status == "delivered"
    assert first.transport == "ghostty-applescript"
    assert second.receipt_id == first.receipt_id
    assert second.status == "duplicate"


def test_send_escapes_line_controls_and_emits_only_requested_return(registered_bridge) -> None:
    bridge, registration, payloads, _state = registered_bridge

    bridge.send(
        registration.id,
        "first\nsecond\tcolumn\rthird",
        submit=False,
        message_id="no-submit-controls",
    )
    bridge.send(
        registration.id,
        "first\nsecond\tcolumn\rthird",
        submit=True,
        message_id="submit-controls",
    )

    draft = payloads[0][1]
    submitted = payloads[1][1]
    assert draft == b"first\\nsecond\\tcolumnthird"
    assert b"\n" not in draft and b"\t" not in draft and b"\r" not in draft
    assert submitted == draft + b"\r"
    assert submitted.count(b"\r") == 1


@pytest.mark.parametrize("terminal_app", ["", "Ghostty", "Terminal", "iTerm", "iTerm2", "tmux"])
def test_every_transport_receives_single_line_data_only(
    registered_bridge,
    terminal_app: str,
) -> None:
    bridge, registration, payloads, _state = registered_bridge
    updated = bridge.register(
        agent="codex",
        tty=registration.tty,
        pid=registration.pid,
        terminal_app=terminal_app,
    )

    bridge.send(
        updated.id,
        "line one\nline two\tvalue",
        message_id=f"transport-controls-{terminal_app or 'tiocsti'}",
    )

    path, payload, normalized_app = payloads[-1]
    assert path == Path(registration.tty)
    assert payload == b"line one\\nline two\\tvalue\r"
    assert payload.count(b"\r") == 1
    assert b"\n" not in payload and b"\t" not in payload
    assert normalized_app in {"", "Ghostty", "Terminal", "iTerm", "iTerm2", "tmux"}


@pytest.mark.parametrize("change", ["payload", "target", "sender", "kind"])
def test_message_id_reuse_with_changed_fingerprint_is_conflict(
    registered_bridge,
    change: str,
) -> None:
    bridge, registration, payloads, _state = registered_bridge
    bridge.send(registration.id, "same body", message_id=f"fingerprint-{change}")

    with pytest.raises(TerminalValidationError, match="conflicts"):
        if change == "payload":
            bridge.send(registration.id, "different body", message_id="fingerprint-payload")
        elif change == "target":
            bridge.send(
                "term-0000000000000000",
                "same body",
                message_id="fingerprint-target",
            )
        elif change == "sender":
            bridge.send(
                registration.id,
                "same body",
                sender=registration.id,
                message_id="fingerprint-sender",
            )
        else:
            bridge.enter(registration.id, message_id="fingerprint-kind")

    assert len(payloads) == 1


def test_forged_live_sender_is_rejected_before_presentation(registered_bridge) -> None:
    bridge, registration, payloads, _state = registered_bridge

    with pytest.raises(TerminalValidationError, match="sender was not found"):
        bridge.send(
            registration.id,
            "sender must be live",
            sender="term-0000000000000000",
            message_id="forged-sender",
        )

    receipt = bridge.history()[0]
    assert receipt.status == "failed"
    assert receipt.sender_id == "term-0000000000000000"
    assert payloads == []


def test_cross_bridge_tty_lock_serializes_messages_and_concurrent_dedupe(tmp_cfg) -> None:
    master_fd, slave_fd = pty.openpty()
    tty = Path(os.ttyname(slave_fd))
    trace: list[str] = []
    calls: list[bytes] = []

    def probe(pid: int) -> ProcessSnapshot:
        return ProcessSnapshot(
            pid=pid,
            uid=os.getuid(),
            tty=tty,
            started_at="native-start:4242:1",
            pgid=pid,
            foreground_pgid=pid,
            command="codex",
        )

    def present(_tty: Path, payload: bytes, *, terminal_app: str) -> str:
        calls.append(payload)
        for byte in payload:
            trace.append(chr(byte))
            time.sleep(0.002)
        return "ghostty-applescript"

    try:
        options = {
            "process_probe": probe,
            "presenter": present,
            "transport_probe": _allow_transport,
            "process_binding_probe": _simulate_process_bound_transport,
        }
        first_bridge = TerminalBridge(tmp_cfg, **options)
        second_bridge = TerminalBridge(tmp_cfg, **options)
        registration = first_bridge.register(agent="codex", tty=tty, pid=4242)

        barrier = threading.Barrier(3)
        errors: list[BaseException] = []

        def send(bridge: TerminalBridge, body: str, message_id: str) -> None:
            try:
                barrier.wait()
                bridge.send(registration.id, body, submit=False, message_id=message_id)
            except BaseException as exc:  # captured for assertion in the main thread
                errors.append(exc)

        threads = [
            threading.Thread(target=send, args=(first_bridge, "AAAAA", "locked-a")),
            threading.Thread(target=send, args=(second_bridge, "BBBBB", "locked-b")),
        ]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=5)

        assert errors == []
        assert "".join(trace) in {"AAAAABBBBB", "BBBBBAAAAA"}

        calls.clear()
        trace.clear()
        barrier = threading.Barrier(3)
        results = []

        def retry(bridge: TerminalBridge) -> None:
            barrier.wait()
            results.append(
                bridge.send(
                    registration.id,
                    "CCCCC",
                    submit=False,
                    message_id="locked-dedupe",
                )
            )

        threads = [
            threading.Thread(target=retry, args=(first_bridge,)),
            threading.Thread(target=retry, args=(second_bridge,)),
        ]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=5)

        assert calls == [b"CCCCC"]
        assert {receipt.status for receipt in results} == {"delivered", "duplicate"}
    finally:
        os.close(slave_fd)
        os.close(master_fd)


def test_send_refuses_when_agent_is_not_foreground(registered_bridge) -> None:
    bridge, registration, payloads, state = registered_bridge
    state["foreground_pgid"] = 9999

    with pytest.raises(TerminalValidationError, match="foreground"):
        bridge.send(registration.id, "do not deliver")

    assert payloads == []


def test_transient_probe_keeps_registration_during_grace_and_refreshes_last_seen(
    registered_bridge,
) -> None:
    bridge, registration, _payloads, state = registered_bridge
    state["probe_unknown"] = True

    assert bridge.list() == [registration]
    with bridge._connect() as conn:
        old = (datetime.now(UTC) - timedelta(minutes=5)).isoformat(timespec="seconds")
        conn.execute(
            "UPDATE terminal_registrations SET last_seen_at = ? WHERE id = ?",
            (old, registration.id),
        )

    assert bridge.list() == []
    with bridge._connect() as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM terminal_registrations WHERE id = ?",
                (registration.id,),
            ).fetchone()[0]
            == 1
        )

    state["probe_unknown"] = False
    refreshed = bridge.list()
    assert len(refreshed) == 1
    assert refreshed[0].last_seen_at != old


def test_rejected_targets_are_failed_receipts_without_bodies(registered_bridge) -> None:
    bridge, registration, payloads, state = registered_bridge
    other_master_fd, other_slave_fd = pty.openpty()
    other_tty = Path(os.ttyname(other_slave_fd))
    attempts = [
        ("term-0000000000000000", "missing secret", "not found"),
    ]
    try:
        for target_id, body, error in attempts:
            with pytest.raises(TerminalValidationError, match=error):
                bridge.send(target_id, body, message_id="rejected-missing")

        state["alive"] = False
        with pytest.raises(TerminalValidationError, match="stale"):
            bridge.send(registration.id, "stale secret", message_id="rejected-stale")

        state["alive"] = True
        state["tty"] = other_tty
        with pytest.raises(TerminalValidationError, match="changed TTY"):
            bridge.send(registration.id, "changed secret", message_id="rejected-changed")

        state["tty"] = Path(registration.tty)
        state["foreground_pgid"] = 9999
        with pytest.raises(TerminalValidationError, match="foreground"):
            bridge.send(registration.id, "background secret", message_id="rejected-background")

        receipts = bridge.history()
        assert {receipt.message_id for receipt in receipts} == {
            "rejected-missing",
            "rejected-stale",
            "rejected-changed",
            "rejected-background",
        }
        assert all(receipt.status == "failed" for receipt in receipts)
        assert all(receipt.error == "TerminalValidationError" for receipt in receipts)
        assert all(not receipt.delivered_at for receipt in receipts)
        assert "secret" not in repr(receipts)
        assert payloads == []
    finally:
        os.close(other_slave_fd)
        os.close(other_master_fd)


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
        pytest.param("\n" * 9_000, id="oversized-after-sanitization"),
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


def test_send_rejects_surrogate_body_without_retaining_codec_context(registered_bridge) -> None:
    bridge, registration, payloads, _state = registered_bridge

    with pytest.raises(TerminalValidationError, match="valid UTF-8") as raised:
        bridge.send(registration.id, "secret\udcffbody")

    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert "secret" not in repr(raised.value)
    assert payloads == []


def test_send_rejects_unbounded_or_malformed_routing_ids(registered_bridge) -> None:
    bridge, registration, payloads, _state = registered_bridge

    with pytest.raises(TerminalValidationError, match="message id"):
        bridge.send(registration.id, "hello", message_id="x" * 129)
    with pytest.raises(TerminalValidationError, match="sender id"):
        bridge.send(registration.id, "hello", sender="not a terminal")
    with pytest.raises(TerminalValidationError, match="target id"):
        bridge.send("not a terminal", "hello")

    assert payloads == []


@pytest.mark.parametrize("invalid_transport", ["", "terminal-applescript", "custom"])
def test_invalid_presenter_transport_is_failed_and_never_retried(
    registered_bridge,
    monkeypatch,
    invalid_transport: str,
) -> None:
    bridge, registration, payloads, _state = registered_bridge
    calls: list[bytes] = []

    def invalid(_tty: Path, payload: bytes, *, terminal_app: str) -> str:
        calls.append(payload)
        return invalid_transport

    monkeypatch.setattr(bridge, "_present", invalid)

    with pytest.raises(TerminalValidationError, match="invalid transport"):
        bridge.send(registration.id, "body", message_id=f"invalid-{invalid_transport or 'empty'}")
    duplicate = bridge.send(
        registration.id,
        "body",
        message_id=f"invalid-{invalid_transport or 'empty'}",
    )

    assert duplicate.status == "duplicate"
    assert bridge.history()[0].status == "failed"
    assert calls == [b"body\r"]
    assert payloads == []


def test_missing_targets_share_bounded_lock_bucket(registered_bridge) -> None:
    bridge, _registration, payloads, _state = registered_bridge

    for index in range(100):
        with pytest.raises(TerminalValidationError, match="not found"):
            bridge.send(
                f"term-{index:016x}",
                "never delivered",
                message_id=f"missing-lock-{index}",
            )

    lockfiles = list(bridge._lock_dir.glob("*.lock"))
    assert len(lockfiles) <= 2
    assert payloads == []


def test_interprocess_lock_wait_is_bounded_before_receipt(
    registered_bridge,
    monkeypatch,
) -> None:
    bridge, registration, payloads, _state = registered_bridge
    ticks = iter((0.0, terminal_live._LOCK_TIMEOUT_SECONDS + 1.0))

    def blocked(_fd: int, _operation: int) -> None:
        raise BlockingIOError(errno.EAGAIN, "busy")

    monkeypatch.setattr(terminal_live.fcntl, "flock", blocked)
    monkeypatch.setattr(terminal_live.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(terminal_live.time, "sleep", lambda _seconds: None)

    with pytest.raises(TerminalValidationError, match="lock timed out"):
        bridge.send(registration.id, "not attempted", message_id="lock-timeout")

    assert bridge.history() == []
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
        bridge = TerminalBridge(
            tmp_cfg,
            process_probe=probe,
            presenter=fail,
            transport_probe=_allow_transport,
            process_binding_probe=_simulate_process_bound_transport,
        )
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


def test_safe_transport_failure_is_preserved_only_in_internal_receipt(registered_bridge) -> None:
    bridge, registration, payloads, _state = registered_bridge

    def fail(_tty: Path, _payload: bytes, *, terminal_app: str) -> str:
        raise TerminalDeliveryError("terminal automation timed out")

    bridge._present = fail

    with pytest.raises(TerminalValidationError, match="terminal automation timed out"):
        bridge.send(registration.id, "secret body", message_id="safe-diagnostic")

    receipt = bridge.history()[0]
    assert receipt.status == "failed"
    assert receipt.error == "terminal automation timed out"
    assert "secret body" not in repr(receipt)
    assert payloads == []


def test_presenter_timeout_is_body_free_and_never_leaves_pending_receipt(tmp_cfg) -> None:
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

    def timeout(_tty: Path, payload: bytes, *, terminal_app: str) -> str:
        raise subprocess.TimeoutExpired(["osascript", payload.decode()], 5)

    try:
        bridge = TerminalBridge(
            tmp_cfg,
            process_probe=probe,
            presenter=timeout,
            transport_probe=_allow_transport,
            process_binding_probe=_simulate_process_bound_transport,
        )
        registration = bridge.register(agent="codex", tty=tty, pid=4242)

        with pytest.raises(TerminalValidationError, match="timed out") as raised:
            bridge.send(
                registration.id,
                "timeout secret body",
                message_id="timeout-failure",
            )

        receipt = bridge.history()[0]
        assert receipt.status == "failed"
        assert receipt.error == "OSError"
        assert receipt.delivered_at == ""
        assert raised.value.__cause__ is None
        assert raised.value.__context__ is None
        assert "timeout secret body" not in repr((raised.value, receipt))
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
        TerminalBridge(
            tmp_cfg,
            process_probe=probe,
            presenter=lambda *_a, **_kw: "ghostty-applescript",
            transport_probe=_allow_transport,
            process_binding_probe=_simulate_process_bound_transport,
        )
        assert (tmp_cfg.state_dir / "terminal_live.db").stat().st_mode & 0o777 == 0o600
    finally:
        os.close(slave_fd)
        os.close(master_fd)


@pytest.mark.parametrize(
    ("stage", "expected_presentations"),
    [
        pytest.param("before_presenter", 0, id="before-presenter"),
        pytest.param("after_presenter", 1, id="after-presenter"),
        pytest.param("before_finalize", 1, id="before-finalize"),
    ],
)
def test_crash_failpoints_recover_pending_as_unknown_without_redelivery(
    tmp_cfg,
    stage: str,
    expected_presentations: int,
) -> None:
    master_fd, slave_fd = pty.openpty()
    tty = Path(os.ttyname(slave_fd))
    payloads: list[bytes] = []

    def probe(pid: int) -> ProcessSnapshot:
        return ProcessSnapshot(
            pid=pid,
            uid=os.getuid(),
            tty=tty,
            started_at="native-start:4242:1",
            pgid=pid,
            foreground_pgid=pid,
            command="codex",
        )

    def present(_tty: Path, payload: bytes, *, terminal_app: str) -> str:
        payloads.append(payload)
        return "ghostty-applescript"

    def failpoint(name: str) -> None:
        if name == stage:
            raise RuntimeError(f"simulated crash at {stage}")

    try:
        bridge = TerminalBridge(
            tmp_cfg,
            process_probe=probe,
            presenter=present,
            transport_probe=_allow_transport,
            process_binding_probe=_simulate_process_bound_transport,
            failpoint=failpoint,
        )
        registration = bridge.register(agent="codex", tty=tty, pid=4242)

        with pytest.raises(RuntimeError, match="simulated crash"):
            bridge.send(registration.id, "crash body", message_id=f"crash-{stage}")

        assert len(payloads) == expected_presentations
        assert bridge.history()[0].status == "pending"
        retry = bridge.send(registration.id, "crash body", message_id=f"crash-{stage}")
        assert retry.status == "unknown"
        assert len(payloads) == expected_presentations

        recovered = TerminalBridge(
            tmp_cfg,
            process_probe=probe,
            presenter=present,
            transport_probe=_allow_transport,
            process_binding_probe=_simulate_process_bound_transport,
            receipt_owner_probe=lambda _pid, _started: False,
        )
        receipt = recovered.history()[0]
        assert receipt.status == "unknown"
        assert receipt.error == "DeliveryStateUnknown"
        assert not receipt.delivered_at
    finally:
        os.close(slave_fd)
        os.close(master_fd)


def test_receipt_schema_migrates_old_pending_rows_to_unknown(tmp_cfg) -> None:
    tmp_cfg.state_dir.mkdir(parents=True, exist_ok=True)
    db = tmp_cfg.state_dir / "terminal_live.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE terminal_receipts (
            receipt_id TEXT PRIMARY KEY,
            message_id TEXT NOT NULL UNIQUE,
            target_id TEXT NOT NULL,
            sender_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            status TEXT NOT NULL,
            transport TEXT NOT NULL,
            error TEXT NOT NULL,
            created_at TEXT NOT NULL,
            delivered_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO terminal_receipts VALUES "
        "('rcpt-old', 'msg-old', 'term-old', '', 'message', 'pending', '', '', ?, '')",
        ((datetime.now(UTC) - timedelta(days=1)).isoformat(),),
    )
    conn.commit()
    conn.close()

    bridge = TerminalBridge(tmp_cfg)

    with bridge._connect() as migrated:
        columns = {row["name"] for row in migrated.execute("PRAGMA table_info(terminal_receipts)")}
    assert {"fingerprint", "owner_pid", "owner_started_at"} <= columns
    receipt = bridge.history()[0]
    assert receipt.status == "unknown"
    assert receipt.error == "DeliveryStateUnknown"


def test_receipt_retention_bounds_final_rows_and_preserves_ambiguous(tmp_cfg, monkeypatch) -> None:
    monkeypatch.setattr(terminal_live, "_MAX_FINAL_RECEIPTS", 2)
    bridge = TerminalBridge(tmp_cfg)
    now = _now_for_test = datetime.now(UTC).isoformat()
    with bridge._connect() as conn:
        for index, status in enumerate(("delivered", "failed", "delivered", "pending", "unknown")):
            conn.execute(
                """
                INSERT INTO terminal_receipts (
                    receipt_id, message_id, target_id, sender_id, kind, status,
                    transport, error, created_at, delivered_at, fingerprint,
                    owner_pid, owner_started_at
                ) VALUES (?, ?, '', '', 'message', ?, '', '', ?, '', ?, 0, '')
                """,
                (f"rcpt-{index}", f"msg-{index}", status, now, f"fingerprint-{index}"),
            )
        bridge._prune_receipts(conn)
        rows = conn.execute("SELECT status FROM terminal_receipts ORDER BY rowid").fetchall()
        tombstones = conn.execute(
            "SELECT message_id, fingerprint FROM terminal_receipt_tombstones"
        ).fetchall()

    statuses = [str(row["status"]) for row in rows]
    assert statuses.count("delivered") + statuses.count("failed") == 2
    assert "pending" in statuses
    assert "unknown" in statuses
    assert [(str(row["message_id"]), str(row["fingerprint"])) for row in tombstones] == [
        ("msg-0", "fingerprint-0")
    ]


def test_pruned_receipt_tombstone_prevents_retry_and_conflicting_reuse(
    registered_bridge,
    monkeypatch,
) -> None:
    bridge, registration, payloads, _state = registered_bridge
    monkeypatch.setattr(terminal_live, "_MAX_FINAL_RECEIPTS", 2)

    first = bridge.send(registration.id, "first", message_id="pruned-first")
    bridge.send(registration.id, "second", message_id="pruned-second")
    bridge.send(registration.id, "third", message_id="pruned-third")

    retry = bridge.send(registration.id, "first", message_id="pruned-first")
    with pytest.raises(TerminalValidationError, match="conflicts"):
        bridge.send(registration.id, "changed", message_id="pruned-first")

    assert retry.receipt_id == first.receipt_id
    assert retry.status == "duplicate"
    assert retry.error == "ReceiptPruned"
    assert len(payloads) == 3


def test_receipt_key_capacity_fails_closed_but_keeps_tombstone_retries(
    registered_bridge,
    monkeypatch,
) -> None:
    bridge, registration, payloads, _state = registered_bridge
    monkeypatch.setattr(terminal_live, "_MAX_FINAL_RECEIPTS", 1)
    monkeypatch.setattr(terminal_live, "_MAX_RECEIPT_KEYS", 2)

    first = bridge.send(registration.id, "first", message_id="capacity-first")
    bridge.send(registration.id, "second", message_id="capacity-second")

    with pytest.raises(TerminalValidationError, match="capacity is exhausted"):
        bridge.send(registration.id, "third", message_id="capacity-third")
    retry = bridge.send(registration.id, "first", message_id="capacity-first")

    assert retry.receipt_id == first.receipt_id
    assert retry.status == "duplicate"
    assert len(payloads) == 2
    assert {receipt.message_id for receipt in bridge.history()} == {"capacity-second"}


def test_native_process_birth_identity_has_subsecond_or_tick_precision() -> None:
    identity = _process_birth_identity(os.getpid())

    assert identity
    assert identity.startswith(("linux-start-ticks:", "darwin-start:"))


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin libproc layout only")
def test_darwin_process_birth_identity_matches_live_libproc_layout() -> None:
    identity = _darwin_process_birth_identity(os.getpid())
    prefix, seconds, microseconds = identity.split(":")

    assert prefix == "darwin-start"
    assert abs(time.time() - int(seconds)) < 60
    assert 0 <= int(microseconds) < 1_000_000
