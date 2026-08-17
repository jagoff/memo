import os
import pty
import stat
import time

import pytest

from memo.terminal_receiver import (
    ReceiverClient,
    ReceiverSession,
    ReceiverSupervisor,
    read_capability_file,
)


def _session():
    pid, fd = pty.fork()
    if pid == 0:
        os.execvp("/bin/cat", ["cat"])  # noqa: S606 - test child, no shell
    return ReceiverSession(fd, pid)


def test_socket_and_state_modes_and_idempotency(tmp_path):
    s = _session()
    sup = ReceiverSupervisor(tmp_path, s)
    path = sup.start()
    try:
        assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o700
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        c = ReceiverClient(path, sup.capability)
        first = c.send(message_id="m1", text="hello\x00\n")
        second = c.send(message_id="m1", text="hello\x00\n")
        assert first == second and first["ok"]
        conflict = c.send(message_id="m1", text="different")
        assert conflict == {"ok": False, "message_id": "m1", "error": "message_id conflict"}
    finally:
        sup.close()
    assert not path.exists()


def test_capability_rejected(tmp_path):
    s = _session()
    sup = ReceiverSupervisor(tmp_path, s)
    path = sup.start()
    try:
        out = ReceiverClient(path, "wrong").send(message_id="x", text="x")
        assert out["ok"] is False and out["error"] == "unauthorized"
    finally:
        sup.close()


def test_dead_child_rejected(tmp_path):
    s = _session()
    sup = ReceiverSupervisor(tmp_path, s)
    path = sup.start()
    try:
        os.kill(s.child_pid, 9)
        for _ in range(20):
            if not s.alive():
                break
            time.sleep(0.01)
        out = ReceiverClient(path, sup.capability).send(message_id="dead", text="x")
        assert out["ok"] is False
    finally:
        sup.close()


def test_frame_limit(tmp_path):
    s = _session()
    sup = ReceiverSupervisor(tmp_path, s)
    path = sup.start()
    try:
        with pytest.raises(ValueError):
            ReceiverClient(path, sup.capability).send(message_id="big", text="x" * 70000)
    finally:
        sup.close()


def test_enter_and_conflicting_message_id_are_deterministic(tmp_path):
    s = _session()
    sup = ReceiverSupervisor(tmp_path, s)
    path = sup.start()
    try:
        c = ReceiverClient(path, sup.capability)
        entered = c.enter(message_id="enter-1")
        assert entered["ok"] is True
        conflict = c.enter(message_id="enter-1")
        assert conflict == entered
        write_conflict = c.send(message_id="enter-1", text="different")
        assert write_conflict["error"] == "message_id conflict"
    finally:
        sup.close()


def test_capability_file_requires_exact_owner_mode_and_content(tmp_path):
    path = tmp_path / "cap"
    path.write_text("secret\n", encoding="ascii")
    path.chmod(0o600)
    assert read_capability_file(path) == "secret"
    path.write_text("", encoding="ascii")
    with pytest.raises(ValueError, match="empty"):
        read_capability_file(path)
    path.write_text("secret", encoding="ascii")
    path.chmod(0o640)
    with pytest.raises(ValueError, match="0600"):
        read_capability_file(path)


def test_session_rejects_invalid_inputs_and_oversized_payload():
    """The rejections a caller can actually trigger, in the order they fire.

    Note the layering: `write` runs the message through terminal_live's 16 KiB
    cap (TerminalValidationError) BEFORE the receiver's own 64 KiB "frame too
    large" ValueError, so the terminal cap is the binding one for this path.
    Asserting the receiver limit here would test a branch `write` cannot reach.
    """
    from memo.errors import TerminalValidationError

    with pytest.raises(ValueError, match="invalid receiver"):
        ReceiverSession(-1, 1)
    s = _session()
    try:
        with pytest.raises(ValueError, match="string"):
            s.write(123)  # type: ignore[arg-type]
        with pytest.raises(TerminalValidationError, match="16 KiB"):
            s.write("x" * (16 * 1024 + 1))
        assert s.alive()
    finally:
        s.close()
        s.close()
