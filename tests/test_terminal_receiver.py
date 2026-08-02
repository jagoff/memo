import os
import pty
import stat
import time

import pytest

from memo.terminal_receiver import ReceiverClient, ReceiverSession, ReceiverSupervisor


def _session():
    pid, fd = pty.fork()
    if pid == 0:
        os.execvp("/bin/cat", ["cat"])
    return ReceiverSession(fd, pid)


def test_socket_and_state_modes_and_idempotency(tmp_path):
    s = _session(); sup = ReceiverSupervisor(tmp_path, s); path = sup.start()
    try:
        assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o700
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        c = ReceiverClient(path, sup.capability)
        first = c.send(message_id="m1", text="hello\x00\n")
        second = c.send(message_id="m1", text="different")
        assert first == second and first["ok"]
    finally:
        sup.close()
    assert not path.exists()


def test_capability_rejected(tmp_path):
    s = _session(); sup = ReceiverSupervisor(tmp_path, s); path = sup.start()
    try:
        out = ReceiverClient(path, "wrong").send(message_id="x", text="x")
        assert out["ok"] is False and out["error"] == "unauthorized"
    finally:
        sup.close()


def test_dead_child_rejected(tmp_path):
    s = _session(); sup = ReceiverSupervisor(tmp_path, s); path = sup.start()
    try:
        os.kill(s.child_pid, 9)
        for _ in range(20):
            if not s.alive(): break
            time.sleep(0.01)
        out = ReceiverClient(path, sup.capability).send(message_id="dead", text="x")
        assert out["ok"] is False
    finally:
        sup.close()


def test_frame_limit(tmp_path):
    s = _session(); sup = ReceiverSupervisor(tmp_path, s); path = sup.start()
    try:
        with pytest.raises(ValueError):
            ReceiverClient(path, sup.capability).send(message_id="big", text="x" * 70000)
    finally:
        sup.close()

