"""Receiver transport: the guard rails, not just the happy path.

Complements test_terminal_receiver.py, which covers the authenticated
send/enter round trip. These exercise what the transport refuses — malformed
frames, unknown ops, bad capability files, dead identities — and the
fork/close lifecycle, so the security-relevant branches are held by tests
rather than by review.
"""

from __future__ import annotations

import json
import os
import pty
import socket
import stat

import pytest

from memo.terminal_receiver import (
    MAX_FRAME,
    ReceiverClient,
    ReceiverSession,
    ReceiverSupervisor,
    read_capability_file,
)

pytestmark = pytest.mark.skipif(os.name != "posix", reason="receiver transport is Unix-only")


def _session() -> ReceiverSession:
    pid, fd = pty.fork()
    if pid == 0:
        os.execvp("/bin/cat", ["cat"])  # noqa: S606 - test child, no shell
    return ReceiverSession(fd, pid)


def _raw_request(path, payload: bytes) -> dict:
    """Send a frame the typed client would refuse to build."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(5.0)
        sock.connect(str(path))
        sock.sendall(payload)
        data = b""
        while not data.endswith(b"\n"):
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
    return json.loads(data.decode("utf-8"))


# --- capability file -------------------------------------------------------


def test_capability_file_must_be_private(tmp_path) -> None:
    path = tmp_path / "cap"
    path.write_text("s3cret", encoding="ascii")
    path.chmod(0o644)

    with pytest.raises(ValueError, match="0600"):
        read_capability_file(path)


def test_capability_file_must_not_be_empty(tmp_path) -> None:
    path = tmp_path / "cap"
    path.write_text("   \n", encoding="ascii")
    path.chmod(0o600)

    with pytest.raises(ValueError, match="empty"):
        read_capability_file(path)


def test_capability_file_is_read_and_stripped(tmp_path) -> None:
    path = tmp_path / "cap"
    path.write_text("  s3cret\n", encoding="ascii")
    path.chmod(0o600)

    assert read_capability_file(path) == "s3cret"


# --- session identity ------------------------------------------------------


def test_session_rejects_an_impossible_identity() -> None:
    with pytest.raises(ValueError, match="invalid receiver PTY identity"):
        ReceiverSession(-1, 4242)
    with pytest.raises(ValueError, match="invalid receiver PTY identity"):
        ReceiverSession(3, 1)


def test_fork_requires_a_command() -> None:
    with pytest.raises(ValueError, match="command is required"):
        ReceiverSession.fork([])
    with pytest.raises(ValueError, match="command is required"):
        ReceiverSession.fork(["   "])


def test_fork_captures_child_identity_and_close_is_idempotent() -> None:
    session = ReceiverSession.fork(["/bin/cat"])
    try:
        assert session.child_start and session.child_tty
        assert session.alive() is True
    finally:
        session.close()
    assert session.alive() is False
    session.close()  # second close must not raise


def test_write_after_close_is_refused() -> None:
    session = ReceiverSession.fork(["/bin/cat"])
    session.close()

    with pytest.raises((OSError, RuntimeError)):
        session.write("anything")


def test_write_rejects_non_text() -> None:
    session = ReceiverSession.fork(["/bin/cat"])
    try:
        with pytest.raises(ValueError, match="must be a string"):
            session.write(b"bytes")  # type: ignore[arg-type]
    finally:
        session.close()


# --- request handling ------------------------------------------------------


def test_malformed_json_is_rejected(tmp_path) -> None:
    supervisor = ReceiverSupervisor(tmp_path, _session())
    path = supervisor.start()
    try:
        assert _raw_request(path, b"{not json\n") == {"ok": False, "error": "invalid request"}
    finally:
        supervisor.close()


def test_a_non_object_request_is_rejected(tmp_path) -> None:
    supervisor = ReceiverSupervisor(tmp_path, _session())
    path = supervisor.start()
    try:
        assert _raw_request(path, b'["not","an","object"]\n') == {
            "ok": False,
            "error": "invalid request",
        }
    finally:
        supervisor.close()


def test_an_oversized_frame_is_rejected_by_the_server(tmp_path) -> None:
    supervisor = ReceiverSupervisor(tmp_path, _session())
    path = supervisor.start()
    try:
        payload = b'{"capability":"x","message_id":"m","op":"write","text":"'
        payload += b"x" * (MAX_FRAME + 10) + b'"}\n'
        assert _raw_request(path, payload) == {"ok": False, "error": "frame too large"}
    finally:
        supervisor.close()


@pytest.mark.parametrize("message_id", ["", 42, "x" * 200, None])
def test_message_id_is_required_and_bounded(tmp_path, message_id) -> None:
    supervisor = ReceiverSupervisor(tmp_path, _session())
    path = supervisor.start()
    try:
        request = {"capability": supervisor.capability, "op": "enter", "message_id": message_id}
        response = _raw_request(path, json.dumps(request).encode("utf-8") + b"\n")
        assert response == {"ok": False, "error": "message_id required"}
    finally:
        supervisor.close()


def test_an_unknown_op_is_refused(tmp_path) -> None:
    supervisor = ReceiverSupervisor(tmp_path, _session())
    path = supervisor.start()
    try:
        request = {"capability": supervisor.capability, "message_id": "u1", "op": "sudo"}
        response = _raw_request(path, json.dumps(request).encode("utf-8") + b"\n")
        assert response["ok"] is False
    finally:
        supervisor.close()


def test_a_non_string_capability_is_unauthorized(tmp_path) -> None:
    supervisor = ReceiverSupervisor(tmp_path, _session())
    path = supervisor.start()
    try:
        request = {"capability": 1234, "message_id": "c1", "op": "enter"}
        response = _raw_request(path, json.dumps(request).encode("utf-8") + b"\n")
        assert response == {"ok": False, "error": "unauthorized"}
    finally:
        supervisor.close()


def test_submit_writes_a_carriage_return(tmp_path) -> None:
    supervisor = ReceiverSupervisor(tmp_path, _session())
    path = supervisor.start()
    try:
        client = ReceiverClient(path, supervisor.capability)
        plain = client.send(message_id="s1", text="abc")
        submitted = client.send(message_id="s2", text="abc", submit=True)
        assert plain["ok"] and submitted["ok"]
        # The submitted frame carries the extra CR.
        assert submitted["bytes"] == plain["bytes"] + 1
    finally:
        supervisor.close()


def test_close_removes_the_socket_and_is_idempotent(tmp_path) -> None:
    supervisor = ReceiverSupervisor(tmp_path, _session())
    path = supervisor.start()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

    supervisor.close()
    assert not path.exists()
    supervisor.close()  # second close must not raise
