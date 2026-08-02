"""Receiver-bound Unix transport for safe Memo terminal delivery.

The supervisor owns a nested PTY and is the only process that writes to its
master.  Senders talk to a short-lived, user-owned Unix socket and must prove
both peer UID and a per-session capability.  The child PID/start identity and
foreground process group are checked immediately before every write.
"""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import hmac
import json
import os
import pty
import secrets
import socket
import struct
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from memo.daemon_common import socket_path_for
from memo.terminal_live import ProcessSnapshot, _process_snapshot, _strip_terminal_controls

MAX_FRAME = 64 * 1024
_MAX_RECEIPTS = 1024
_SOCKET_TIMEOUT = 5.0
_MESSAGE_ID_MAX = 128


def read_capability_file(path: Path | str) -> str:
    """Read a mode-0600 capability without putting it in process argv."""

    capability_path = Path(path)
    info = capability_path.stat()
    if info.st_uid != os.getuid() or info.st_mode & 0o777 != 0o600:
        raise ValueError("capability file must be owned by the current user and mode 0600")
    with capability_path.open(encoding="ascii") as stream:
        capability = stream.read().strip()
    if not capability:
        raise ValueError("capability file is empty")
    return capability


def _peer_uid(conn: socket.socket) -> int | None:
    """Return the authenticated peer UID, or ``None`` if unsupported."""

    if hasattr(socket, "SO_PEERCRED"):
        raw = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
        return struct.unpack("3i", raw)[1]
    if hasattr(conn, "getpeereid"):
        return conn.getpeereid()[0]  # type: ignore[attr-defined]
    if sys.platform == "darwin":
        euid = ctypes.c_uint()
        egid = ctypes.c_uint()
        libc = ctypes.CDLL(None, use_errno=True)
        getpeereid = libc.getpeereid
        getpeereid.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_uint),
        ]
        getpeereid.restype = ctypes.c_int
        if getpeereid(conn.fileno(), ctypes.byref(euid), ctypes.byref(egid)) == 0:
            return int(euid.value)
    return None


def _fingerprint(request: dict[str, Any]) -> str:
    material = json.dumps(
        {
            "op": request.get("op"),
            "text": request.get("text"),
            "submit": request.get("submit"),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


class ReceiverSession:
    """Own a PTY master and validate child identity before every write."""

    def __init__(
        self,
        master_fd: int,
        child_pid: int,
        *,
        child_start: str | None = None,
        child_tty: str | None = None,
    ) -> None:
        if master_fd < 0 or child_pid <= 1:
            raise ValueError("invalid receiver PTY identity")
        self.master_fd = master_fd
        self.child_pid = child_pid
        self.child_start = child_start
        self.child_tty = child_tty
        self._closed = False
        self._lock = threading.Lock()
        self._capture_identity()

    @classmethod
    def fork(cls, argv: list[str]) -> ReceiverSession:
        if not argv or not argv[0].strip():
            raise ValueError("receiver command is required")
        pid, fd = pty.fork()
        if pid == 0:
            os.execvp(argv[0], argv)  # noqa: S606 - explicit receiver command, no shell
        session = cls(fd, pid)
        if session.child_start is None or session.child_tty is None:
            session.close()
            raise RuntimeError("receiver child identity could not be established")
        return session

    def _capture_identity(self) -> None:
        for _ in range(100):
            snapshot = _process_snapshot(self.child_pid)
            if isinstance(snapshot, ProcessSnapshot):
                self.child_start = str(snapshot.started_at)
                self.child_tty = str(snapshot.tty)
                return
            time.sleep(0.01)

    def alive(self) -> bool:
        if self._closed:
            return False
        snapshot = _process_snapshot(self.child_pid)
        return self._identity_matches(snapshot, require_foreground=True)

    def _identity_matches(
        self, snapshot: ProcessSnapshot | object, *, require_foreground: bool
    ) -> bool:
        if not isinstance(snapshot, ProcessSnapshot):
            return False
        if self.child_start is None or self.child_tty is None:
            return False
        return (
            hmac.compare_digest(str(snapshot.started_at), self.child_start)
            and str(snapshot.tty) == self.child_tty
            and (not require_foreground or snapshot.pgid == snapshot.foreground_pgid)
        )

    def _write_bytes(self, payload: bytes) -> int:
        if len(payload) > MAX_FRAME:
            raise ValueError("frame too large")
        with self._lock:
            if not self.alive():
                raise RuntimeError("stale or dead child")
            written = 0
            while written < len(payload):
                try:
                    count = os.write(self.master_fd, payload[written:])
                except OSError:
                    raise
                if count <= 0:
                    raise OSError("receiver PTY write made no progress")
                written += count
            return written

    def write(self, text: str, *, submit: bool = False) -> int:
        if not isinstance(text, str):
            raise ValueError("receiver text must be a string")
        sanitized = _strip_terminal_controls(text)
        payload = sanitized.encode("utf-8") + (b"\r" if submit else b"")
        return self._write_bytes(payload)

    def enter(self) -> int:
        return self._write_bytes(b"\r")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            with contextlib.suppress(OSError):
                os.close(self.master_fd)
            snapshot = _process_snapshot(self.child_pid)
            if self._identity_matches(snapshot, require_foreground=False):
                with contextlib.suppress(OSError, ProcessLookupError):
                    os.killpg(os.getpgid(self.child_pid), 15)
            for _ in range(20):
                with contextlib.suppress(ChildProcessError, OSError):
                    waited, _ = os.waitpid(self.child_pid, os.WNOHANG)
                    if waited == self.child_pid:
                        return
                time.sleep(0.01)
            snapshot = _process_snapshot(self.child_pid)
            if self._identity_matches(snapshot, require_foreground=False):
                with contextlib.suppress(OSError, ProcessLookupError):
                    os.killpg(os.getpgid(self.child_pid), 9)
            with contextlib.suppress(ChildProcessError, OSError):
                os.waitpid(self.child_pid, os.WNOHANG)


class ReceiverSupervisor:
    """Authenticated newline-delimited JSON receiver server."""

    def __init__(self, state_dir: Path | str, session: ReceiverSession) -> None:
        if os.name != "posix":
            raise RuntimeError("receiver transport requires Unix")
        self.state_dir = Path(state_dir)
        self.capability = secrets.token_urlsafe(32)
        self.socket_path = socket_path_for(self.state_dir, f"receiver-{secrets.token_hex(8)}")
        self.session = session
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._receipts: OrderedDict[str, tuple[str, dict[str, Any]]] = OrderedDict()
        self._receipts_lock = threading.Lock()

    def start(self) -> Path:
        self.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.state_dir.chmod(0o700)
        self.socket_path.unlink(missing_ok=True)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(self.socket_path))
        self.socket_path.chmod(0o600)
        sock.listen(8)
        sock.settimeout(0.2)
        self._sock = sock
        self._thread = threading.Thread(target=self._serve, name="memo-receiver", daemon=True)
        self._thread.start()
        return self.socket_path

    def _serve(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            conn.settimeout(_SOCKET_TIMEOUT)
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    @staticmethod
    def _response(conn: socket.socket, response: dict[str, Any]) -> None:
        conn.sendall(json.dumps(response, separators=(",", ":")).encode("utf-8") + b"\n")

    def _handle(self, conn: socket.socket) -> None:
        try:
            if _peer_uid(conn) != os.getuid():
                self._response(conn, {"ok": False, "error": "unauthorized"})
                return
            buf = b""
            while len(buf) <= MAX_FRAME:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf += chunk
                if b"\n" not in buf:
                    continue
                line = buf.split(b"\n", 1)[0]
                if len(line) > MAX_FRAME:
                    self._response(conn, {"ok": False, "error": "frame too large"})
                    return
                try:
                    request = json.loads(line.decode("utf-8"))
                except (ValueError, UnicodeError):
                    self._response(conn, {"ok": False, "error": "invalid request"})
                    return
                self._response(conn, self._dispatch(request))
                return
            self._response(conn, {"ok": False, "error": "frame too large"})
        except (OSError, TimeoutError):
            return
        finally:
            conn.close()

    def _dispatch(self, request: Any) -> dict[str, Any]:
        if not isinstance(request, dict):
            return {"ok": False, "error": "invalid request"}
        capability = request.get("capability")
        if not isinstance(capability, str) or not hmac.compare_digest(capability, self.capability):
            return {"ok": False, "error": "unauthorized"}
        message_id = request.get("message_id")
        if not isinstance(message_id, str) or not message_id or len(message_id) > _MESSAGE_ID_MAX:
            return {"ok": False, "error": "message_id required"}
        fingerprint = _fingerprint(request)
        with self._receipts_lock:
            existing = self._receipts.get(message_id)
            if existing is not None:
                old_fingerprint, old_response = existing
                if not hmac.compare_digest(old_fingerprint, fingerprint):
                    return {"ok": False, "message_id": message_id, "error": "message_id conflict"}
                return old_response
            try:
                op = request.get("op")
                if op == "write" and isinstance(request.get("text"), str):
                    count = self.session.write(
                        request["text"], submit=bool(request.get("submit", False))
                    )
                elif op == "enter":
                    count = self.session.enter()
                else:
                    raise ValueError("invalid request")
                response = {"ok": True, "message_id": message_id, "bytes": count}
            except (OSError, RuntimeError, TypeError, UnicodeError, ValueError) as exc:
                response = {"ok": False, "message_id": message_id, "error": str(exc)}
            self._receipts[message_id] = (fingerprint, response)
            self._receipts.move_to_end(message_id)
            while len(self._receipts) > _MAX_RECEIPTS:
                self._receipts.popitem(last=False)
            return response

    def close(self) -> None:
        self._stop.set()
        if self._sock is not None:
            with contextlib.suppress(OSError):
                self._sock.close()
        if self._thread is not None:
            self._thread.join(timeout=1)
        self.socket_path.unlink(missing_ok=True)
        self.session.close()


class ReceiverClient:
    """Authenticated client used by the experimental receiver CLI/MCP path."""

    def __init__(
        self, socket_path: Path | str, capability: str, *, timeout: float = _SOCKET_TIMEOUT
    ):
        self.socket_path = Path(socket_path)
        self.capability = capability
        self.timeout = timeout

    def send(self, *, message_id: str, text: str, submit: bool = False) -> dict[str, Any]:
        return self._request(
            {"op": "write", "message_id": message_id, "text": text, "submit": submit}
        )

    def enter(self, *, message_id: str) -> dict[str, Any]:
        return self._request({"op": "enter", "message_id": message_id})

    def _request(self, fields: dict[str, Any]) -> dict[str, Any]:
        request = {"capability": self.capability, **fields}
        raw = json.dumps(request, separators=(",", ":")).encode("utf-8")
        if len(raw) > MAX_FRAME:
            raise ValueError("frame too large")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(self.timeout)
            sock.connect(str(self.socket_path))
            sock.sendall(raw + b"\n")
            data = b""
            while not data.endswith(b"\n"):
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
                if len(data) > MAX_FRAME:
                    raise ValueError("frame too large")
        if not data.endswith(b"\n"):
            raise RuntimeError("receiver returned an incomplete response")
        response = json.loads(data.decode("utf-8"))
        if not isinstance(response, dict):
            raise RuntimeError("receiver returned an invalid response")
        return response
