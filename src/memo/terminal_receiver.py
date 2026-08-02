"""Receiver-bound Unix transport for terminal/PTY writes.

The API is intentionally small: :class:`ReceiverSupervisor` owns a socket and
PTY child, while :class:`ReceiverClient` sends authenticated JSON messages.
"""
from __future__ import annotations

import json
import os
import pty
import secrets
import socket
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

MAX_FRAME = 64 * 1024


def _sanitize(value: Any) -> Any:
    if isinstance(value, str):
        # Keep printable text plus common whitespace; reject NUL and terminal
        # control sequences before they reach a PTY.
        return "".join(c for c in value if c in "\n\r\t" or ord(c) >= 32 and ord(c) != 127)
    if isinstance(value, dict):
        return {_sanitize(str(k)): _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    return value


def _peer_uid(conn: socket.socket) -> int | None:
    if hasattr(socket, "SO_PEERCRED"):
        import struct
        raw = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
        return struct.unpack("3i", raw)[1]
    if hasattr(conn, "getpeereid"):
        return conn.getpeereid()[0]  # type: ignore[attr-defined]
    return None


def _proc_start(pid: int) -> str | None:
    if os.name != "posix":
        return None
    try:
        if Path("/proc").is_dir():
            fields = (Path(f"/proc/{pid}/stat").read_text()).split()
            return fields[21]
    except (OSError, IndexError):
        return None
    return None


class ReceiverSession:
    """Owns a PTY master and validates child identity before every write."""

    def __init__(self, master_fd: int, child_pid: int, *, child_start: str | None = None):
        self.master_fd = master_fd
        self.child_pid = child_pid
        self.child_start = child_start if child_start is not None else _proc_start(child_pid)
        self._closed = False
        self._lock = threading.Lock()

    @classmethod
    def fork(cls, argv: list[str]) -> "ReceiverSession":
        pid, fd = pty.fork()
        if pid == 0:
            os.execvp(argv[0], argv)
        return cls(fd, pid)

    def alive(self) -> bool:
        if self._closed:
            return False
        try:
            os.kill(self.child_pid, 0)
        except OSError:
            return False
        if self.child_start is not None and _proc_start(self.child_pid) != self.child_start:
            return False
        return True

    def write(self, text: str) -> int:
        text = _sanitize(text)
        if not isinstance(text, str) or len(text.encode()) > MAX_FRAME:
            raise ValueError("frame too large")
        with self._lock:
            if not self.alive():
                raise RuntimeError("stale or dead child")
            return os.write(self.master_fd, text.encode())

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                try:
                    os.close(self.master_fd)
                except OSError:
                    pass


class ReceiverSupervisor:
    """Authenticated newline-delimited JSON receiver server."""

    def __init__(self, state_dir: Path | str, session: ReceiverSession):
        if os.name != "posix":
            raise RuntimeError("receiver transport requires Unix")
        self.state_dir = Path(state_dir)
        self.socket_path = self.state_dir / "receiver.sock"
        self.session = session
        self.capability = secrets.token_urlsafe(32)
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._receipts: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._receipts_lock = threading.Lock()

    def start(self) -> Path:
        self.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.state_dir, 0o700)
        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(self.socket_path))
        os.chmod(self.socket_path, 0o600)
        sock.listen(8)
        sock.settimeout(0.2)
        self._sock = sock
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        return self.socket_path

    def _serve(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        try:
            if _peer_uid(conn) not in (None, os.getuid()):
                return
            buf = b""
            while len(buf) <= MAX_FRAME:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf += chunk
                if b"\n" in buf:
                    line, _, _ = buf.partition(b"\n")
                    if len(line) > MAX_FRAME:
                        return
                    try:
                        req = json.loads(line.decode())
                    except (ValueError, UnicodeError):
                        return
                    resp = self._dispatch(req)
                    conn.sendall(json.dumps(resp, separators=(",", ":")).encode() + b"\n")
                    return
        finally:
            conn.close()

    def _dispatch(self, req: Any) -> dict[str, Any]:
        if not isinstance(req, dict) or req.get("capability") != self.capability:
            return {"ok": False, "error": "unauthorized"}
        mid = req.get("message_id")
        if not isinstance(mid, str) or not mid:
            return {"ok": False, "error": "message_id required"}
        with self._receipts_lock:
            if mid in self._receipts:
                return self._receipts[mid]
        if req.get("op") != "write" or not isinstance(req.get("text"), str):
            out = {"ok": False, "error": "invalid request"}
        else:
            try:
                n = self.session.write(req["text"])
                out = {"ok": True, "message_id": mid, "bytes": n}
            except Exception as exc:
                out = {"ok": False, "message_id": mid, "error": str(exc)}
        with self._receipts_lock:
            self._receipts[mid] = out
            self._receipts.move_to_end(mid)
            while len(self._receipts) > 1024:
                self._receipts.popitem(last=False)
        return out

    def close(self) -> None:
        self._stop.set()
        if self._sock is not None:
            self._sock.close()
        if self._thread is not None:
            self._thread.join(timeout=1)
        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass
        self.session.close()


class ReceiverClient:
    def __init__(self, socket_path: Path | str, capability: str):
        self.socket_path, self.capability = Path(socket_path), capability

    def send(self, *, message_id: str, text: str) -> dict[str, Any]:
        req = {"capability": self.capability, "message_id": message_id, "op": "write", "text": text}
        raw = json.dumps(req, separators=(",", ":")).encode()
        if len(raw) > MAX_FRAME:
            raise ValueError("frame too large")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(str(self.socket_path)); s.sendall(raw + b"\n")
            data = b""
            while not data.endswith(b"\n"):
                chunk = s.recv(4096)
                if not chunk: break
                data += chunk
        return json.loads(data.decode())
