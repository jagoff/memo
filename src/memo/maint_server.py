"""Maintenance daemon — keeps the synthesis LLM out of the MCP resident set.

`Memory.consolidate` clusters near-duplicate memorias and runs the synthesis
LLM (`MLXChat`, multi-GB, distinct from the embedder) over each cluster to
propose merges. It is a READ/propose step — it writes nothing; the user
applies a proposal later via `memo update`/`memo delete` (a cheap, in-process
transactional write that deliberately does NOT route here).

Running that LLM inside `memo-mcp` would pin a multi-GB model in the MCP
server's resident set for an off-request maintenance verb. This daemon hosts
it in its own process instead: `Memory.consolidate` ships the propose request
over `maint.sock`, the daemon runs the LLM and returns the proposals, and the
MCP server stays lean. Same AF_UNIX/JSON framing as the other daemons
(`embed_protocol`); same graceful degradation (daemon down -> in-process).

Protocol::

    {"op": "ping"}                                          -> {"ok": true, "kind": "maint"}
    {"op": "consolidate", "params": {threshold, max_clusters, type_}}
                                                            -> {"ok": true, "proposals": [...]}
    on error                                                -> {"error": "<message>"}

Gated OFF by default (`MEMO_MAINT_VIA_DAEMON`).
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import socketserver
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

_MAX_LINE_BYTES = 1 << 20

# A runner takes (op, params) and returns a JSON-serializable result dict.
# Injectable so tests exercise the daemon without loading MLXChat.
MaintRunner = Callable[[str, dict[str, Any]], dict[str, Any]]


def _socket_path(state_dir: Path) -> Path:
    return state_dir / "maint.sock"


def _pid_file(state_dir: Path) -> Path:
    return state_dir / "maint-daemon.pid"


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _read_pid(state_dir: Path) -> int | None:
    pf = _pid_file(state_dir)
    if not pf.is_file():
        return None
    try:
        return int(pf.read_text().strip())
    except (ValueError, OSError):
        return None


def _cleanup(state_dir: Path) -> None:
    _socket_path(state_dir).unlink(missing_ok=True)
    _pid_file(state_dir).unlink(missing_ok=True)


class _MaintHandler(socketserver.StreamRequestHandler):
    server: "_MaintServer"  # type: ignore[assignment]

    def handle(self) -> None:
        try:
            line = self.rfile.readline(_MAX_LINE_BYTES)
            if not line:
                return
            try:
                req = json.loads(line.decode("utf-8", errors="replace").strip())
            except Exception:
                self._write({"error": "malformed JSON request"})
                return
            if not isinstance(req, dict):
                self._write({"error": "request must be a JSON object"})
                return
            op = str(req.get("op") or "").strip()
            if op in ("ping",):
                self._write({"ok": True, "kind": "maint"})
                return
            if op == "consolidate":
                params = req.get("params") or {}
                if not isinstance(params, dict):
                    self._write({"error": "params must be a JSON object"})
                    return
                # Serialize: a single synthesis LLM, one heavy job at a time.
                with self.server.lock:
                    try:
                        result = self.server.runner(op, params)
                    except Exception as exc:  # noqa: BLE001 — report, never crash the daemon
                        self._write({"error": f"{type(exc).__name__}: {exc}"})
                        return
                self._write({"ok": True, **result})
                return
            self._write({"error": f"unknown op: {op!r}"})
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    def _write(self, payload: dict[str, Any]) -> None:
        with contextlib.suppress(BrokenPipeError, ConnectionResetError, OSError):
            self.wfile.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))


class _MaintServer(socketserver.ThreadingUnixStreamServer):
    def __init__(self, sock_path: str, runner: MaintRunner) -> None:
        self.runner = runner
        self.lock = threading.Lock()
        super().__init__(sock_path, _MaintHandler)


def _default_runner(cfg: Any) -> MaintRunner:
    """Production runner: a persistent Memory in THIS process whose MLXChat
    warms once and stays resident here (not in memo-mcp)."""
    from memo.memory import Memory

    mem = Memory(cfg)

    def run(op: str, params: dict[str, Any]) -> dict[str, Any]:
        if op == "consolidate":
            # Call the in-process path directly so the daemon never re-routes
            # to itself even if MEMO_MAINT_VIA_DAEMON is set in its environment.
            proposals = mem._consolidate_in_process(
                threshold=float(params.get("threshold", 0.85)),
                max_clusters=int(params.get("max_clusters", 50)),
                type_=params.get("type_"),
            )
            return {"proposals": proposals}
        raise ValueError(f"unsupported maint op: {op!r}")

    return run


def run_server(state_dir: Path | None = None, *, runner: MaintRunner | None = None) -> None:
    """Start the maintenance daemon. Invoked by `memo maint-daemon _serve`."""
    from memo.config import Config

    cfg = Config.from_env()
    if state_dir is None:
        state_dir = cfg.state_dir
    state_dir.mkdir(parents=True, exist_ok=True)

    sock_path = _socket_path(state_dir)
    pid_file = _pid_file(state_dir)

    existing = _read_pid(state_dir)
    if existing is not None and _is_pid_alive(existing):
        print("maint-daemon: already running", file=sys.stderr)
        sys.exit(0)

    sock_path.unlink(missing_ok=True)
    pid_file.unlink(missing_ok=True)

    try:
        server = _MaintServer(str(sock_path), runner or _default_runner(cfg))
    except OSError as exc:
        print(f"maint-daemon: bind failed ({exc}), exiting", file=sys.stderr)
        sys.exit(0)

    pid_file.write_text(str(os.getpid()))

    shutdown_event = threading.Event()

    def _sigterm(signum: int, frame: Any) -> None:
        shutdown_event.set()

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    server_thread = threading.Thread(target=server.serve_forever, name="maint-daemon-serve", daemon=True)
    server_thread.start()
    try:
        while not shutdown_event.wait(timeout=1.0):
            pass
    finally:
        with contextlib.suppress(Exception):
            server.shutdown()
        with contextlib.suppress(Exception):
            server.server_close()
        server_thread.join(timeout=5.0)
        _cleanup(state_dir)
