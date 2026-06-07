"""Ingest worker daemon — a scheduler, not a second store.

Batch ingest (repo indexing, bulk capture) embeds many chunks and writes
many rows. Run inline on the MCP worker thread it does two bad things: it
holds the recall daemon's shared embedder lock for tens of seconds (the
"53s tail" that starves interactive recall), and it blocks the MCP worker
threadpool. This daemon moves that work into its own process with its own
SINGLE serialized writer, so batch ingest never contends with the request
path. It is a *scheduler* — it still writes to the one `memvec.db` via the
normal `VecStore`/`RepoCorpus` writer; it does NOT introduce a second store.

Protocol (AF_UNIX, newline-delimited JSON — same framing as
``recall_server`` via ``embed_protocol``)::

    {"op": "ping"}                                  -> {"ok": true, "kind": "ingest", "jobs": N}
    {"op": "enqueue", "kind": "repo", "payload": {...}}  -> {"ok": true, "job_id": "..."}
    {"op": "status", "job_id": "..."}               -> {"state": "queued|running|done|error", ...}
    on error                                        -> {"error": "<message>"}

Gated OFF by default: `Memory.repo_index` only routes here when
``MEMO_INGEST_VIA_DAEMON=1`` AND the socket is reachable; otherwise it runs
the index in-process exactly as before (graceful degradation — nothing
breaks if the daemon is down).

Lifecycle mirrors the recall daemon: ``memo ingest-daemon start|stop|status``.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import socketserver
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from memo import embed_protocol
from memo.daemon_common import (
    cleanup,
    daemon_paths,
    is_pid_alive,
    read_pid,
    serve_until_shutdown,
)

# Back-compat alias for the CLI daemon wrapper.
_is_pid_alive = is_pid_alive

# Cap a single request line; enqueue payloads are small JSON (index kwargs).
_MAX_LINE_BYTES = 1 << 20

# A job runner takes (kind, payload) and returns the job result dict, or
# raises. Injectable so tests exercise the daemon without MLX / real clones.
JobRunner = Callable[[str, dict[str, Any]], dict[str, Any]]


def _socket_path(state_dir: Path) -> Path:
    return daemon_paths(state_dir, "ingest")[0]


def _pid_file(state_dir: Path) -> Path:
    return daemon_paths(state_dir, "ingest")[1]


def _read_pid(state_dir: Path) -> int | None:
    return read_pid(_pid_file(state_dir))


def _cleanup(state_dir: Path) -> None:
    cleanup(_socket_path(state_dir), _pid_file(state_dir))


class _JobBook:
    """Thread-safe registry of job state + a single serialized work queue.

    One worker thread drains the queue so all batch writes to the one DB are
    serialized (no two index runs interleave their transactions). Job records
    are retained (bounded) so a caller can poll `status` after completion.
    """

    def __init__(self, runner: JobRunner, *, retain: int = 256) -> None:
        self._runner = runner
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._order: deque[str] = deque()
        self._retain = retain
        self._queue: deque[str] = deque()
        self._wakeup = threading.Condition(self._lock)
        self._worker = threading.Thread(target=self._drain, name="ingest-worker", daemon=True)
        self._stop = False
        self._worker.start()

    def enqueue(self, kind: str, payload: dict[str, Any]) -> str:
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = {
                "state": "queued",
                "kind": kind,
                "payload": payload,
                "result": None,
                "error": None,
                "enqueued_at": time.time(),
            }
            self._order.append(job_id)
            self._queue.append(job_id)
            self._evict_locked()
            self._wakeup.notify()
        return job_id

    def status(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                return None
            # Return a copy without the (possibly large) payload echo.
            return {k: v for k, v in rec.items() if k != "payload"}

    def count(self) -> int:
        with self._lock:
            return len(self._jobs)

    def shutdown(self) -> None:
        with self._lock:
            self._stop = True
            self._wakeup.notify_all()

    def _evict_locked(self) -> None:
        while len(self._order) > self._retain:
            old = self._order.popleft()
            # Never evict a job still queued/running.
            rec = self._jobs.get(old)
            if rec and rec["state"] in ("done", "error"):
                self._jobs.pop(old, None)
            else:
                self._order.append(old)
                break

    def _drain(self) -> None:
        while True:
            with self._lock:
                while not self._queue and not self._stop:
                    self._wakeup.wait()
                if self._stop and not self._queue:
                    return
                job_id = self._queue.popleft()
                rec = self._jobs.get(job_id)
                if rec is None:
                    continue
                rec["state"] = "running"
                rec["started_at"] = time.time()
                kind, payload = rec["kind"], rec["payload"]
            # Run OUTSIDE the lock so status polls stay responsive.
            try:
                result = self._runner(kind, payload)
                with self._lock:
                    rec["state"] = "done"
                    rec["result"] = result
                    rec["finished_at"] = time.time()
            except Exception as exc:
                with self._lock:
                    rec["state"] = "error"
                    rec["error"] = f"{type(exc).__name__}: {exc}"
                    rec["finished_at"] = time.time()


class _IngestHandler(socketserver.StreamRequestHandler):
    server: _IngestServer  # type: ignore[assignment]
    # Backstop so a stalled client can't park a handler thread forever
    # (socket.timeout raises in readline → caught by the OSError handler below).
    timeout = 5.0

    def handle(self) -> None:
        try:
            line = self.rfile.readline(_MAX_LINE_BYTES)
            if not line:
                return
            try:
                req = json.loads(line.decode("utf-8", errors="replace").strip())
            except json.JSONDecodeError:
                self._write({"error": "malformed JSON request"})
                return
            if not isinstance(req, dict):
                self._write({"error": "request must be a JSON object"})
                return
            op = str(req.get("op") or "").strip()
            if op == embed_protocol.OP_PING or op == "ping":
                self._write({"ok": True, "kind": "ingest", "jobs": self.server.book.count()})
            elif op == "enqueue":
                kind = str(req.get("kind") or "").strip()
                payload = req.get("payload")
                if kind not in self.server.allowed_kinds:
                    self._write({"error": f"unknown kind: {kind!r}"})
                    return
                if not isinstance(payload, dict):
                    self._write({"error": "payload must be a JSON object"})
                    return
                job_id = self.server.book.enqueue(kind, payload)
                self._write({"ok": True, "job_id": job_id})
            elif op == "status":
                job_id = str(req.get("job_id") or "")
                st = self.server.book.status(job_id)
                self._write(st if st is not None else {"error": "unknown job_id"})
            else:
                self._write({"error": f"unknown op: {op!r}"})
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    def _write(self, payload: dict[str, Any]) -> None:
        with contextlib.suppress(BrokenPipeError, ConnectionResetError, OSError):
            self.wfile.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))


class _IngestServer(socketserver.ThreadingUnixStreamServer):
    """AF_UNIX server fronting a single serialized job worker."""

    allowed_kinds = frozenset({"repo"})

    def __init__(self, sock_path: str, book: _JobBook) -> None:
        self.book = book
        super().__init__(sock_path, _IngestHandler)


def _default_runner(cfg: Any) -> JobRunner:
    """Build the production runner: execute the batch op in THIS process,
    writing to the one DB via the normal RepoCorpus writer."""

    def run(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        if kind == "repo":
            from memo.repo_index import RepoCorpus

            corpus = RepoCorpus(cfg)
            return corpus.index(**payload)
        raise ValueError(f"unsupported ingest kind: {kind!r}")

    return run


def run_server(state_dir: Path | None = None, *, runner: JobRunner | None = None) -> None:
    """Start the ingest daemon. Invoked by `memo ingest-daemon _serve`."""
    from memo.config import Config

    cfg = Config.from_env()
    if state_dir is None:
        state_dir = cfg.state_dir
    state_dir.mkdir(parents=True, exist_ok=True)

    sock_path = _socket_path(state_dir)
    pid_file = _pid_file(state_dir)

    existing = _read_pid(state_dir)
    if existing is not None and is_pid_alive(existing):
        print("ingest-daemon: already running", file=sys.stderr)
        sys.exit(0)

    sock_path.unlink(missing_ok=True)
    pid_file.unlink(missing_ok=True)

    book = _JobBook(runner or _default_runner(cfg))
    try:
        server = _IngestServer(str(sock_path), book)
    except OSError as exc:
        print(f"ingest-daemon: bind failed ({exc}), exiting", file=sys.stderr)
        sys.exit(0)

    pid_file.write_text(str(os.getpid()))

    shutdown_event = threading.Event()

    def _sigterm(signum: int, frame: Any) -> None:
        shutdown_event.set()

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    def _on_shutdown() -> None:
        book.shutdown()
        _cleanup(state_dir)

    serve_until_shutdown(
        server,
        shutdown_event,
        name="ingest-daemon-serve",
        on_shutdown=_on_shutdown,
    )
