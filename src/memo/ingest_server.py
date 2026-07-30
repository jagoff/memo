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
import hashlib
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
from memo.ingest_ledger import IngestFailureLedger

# Back-compat alias for the CLI daemon wrapper.
_is_pid_alive = is_pid_alive

# Cap a single request line; enqueue payloads are small JSON (index kwargs).
_MAX_LINE_BYTES = 1 << 20

# A job runner takes (kind, payload) and returns the job result dict, or
# raises. Injectable so tests exercise the daemon without MLX / real clones.
JobRunner = Callable[[str, dict[str, Any]], dict[str, Any]]


def _job_fingerprint(kind: str, payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        {"kind": kind, "payload": payload},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


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

    def __init__(
        self,
        runner: JobRunner,
        *,
        retain: int = 256,
        ledger: IngestFailureLedger | None = None,
        quarantine_threshold: int = 3,
    ) -> None:
        self._runner = runner
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._order: deque[str] = deque()
        self._retain = retain
        self._queue: deque[str] = deque()
        self._active_by_fingerprint: dict[str, str] = {}
        self._ledger = ledger
        self._quarantine_threshold = max(0, int(quarantine_threshold))
        self._fatal_failures = ledger.fatal_failure_counts() if ledger is not None else {}
        self._wakeup = threading.Condition(self._lock)
        self._stop = False
        self._worker: threading.Thread | None = None
        self._worker_starts = 0
        with self._lock:
            self._start_worker_locked()

    def enqueue(self, kind: str, payload: dict[str, Any]) -> str:
        return self.enqueue_receipt(kind, payload)["job_id"]

    def enqueue_receipt(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        fingerprint = _job_fingerprint(kind, payload)
        job_id = uuid.uuid4().hex
        with self._lock:
            self._ensure_worker_locked()
            active_id = self._active_by_fingerprint.get(fingerprint)
            active = self._jobs.get(active_id) if active_id else None
            if active is not None and active["state"] in {"queued", "running"}:
                return {
                    "job_id": str(active_id),
                    "deduplicated": True,
                    "state": str(active["state"]),
                    "fingerprint": fingerprint,
                }
            quarantined = (
                self._quarantine_threshold > 0
                and self._fatal_failures.get(fingerprint, 0) >= self._quarantine_threshold
            )
            self._jobs[job_id] = {
                "state": "quarantined" if quarantined else "queued",
                "kind": kind,
                "payload": payload,
                "result": None,
                "error": (
                    "job fingerprint quarantined after repeated fatal worker failures"
                    if quarantined
                    else None
                ),
                "fingerprint": fingerprint,
                "deduplicated": False,
                "enqueued_at": time.time(),
            }
            self._order.append(job_id)
            if quarantined:
                self._append_event_locked(
                    {
                        "event": "quarantined",
                        "job_id": job_id,
                        "kind": kind,
                        "fingerprint": fingerprint,
                        "fatal_failures": self._fatal_failures.get(fingerprint, 0),
                    }
                )
            else:
                self._queue.append(job_id)
                self._active_by_fingerprint[fingerprint] = job_id
                self._append_event_locked(
                    {
                        "event": "queued",
                        "job_id": job_id,
                        "kind": kind,
                        "fingerprint": fingerprint,
                    }
                )
            self._evict_locked()
            self._wakeup.notify()
        return {
            "job_id": job_id,
            "deduplicated": False,
            "state": "quarantined" if quarantined else "queued",
            "fingerprint": fingerprint,
        }

    def status(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            self._ensure_worker_locked()
            rec = self._jobs.get(job_id)
            if rec is None:
                return None
            # Return a copy without the (possibly large) payload echo.
            return {k: v for k, v in rec.items() if k != "payload"}

    def count(self) -> int:
        with self._lock:
            self._ensure_worker_locked()
            return len(self._jobs)

    def health(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_worker_locked()
            return {
                "worker_alive": bool(self._worker and self._worker.is_alive()),
                "worker_starts": self._worker_starts,
                "queued": len(self._queue),
                "quarantined_fingerprints": sum(
                    count >= self._quarantine_threshold
                    for count in self._fatal_failures.values()
                )
                if self._quarantine_threshold
                else 0,
                "ledger": self._ledger.health() if self._ledger is not None else None,
            }

    def shutdown(self) -> None:
        with self._lock:
            self._stop = True
            self._wakeup.notify_all()

    def _evict_locked(self) -> None:
        while len(self._order) > self._retain:
            old = self._order.popleft()
            # Never evict a job still queued/running.
            rec = self._jobs.get(old)
            if rec and rec["state"] in ("done", "error", "quarantined"):
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
                self._append_event_locked(
                    {
                        "event": "running",
                        "job_id": job_id,
                        "kind": kind,
                        "fingerprint": rec["fingerprint"],
                    }
                )
            # Run OUTSIDE the lock so status polls stay responsive.
            try:
                result = self._runner(kind, payload)
                with self._lock:
                    rec["state"] = "done"
                    rec["result"] = result
                    rec["finished_at"] = time.time()
                    self._active_by_fingerprint.pop(str(rec["fingerprint"]), None)
                    self._append_event_locked(
                        {
                            "event": "done",
                            "job_id": job_id,
                            "kind": kind,
                            "fingerprint": rec["fingerprint"],
                        }
                    )
            except BaseException as exc:
                fatal = not isinstance(exc, Exception)
                with self._lock:
                    rec["state"] = "error"
                    rec["error"] = f"{type(exc).__name__}: {exc}"
                    rec["finished_at"] = time.time()
                    rec["fatal"] = fatal
                    fingerprint = str(rec["fingerprint"])
                    self._active_by_fingerprint.pop(fingerprint, None)
                    if fatal:
                        self._fatal_failures[fingerprint] = (
                            self._fatal_failures.get(fingerprint, 0) + 1
                        )
                    self._append_event_locked(
                        {
                            "event": "error",
                            "job_id": job_id,
                            "kind": kind,
                            "fingerprint": fingerprint,
                            "fatal": fatal,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )

    def _start_worker_locked(self) -> None:
        self._worker = threading.Thread(
            target=self._drain,
            name="ingest-worker",
            daemon=True,
        )
        self._worker_starts += 1
        self._worker.start()

    def _ensure_worker_locked(self) -> None:
        if not self._stop and (self._worker is None or not self._worker.is_alive()):
            self._start_worker_locked()
            self._append_event_locked(
                {
                    "event": "worker_restart",
                    "worker_starts": self._worker_starts,
                }
            )

    def _append_event_locked(self, event: dict[str, Any]) -> None:
        if self._ledger is not None:
            self._ledger.append(event)


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
                self._write(
                    {
                        "ok": True,
                        "kind": "ingest",
                        "jobs": self.server.book.count(),
                        "health": self.server.book.health(),
                    }
                )
            elif op == "enqueue":
                kind = str(req.get("kind") or "").strip()
                payload = req.get("payload")
                if kind not in self.server.allowed_kinds:
                    self._write({"error": f"unknown kind: {kind!r}"})
                    return
                if not isinstance(payload, dict):
                    self._write({"error": "payload must be a JSON object"})
                    return
                receipt = self.server.book.enqueue_receipt(kind, payload)
                self._write({"ok": True, **receipt})
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

    # Guard the start critical section. Without it, two concurrent starts can
    # both pass the dead-PID check below, then race sock_path.unlink + bind (one
    # unlinks the socket the other just bound, orphaning a live daemon). A
    # non-blocking exclusive flock on a dedicated lock file admits exactly one
    # starter; a loser exits at once. The fd is held for the process lifetime
    # (released by the OS on exit).
    import fcntl

    lock_path = pid_file.with_name(pid_file.name + ".lock")
    lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(lock_fd)
        print("ingest-daemon: another instance is starting", file=sys.stderr)
        sys.exit(0)

    existing = _read_pid(state_dir)
    if existing is not None and is_pid_alive(existing):
        print("ingest-daemon: already running", file=sys.stderr)
        sys.exit(0)

    sock_path.unlink(missing_ok=True)
    pid_file.unlink(missing_ok=True)

    from memo.flags import flag_int

    ledger = IngestFailureLedger(state_dir / "ingest-jobs.jsonl")
    book = _JobBook(
        runner or _default_runner(cfg),
        ledger=ledger,
        quarantine_threshold=int(flag_int("MEMO_INGEST_QUARANTINE_THRESHOLD") or 0),
    )
    try:
        server = _IngestServer(str(sock_path), book)
    except OSError as exc:
        print(f"ingest-daemon: bind failed ({exc}), exiting", file=sys.stderr)
        sys.exit(0)

    tmp_pid = pid_file.with_suffix(pid_file.suffix + ".tmp")
    tmp_pid.write_text(str(os.getpid()))
    os.replace(tmp_pid, pid_file)

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
