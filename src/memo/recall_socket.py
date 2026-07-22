from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import signal
import socketserver
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from memo.daemon_common import cleanup, daemon_paths, is_pid_alive, read_pid
from memo.daemon_common import serve_until_shutdown as _serve_until_shutdown
from memo.embed_protocol import MAX_LINE_BYTES
from memo.mlx_gpu import gpu_deadline, set_process_gpu_priority
from memo.recall_logic import _recall_logic
from memo.recall_stats import _STATS_DEFAULT_PERSIST_INTERVAL_S, _DaemonStats, _stats_persister

_log = logging.getLogger(__name__)
# Same cap on both ends of the wire — see ``embed_protocol.MAX_LINE_BYTES``.
_MAX_LINE_BYTES = MAX_LINE_BYTES

# A recall that waits longer than this for the priority lock gets one
# structured stderr line (`recall_lock_wait` / `recall_lock_bail`) so the
# next latency tail is diagnosable from the daemon log alone.
_LOCK_WAIT_LOG_MS = 500.0

# Daemon-busy marker: returned instead of "{}" on the warming and lock-bail
# paths so the hook client can tell "daemon can't serve right now" (fall back
# to the subprocess path) from a legit empty recall. Backward-compatible on
# the wire: an old client prints it verbatim, and a dict without
# hookSpecificOutput injects nothing — same net effect as the old "{}".
BUSY_RESPONSE = '{"busy": true}'


def _embedder_model_identity(mem: Any, cfg: Any) -> str:
    """Resolve the wire identity, preferring the store's exact vector owner.

    Production ``Memory`` instances stamp the backend-specific identity on
    their store (including an ST revision). Lightweight adapters and test
    doubles may only expose the legacy config model, so retain that protocol
    compatibility as a fallback.
    """
    store = getattr(mem, "store", None)
    store_model = getattr(store, "embedder_model", None)
    if isinstance(store_model, str) and store_model:
        return store_model
    return str(cfg.embedder_model)


def _log_lock_contention(event: str, wait_ms: float, held_by: str | None) -> None:
    """One structured line per contended recall — grep the daemon stderr log
    for ``recall_lock`` to see who a slow/bailed recall waited behind."""
    print(
        json.dumps(
            {"event": event, "op": "recall", "wait_ms": round(wait_ms, 1), "held_by": held_by},
            ensure_ascii=False,
        ),
        file=sys.stderr,
    )


def _socket_path(state_dir: Path) -> Path:
    return daemon_paths(state_dir, "recall")[0]


def _pid_file(state_dir: Path) -> Path:
    return daemon_paths(state_dir, "recall")[1]


def _read_pid(state_dir: Path) -> int | None:
    return read_pid(_pid_file(state_dir))


class _RecallHandler(socketserver.StreamRequestHandler):
    server: _RecallServer  # type: ignore[assignment]
    timeout = 5.0
    _stats_recorded = False

    def _write_response(self, result: str, *, debug: bool) -> bool:
        try:
            self.wfile.write((result + "\n").encode("utf-8"))
            return True
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            if debug:
                print(
                    f"# recall-daemon: client disconnected before response: {exc}", file=sys.stderr
                )
            return False

    def _record_stats_once(self, *, started_at: float, op: str, error: bool) -> None:
        if self._stats_recorded:
            return
        self._stats_recorded = True
        stats = getattr(self.server, "_stats", None)
        if stats is not None:
            latency_ms = (time.time() - started_at) * 1000.0
            stats.record(op, latency_ms, error=error)

    def _write_tracked_response(
        self,
        result: str,
        *,
        debug: bool,
        started_at: float,
        op: str,
        error: bool,
    ) -> bool:
        """Publish request metrics before its response becomes observable."""
        self._record_stats_once(started_at=started_at, op=op, error=error)
        return self._write_response(result, debug=debug)

    def _embed_query(self, req: dict[str, Any]) -> str:
        text = str(req.get("text") or "")
        if not text.strip():
            return json.dumps({"error": "embed_query: empty text"})
        vec = self.server._mem.embedder.embed_query(text)
        return json.dumps(
            {
                "vector": vec,
                "dim": len(vec),
                "dims": len(vec),
                "model": _embedder_model_identity(self.server._mem, self.server._cfg),
            },
            ensure_ascii=False,
        )

    def _search(self, req: dict[str, Any]) -> str:
        """Warm structured search for programmatic readers (memflow bridge).

        Same hybrid retrieval as ``memo search`` but served from the hot daemon
        (~0.7s vs ~9s cold CLI) and emitting ``{"results": [...]}``. Attributes
        the consult to ``client`` so the layer counts in ``memo usefulness``.
        """
        prompt = str(req.get("prompt") or req.get("query") or "").strip()
        if not prompt:
            return json.dumps({"error": "search: empty prompt", "results": []})
        from memo.flags import flag_int

        limit = req.get("limit")
        limit = int(limit) if isinstance(limit, (int, float)) else 5
        type_ = req.get("type") or None
        client = req.get("client") or None
        t0 = time.time()
        timeout_ms = flag_int("MEMO_RECALL_LOCK_TIMEOUT_MS")
        timeout_s = 2.5 if timeout_ms is None else max(0.0, timeout_ms / 1000.0)
        _we = getattr(self.server, "_warm_event", None)
        if _we is not None:
            _we.wait(timeout=timeout_s)
        # priority=0: only the interactive `recall` op (5s hook budget) outranks
        # the queue — the memflow bridge degrades gracefully on a bail.
        if not self.server._priority_lock.acquire(priority=0, timeout=timeout_s, label="search"):
            return json.dumps({"error": "search: timeout acquiring lock", "results": []})
        try:
            # No cross-encoder rerank: it adds ~6s and a fallback bridge wants a
            # fast hybrid shortlist, not a precision re-sort. Caller score-filters.
            with gpu_deadline(timeout_s):
                hits = self.server._mem.search(
                    prompt, limit=limit, type_=type_, mode="hybrid", disable_reranker=True
                )
        except TimeoutError:
            return json.dumps({"error": "search: timeout acquiring lock", "results": []})
        finally:
            self.server._priority_lock.release()
        results = []
        for h in hits[:limit]:
            d = h.to_dict()
            d["kind"] = d.get("type")
            results.append(d)
        try:
            from memo.dashboard import append_recall_log

            append_recall_log(
                self.server._cfg.state_dir,
                prompt=prompt,
                hits=results,
                via="daemon",
                client=client,
                source=(req.get("source") or None),
                latency_ms=int((time.time() - t0) * 1000),
            )
        except Exception:
            _log.debug("recall socket: stats recording failed", exc_info=True)
        return json.dumps({"results": results}, ensure_ascii=False)

    def _embed_batch(self, req: dict[str, Any]) -> str:
        texts = req.get("texts")
        if not isinstance(texts, list):
            return json.dumps({"error": "embed_batch: `texts` must be a list"})
        if not texts:
            return json.dumps(
                {
                    "vectors": [],
                    "dim": 0,
                    "dims": 0,
                    "model": _embedder_model_identity(self.server._mem, self.server._cfg),
                }
            )
        if not all(isinstance(t, str) for t in texts):
            return json.dumps({"error": "embed_batch: every element of `texts` must be a string"})
        from memo.flags import flag_int

        chunk_flag = flag_int("MEMO_EMBED_BATCH_CHUNK")
        chunk = max(1, 32 if chunk_flag is None else chunk_flag)
        _we = getattr(self.server, "_warm_event", None)
        if _we is not None:
            _we.wait(timeout=60.0)
        vectors: list[Any] = []
        for i in range(0, len(texts), chunk):
            if self.server._priority_lock.acquire(priority=0, timeout=60.0, label="embed_batch"):
                try:
                    with gpu_deadline(60.0):
                        vectors.extend(self.server._mem.embedder.embed(texts[i : i + chunk]))
                except TimeoutError:
                    return json.dumps({"error": "embed_batch: timeout acquiring lock"})
                finally:
                    self.server._priority_lock.release()
            else:
                return json.dumps({"error": "embed_batch: timeout acquiring lock"})
        dim = len(vectors[0]) if vectors else 0
        return json.dumps(
            {
                "vectors": vectors,
                "dim": dim,
                "dims": dim,
                "model": _embedder_model_identity(self.server._mem, self.server._cfg),
            },
            ensure_ascii=False,
        )

    def _ping(self) -> str:
        stats = getattr(self.server, "_stats", None)
        snap = stats.snapshot() if stats is not None else {}
        return json.dumps(
            {
                "ok": True,
                "model": _embedder_model_identity(self.server._mem, self.server._cfg),
                "dims": self.server._cfg.embedder_dims,
                "started_at": snap.get("started_at"),
                "uptime_s": snap.get("uptime_s"),
            }
        )

    def _stats(self) -> str:
        stats = getattr(self.server, "_stats", None)
        if stats is None:
            return json.dumps({"error": "stats not initialised"})
        return json.dumps(stats.snapshot(), ensure_ascii=False)

    def handle(self) -> None:
        t0 = time.time()
        from memo.flags import flag_bool

        debug = flag_bool("MEMO_RECALL_DEBUG")
        op = "parse"
        error = False
        self._stats_recorded = False

        try:
            try:
                line = self.rfile.readline(_MAX_LINE_BYTES)
                if not line:
                    self._write_tracked_response(
                        "{}", debug=debug, started_at=t0, op=op, error=error
                    )
                    return
                if len(line) >= _MAX_LINE_BYTES and not line.endswith(b"\n"):
                    error = True
                    self._write_tracked_response(
                        "{}", debug=debug, started_at=t0, op=op, error=error
                    )
                    return
                req = json.loads(line.decode("utf-8", errors="replace").strip())
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError, OSError) as exc:
                error = True
                print(f"# recall-daemon: parse error: {type(exc).__name__}: {exc}", file=sys.stderr)
                self._write_tracked_response(
                    "{}", debug=debug, started_at=t0, op=op, error=error
                )
                return

            if not isinstance(req, dict):
                error = True
                self._write_tracked_response(
                    "{}", debug=debug, started_at=t0, op=op, error=error
                )
                return

            op = str(req.get("op") or "recall").strip()
            priority = 1 if op == "recall" else 0
            log_fn: Callable[[], None] | None = None
            try:
                if op == "recall":
                    # Warming up: bail fast with the busy marker — the hook
                    # client falls through to the subprocess path instead of
                    # queueing behind the cold MLX load (a bare "{}" here was
                    # indistinguishable from a legit empty recall, so the
                    # fallback never ran during warmup). getattr: embedded/
                    # test servers without the event are treated as warm.
                    _we = getattr(self.server, "_warm_event", None)
                    if _we is not None and not _we.is_set():
                        _log_lock_contention("recall_warming", 0.0, "warmup")
                        self._write_tracked_response(
                            BUSY_RESPONSE, debug=debug, started_at=t0, op=op, error=error
                        )
                        return
                    prompt = (req.get("prompt") or "").strip()
                    cwd = req.get("cwd") or None
                    _sid = req.get("session_id") or None
                    _turn = req.get("turn")
                    _turn = int(_turn) if isinstance(_turn, (int, float)) else None
                    _client = req.get("client") or None
                    if not prompt:
                        self._write_tracked_response(
                            "{}", debug=debug, started_at=t0, op=op, error=error
                        )
                        return
                    from memo.flags import flag_int

                    timeout_ms = flag_int("MEMO_RECALL_LOCK_TIMEOUT_MS")
                    timeout_s = 2.5 if timeout_ms is None else max(0.0, timeout_ms / 1000.0)
                    lock = self.server._priority_lock
                    # Snapshot the current holder BEFORE waiting: after a
                    # successful acquire the holder is us, so this is the only
                    # cheap record of what a slow recall waited behind.
                    held_by = getattr(lock, "holder", None)
                    wait_t0 = time.monotonic()
                    acquired = lock.acquire(priority=priority, timeout=timeout_s, label="recall")
                    wait_ms = (time.monotonic() - wait_t0) * 1000.0
                    if not acquired:
                        _log_lock_contention(
                            "recall_lock_bail",
                            wait_ms,
                            getattr(lock, "holder", None) or held_by,
                        )
                        if debug:
                            print(
                                f"# recall-daemon: lock busy >{timeout_s:.1f}s, bailing busy",
                                file=sys.stderr,
                            )
                        self._write_tracked_response(
                            BUSY_RESPONSE, debug=debug, started_at=t0, op=op, error=error
                        )
                        return
                    if wait_ms > _LOCK_WAIT_LOG_MS:
                        _log_lock_contention("recall_lock_wait", wait_ms, held_by)
                    try:
                        # Bound the GPU flock wait to the recall budget: a
                        # busy GPU raises TimeoutError (caught below → error
                        # response) instead of wedging this thread inside
                        # the PriorityLock, which starved every other op
                        # (the observed recall_lock_bail storms).
                        with gpu_deadline(timeout_s):
                            result, log_fn = _recall_logic(
                                prompt,
                                cwd,
                                self.server._mem,
                                self.server._cfg,
                                debug,
                                t0=t0,
                                session_id=_sid,
                                turn=_turn,
                                client=_client,
                                micro_embedder=self.server._micro_embedder,
                            )
                    finally:
                        self.server._priority_lock.release()
                elif op == "embed_query":
                    from memo.flags import flag_int

                    lock_timeout_flag = flag_int("MEMO_EMBED_LOCK_TIMEOUT_MS")
                    _embed_timeout_s = max(
                        0.1, (60000 if lock_timeout_flag is None else lock_timeout_flag) / 1000.0
                    )
                    # Not latency-bound: wait out the warmup instead of racing
                    # the background model load.
                    _we = getattr(self.server, "_warm_event", None)
                    if _we is not None:
                        _we.wait(timeout=_embed_timeout_s)
                    # priority=0: embed_query callers (save dedup, memflow vec
                    # indexing, dream/eval passes) are not latency-bound — see
                    # MEMO_EMBED_LOCK_TIMEOUT_MS. At priority=1 a burst of these
                    # queued AT recall's priority with a 60s timeout, so an
                    # interactive recall burned its whole 2500ms budget behind
                    # them and bailed empty (the p95 tail).
                    if self.server._priority_lock.acquire(
                        priority=0, timeout=_embed_timeout_s, label="embed_query"
                    ):
                        try:
                            with gpu_deadline(_embed_timeout_s):
                                result = self._embed_query(req)
                        finally:
                            self.server._priority_lock.release()
                    else:
                        result = json.dumps({"error": "embed_query: timeout acquiring lock"})
                elif op == "search":
                    result = self._search(req)
                elif op == "embed_batch":
                    result = self._embed_batch(req)
                elif op == "ping":
                    result = self._ping()
                elif op == "stats":
                    result = self._stats()
                else:
                    error = True
                    result = json.dumps({"error": f"unknown op: {op!r}"})
            except Exception as exc:
                error = True
                print(
                    f"# recall-daemon: handler error (op={op}): {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                result = json.dumps({"error": f"{type(exc).__name__}: {exc}"})

            delivered = self._write_tracked_response(
                result, debug=debug, started_at=t0, op=op, error=error
            )
            if delivered and log_fn is not None:
                log_fn()
        finally:
            self._record_stats_once(started_at=t0, op=op, error=error)


# Lock acquisition order (to avoid deadlocks):
# 1. GPU lock (mlx_gpu._GPU_LOCK) - outermost, cross-process
# 2. Individual module locks (chat_lock, reranker_lock, load_lock, etc.) - inner
# Never acquire a module lock while holding GPU lock; GPU lock is only
# for MLX device serialization, not for general state protection.


class PriorityLock:
    """Priority lock with high-priority preemption.

    High-priority requests (priority > 0) jump ahead of normal requests.
    Used by recall daemon to prioritize interactive requests over background work.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._high_priority_waiters = 0
        self._busy = False
        self._holder_label: str | None = None

    def acquire(
        self, priority: int = 0, timeout: float | None = None, label: str | None = None
    ) -> bool:
        end_time = (time.time() + timeout) if timeout is not None else None
        with self._lock:
            if priority > 0:
                self._high_priority_waiters += 1
            try:
                while self._busy or (priority == 0 and self._high_priority_waiters > 0):
                    wait_timeout = None
                    if end_time is not None:
                        wait_timeout = max(0, end_time - time.time())
                        if wait_timeout <= 0:
                            return False
                    if not self._cond.wait(timeout=wait_timeout):
                        return False
                self._busy = True
                self._holder_label = label
                return True
            finally:
                if priority > 0:
                    self._high_priority_waiters -= 1
                    # Wake priority-0 waiters that re-slept on the
                    # high-priority gate — a timed-out high waiter must not
                    # leave them sleeping until their own timeout.
                    self._cond.notify_all()

    def release(self) -> None:
        with self._lock:
            self._busy = False
            self._holder_label = None
            self._cond.notify_all()

    @property
    def holder(self) -> str | None:
        """Label of the op currently holding the lock (None when free).

        Observability only — the value can go stale the moment it is read.
        """
        with self._lock:
            return self._holder_label


class _SimpleLockWrapper:
    """Wrapper for threading.Lock that matches PriorityLock interface."""

    def __init__(self, lock: threading.Lock) -> None:
        self._lock = lock
        self._holder_label: str | None = None

    def acquire(
        self, priority: int = 0, timeout: float | None = None, label: str | None = None
    ) -> bool:
        # Ignores priority parameter for simple lock
        acquired = self._lock.acquire(timeout=timeout if timeout is not None else -1.0)
        if acquired:
            self._holder_label = label
        return acquired

    def release(self) -> None:
        self._holder_label = None
        self._lock.release()

    @property
    def holder(self) -> str | None:
        """Best-effort label of the current holder (unsynchronised read)."""
        return self._holder_label


class _RecallServer(socketserver.ThreadingUnixStreamServer):
    def __init__(self, sock_path: str, cfg: Any, mem: Any) -> None:
        self._cfg = cfg
        self._mem = mem
        from memo.flags import flag_bool, flag_str

        if flag_bool("MEMO_RECALL_PRIORITY_ENABLED"):
            self._priority_lock: PriorityLock | _SimpleLockWrapper = PriorityLock()
        else:
            # Simple lock fallback when priority disabled
            self._priority_lock = _SimpleLockWrapper(threading.Lock())
        # Pre-set: only run_server's background warmup replaces this with an
        # unset event; embedded/test servers stay warm-by-default.
        self._warm_event = threading.Event()
        self._warm_event.set()

        self._micro_embedder = None
        micro_model = flag_str("MEMO_MICRO_EMBEDDER_MODEL")
        if micro_model:
            try:
                from memo.embedder import MicroEmbedder

                self._micro_embedder = MicroEmbedder(micro_model, expected_dims=cfg.embedder_dims)
            except Exception as exc:
                print(f"# recall-daemon: failed to init micro-embedder: {exc}", file=sys.stderr)

        self._stats = _DaemonStats(
            started_at=time.time(),
            model=_embedder_model_identity(mem, cfg),
            dims=cfg.embedder_dims,
        )
        super().__init__(sock_path, _RecallHandler)

    def server_close(self) -> None:
        super().server_close()


def _cleanup(state_dir: Path) -> None:
    cleanup(_socket_path(state_dir), _pid_file(state_dir))


def _warmup_embedder(mem: Any) -> float | None:
    """Force the embedder's cold model load BEFORE the socket is bound.

    The embedder loads lazily on first use (``MLXEmbedder._ensure_loaded``).
    Without this warm-up, the first request after a daemon (re)start paid the
    multi-second MLX load while HOLDING the priority lock, so every queued
    recall either waited behind it or hit its 2500ms bail and returned empty —
    the post-restart p95 tail. Running the load here, before ``_RecallServer``
    binds the socket, means no client can even connect during the load: a cold
    start never counts against a queued recall's budget (clients fall back to
    the subprocess path exactly as when the daemon is down).

    MLX imports stay deferred (they happen inside the embed call). Failure is
    non-fatal: the first request falls back to the old lazy load.

    Returns the warm-up duration in ms, or None on failure.
    """
    t0 = time.monotonic()
    try:
        mem.embedder.embed(["memo recall-daemon warm-up"])
    except Exception as exc:
        print(
            f"# recall-daemon: warm-up failed ({type(exc).__name__}: {exc}); "
            "first request will lazy-load",
            file=sys.stderr,
        )
        return None
    return (time.monotonic() - t0) * 1000.0


def run_server(state_dir: Path | None = None) -> None:
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
    lock_path = pid_file.with_name(pid_file.name + ".lock")
    lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(lock_fd)
        print("recall-daemon: another instance is starting", file=sys.stderr)
        sys.exit(0)

    shutdown_event = threading.Event()

    def _sigterm(signum: int, frame: Any) -> None:
        shutdown_event.set()

    try:
        signal.signal(signal.SIGTERM, _sigterm)
        signal.signal(signal.SIGINT, _sigterm)
        _run_server_locked(cfg, state_dir, sock_path, pid_file, shutdown_event)
    finally:
        os.close(lock_fd)


def _bind_recall_server(sock_path: Path, cfg: Any, mem: Any) -> Any:
    try:
        return _RecallServer(str(sock_path), cfg, mem)
    except OSError as exc:
        print(f"recall-daemon: bind failed ({exc}), exiting", file=sys.stderr)
        sys.exit(0)


def _run_server_locked(
    cfg: Any,
    state_dir: Path,
    sock_path: Path,
    pid_file: Path,
    shutdown_event: threading.Event,
) -> None:
    """Run all post-flock setup under one resource-ownership boundary."""

    from memo.memory import Memory

    mem: Any | None = None
    server: Any | None = None
    try:
        existing_pid = _read_pid(state_dir)
        if existing_pid is not None and is_pid_alive(existing_pid):
            print("recall-daemon: already running", file=sys.stderr)
            sys.exit(0)

        sock_path.unlink(missing_ok=True)
        pid_file.unlink(missing_ok=True)

        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        # The resident daemon is the machine's latency-critical GPU user: its
        # warmup + query embeds take the fast lane on the cross-process GPU
        # flock so batch jobs yield instead of starving latency-critical recall.
        set_process_gpu_priority(True)
        mem = Memory(cfg)

        # Bind before warmup so cold-start socket probes succeed immediately.
        server = _bind_recall_server(sock_path, cfg, mem)

        warm_event = threading.Event()
        server._warm_event = warm_event
        pid_file.write_text(str(os.getpid()))
        from memo.flags import flag_bool, flag_float

        if flag_bool("MEMO_RECALL_DEBUG"):
            print(f"# recall-daemon: listening on {sock_path}", file=sys.stderr)

        interval = flag_float("MEMO_EMBEDDER_STATS_INTERVAL_S") or _STATS_DEFAULT_PERSIST_INTERVAL_S
        if interval > 0:
            threading.Thread(
                target=_stats_persister,
                args=(state_dir, server._stats, interval, shutdown_event),
                daemon=True,
                name="recall-daemon-stats-persister",
            ).start()

        def _warm_bg() -> None:
            try:
                warm_ms = _warmup_embedder(mem)
                if warm_ms is not None:
                    print(
                        f"# recall-daemon: embedder warm in {warm_ms:.0f}ms",
                        file=sys.stderr,
                    )
            finally:
                # Set even on failure — handlers fall back to the lazy load.
                warm_event.set()

        threading.Thread(target=_warm_bg, name="embedder-warmup", daemon=True).start()

        _serve_until_shutdown(
            server,
            shutdown_event,
            name="recall-daemon-serve",
            on_shutdown=lambda: _cleanup(state_dir),
        )
    finally:
        shutdown_event.set()
        if server is not None:
            with contextlib.suppress(Exception):
                server.server_close()
        if mem is not None:
            with contextlib.suppress(Exception):
                mem.close()
        with contextlib.suppress(Exception):
            _cleanup(state_dir)
