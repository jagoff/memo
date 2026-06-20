from __future__ import annotations

import json
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
from memo.recall_logic import _recall_logic
from memo.recall_stats import _STATS_DEFAULT_PERSIST_INTERVAL_S, _DaemonStats, _stats_persister

_MAX_LINE_BYTES = 1 << 20


def _socket_path(state_dir: Path) -> Path:
    return daemon_paths(state_dir, "recall")[0]


def _pid_file(state_dir: Path) -> Path:
    return daemon_paths(state_dir, "recall")[1]


def _read_pid(state_dir: Path) -> int | None:
    return read_pid(_pid_file(state_dir))


class _RecallHandler(socketserver.StreamRequestHandler):
    server: _RecallServer  # type: ignore[assignment]
    timeout = 5.0

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
                "model": self.server._cfg.embedder_model,
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
        timeout_s = max(0.1, (flag_int("MEMO_RECALL_LOCK_TIMEOUT_MS") or 2500) / 1000.0)
        if not self.server._priority_lock.acquire(priority=1, timeout=timeout_s):
            return json.dumps({"error": "search: timeout acquiring lock", "results": []})
        try:
            # No cross-encoder rerank: it adds ~6s and a fallback bridge wants a
            # fast hybrid shortlist, not a precision re-sort. Caller score-filters.
            hits = self.server._mem.search(
                prompt, limit=limit, type_=type_, mode="hybrid", disable_reranker=True
            )
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
            pass
        return json.dumps({"results": results}, ensure_ascii=False)

    def _embed_batch(self, req: dict[str, Any]) -> str:
        texts = req.get("texts")
        if not isinstance(texts, list):
            return json.dumps({"error": "embed_batch: `texts` must be a list"})
        if not texts:
            return json.dumps(
                {"vectors": [], "dim": 0, "dims": 0, "model": self.server._cfg.embedder_model}
            )
        if not all(isinstance(t, str) for t in texts):
            return json.dumps({"error": "embed_batch: every element of `texts` must be a string"})
        from memo.flags import flag_int

        chunk = max(1, flag_int("MEMO_EMBED_BATCH_CHUNK") or 32)
        vectors: list[Any] = []
        for i in range(0, len(texts), chunk):
            if self.server._priority_lock.acquire(priority=0, timeout=60.0):
                try:
                    vectors.extend(self.server._mem.embedder.embed(texts[i : i + chunk]))
                finally:
                    self.server._priority_lock.release()
            else:
                return json.dumps({"error": "embed_batch: timeout acquiring lock"})
        dim = len(vectors[0]) if vectors else 0
        return json.dumps(
            {"vectors": vectors, "dim": dim, "dims": dim, "model": self.server._cfg.embedder_model},
            ensure_ascii=False,
        )

    def _ping(self) -> str:
        stats = getattr(self.server, "_stats", None)
        snap = stats.snapshot() if stats is not None else {}
        return json.dumps(
            {
                "ok": True,
                "model": self.server._cfg.embedder_model,
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
        try:
            try:
                line = self.rfile.readline(_MAX_LINE_BYTES)
                if not line:
                    self._write_response("{}", debug=debug)
                    return
                if len(line) >= _MAX_LINE_BYTES and not line.endswith(b"\n"):
                    error = True
                    self._write_response("{}", debug=debug)
                    return
                req = json.loads(line.decode("utf-8", errors="replace").strip())
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError, OSError) as exc:
                error = True
                print(f"# recall-daemon: parse error: {type(exc).__name__}: {exc}", file=sys.stderr)
                self._write_response("{}", debug=debug)
                return

            if not isinstance(req, dict):
                error = True
                self._write_response("{}", debug=debug)
                return

            op = str(req.get("op") or "recall").strip()
            priority = 1 if op == "recall" else 0
            log_fn: Callable[[], None] | None = None
            try:
                if op == "recall":
                    prompt = (req.get("prompt") or "").strip()
                    cwd = req.get("cwd") or None
                    _sid = req.get("session_id") or None
                    _turn = req.get("turn")
                    _turn = int(_turn) if isinstance(_turn, (int, float)) else None
                    _client = req.get("client") or None
                    if not prompt:
                        self._write_response("{}", debug=debug)
                        return
                    from memo.flags import flag_int

                    timeout_s = max(0.1, (flag_int("MEMO_RECALL_LOCK_TIMEOUT_MS") or 2500) / 1000.0)
                    if not self.server._priority_lock.acquire(priority=priority, timeout=timeout_s):
                        if debug:
                            print(
                                f"# recall-daemon: lock busy >{timeout_s:.1f}s, bailing empty",
                                file=sys.stderr,
                            )
                        self._write_response("{}", debug=debug)
                        return
                    try:
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
                    if self.server._priority_lock.acquire(priority=1, timeout=5.0):
                        try:
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

            delivered = self._write_response(result, debug=debug)
            if delivered and log_fn is not None:
                log_fn()
        finally:
            latency_ms = (time.time() - t0) * 1000.0
            stats = getattr(self.server, "_stats", None)
            if stats is not None:
                stats.record(op, latency_ms, error=error)


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

    def acquire(self, priority: int = 0, timeout: float | None = None) -> bool:
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
                            if priority > 0:
                                self._high_priority_waiters -= 1
                            return False
                    if not self._cond.wait(timeout=wait_timeout):
                        if priority > 0:
                            self._high_priority_waiters -= 1
                        return False
                self._busy = True
                if priority > 0:
                    self._high_priority_waiters -= 1
                return True
            except Exception:
                if priority > 0:
                    self._high_priority_waiters -= 1
                raise

    def release(self) -> None:
        with self._lock:
            self._busy = False
            self._cond.notify_all()


class _SimpleLockWrapper:
    """Wrapper for threading.Lock that matches PriorityLock interface."""
    def __init__(self, lock: threading.Lock) -> None:
        self._lock = lock

    def acquire(self, priority: int = 0, timeout: float | None = None) -> bool:
        # Ignores priority parameter for simple lock
        return self._lock.acquire(timeout=timeout if timeout is not None else -1)

    def release(self) -> None:
        self._lock.release()


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

        self._micro_embedder = None
        micro_model = flag_str("MEMO_MICRO_EMBEDDER_MODEL")
        if micro_model:
            try:
                from memo.embedder import MicroEmbedder

                self._micro_embedder = MicroEmbedder(micro_model)
            except Exception as exc:
                print(f"# recall-daemon: failed to init micro-embedder: {exc}", file=sys.stderr)

        self._stats = _DaemonStats(
            started_at=time.time(), model=cfg.embedder_model, dims=cfg.embedder_dims
        )
        super().__init__(sock_path, _RecallHandler)

    def server_close(self) -> None:
        super().server_close()


def _cleanup(state_dir: Path) -> None:
    cleanup(_socket_path(state_dir), _pid_file(state_dir))


def run_server(state_dir: Path | None = None) -> None:
    from memo.config import Config
    from memo.memory import Memory

    cfg = Config.from_env()
    if state_dir is None:
        state_dir = cfg.state_dir
    state_dir.mkdir(parents=True, exist_ok=True)
    sock_path = _socket_path(state_dir)
    pid_file = _pid_file(state_dir)

    existing_pid = _read_pid(state_dir)
    if existing_pid is not None and is_pid_alive(existing_pid):
        print("recall-daemon: already running", file=sys.stderr)
        sys.exit(0)

    sock_path.unlink(missing_ok=True)
    pid_file.unlink(missing_ok=True)

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    mem = Memory(cfg)

    try:
        server = _RecallServer(str(sock_path), cfg, mem)
    except OSError as exc:
        print(f"recall-daemon: bind failed ({exc}), exiting", file=sys.stderr)
        sys.exit(0)

    pid_file.write_text(str(os.getpid()))
    shutdown_event = threading.Event()

    def _sigterm(signum: int, frame: Any) -> None:
        shutdown_event.set()

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    from memo.flags import flag_bool, flag_float

    debug = flag_bool("MEMO_RECALL_DEBUG")
    if debug:
        print(f"# recall-daemon: listening on {sock_path}", file=sys.stderr)

    interval = flag_float("MEMO_EMBEDDER_STATS_INTERVAL_S") or _STATS_DEFAULT_PERSIST_INTERVAL_S
    if interval > 0:
        persister = threading.Thread(
            target=_stats_persister,
            args=(state_dir, server._stats, interval),
            daemon=True,
            name="recall-daemon-stats-persister",
        )
        persister.start()

    _serve_until_shutdown(
        server, shutdown_event, name="recall-daemon-serve", on_shutdown=lambda: _cleanup(state_dir)
    )
