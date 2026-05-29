"""Recall daemon — persistent Unix socket server for low-latency recall.

Keeps the MLX embedder in RAM and answers recall requests via a Unix domain
socket in <200 ms (vs 1-2 s cold subprocess per prompt).

Protocol
--------
One JSON line in → one JSON line out (newline-delimited). Each request is
dispatched on its `op` field; legacy clients that omit `op` default to
recall and stay compatible with the original schema.

Request shapes:
    {"op": "recall", "prompt": "...", "cwd": "..."}    # default if op omitted
    {"op": "embed_query", "text": "..."}               # asymmetric query embedding
    {"op": "embed_batch", "texts": ["...", ...]}       # symmetric doc embedding
    {"op": "ping"}                                     # warm-state probe
    {"op": "stats"}                                    # per-op counters + p50/95/99

Response shapes:
    recall (no injection):   {}
    recall (with injection): {"hookSpecificOutput": {...}}
    embed_query:             {"vector": [...], "dim": N, "model": "..."}
    embed_batch:             {"vectors": [[...]], "dim": N, "model": "..."}
    ping:                    {"ok": true, "model": "...", "dims": N,
                              "started_at": <epoch>, "uptime_s": N}
    stats:                   {"started_at": <epoch>, "uptime_s": N,
                              "model": "...", "dims": N,
                              "ops": {op: {count, errors, samples,
                                            p50_ms, p95_ms, p99_ms}}}
    on error:                {"error": "<message>"}

The daemon also persists `embed_daemon_stats.json` in `state_dir`
every `MEMO_EMBEDDER_STATS_INTERVAL_S` seconds (default 60) so peers
(synapse_doctor, dashboards) can read metrics without opening the
socket.

This is the shared-embedder sidecar surface: any in-process or peer
(synapse, memflow) can reuse the one warm MLX instance instead of loading
its own copy. `memo.embedder_client` is the client adapter (socket-first,
in-process fallback). See `src/memo/embedder_client.py`.

Usage
-----
    memo recall-daemon start    # background daemon
    memo recall-daemon stop     # SIGTERM the PID
    memo recall-daemon status   # running/stopped

The daemon is started automatically by the SessionStart hook in hooks.json.
"""

from __future__ import annotations

import json
import os
import signal
import socketserver
import sys
import threading
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any

_STATS_SAMPLE_CAP = 1024
_STATS_DEFAULT_PERSIST_INTERVAL_S = 60.0
# Cap a single request/response line so a client that never sends a newline
# can't make us buffer unboundedly. Requests are small JSON ({op, prompt,
# cwd}); 1 MiB is far above any legitimate prompt.
_MAX_LINE_BYTES = 1 << 20


def _percentile(sorted_values: list[float], pct: int) -> float | None:
    """Return the linear-interpolated percentile of a sorted list (or None)."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * pct / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


class _DaemonStats:
    """Thread-safe in-memory metrics for the recall daemon.

    Per-op: request count, error count, latency reservoir bounded to the
    most-recent `_STATS_SAMPLE_CAP` samples for p50/p95/p99 computation.
    """

    def __init__(self, started_at: float, model: str, dims: int) -> None:
        self._started_at = started_at
        self._model = model
        self._dims = dims
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}
        self._errors: dict[str, int] = {}
        self._latencies: dict[str, deque[float]] = {}

    def record(self, op: str, latency_ms: float, *, error: bool = False) -> None:
        with self._lock:
            self._counts[op] = self._counts.get(op, 0) + 1
            if error:
                self._errors[op] = self._errors.get(op, 0) + 1
            buf = self._latencies.get(op)
            if buf is None:
                buf = deque(maxlen=_STATS_SAMPLE_CAP)
                self._latencies[op] = buf
            buf.append(latency_ms)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            ops: dict[str, dict[str, Any]] = {}
            for op, count in self._counts.items():
                lat = sorted(self._latencies.get(op) or [])
                ops[op] = {
                    "count": count,
                    "errors": self._errors.get(op, 0),
                    "samples": len(lat),
                    "p50_ms": _percentile(lat, 50),
                    "p95_ms": _percentile(lat, 95),
                    "p99_ms": _percentile(lat, 99),
                }
        return {
            "started_at": self._started_at,
            "uptime_s": int(time.time() - self._started_at),
            "model": self._model,
            "dims": self._dims,
            "ops": ops,
        }


def _stats_file(state_dir: Path) -> Path:
    return state_dir / "embed_daemon_stats.json"


def _stats_persister(state_dir: Path, stats: _DaemonStats, interval_s: float) -> None:
    """Write snapshot to disk periodically. Runs as a daemon thread."""
    target = _stats_file(state_dir)
    while True:
        time.sleep(interval_s)
        try:
            snap = stats.snapshot()
            tmp = target.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(snap, indent=2))
            tmp.replace(target)
        except Exception as exc:
            if os.environ.get("MEMO_RECALL_DEBUG") == "1":
                print(f"# recall-daemon: stats persist failed: {exc}", file=sys.stderr)


def _socket_path(state_dir: Path) -> Path:
    return state_dir / "recall.sock"


def _pid_file(state_dir: Path) -> Path:
    return state_dir / "recall-daemon.pid"


def _is_pid_alive(pid: int) -> bool:
    """Return True if a process with this PID is running."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _read_pid(state_dir: Path) -> int | None:
    """Read the PID from the PID file. Returns None if missing or invalid."""
    pf = _pid_file(state_dir)
    if not pf.is_file():
        return None
    try:
        return int(pf.read_text().strip())
    except (ValueError, OSError):
        return None


def _apply_project_boost(hits: list[Any], project_tag: str | None, project_boost: float) -> list[Any]:
    """Return hits re-ranked with an additive project boost.

    MemoryRecord is frozen, so boost by creating replacement records instead
    of mutating the score field in place.
    """
    if not project_tag:
        return list(hits)

    boosted: list[Any] = []
    for h in hits:
        if h.score is not None and project_tag in (h.tags or []):
            boosted.append(replace(h, score=h.score + project_boost))
        else:
            boosted.append(h)
    boosted.sort(key=lambda h: (h.score or 0.0), reverse=True)
    return boosted


def _recall_logic(
    prompt: str,
    cwd: str | None,
    mem: Any,
    cfg: Any,
    debug: bool = False,
    t0: float | None = None,
) -> str:
    """Run recall search and return a JSON string to write back on the socket.

    Mirrors the logic in cli.py:recall_hook but operates on a pre-loaded
    Memory instance (the daemon's persistent one).
    """
    import os as _os

    top_k = int(_os.environ.get("MEMO_RECALL_TOP_K", "3"))
    min_sim = float(_os.environ.get("MEMO_RECALL_MIN_SIM", "0.6"))
    body_chars = int(_os.environ.get("MEMO_RECALL_BODY_CHARS", "240"))
    token_budget = int(_os.environ.get("MEMO_RECALL_TOKEN_BUDGET", "0") or 0)
    project_boost = float(_os.environ.get("MEMO_RECALL_PROJECT_BOOST", "0.15"))
    mode = _os.environ.get("MEMO_RECALL_MODE", "vec")
    min_body_chars = int(_os.environ.get("MEMO_RECALL_MIN_BODY_CHARS", "40"))

    # Project boost
    project_tag = None
    if project_boost > 0 and cwd:
        try:
            from memo.project import current_project_tag
            project_tag = current_project_tag(cwd)
        except Exception:
            project_tag = None

    search_k = top_k * 3 if project_tag else top_k

    try:
        hits = mem.search(prompt, limit=search_k, mode=mode)
    except Exception as exc:
        if debug:
            print(f"# recall-daemon: search failed: {exc}", file=sys.stderr)
        return "{}"

    # Project boost
    if project_tag:
        hits = _apply_project_boost(hits, project_tag, project_boost)
    hits = hits[:top_k]

    # Similarity floor
    relevant = [h for h in hits if h.score is None or h.score >= min_sim]

    # Body stub filter
    if min_body_chars > 0:
        relevant = [h for h in relevant if len((h.body or "").strip()) >= min_body_chars]

    if not relevant:
        return "{}"

    # Format markdown additionalContext
    header = "## Relevant memories from your past (memo)"
    footer = "_Use `/memo get <id>` to see full content._"
    lines = [header, ""]
    budget_chars = token_budget * 4 if token_budget > 0 else None
    used_chars = 0

    for h in relevant:
        score_tag = f" (score {h.score:.2f})" if h.score is not None else ""
        body = (h.body or "").strip().replace("\n", " ")
        if len(body) > body_chars:
            body = body[:body_chars].rstrip() + "…"
        block_lines = [f"**[{h.id[:8]}] {h.title}**{score_tag}"]
        if h.tags:
            block_lines.append(f"_tags_: {', '.join(h.tags)}")
        if body:
            block_lines.append(f"> {body}")
        block_lines.append("")
        block = "\n".join(block_lines)

        if budget_chars is None:
            lines.extend(block_lines)
        else:
            remaining = budget_chars - used_chars
            if remaining <= 0:
                break
            if len(block) <= remaining:
                lines.extend(block_lines)
                used_chars += len(block)
            else:
                break

    lines.append(footer)

    # Log to recall.log
    latency_ms: int | None = None
    if t0 is not None:
        latency_ms = int((time.time() - t0) * 1000)
    try:
        from memo.dashboard import append_recall_log
        append_recall_log(
            cfg.state_dir,
            prompt=prompt,
            hits=[{"id": h.id, "score": h.score, "title": h.title} for h in relevant],
            mode=mode,
            latency_ms=latency_ms,
            via="daemon",
        )
    except Exception:
        pass

    output = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": "\n".join(lines),
        }
    }
    return json.dumps(output, ensure_ascii=False)


class _RecallHandler(socketserver.StreamRequestHandler):
    """Handle one connection: read a JSON line, respond with a JSON line."""

    server: _RecallServer  # type: ignore[assignment]

    def _write_response(self, result: str, *, debug: bool) -> None:
        try:
            self.wfile.write((result + "\n").encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            if debug:
                print(f"# recall-daemon: client disconnected before response: {exc}", file=sys.stderr)

    def _embed_query(self, req: dict[str, Any]) -> str:
        text = str(req.get("text") or "")
        if not text.strip():
            return json.dumps({"error": "embed_query: empty text"})
        with self.server._lock:
            vec = self.server._mem.embedder.embed_query(text)
        return json.dumps({
            "vector": vec,
            "dim": len(vec),
            "model": self.server._cfg.embedder_model,
        }, ensure_ascii=False)

    def _embed_batch(self, req: dict[str, Any]) -> str:
        texts = req.get("texts")
        if not isinstance(texts, list):
            return json.dumps({"error": "embed_batch: `texts` must be a list"})
        if not texts:
            return json.dumps({
                "vectors": [],
                "dim": 0,
                "model": self.server._cfg.embedder_model,
            })
        if not all(isinstance(t, str) for t in texts):
            return json.dumps({"error": "embed_batch: every element of `texts` must be a string"})
        with self.server._lock:
            vectors = self.server._mem.embedder.embed(texts)
        dim = len(vectors[0]) if vectors else 0
        return json.dumps({
            "vectors": vectors,
            "dim": dim,
            "model": self.server._cfg.embedder_model,
        }, ensure_ascii=False)

    def _ping(self) -> str:
        stats = getattr(self.server, "_stats", None)
        snap = stats.snapshot() if stats is not None else {}
        return json.dumps({
            "ok": True,
            "model": self.server._cfg.embedder_model,
            "dims": self.server._cfg.embedder_dims,
            "started_at": snap.get("started_at"),
            "uptime_s": snap.get("uptime_s"),
        })

    def _stats(self) -> str:
        stats = getattr(self.server, "_stats", None)
        if stats is None:
            return json.dumps({"error": "stats not initialised"})
        return json.dumps(stats.snapshot(), ensure_ascii=False)

    def handle(self) -> None:
        t0 = time.time()
        debug = os.environ.get("MEMO_RECALL_DEBUG") == "1"
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
            except Exception as exc:
                error = True
                if debug:
                    print(f"# recall-daemon: parse error: {exc}", file=sys.stderr)
                self._write_response("{}", debug=debug)
                return

            if not isinstance(req, dict):
                error = True
                self._write_response("{}", debug=debug)
                return

            # Default `op` to "recall" so legacy clients (no `op` field) keep
            # the original prompt/cwd contract working.
            op = str(req.get("op") or "recall").strip()

            try:
                if op == "recall":
                    prompt = (req.get("prompt") or "").strip()
                    cwd = req.get("cwd") or None
                    if not prompt:
                        self._write_response("{}", debug=debug)
                        return
                    with self.server._lock:
                        result = _recall_logic(
                            prompt, cwd, self.server._mem, self.server._cfg, debug, t0=t0,
                        )
                elif op == "embed_query":
                    result = self._embed_query(req)
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
                if debug:
                    print(f"# recall-daemon: handler error (op={op}): {exc}", file=sys.stderr)
                result = json.dumps({"error": f"{type(exc).__name__}: {exc}"})

            self._write_response(result, debug=debug)
        finally:
            latency_ms = (time.time() - t0) * 1000.0
            stats = getattr(self.server, "_stats", None)
            if stats is not None:
                stats.record(op, latency_ms, error=error)


class _RecallServer(socketserver.ThreadingUnixStreamServer):
    """Unix domain socket server with persistent Memory."""

    def __init__(self, sock_path: str, cfg: Any, mem: Any) -> None:
        self._cfg = cfg
        self._mem = mem
        self._lock = threading.Lock()
        self._stats = _DaemonStats(
            started_at=time.time(),
            model=cfg.embedder_model,
            dims=cfg.embedder_dims,
        )
        # SO_REUSEADDR is a no-op for AF_UNIX sockets; the actual guard against
        # stale files is the explicit unlink in run_server before bind().
        super().__init__(sock_path, _RecallHandler)

    def server_close(self) -> None:
        super().server_close()


def _cleanup(state_dir: Path) -> None:
    _socket_path(state_dir).unlink(missing_ok=True)
    _pid_file(state_dir).unlink(missing_ok=True)


def run_server(state_dir: Path | None = None) -> None:
    """Start the recall daemon. Called by `memo recall-daemon _serve` (internal)."""
    from memo.config import Config
    from memo.memory import Memory

    cfg = Config.from_env()
    if state_dir is None:
        state_dir = cfg.state_dir

    state_dir.mkdir(parents=True, exist_ok=True)
    sock_path = _socket_path(state_dir)
    pid_file = _pid_file(state_dir)

    # Check if already running
    existing_pid = _read_pid(state_dir)
    if existing_pid is not None and _is_pid_alive(existing_pid):
        print("recall-daemon: already running", file=sys.stderr)
        sys.exit(0)

    # Stale files cleanup
    sock_path.unlink(missing_ok=True)
    pid_file.unlink(missing_ok=True)

    # Load Memory (triggers embedder warm)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    mem = Memory(cfg)

    # Write PID file
    pid_file.write_text(str(os.getpid()))

    try:
        server = _RecallServer(str(sock_path), cfg, mem)
    except OSError as exc:
        # Another instance won the TOCTOU race and already bound the socket.
        print(f"recall-daemon: bind failed ({exc}), exiting", file=sys.stderr)
        pid_file.unlink(missing_ok=True)
        sys.exit(0)

    def _sigterm(signum: int, frame: Any) -> None:
        # server.shutdown() deadlocks when called from a signal handler because
        # it waits for serve_forever() to exit, which is blocked in the same
        # thread. Use os._exit() after cleanup instead.
        _cleanup(state_dir)
        os._exit(0)

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    debug = os.environ.get("MEMO_RECALL_DEBUG") == "1"
    if debug:
        print(f"# recall-daemon: listening on {sock_path}", file=sys.stderr)

    try:
        interval = float(
            os.environ.get("MEMO_EMBEDDER_STATS_INTERVAL_S")
            or _STATS_DEFAULT_PERSIST_INTERVAL_S
        )
    except ValueError:
        interval = _STATS_DEFAULT_PERSIST_INTERVAL_S
    if interval > 0:
        persister = threading.Thread(
            target=_stats_persister,
            args=(state_dir, server._stats, interval),
            daemon=True,
            name="recall-daemon-stats-persister",
        )
        persister.start()

    try:
        server.serve_forever()
    finally:
        _cleanup(state_dir)


def _send_request(state_dir: Path, payload: dict[str, Any], timeout: float) -> str | None:
    """Send one JSON-line request to the daemon, return the JSON-line response.

    Returns `None` if the daemon socket is missing, refused, or times out so
    callers can transparently fall back to in-process execution.
    """
    import socket

    sock_path = _socket_path(state_dir)
    if not sock_path.exists():
        return None

    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(str(sock_path))
            req = json.dumps(payload, ensure_ascii=False)
            sock.sendall((req + "\n").encode("utf-8"))
            buf = b""
            while b"\n" not in buf:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                buf += chunk
                if len(buf) >= _MAX_LINE_BYTES:
                    break  # runaway response — stop buffering
        line = buf.decode("utf-8", errors="replace").strip()
        return line if line else None
    except (FileNotFoundError, ConnectionRefusedError, OSError, TimeoutError):
        return None
    except Exception:
        return None


def connect_and_recall(state_dir: Path, prompt: str, cwd: str | None, timeout: float = 1.0) -> str | None:
    """Try to get a recall result from a running daemon.

    Returns the JSON response string on success, None if the daemon is not
    reachable (caller should fall back to subprocess logic).
    """
    return _send_request(state_dir, {"prompt": prompt, "cwd": cwd or ""}, timeout)


def connect_and_send(state_dir: Path, payload: dict[str, Any], timeout: float = 5.0) -> str | None:
    """Public socket helper for non-recall ops (embed/embed_batch/ping).

    Thin wrapper over `_send_request` so callers (embedder_client,
    `memo embed-daemon stats`) don't reach for the private name.
    """
    return _send_request(state_dir, payload, timeout)
