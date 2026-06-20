"""Trinity Daemon — resident process for memo, synapse and memflow state.

Evolves the recall server into a central state cache to achieve sub-second
latency for the entire intelligence suite.
"""

from __future__ import annotations

import json
import logging
import socketserver
import threading
import time
from typing import Any

# Re-use helpers from the original recall_server where possible, 
# but this will eventually replace it.
from memo.recall_server import (
    _MAX_LINE_BYTES,
    PriorityLock,
    _DaemonStats,
    _recall_logic,
)

_LOG = logging.getLogger("memo.trinity_server")


class TrinityServer(socketserver.ThreadingUnixStreamServer):
    """Unix domain socket server caching global Trinity state."""

    def __init__(self, sock_path: str, cfg: Any, mem: Any) -> None:
        self._cfg = cfg
        self._mem = mem
        
        from memo.flags import flag_bool
        if flag_bool("MEMO_RECALL_PRIORITY_ENABLED"):
            self._priority_lock = PriorityLock()
        else:
            self._priority_lock = self._make_fake_priority_lock()

        self._stats = _DaemonStats(
            started_at=time.time(),
            model=cfg.embedder_model,
            dims=cfg.embedder_dims,
        )

        # Trinity State Cache
        self._state_cache: dict[str, Any] = {
            "synapse_packet": None,
            "last_synapse_fetch": 0.0,
            "memflow_sessions": {},
            "last_memflow_fetch": 0.0,
        }
        self._cache_lock = threading.Lock()

        super().__init__(sock_path, TrinityHandler)

    def _make_fake_priority_lock(self) -> Any:
        class FakePriorityLock:
            def __init__(self, lock: threading.Lock):
                self._lock = lock
            def acquire(self, priority: int = 0, timeout: float | None = None) -> bool:
                return self._lock.acquire(timeout=timeout if timeout is not None else -1)
            def release(self) -> None:
                self._lock.release()
        return FakePriorityLock(threading.Lock())

    def get_cached_packet(self) -> dict[str, Any] | None:
        """Fetch or return cached synapse packet."""
        from memo.flags import flag_int
        ttl = flag_int("MEMO_TRINITY_CACHE_TTL") or 15
        
        with self._cache_lock:
            now = time.time()
            if self._state_cache["synapse_packet"] and (now - self._state_cache["last_synapse_fetch"]) < ttl:
                return self._state_cache["synapse_packet"]
        
        # Cache miss or stale: fetch fresh
        try:
            from memo.synapse_client import get_packet
            packet = get_packet(self._cfg)
            if packet:
                with self._cache_lock:
                    self._state_cache["synapse_packet"] = packet
                    self._state_cache["last_synapse_fetch"] = time.time()
                return packet
        except Exception as exc:
            _LOG.debug("TrinityServer: synapse fetch failed: %s", exc)
        
        return self._state_cache["synapse_packet"] # return stale if fetch failed


class TrinityHandler(socketserver.StreamRequestHandler):
    """JSON-RPC handler for Trinity operations."""

    server: TrinityServer  # narrow socketserver's BaseServer-typed attr

    def handle(self) -> None:
        for line in self.rfile:
            if len(line) > _MAX_LINE_BYTES:
                _LOG.warning("trinity-daemon: request line too long, closing")
                break
            
            t0 = time.time()
            op = "unknown"
            error = False
            try:
                req = json.loads(line.decode("utf-8"))
                op = str(req.get("op") or "recall").strip()
                
                if op == "trinity_briefing":
                    result = self._trinity_briefing(req)
                elif op == "recall":
                    # Delegate to original recall logic but with cache awareness
                    result = self._recall_with_cache(req, t0)
                elif op == "ping":
                    result = json.dumps({"ok": True, "trinity": True})
                elif op == "stats":
                    result = json.dumps(self.server._stats.snapshot())
                else:
                    # Fallback to standard embedder ops
                    result = self._delegate_to_embedder(op, req)
                
                self._write_response(result)
            except Exception as exc:
                error = True
                _LOG.error("trinity-daemon: handler error: %s", exc, exc_info=True)
                self._write_response(json.dumps({"error": str(exc)}))
            finally:
                latency_ms = (time.time() - t0) * 1000
                self.server._stats.record(op, latency_ms, error=error)

    def _trinity_briefing(self, req: dict[str, Any]) -> str:
        """Fast briefing using cached state."""
        packet = self.server.get_cached_packet()
        # In a real impl, we'd call briefing logic here but passing the packet
        # to avoid the 4s subprocess hit.
        return json.dumps({
            "status": "ok",
            "source": "trinity_cache",
            "synapse_ready": packet is not None,
            "packet": packet
        }, ensure_ascii=False)

    def _recall_with_cache(self, req: dict[str, Any], t0: float) -> str:
        # Similar to recall_server.py handle() but uses TrinityServer's mem/cfg
        prompt = (req.get("prompt") or "").strip()
        if not prompt:
            return "{}"
        
        priority = 1
        from memo.flags import flag_int
        timeout_s = max(0.1, (flag_int("MEMO_RECALL_LOCK_TIMEOUT_MS") or 2500) / 1000.0)
        
        if not self.server._priority_lock.acquire(priority=priority, timeout=timeout_s):
            return "{}"
            
        try:
            # We also pass the micro_embedder if available (omitted for brevity in this draft)
            result, log_fn = _recall_logic(
                prompt,
                req.get("cwd"),
                self.server._mem,
                self.server._cfg,
                t0=t0,
            )
            if log_fn:
                log_fn()
            return result
        finally:
            self.server._priority_lock.release()

    def _delegate_to_embedder(self, op: str, req: dict[str, Any]) -> str:
        if op == "embed_query":
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
        if op == "embed_batch":
            texts = req.get("texts")
            if not isinstance(texts, list):
                return json.dumps({"error": "embed_batch: `texts` must be a list"})
            if not all(isinstance(t, str) for t in texts):
                return json.dumps({"error": "embed_batch: every element of `texts` must be a string"})
            vectors = self.server._mem.embedder.embed(texts) if texts else []
            dim = len(vectors[0]) if vectors else 0  # type: ignore[union-attr]
            return json.dumps(
                {
                    "vectors": vectors,
                    "dim": dim,
                    "dims": dim,
                    "model": self.server._cfg.embedder_model,
                },
                ensure_ascii=False,
            )
        return json.dumps({"error": f"unknown op: {op!r}"})

    def _write_response(self, text: str) -> None:
        self.wfile.write(text.encode("utf-8") + b"\n")
        self.wfile.flush()
