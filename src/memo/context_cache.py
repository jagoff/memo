from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any


class TurnContextCache:
    """Small process-local TTL cache for read-only context envelopes."""

    def __init__(self, *, max_size: int = 128, ttl_s: int = 60) -> None:
        self.max_size = max(0, int(max_size))
        self.ttl_s = max(0, int(ttl_s))
        self._items: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        # Shared across FastMCP's worker threads (read-only tools mutate it on
        # every hit via move_to_end / eviction), so the OrderedDict needs a lock.
        self._lock = threading.Lock()

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.max_size or not self.ttl_s:
            return None
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            expires_at, value = item
            if expires_at < time.monotonic():
                self._items.pop(key, None)
                return None
            self._items.move_to_end(key)
            return dict(value)

    def set(self, key: str, value: dict[str, Any]) -> None:
        if not self.max_size or not self.ttl_s:
            return
        with self._lock:
            self._items[key] = (time.monotonic() + self.ttl_s, dict(value))
            self._items.move_to_end(key)
            while len(self._items) > self.max_size:
                self._items.popitem(last=False)


def stable_cache_key(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


_CONTEXT_CACHE = TurnContextCache()


def context_cache() -> TurnContextCache:
    from memo.flags import flag_int

    ttl = flag_int("MEMO_CONTEXT_CACHE_TTL") or 0
    _CONTEXT_CACHE.ttl_s = max(0, ttl)
    return _CONTEXT_CACHE
