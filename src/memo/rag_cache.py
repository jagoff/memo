"""Session-scoped RAG context cache.

Caches the retrieval half of ask() (the assembled context + sources) so
repeated asks within the same session don't re-run search/rerank. Entries
are keyed by session_id + query and invalidated three ways:

- TTL (default 5 min) — bounds staleness for a live session.
- corpus_version mismatch — any save/update changes the corpus fingerprint,
  so a stale retrieval is never served after the corpus moves.
- LRU eviction — bounds memory; oldest entry drops when full.

Process-local and read-model only: losing it costs a recompute, never data.
`now` is injected by the caller (time.time()) so the cache is deterministic
under test.
"""

from __future__ import annotations

from typing import Any


class RagContextCache:
    def __init__(self, *, ttl_s: float = 300.0, max_entries: int = 128) -> None:
        self._ttl = float(ttl_s)
        self._max = int(max_entries)
        # key -> (expires_at, corpus_version, value)
        self._store: dict[str, tuple[float, str, Any]] = {}

    def get(self, key: str, *, corpus_version: str, now: float) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, cached_version, value = entry
        if now >= expires_at or cached_version != corpus_version:
            self._store.pop(key, None)
            return None
        return value

    def put(self, key: str, value: Any, *, corpus_version: str, now: float) -> None:
        if key not in self._store and len(self._store) >= self._max:
            # Evict the entry closest to expiry (oldest write under a fixed TTL).
            oldest = min(self._store, key=lambda k: self._store[k][0])
            self._store.pop(oldest, None)
        self._store[key] = (now + self._ttl, corpus_version, value)

    def clear(self) -> None:
        self._store.clear()
