"""Cache-tier mode — memo as a bounded, evictable cache fronting an
authoritative backing store.

memo's default identity is a **durable** semantic memory: the local vault is
the source of truth and nothing is ever evicted automatically (see CLAUDE.md
"Source of truth — role & contract"). This module implements the OPT-IN
inversion of that contract: when `MEMO_CACHE_MODE != off`, the local store
becomes a *derived cache* in front of an authoritative backing store
(`MEMO_CACHE_BACKEND`, e.g. Memflow). That introduces four behaviours memo
otherwise lacks:

  - capacity bound + eviction (this module)
  - read-through on a local miss (wired in `Memory.search`)
  - a write policy: write-through / write-back (wired in `Memory.save`)
  - coherence/invalidation against the backing store

`CachePolicy` reads the `MEMO_CACHE_*` flags (registered in `flags.py`).
`CacheManager` owns eviction; the backing-store client is injected as a
`CacheBackend` so this module stays independent of the concrete backend.

Eviction reuses existing store primitives — `VecStore.count()`,
`VecStore.eviction_candidates()` (hit-count / last-access ordering) — and
routes the actual removal through `Memory.delete()` so the vault file, vec
index, FTS row, history log, and graph edges all stay consistent.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from memo.flags import flag_int, flag_str

_log = logging.getLogger(__name__)

_VALID_MODES = {"off", "read_through", "write_through", "write_back"}
_VALID_EVICTION = {"lru", "lfu", "ttl"}

# extra-bag key marking a memoria written locally but not yet persisted to the
# backing store (write-back mode). Such entries must be flushed before they can
# be evicted, or the write is lost.
CACHE_DIRTY_KEY = "_cache_dirty"


@dataclass(frozen=True)
class CachePolicy:
    """Resolved cache-tier configuration. `mode == "off"` (the default)
    means memo behaves as a durable store and every method here no-ops."""

    mode: str = "off"
    max_entries: int = 0  # 0 = unbounded (durable behavior)
    eviction: str = "lru"
    ttl_days: int = 0  # 0 = no freshness window
    backend: str = "memflow"

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> CachePolicy:
        mode = (flag_str("MEMO_CACHE_MODE", env=env) or "off").strip().lower()
        if mode not in _VALID_MODES:
            _log.warning("invalid MEMO_CACHE_MODE=%r; falling back to 'off'", mode)
            mode = "off"
        eviction = (flag_str("MEMO_CACHE_EVICTION", env=env) or "lru").strip().lower()
        if eviction not in _VALID_EVICTION:
            _log.warning("invalid MEMO_CACHE_EVICTION=%r; falling back to 'lru'", eviction)
            eviction = "lru"
        return cls(
            mode=mode,
            max_entries=max(0, flag_int("MEMO_CACHE_MAX_ENTRIES", env=env) or 0),
            eviction=eviction,
            ttl_days=max(0, flag_int("MEMO_CACHE_TTL_DAYS", env=env) or 0),
            backend=(flag_str("MEMO_CACHE_BACKEND", env=env) or "memflow").strip().lower(),
        )

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def write_through(self) -> bool:
        return self.mode == "write_through"

    @property
    def write_back(self) -> bool:
        return self.mode == "write_back"

    @property
    def read_through(self) -> bool:
        # every enabled mode reads through on a miss; only the write side differs
        return self.enabled


class CacheBackend(Protocol):
    """The authoritative store the cache fronts. Implemented in `sync.py`
    (Memflow client / remote vault) and injected into `CacheManager` +
    `Memory`. Kept as a Protocol so this module has no backend dependency."""

    def push(self, record: Any) -> bool:
        """Persist a memoria to the backing store. Returns True on success."""
        ...

    def fetch(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        """Read-through: fetch candidate memorias from the backing store."""
        ...

    def has_current(self, id_: str, body_hash: str) -> bool:
        """Coherence check: does the backing store already hold this exact
        (id, body_hash)? Used to decide if a clean local copy can be dropped
        without a flush."""
        ...


def _days_since(ts_raw: str | None, *, fallback: str | None = None) -> int | None:
    """Whole days since an ISO timestamp (or `fallback` if the first is None).
    Returns None when neither parses."""
    raw = ts_raw or fallback
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        return (datetime.now(UTC) - ts).days
    except Exception:
        return None


class CacheManager:
    """Owns capacity-bound eviction for cache-tier mode.

    No-ops entirely when the policy is disabled or unbounded, so it is safe
    to construct and call unconditionally from `Memory`.
    """

    def __init__(
        self,
        memory: Any,
        policy: CachePolicy | None = None,
        backend: CacheBackend | None = None,
    ) -> None:
        self.memory = memory
        self.policy = policy or CachePolicy.from_env()
        self.backend = backend
        # True once a backend has been resolved (or was injected). Lets
        # `ensure_backend` build it lazily on first need without rebuilding.
        self._backend_loaded = backend is not None

    def ensure_backend(self) -> CacheBackend | None:
        """Lazily build the configured backing-store client on first need.
        Returns None when cache mode is off. Built once, then memoized."""
        if not self.policy.enabled:
            return None
        if not self._backend_loaded:
            from memo.cache_backend import make_backend

            self.backend = make_backend(self.policy.backend)
            self._backend_loaded = True
        return self.backend

    # -- flush (write-back safety) -----------------------------------------

    def _flush(self, memoria_id: str) -> bool:
        """Push a memoria to the backing store before eviction. Returns True
        if the local copy is safe to drop (clean, or successfully flushed)."""
        rec = self.memory.get(memoria_id)
        if rec is None:
            return True  # already gone
        extra = getattr(rec, "extra", None) or {}
        if not extra.get(CACHE_DIRTY_KEY):
            return True  # clean — backing store already has it (or never dirty)
        if self.backend is None:
            # Dirty but nowhere to flush: refuse to evict, never lose a write.
            _log.warning(
                "cache: memoria %s is dirty but no backend configured; "
                "skipping eviction to avoid data loss",
                memoria_id[:8],
            )
            return False
        try:
            return bool(self.backend.push(rec))
        except Exception as exc:
            _log.warning("cache: flush of %s failed: %s", memoria_id[:8], exc)
            return False

    # -- eviction ----------------------------------------------------------

    def evict_if_needed(self, *, exclude_types: set[str] | None = None) -> list[str]:
        """Reclaim local capacity down to `max_entries` under the configured
        replacement policy. Returns the ids actually evicted.

        Dirty (write-back, un-flushed) entries are flushed to the backing
        store first; any that can't be flushed are skipped rather than lost.
        Removal routes through `Memory.delete()` so vault file + indexes +
        history stay consistent.
        """
        if not self.policy.enabled or self.policy.max_entries <= 0:
            return []
        self.ensure_backend()  # so dirty entries can be flushed before removal
        store = self.memory.store
        overflow = store.count() - self.policy.max_entries
        if overflow <= 0:
            return []

        # Pull a generous candidate pool (coldest-first). We may skip some
        # (un-flushable dirty / ttl-too-fresh), so over-fetch to still reach
        # `overflow` removals when possible.
        pool = store.eviction_candidates(
            self.policy.eviction,
            overflow * 3 + 10,
            exclude_types=exclude_types,
        )
        evicted: list[str] = []
        for cand in pool:
            if len(evicted) >= overflow:
                break
            # ttl policy only evicts entries past the freshness window.
            if self.policy.eviction == "ttl" and self.policy.ttl_days > 0:
                age = _days_since(cand.get("last_accessed"), fallback=cand.get("updated"))
                if age is None or age < self.policy.ttl_days:
                    continue
            if not self._flush(cand["id"]):
                continue  # dirty + un-flushable: keep it
            if self.memory.delete(cand["id"]):
                evicted.append(cand["id"])
        if evicted:
            _log.info(
                "cache: evicted %d entr%s (policy=%s)",
                len(evicted),
                "y" if len(evicted) == 1 else "ies",
                self.policy.eviction,
            )
        return evicted

    # -- flush -------------------------------------------------------------

    def flush_all(self) -> dict[str, Any]:
        """Push every dirty (write-back, un-persisted) memoria to the backing
        store and clear its dirty flag. Returns counts. No-op when disabled or
        no backend is configured.
        """
        result = {"flushed": 0, "failed": 0, "dirty_remaining": 0}
        backend = self.ensure_backend()
        if not self.policy.enabled or backend is None:
            return result
        store = self.memory.store
        # `eviction_candidates` enumerates all rows; we filter to dirty ones.
        rows = store.eviction_candidates("lru", store.count() + 1)
        for row in rows:
            rec = self.memory.get(row["id"])
            if rec is None or not (getattr(rec, "extra", None) or {}).get(CACHE_DIRTY_KEY):
                continue
            try:
                ok = bool(backend.push(rec))
            except Exception as exc:
                _log.warning("cache: flush_all push of %s failed: %s", row["id"][:8], exc)
                ok = False
            if ok:
                merged = dict(rec.extra or {})
                merged.pop(CACHE_DIRTY_KEY, None)
                with contextlib.suppress(Exception):
                    self.memory.update(row["id"], extra=merged)
                result["flushed"] += 1
            else:
                result["failed"] += 1
                result["dirty_remaining"] += 1
        return result

    # -- stats -------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Snapshot for the `memo_cache_stats` MCP tool."""
        store = self.memory.store
        total = store.count()
        cap = self.policy.max_entries
        return {
            "mode": self.policy.mode,
            "enabled": self.policy.enabled,
            "backend": self.policy.backend if self.policy.enabled else None,
            "eviction": self.policy.eviction,
            "ttl_days": self.policy.ttl_days,
            "entries": total,
            "max_entries": cap,
            "over_capacity": max(0, total - cap) if cap > 0 else 0,
        }
