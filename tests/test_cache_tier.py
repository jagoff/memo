"""Cache-tier mode tests — memo as a bounded cache fronting a backing store.

Covers the opt-in inversion of memo's durable contract:
  - MEMO_CACHE_MODE=off (default) changes nothing (no eviction, durable).
  - hit tracking (access_count / last_accessed) on reads.
  - capacity-bound eviction under lru / lfu / ttl.
  - write-through / write-back policy + dirty-flush safety.
  - read-through materialization on a local miss.

Backend is a deterministic fake injected via `mem._cache` so tests don't
shell out to the real `memflow` binary.
"""

from __future__ import annotations

from memo.cache import CACHE_DIRTY_KEY, CacheManager, CachePolicy
from memo.cache_backend import NullBackend


class FakeBackend:
    """In-memory CacheBackend double."""

    def __init__(self, fetch_result=None, push_ok=True):
        self.pushed: list[str] = []
        self._fetch = fetch_result or []
        self.push_ok = push_ok

    def push(self, record):
        self.pushed.append(record.id)
        return self.push_ok

    def fetch(self, query, *, limit=10):
        return list(self._fetch)

    def has_current(self, id_, body_hash):
        return False


def _install_cache(mem, policy, backend=None):
    """Force a specific cache policy + backend onto a Memory (bypasses env)."""
    mem._cache = CacheManager(mem, policy, backend=backend)
    return mem._cache


def _save(mem, content, title):
    return mem.save(content=content, title=title, type_="note", auto_derive=False)


# ── default: off = durable, no behavior change ───────────────────────────────


def test_cache_off_is_default_and_does_not_evict(mock_memory):
    # No MEMO_CACHE_* env set: policy resolves to off, eviction never runs.
    assert mock_memory.cache.policy.mode == "off"
    assert mock_memory.cache.policy.enabled is False
    for i in range(6):
        _save(mock_memory, f"durable fact number {i}", f"fact {i}")
    # Nothing evicted even though we wrote many — durable behavior intact.
    assert mock_memory.store.count() == 6
    assert mock_memory.cache.evict_if_needed() == []


# ── hit tracking ─────────────────────────────────────────────────────────────


def test_search_records_access_hits(mock_memory):
    _save(mock_memory, "the capital of france is paris", "france capital")
    _save(mock_memory, "unrelated note about rust borrow checker", "rust")
    hits = mock_memory.search("capital of france", limit=5)
    assert hits, "expected at least one hit to record access for"
    top = hits[0].id
    acc = mock_memory.store.get_access(top)
    assert acc["access_count"] >= 1
    assert acc["last_accessed"] is not None
    # Another search bumps the count again.
    mock_memory.search("capital of france", limit=5)
    assert mock_memory.store.get_access(top)["access_count"] >= 2


def test_lifecycle_access_count_reads_from_access_table(mock_memory):
    rec = _save(mock_memory, "a fact that gets read a lot", "hot fact")
    for _ in range(3):
        mock_memory.store.touch([rec.id])
    assert mock_memory.lifecycle.get_access_count(rec.id) == 3


# ── eviction ─────────────────────────────────────────────────────────────────


def test_eviction_lru_drops_coldest(mock_memory):
    ids = [_save(mock_memory, f"body number {i}", f"title {i}").id for i in range(5)]
    # Make the last two hot; the first three stay cold.
    mock_memory.store.touch([ids[3], ids[4]])
    mock_memory.store.touch([ids[4]])
    _install_cache(mock_memory, CachePolicy(mode="read_through", max_entries=3, eviction="lru"))
    evicted = mock_memory.cache.evict_if_needed()
    assert len(evicted) == 2
    assert mock_memory.store.count() == 3
    # The hot ones survive; cold ones go.
    assert ids[4] not in evicted and ids[3] not in evicted
    assert set(evicted).issubset({ids[0], ids[1], ids[2]})


def test_eviction_lfu_drops_least_frequent(mock_memory):
    ids = [_save(mock_memory, f"lfu body {i}", f"lfu {i}").id for i in range(4)]
    mock_memory.store.touch([ids[0]] * 1 or [ids[0]])  # 1 hit
    for _ in range(5):
        mock_memory.store.touch([ids[1]])  # very frequent
    for _ in range(3):
        mock_memory.store.touch([ids[2]])
    # ids[3] never touched -> least frequent -> first evicted.
    _install_cache(mock_memory, CachePolicy(mode="read_through", max_entries=3, eviction="lfu"))
    evicted = mock_memory.cache.evict_if_needed()
    assert evicted == [ids[3]]


def test_eviction_disabled_when_unbounded(mock_memory):
    for i in range(4):
        _save(mock_memory, f"x{i}", f"t{i}")
    _install_cache(mock_memory, CachePolicy(mode="read_through", max_entries=0))
    assert mock_memory.cache.evict_if_needed() == []
    assert mock_memory.store.count() == 4


# ── write policy ─────────────────────────────────────────────────────────────


def test_write_through_pushes_to_backend(mock_memory):
    backend = FakeBackend(push_ok=True)
    _install_cache(mock_memory, CachePolicy(mode="write_through", max_entries=0), backend)
    rec = _save(mock_memory, "write-through fact", "wt")
    assert rec.id in backend.pushed
    # Clean: a successful push leaves no dirty flag.
    stored = mock_memory.store.get(rec.id)
    assert not (stored.get("extra") or {}).get(CACHE_DIRTY_KEY)


def test_write_through_marks_dirty_on_push_failure(mock_memory):
    backend = FakeBackend(push_ok=False)
    _install_cache(mock_memory, CachePolicy(mode="write_through", max_entries=0), backend)
    rec = _save(mock_memory, "fact whose push fails", "wt-fail")
    stored = mock_memory.store.get(rec.id)
    assert (stored.get("extra") or {}).get(CACHE_DIRTY_KEY) is True


def test_write_back_marks_dirty_then_flush_clears(mock_memory):
    backend = FakeBackend(push_ok=True)
    _install_cache(mock_memory, CachePolicy(mode="write_back", max_entries=0), backend)
    rec = _save(mock_memory, "write-back fact", "wb")
    # Dirty on save (not yet pushed).
    assert (mock_memory.store.get(rec.id).get("extra") or {}).get(CACHE_DIRTY_KEY) is True
    assert rec.id not in backend.pushed
    # Flush pushes + clears the dirty flag.
    result = mock_memory.cache.flush_all()
    assert result["flushed"] >= 1
    assert rec.id in backend.pushed
    assert not (mock_memory.store.get(rec.id).get("extra") or {}).get(CACHE_DIRTY_KEY)


def test_dirty_entry_not_evicted_without_backend(mock_memory):
    # write_back with a no-op backend (push always fails): dirty entries must
    # not be evicted (would lose the write). NullBackend.push() returns False.
    _install_cache(
        mock_memory,
        CachePolicy(mode="write_back", max_entries=1, backend="none"),
        backend=NullBackend(),
    )
    r0 = _save(mock_memory, "dirty one", "d0")  # dirty (write_back)
    _save(mock_memory, "clean-ish two", "d1")  # also dirty under write_back
    # Both are dirty and there's no backend, so eviction can flush none of
    # them — capacity stays violated rather than losing a write.
    evicted = mock_memory.cache.evict_if_needed()
    assert r0.id not in evicted
    assert mock_memory.store.get(r0.id) is not None


# ── read-through ─────────────────────────────────────────────────────────────


def test_read_through_materializes_backend_hit(mock_memory):
    backend = FakeBackend(
        fetch_result=[
            {
                "id": "ext-1",
                "title": "Backing-only fact",
                "type": "note",
                "body": "this fact lives only in the backing store",
                "tags": [],
                "score": 0.9,
                "from_backend": True,
            }
        ]
    )
    _install_cache(mock_memory, CachePolicy(mode="read_through", max_entries=0), backend)
    # Local store is empty for this query; read_through pulls + materializes.
    out = mock_memory.search("backing store fact", limit=5, read_through=True)
    assert any("backing store" in r.body for r in out)
    # Materialized locally: a plain (no read-through) search now hits it.
    local = mock_memory.search("backing store fact", limit=5)
    assert any("backing store" in r.body for r in local)


def test_read_through_not_triggered_without_flag(mock_memory):
    backend = FakeBackend(
        fetch_result=[
            {
                "id": "ext-2",
                "title": "should not appear",
                "type": "note",
                "body": "must not be fetched without the flag",
                "tags": [],
            }
        ]
    )
    _install_cache(mock_memory, CachePolicy(mode="read_through", max_entries=0), backend)
    out = mock_memory.search("anything", limit=5)  # read_through defaults False
    assert mock_memory.store.count() == 0
    assert out == []


def test_read_through_fill_is_clean_not_dirty(mock_memory):
    # Even under write_back, a read-through fill mirrors the backing store and
    # must stay clean (skip write policy).
    backend = FakeBackend(
        fetch_result=[
            {
                "id": "ext-3",
                "title": "filled",
                "type": "note",
                "body": "filled from backing store",
                "tags": [],
            }
        ]
    )
    _install_cache(mock_memory, CachePolicy(mode="write_back", max_entries=0), backend)
    out = mock_memory.search("filled backing", limit=5, read_through=True)
    filled = [r for r in out if "filled" in r.body]
    assert filled
    assert not (filled[0].extra or {}).get(CACHE_DIRTY_KEY)


# ── stats ────────────────────────────────────────────────────────────────────


def test_cache_stats_shape(mock_memory):
    # Save under the default (off) policy so eviction doesn't trim during the
    # writes, then install a tighter cap to observe over_capacity reporting.
    for i in range(3):
        _save(mock_memory, f"s{i}", f"st{i}")
    _install_cache(mock_memory, CachePolicy(mode="read_through", max_entries=2, eviction="lru"))
    stats = mock_memory.cache.stats()
    assert stats["enabled"] is True
    assert stats["mode"] == "read_through"
    assert stats["entries"] == 3
    assert stats["max_entries"] == 2
    assert stats["over_capacity"] == 1


def test_policy_from_env_validates(monkeypatch):
    monkeypatch.setenv("MEMO_CACHE_MODE", "bogus")
    monkeypatch.setenv("MEMO_CACHE_EVICTION", "nonsense")
    p = CachePolicy.from_env()
    assert p.mode == "off"  # invalid mode falls back to off
    assert p.eviction == "lru"  # invalid eviction falls back to lru
