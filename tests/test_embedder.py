"""Unit tests for the embedder's non-MLX surface.

The MLX forward pass is gated by ``@pytest.mark.requires_mlx`` elsewhere; this
module covers the pure logic that runs everywhere: the LRU fallback cache, the
embedding validator, and the query-side asymmetric-prefix contract (MLX
invariant #1). ``MLXEmbedder()`` construction is lazy — it does not load the
model — so we can stub ``.embed`` and exercise ``embed_query`` without MLX.
"""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from types import ModuleType

import pytest

from memo.embedder import (
    _QUERY_INSTRUCTION_PREFIX,
    MLXEmbedder,
    _SimpleLRU,
    assert_valid_embedding,
)


def test_load_resolves_exact_snapshot_before_mlx_lm(monkeypatch):
    calls: dict[str, str] = {}
    sha = "e" * 40
    hf = ModuleType("huggingface_hub")
    mlx_lm = ModuleType("mlx_lm")

    def snapshot_download(*, repo_id: str, revision: str) -> str:
        calls.update(repo_id=repo_id, revision=revision)
        return "/cache/embedder-snapshot"

    def load(path: str):
        calls["load_path"] = path
        return object(), object()

    hf.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    mlx_lm.load = load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hf)
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)

    emb = MLXEmbedder(model_path="someone/embedder", revision=sha, expected_dims=4)
    emb._ensure_loaded()

    assert calls == {
        "repo_id": "someone/embedder",
        "revision": sha,
        "load_path": "/cache/embedder-snapshot",
    }


# -- _SimpleLRU ------------------------------------------------------------


def test_simplelru_get_put_roundtrip():
    lru = _SimpleLRU(4)
    assert lru.get("missing") is None
    lru.put("a", [1.0])
    assert lru.get("a") == [1.0]


def test_simplelru_evicts_least_recently_used():
    lru = _SimpleLRU(2)
    lru.put("a", 1)
    lru.put("b", 2)
    lru.get("a")  # touch a → b is now LRU
    lru.put("c", 3)  # evicts b
    assert lru.get("b") is None
    assert lru.get("a") == 1
    assert lru.get("c") == 3


def test_simplelru_minimum_capacity_is_one():
    lru = _SimpleLRU(0)  # clamped to 1
    lru.put("a", 1)
    lru.put("b", 2)
    assert lru.get("a") is None
    assert lru.get("b") == 2


def test_simplelru_is_thread_safe_under_concurrent_get_put():
    """get/put run in embed_query OUTSIDE gpu_guard; under the FastMCP HTTP
    threadpool concurrent memo_search/memo_ask share one embedder. Without a
    lock, a get whose key another thread evicts (popitem) between the
    ``key not in self._d`` check and ``move_to_end`` KeyErrors. A tiny cap forces
    constant eviction and a lowered thread-switch interval widens that window so
    the race surfaces; with the lock the test always passes."""
    import sys
    import threading

    lru = _SimpleLRU(4)  # tiny cap → nearly every put triggers popitem eviction
    key_space = 32  # >> cap so gets straddle the LRU eviction boundary
    n_threads = 8
    errors: list[BaseException] = []
    barrier = threading.Barrier(n_threads)

    def worker(base: int) -> None:
        barrier.wait()  # release all threads together to maximise contention
        try:
            for i in range(4000):
                lru.put(f"k{(base + i) % key_space}", i)
                # get low-index keys that live near the LRU end — exactly the
                # ones a concurrent put is about to popitem.
                lru.get(f"k{i % key_space}")
                lru.get(f"k{(i * 7) % key_space}")
        except BaseException as exc:  # capture any race error, incl. KeyError
            errors.append(exc)

    _prev_switch = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # force frequent GIL handoffs mid-operation
    try:
        threads = [threading.Thread(target=worker, args=(b * 4,)) for b in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        sys.setswitchinterval(_prev_switch)

    assert not errors, f"concurrent get/put raced: {errors!r}"


# -- assert_valid_embedding ------------------------------------------------


def test_assert_valid_embedding_accepts_unit_vector():
    assert_valid_embedding([1.0, 0.0, 0.0, 0.0], 4)  # norm == 1.0, no raise


def test_assert_valid_embedding_rejects_wrong_dim():
    with pytest.raises(ValueError, match="dim mismatch"):
        assert_valid_embedding([1.0, 0.0], 4, context="repo-index")


def test_assert_valid_embedding_rejects_unnormalised():
    with pytest.raises(ValueError, match="norm out of"):
        assert_valid_embedding([5.0, 5.0, 5.0, 5.0], 4)


def test_assert_valid_embedding_context_in_message():
    with pytest.raises(ValueError, match=r"\[ctx-tag\]"):
        assert_valid_embedding([0.0, 0.0], 4, context="ctx-tag")


# -- embed_query: asymmetric prefix (MLX invariant #1) ---------------------


def test_embed_query_prepends_instruction_prefix_and_strips():
    emb = MLXEmbedder(expected_dims=4)
    seen: list[list[str]] = []

    def fake_embed(inputs):
        seen.append(list(inputs))
        return [[1.0, 0.0, 0.0, 0.0]]

    emb.embed = fake_embed  # type: ignore[method-assign]
    out = emb.embed_query("  hello world  ")

    assert out == [1.0, 0.0, 0.0, 0.0]
    # stripped once, then prefixed with the asymmetric-retrieval instruction
    assert seen == [[_QUERY_INSTRUCTION_PREFIX + "hello world"]]


def test_embed_query_caches_when_enabled(monkeypatch):
    monkeypatch.setenv("MEMO_QUERY_CACHE_SIZE", "8")
    emb = MLXEmbedder(expected_dims=4)
    calls: list[list[str]] = []

    def fake_embed(inputs):
        calls.append(list(inputs))
        return [[1.0, 0.0, 0.0, 0.0]]

    emb.embed = fake_embed  # type: ignore[method-assign]
    # Unique key avoids collisions with the process-global shared cache.
    q = "test_embed_query_caches_when_enabled unique probe"
    first = emb.embed_query(q)
    second = emb.embed_query(q)
    assert first == second
    assert len(calls) == 1  # second call served from cache


def test_cache_size_param_overrides_off_env(monkeypatch):
    # Explicit cache_size wins over the raw env default (which is off/0). This
    # is how flags-aware callers (Memory facade, MCP daemon) wire the registry
    # default into the foundation-module embedder.
    monkeypatch.delenv("MEMO_QUERY_CACHE_SIZE", raising=False)
    assert MLXEmbedder(expected_dims=4, cache_size=16)._query_cache is not None


def test_cache_size_param_zero_disables_even_with_env(monkeypatch):
    monkeypatch.setenv("MEMO_QUERY_CACHE_SIZE", "500")  # env says on...
    assert MLXEmbedder(expected_dims=4, cache_size=0)._query_cache is None  # ...0 wins


def test_memory_facade_enables_query_cache_by_default(tmp_cfg, monkeypatch):
    # Regression: the embedder reads MEMO_QUERY_CACHE_SIZE raw (default off),
    # so before the facade passed the registry default (256) the query cache
    # was silently disabled on every Memory-backed path (recall hook, CLI).
    monkeypatch.delenv("MEMO_QUERY_CACHE_SIZE", raising=False)
    from memo.memory import Memory

    mem = Memory(tmp_cfg)
    assert mem.embedder._query_cache is not None


def test_embed_query_no_cache_calls_embed_each_time(monkeypatch):
    monkeypatch.delenv("MEMO_QUERY_CACHE_SIZE", raising=False)
    emb = MLXEmbedder(expected_dims=4)
    assert emb._query_cache is None
    calls: list[list[str]] = []

    def fake_embed(inputs):
        calls.append(list(inputs))
        return [[1.0, 0.0, 0.0, 0.0]]

    emb.embed = fake_embed  # type: ignore[method-assign]
    emb.embed_query("z")
    emb.embed_query("z")
    assert len(calls) == 2


# -- misc contracts --------------------------------------------------------


def test_dims_property_returns_expected_dims():
    assert MLXEmbedder(expected_dims=2560).dims == 2560


def test_unload_is_idempotent_without_mlx():
    emb = MLXEmbedder(expected_dims=4)
    emb.unload()
    emb.unload()  # no model loaded, no MLX → must not raise
    assert emb._model is None


def test_unload_cold_embedder_does_not_wait_for_gpu(monkeypatch):
    """Closing a never-loaded embedder must not contend on the machine GPU lock."""
    fake_core = types.ModuleType("mlx.core")
    fake_core.clear_cache = lambda: None  # type: ignore[attr-defined]
    fake_mlx = types.ModuleType("mlx")
    fake_mlx.core = fake_core  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)

    @contextmanager
    def unexpected_gpu_guard():
        raise AssertionError("a cold embedder must not acquire the GPU lock")
        yield  # pragma: no cover

    monkeypatch.setattr("memo.embedder.gpu_guard", unexpected_gpu_guard)

    MLXEmbedder(expected_dims=4).unload()


# -- MicroEmbedder load-failure contract -------------------------------------


def test_micro_embedder_load_failure_raises_instead_of_zero_vectors(monkeypatch):
    """A MicroEmbedder whose model failed to load must surface the failure —
    silently returning all-zero vectors made recall score every candidate
    equally (and return empty) instead of the caller falling back to BM25."""
    from memo.embedder import MicroEmbedder

    micro = MicroEmbedder("stub/micro-that-never-loads", expected_dims=4)
    # Simulate a failed load: _ensure_loaded ran but _model stayed None.
    monkeypatch.setattr(MicroEmbedder, "_ensure_loaded", lambda self: None)

    assert micro.is_warm is False
    assert micro.embed([]) == []  # empty input stays a cheap no-op
    with pytest.raises(RuntimeError, match="failed to load"):
        micro.embed(["hola"])
    with pytest.raises(RuntimeError, match="failed to load"):
        micro.embed_query("hola")
