"""Tests for MEMO_RERANK_ADAPTIVE_POOL adaptive pool sizing.

Verifies that, when the flag is on:
- high-variance vec scores → wider rerank candidate pool (int(rerank_input_k * 1.5), capped at 200)
- low-variance vec scores  → narrower pool (max(limit + 5, 15))
- medium-variance          → pool unchanged (rerank_input_k)

The tests monkeypatch `store.search`, `store.search_bm25`, and `_rrf_fuse` so
no real MLX or DB is needed, and capture the `limit` arg passed to `_rrf_fuse`
to assert on pool size.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Helper: build a list of fake vec-hit dicts with controlled scores
# ---------------------------------------------------------------------------

def _vec_hits(scores: list[float]) -> list[dict[str, Any]]:
    return [
        {
            "id": f"id{i}",
            "path": f"x/id{i}.md",
            "title": f"t{i}",
            "type": "note",
            "tags": [],
            "created": "2026-01-01",
            "updated": "2026-01-01",
            "extra": {},
            "score": s,
        }
        for i, s in enumerate(scores)
    ]


def _stddev(scores: list[float]) -> float:
    mean = sum(scores) / len(scores)
    var = sum((s - mean) ** 2 for s in scores) / len(scores)
    return math.sqrt(var)


# ---------------------------------------------------------------------------
# Verify our test data actually has the right variance profile
# ---------------------------------------------------------------------------

def test_high_variance_fixture():
    scores = [0.95, 0.20, 0.85, 0.15, 0.90, 0.10, 0.80, 0.12]
    assert _stddev(scores) > 0.15, "fixture must be high-variance"


def test_low_variance_fixture():
    scores = [0.60, 0.61, 0.60, 0.61, 0.60, 0.60, 0.61, 0.60]
    assert _stddev(scores) < 0.05, "fixture must be low-variance"


def test_medium_variance_fixture():
    scores = [0.90, 0.65, 0.80, 0.60, 0.85, 0.70, 0.75, 0.72]
    assert 0.05 <= _stddev(scores) <= 0.15, "fixture must be medium-variance"


# ---------------------------------------------------------------------------
# Factory: isolated Memory with reranker_enabled=True and a known rerank_input_k
# ---------------------------------------------------------------------------

def _make_mem(tmp_path, monkeypatch, *, rerank_input_k: int = 20):
    """Build an isolated Memory with reranker_enabled=True and stub embedder."""
    from memo.config import Config
    from memo.memory import Memory

    data = tmp_path / "data"
    vault = tmp_path / "vault"
    state = tmp_path / "state"
    data.mkdir()
    (vault / "Obsidian" / "AI" / "memory").mkdir(parents=True)
    state.mkdir()
    monkeypatch.setenv("MEMO_CONFIG_FILE", str(tmp_path / "memo-config.toml"))

    cfg = Config(
        data_dir=data,
        vault_path=vault,
        state_dir=state,
        reranker_enabled=True,
        rerank_input_k=rerank_input_k,
    )
    mem = Memory(cfg)

    def _fake_embedding(text: str) -> list[float]:
        digest = hashlib.sha256((text or "").encode("utf-8")).digest()
        values = [
            ((digest[i % len(digest)] / 255.0) * 2.0) - 1.0
            for i in range(cfg.embedder_dims)
        ]
        norm = sum(v * v for v in values) ** 0.5
        return [v / norm for v in values]

    mem.embedder.embed = lambda inputs: [_fake_embedding(t) for t in inputs]
    mem.embedder.embed_query = lambda query: _fake_embedding(query)
    return mem


# ---------------------------------------------------------------------------
# Helper: run search() and capture the limit passed to _rrf_fuse
# ---------------------------------------------------------------------------

def _captured_pool(
    tmp_path,
    monkeypatch,
    vec_scores: list[float],
    *,
    adaptive: str = "1",
    rerank_input_k: int = 20,
    search_limit: int | None = None,
) -> int:
    """Build a Memory, run search() in hybrid mode, and return the `limit`
    arg that was passed to `_rrf_fuse` (= the effective rerank pool size)."""
    mem = _make_mem(tmp_path, monkeypatch, rerank_input_k=rerank_input_k)
    hits = _vec_hits(vec_scores)
    captured: dict[str, Any] = {}

    # Patch store.search to return our controlled vec hits
    monkeypatch.setattr(mem.store, "search", lambda *a, **kw: hits)
    # Patch store.search_bm25 to return empty
    monkeypatch.setattr(mem.store, "search_bm25", lambda *a, **kw: [])

    # Patch _rrf_fuse at the module level where search_ops imports it
    import memo.memory.search_ops as _sops

    def _capturing_fuse(*_lists, limit, **kw):
        captured["limit"] = limit
        return []

    monkeypatch.setattr(_sops, "_rrf_fuse", _capturing_fuse)

    env_patch = {
        "MEMO_RERANK_ADAPTIVE_POOL": adaptive,
        # Disable features that open extra DB connections / do extra work
        "MEMO_HEALTH_SCORES_DISABLED": "1",
        "MEMO_FEEDBACK_DISABLED": "1",
    }
    with patch.dict("os.environ", env_patch):
        kw: dict[str, Any] = {"mode": "hybrid", "disable_reranker": True}
        if search_limit is not None:
            kw["limit"] = search_limit
        mem.search("test query", **kw)

    mem.close()
    return captured.get("limit", -1)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

def test_high_variance_widens_pool(tmp_path, monkeypatch):
    """High-variance scores → pool = min(int(rerank_input_k * 1.5), 200)."""
    scores = [0.95, 0.20, 0.85, 0.15, 0.90, 0.10, 0.80, 0.12]
    assert _stddev(scores) > 0.15

    pool = _captured_pool(tmp_path, monkeypatch, scores, rerank_input_k=20)

    # Expected: min(int(20 * 1.5), 200) = 30
    assert pool == min(int(20 * 1.5), 200), f"expected 30, got {pool}"


def test_low_variance_shrinks_pool(tmp_path, monkeypatch):
    """Low-variance tight cluster → pool = max(limit + 5, 15)."""
    scores = [0.60, 0.61, 0.60, 0.61, 0.60, 0.60, 0.61, 0.60]
    assert _stddev(scores) < 0.05

    pool = _captured_pool(
        tmp_path, monkeypatch, scores, rerank_input_k=20, search_limit=10
    )

    # limit=10 → max(10 + 5, 15) = 15
    assert pool == max(10 + 5, 15), f"expected 15, got {pool}"


def test_medium_variance_keeps_pool_unchanged(tmp_path, monkeypatch):
    """Medium-variance → pool = rerank_input_k unchanged."""
    scores = [0.90, 0.65, 0.80, 0.60, 0.85, 0.70, 0.75, 0.72]
    assert 0.05 <= _stddev(scores) <= 0.15

    pool = _captured_pool(tmp_path, monkeypatch, scores, rerank_input_k=20)

    assert pool == 20, f"expected 20 (unchanged), got {pool}"


def test_adaptive_pool_disabled_by_default(tmp_path, monkeypatch):
    """When MEMO_RERANK_ADAPTIVE_POOL=0, pool is always rerank_input_k."""
    scores = [0.95, 0.20, 0.85, 0.15, 0.90, 0.10]  # high-variance
    assert _stddev(scores) > 0.15

    pool = _captured_pool(
        tmp_path, monkeypatch, scores, adaptive="0", rerank_input_k=20
    )

    # Must stay at 20, not widened to 30
    assert pool == 20, f"expected 20 (flag off), got {pool}"


def test_high_variance_pool_capped_at_200(tmp_path, monkeypatch):
    """Pool must not exceed 200 even when 1.5x would exceed it."""
    scores = [0.95, 0.20, 0.85, 0.15, 0.90, 0.10, 0.80, 0.12]
    assert _stddev(scores) > 0.15

    # rerank_input_k=150 → 1.5x = 225 → capped at 200
    pool = _captured_pool(
        tmp_path, monkeypatch, scores, rerank_input_k=150
    )

    assert pool == 200, f"expected 200 (cap), got {pool}"
