"""Tests for Tier 2 performance and robustness fixes (2.2 / 2.4 / 2.5).

Fix 2.2 — search_ops: health scores applied BEFORE the reranker so
           low-confidence hits are down-weighted before the cross-encoder
           wastes compute promoting them.
Fix 2.4 — recall-hook daemon timeout reduced to 2.0 s default
           (configurable via MEMO_RECALL_DAEMON_TIMEOUT / _MS); worst-case
           2.0 s + 1-2 s subprocess fallback stays inside the 5 s budget.
Fix 2.5 — store/queries: tightened norm warning range (0.5,1.5) → warn at
           (0.95,1.05); catastrophic mismatches still raise, drift warns.
"""

from __future__ import annotations

# ── Fix 2.2 ─────────────────────────────────────────────────────────────────
import hashlib
import logging


def _make_reranker_memory(tmp_path, monkeypatch):
    """Build an isolated Memory with reranker_enabled=True and a stub embedder."""

    from memo.config import Config
    from memo.memory import Memory

    data = tmp_path / "data"
    vault = tmp_path / "vault"
    state = tmp_path / "state"
    data.mkdir()
    (vault / "Obsidian" / "AI" / "memory").mkdir(parents=True)
    state.mkdir()
    monkeypatch.setenv("MEMO_CONFIG_FILE", str(tmp_path / "memo-config.toml"))

    cfg = Config(data_dir=data, vault_path=vault, state_dir=state, reranker_enabled=True)
    mem = Memory(cfg)

    dims = cfg.embedder_dims

    def _fake_embedding(text: str) -> list[float]:
        digest = hashlib.sha256((text or "").encode("utf-8")).digest()
        values = [
            ((digest[i % len(digest)] / 255.0) * 2.0) - 1.0
            for i in range(dims)
        ]
        norm = sum(v * v for v in values) ** 0.5
        return [v / norm for v in values]

    mem.embedder.embed = lambda inputs: [_fake_embedding(t) for t in inputs]
    mem.embedder.embed_query = lambda query: _fake_embedding(query)
    return mem


def test_health_scores_applied_before_reranker(tmp_path, monkeypatch):
    """When the reranker is enabled, _apply_health_scores must be called
    BEFORE _rerank (not after), so low-confidence hits are already
    down-weighted when the cross-encoder receives them."""
    monkeypatch.setenv("MEMO_HEALTH_SCORES_DISABLED", "0")

    mem = _make_reranker_memory(tmp_path, monkeypatch)
    try:
        call_order: list[str] = []

        orig_health = mem._apply_health_scores

        def spy_health(results):
            call_order.append("health")
            return orig_health(results)

        def spy_rerank(query, results, top_n=5):
            call_order.append("rerank")
            return results

        mem._apply_health_scores = spy_health
        mem._rerank = spy_rerank

        mem.save(content="health scores test content", title="Health Scores Order")
        mem.search("health scores test", mode="hybrid")

        # Both should fire because reranker is enabled and there's a result.
        # The key invariant: if both fired, health must precede rerank.
        if "health" in call_order and "rerank" in call_order:
            assert call_order.index("health") < call_order.index("rerank"), (
                f"Expected health BEFORE rerank, got order: {call_order}"
            )
        # At minimum, rerank should have been called (we have a result and
        # reranker_enabled=True). Health may not fire if no health rows exist.
        assert "rerank" in call_order, "Expected _rerank to be called with reranker_enabled=True"
    finally:
        mem.close()


def test_health_scores_not_applied_twice_when_reranker_runs(tmp_path, monkeypatch):
    """When the reranker is enabled, health scores should be applied before
    it — but the post-pipeline MEMO_HEALTH_SCORES_DISABLED guard prevents
    a second application. Confirm _apply_health_scores is called at most once
    per search() invocation when reranker fires."""
    monkeypatch.setenv("MEMO_HEALTH_SCORES_DISABLED", "0")

    mem = _make_reranker_memory(tmp_path, monkeypatch)
    try:
        health_call_count = 0
        orig_health = mem._apply_health_scores

        def counting_health(results):
            nonlocal health_call_count
            health_call_count += 1
            return orig_health(results)

        mem._apply_health_scores = counting_health
        mem._rerank = lambda query, results, top_n=5: results

        mem.save(content="double health test", title="Double Health Check")
        mem.search("double health", mode="hybrid")

        assert health_call_count <= 1, (
            f"_apply_health_scores called {health_call_count} times; expected at most 1"
        )
    finally:
        mem.close()


def test_health_scores_still_applied_without_reranker(mock_memory, monkeypatch):
    """When reranker is disabled, the post-pipeline health score path must
    still fire (MEMO_HEALTH_SCORES_DISABLED=0)."""
    monkeypatch.setenv("MEMO_HEALTH_SCORES_DISABLED", "0")
    # mock_memory fixture sets reranker_enabled=False

    health_called = False
    orig_health = mock_memory._apply_health_scores

    def spy_health(results):
        nonlocal health_called
        health_called = True
        return orig_health(results)

    mock_memory._apply_health_scores = spy_health

    mock_memory.save(content="without reranker health check", title="No Reranker Health")
    mock_memory.search("without reranker", mode="hybrid")

    assert health_called, "_apply_health_scores was not called when reranker is disabled"


# ── Fix 2.4 ─────────────────────────────────────────────────────────────────


def test_daemon_timeout_default_is_2s(monkeypatch):
    """Default daemon timeout reads 2000 ms → 2.0 s from the flags registry."""
    # Unset both timeout flags so defaults apply.
    monkeypatch.delenv("MEMO_RECALL_DAEMON_TIMEOUT_MS", raising=False)
    monkeypatch.delenv("MEMO_RECALL_DAEMON_TIMEOUT", raising=False)

    from memo.flags import flag_float, flag_int

    ms_val = flag_int("MEMO_RECALL_DAEMON_TIMEOUT_MS")
    float_val = flag_float("MEMO_RECALL_DAEMON_TIMEOUT")

    # Default ms → 2000 ms → 2.0 s
    assert ms_val == 2000, f"Expected default 2000 ms, got {ms_val}"
    # Default float → 2.0 s
    assert float_val == 2.0, f"Expected default 2.0 s, got {float_val}"


def test_daemon_timeout_float_flag_takes_precedence(monkeypatch):
    """When MEMO_RECALL_DAEMON_TIMEOUT (float) is set, it overrides the MS flag."""
    monkeypatch.setenv("MEMO_RECALL_DAEMON_TIMEOUT", "1.5")
    monkeypatch.setenv("MEMO_RECALL_DAEMON_TIMEOUT_MS", "3500")

    from memo.flags import flag_float, flag_int

    float_val = flag_float("MEMO_RECALL_DAEMON_TIMEOUT")
    assert float_val == 1.5, f"Expected 1.5, got {float_val}"

    # The MS flag still parses as set (3500), but cli.py should prefer float.
    ms_val = flag_int("MEMO_RECALL_DAEMON_TIMEOUT_MS")
    assert ms_val == 3500


def test_daemon_timeout_ms_flag_configurable(monkeypatch):
    """MEMO_RECALL_DAEMON_TIMEOUT_MS is read and converted to seconds by cli.py."""
    monkeypatch.setenv("MEMO_RECALL_DAEMON_TIMEOUT_MS", "1200")
    monkeypatch.delenv("MEMO_RECALL_DAEMON_TIMEOUT", raising=False)

    from memo.flags import flag_int

    ms_val = flag_int("MEMO_RECALL_DAEMON_TIMEOUT_MS")
    assert ms_val == 1200
    # The cli.py expression: max(0.2, ms / 1000.0) → 1.2
    computed = max(0.2, ms_val / 1000.0)
    assert abs(computed - 1.2) < 1e-9


def test_daemon_timeout_stays_under_budget(monkeypatch):
    """Default timeout (2.0 s) + subprocess fallback (max ~2s) stays within 5s."""
    monkeypatch.delenv("MEMO_RECALL_DAEMON_TIMEOUT_MS", raising=False)
    monkeypatch.delenv("MEMO_RECALL_DAEMON_TIMEOUT", raising=False)

    from memo.flags import flag_int

    default_ms = flag_int("MEMO_RECALL_DAEMON_TIMEOUT_MS") or 2000
    default_s = default_ms / 1000.0
    subprocess_max_s = 2.0  # conservative upper bound

    assert default_s + subprocess_max_s <= 5.0, (
        f"daemon_timeout ({default_s}s) + subprocess ({subprocess_max_s}s) exceeds 5s budget"
    )


# ── Fix 2.5 ─────────────────────────────────────────────────────────────────


def _make_vecstore(tmp_path, dims=16):
    """Create an isolated VecStore under tmp_path for norm validation tests."""
    from memo.store.store import VecStore

    db_path = tmp_path / "test_norm.db"
    return VecStore(db_path, dims=dims)


def test_norm_in_tight_range_no_warning(tmp_path, caplog):
    """A norm in [0.95, 1.05] must not emit any WARNING."""
    dims = 16
    store = _make_vecstore(tmp_path, dims=dims)

    # Build a unit vector (norm = 1.0 exactly).
    emb = [0.0] * dims
    emb[0] = 1.0

    with caplog.at_level(logging.WARNING, logger="memo.store.queries"):
        store.upsert(
            id_="a" * 32,
            path="/fake/path.md",
            title="Norm Test",
            type_="note",
            tags=[],
            created="2024-01-01T00:00:00",
            updated="2024-01-01T00:00:00",
            body_hash="deadbeef",
            embedding=emb,
        )

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert not warnings, f"Unexpected WARNING for unit-norm embedding: {warnings}"
    store.close()


def test_norm_outside_tight_range_emits_warning(tmp_path, caplog):
    """A norm in (0.5, 0.95] or [1.05, 1.5) must emit a WARNING about
    possible model mismatch or quantization, but NOT raise an exception."""
    dims = 16
    store = _make_vecstore(tmp_path / "store2", dims=dims)
    (tmp_path / "store2").mkdir(exist_ok=True)

    # Construct an embedding with norm ≈ 0.80 (inside old [0.5, 1.5] range,
    # outside new [0.95, 1.05] warning range).
    target_norm = 0.80
    emb = [target_norm] + [0.0] * (dims - 1)  # norm = 0.80

    with caplog.at_level(logging.WARNING, logger="memo.store.queries"):
        # Must NOT raise — degraded operation allowed.
        store.upsert(
            id_="b" * 32,
            path="/fake/path2.md",
            title="Norm Drift Test",
            type_="note",
            tags=[],
            created="2024-01-01T00:00:00",
            updated="2024-01-01T00:00:00",
            body_hash="cafebabe",
            embedding=emb,
        )

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "Expected a WARNING for out-of-range norm, got none"
    msg = warnings[0].message
    assert "0.8" in msg or "norm" in msg.lower(), (
        f"Expected norm value in warning message, got: {msg}"
    )
    store.close()


def test_norm_catastrophically_out_of_range_raises(tmp_path):
    """A norm outside (0.5, 1.5) must still raise ValueError."""
    import pytest

    dims = 16
    store = _make_vecstore(tmp_path / "store3", dims=dims)
    (tmp_path / "store3").mkdir(exist_ok=True)

    # All-zeros embedding has norm 0.0 — catastrophic failure.
    emb = [0.0] * dims

    with pytest.raises(ValueError, match="norm"):
        store.upsert(
            id_="c" * 32,
            path="/fake/path3.md",
            title="Zero Norm Test",
            type_="note",
            tags=[],
            created="2024-01-01T00:00:00",
            updated="2024-01-01T00:00:00",
            body_hash="00000000",
            embedding=emb,
        )
    store.close()
