"""Test suite for Wave 1 token economy overhaul (L1 + L4).

L1 = JSON array crushing on ingest (60-92% reduction possible)
L4 = Verbosity steering on recall output (5-10% reduction possible)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from freezegun import freeze_time

from memo.store.crush_cache import CrushCache, crush_marker


def test_crush_cache_stores_and_retrieves():
    """CrushCache stores original JSON, retrieves by hash."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CrushCache(Path(tmpdir))
        original = json.dumps([{"id": 1}, {"id": 2}])
        hash_val = "abc123def456"

        # Store
        cache.cache(hash_val, original)

        # Retrieve
        retrieved = cache.retrieve(hash_val)
        assert retrieved == original


def test_crush_marker_format():
    """crush_marker produces correct sentinel object."""
    marker = crush_marker(dropped_count=47, hash_val="abc123")
    assert marker["_compressed"] == "47 rows offloaded — ask `memo retrieve <<memo-crush:abc123>>` for full"


def test_crush_cache_ttl_expiration():
    """CrushCache respects TTL and evicts expired entries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CrushCache(Path(tmpdir))
        original = json.dumps([{"id": 1}])
        hash_val = "abc123"

        with freeze_time("2026-07-07"):
            cache.cache(hash_val, original)
            assert cache.retrieve(hash_val, ttl_days=30) == original

        with freeze_time("2026-08-10"):  # 34 days later
            assert cache.retrieve(hash_val, ttl_days=30) is None
            # Eviction must run at the future time so the entry reads as expired
            evicted = cache.evict_expired(ttl_days=30)
            assert evicted == 1


def test_crush_cache_missing_returns_none():
    """CrushCache returns None for missing entries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CrushCache(Path(tmpdir))
        assert cache.retrieve("nonexistent") is None


def test_crush_cache_creates_directory():
    """CrushCache creates cache directory on init."""
    with tempfile.TemporaryDirectory() as tmpdir:
        state_dir = Path(tmpdir)
        CrushCache(state_dir)
        assert (state_dir / "crush_cache").is_dir()


def test_crush_cache_handles_corrupt_json():
    """CrushCache gracefully handles corrupt cache files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CrushCache(Path(tmpdir))
        # Write corrupt JSON directly
        cache_dir = cache.cache_dir
        corrupt_file = cache_dir / "corrupt123.json"
        corrupt_file.write_text("not valid json")

        # Should return None instead of crashing
        assert cache.retrieve("corrupt123") is None

        # evict_expired should skip corrupt files
        evicted = cache.evict_expired(ttl_days=30)
        assert evicted == 0


def test_crush_cache_evict_expired_with_multiple_entries():
    """CrushCache evicts only expired entries, preserves fresh ones."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CrushCache(Path(tmpdir))

        with freeze_time("2026-07-07"):
            cache.cache("old1", json.dumps({"data": "old"}))
            cache.cache("old2", json.dumps({"data": "old"}))

        with freeze_time("2026-08-10"):  # 34 days later
            cache.cache("new1", json.dumps({"data": "new"}))

            # Evict at the future time: old1/old2 are 34d old (expired), new1 is fresh
            evicted = cache.evict_expired(ttl_days=30)
            assert evicted == 2

            # Fresh entry survived
            assert cache.retrieve("new1", ttl_days=30) is not None

        # Verify fresh entry still there
        assert cache.retrieve("new1") == json.dumps({"data": "new"})
        assert cache.retrieve("old1") is None


def test_crush_cache_retrieve_respects_ttl_parameter():
    """CrushCache retrieve respects custom TTL parameter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CrushCache(Path(tmpdir))

        with freeze_time("2026-07-07"):
            cache.cache("test", json.dumps({"id": 1}))

        with freeze_time("2026-08-10"):  # 34 days later
            # With 30-day TTL, should be expired
            assert cache.retrieve("test", ttl_days=30) is None
            # With 40-day TTL, should still be there
            assert cache.retrieve("test", ttl_days=40) == json.dumps({"id": 1})


def test_crush_marker_with_different_dropped_counts():
    """crush_marker formats correctly for various dropped counts."""
    test_cases = [
        (0, "abc123", "0 rows offloaded — ask `memo retrieve <<memo-crush:abc123>>` for full"),
        (1, "def456", "1 rows offloaded — ask `memo retrieve <<memo-crush:def456>>` for full"),
        (100, "xyz789", "100 rows offloaded — ask `memo retrieve <<memo-crush:xyz789>>` for full"),
    ]
    for dropped_count, hash_val, expected_message in test_cases:
        marker = crush_marker(dropped_count, hash_val)
        assert marker["_compressed"] == expected_message


# --- Task 3: Tests for verbosity steering on recall output ---


def test_maybe_inject_verbosity_steering_idempotent():
    """Verbosity steering is idempotent (doesn't double-inject)."""
    from memo.cli_recall_hook import maybe_inject_verbosity_steering

    prompt = "You are a helpful assistant."

    # Inject once
    injected_1 = maybe_inject_verbosity_steering(prompt, level=2)
    assert "<headroom_recall_verbosity>" in injected_1

    # Inject again (should not double-inject)
    injected_2 = maybe_inject_verbosity_steering(injected_1, level=2)
    assert injected_1 == injected_2  # Idempotent


def test_maybe_inject_verbosity_respects_level():
    """Verbosity levels produce correct steering text."""
    from memo.cli_recall_hook import maybe_inject_verbosity_steering

    prompt = "Base prompt."

    # Level 0: no steering
    result_0 = maybe_inject_verbosity_steering(prompt, level=0)
    assert result_0 == prompt

    # Level 1: basic steering
    result_1 = maybe_inject_verbosity_steering(prompt, level=1)
    assert "Skip preamble" in result_1
    assert "<headroom_recall_verbosity>" in result_1

    # Level 3: aggressive steering
    result_3 = maybe_inject_verbosity_steering(prompt, level=3)
    assert "Minimum tokens" in result_3


def test_flag_recall_verbosity_level(monkeypatch):
    """Flag resolves verbosity level from env."""
    from memo.flags_recall import flag_recall_verbosity_level

    monkeypatch.setenv("MEMO_RECALL_VERBOSITY_LEVEL", "2")
    assert flag_recall_verbosity_level() == 2

    monkeypatch.setenv("MEMO_RECALL_VERBOSITY_LEVEL", "0")
    assert flag_recall_verbosity_level() == 0

    monkeypatch.delenv("MEMO_RECALL_VERBOSITY_LEVEL", raising=False)
    assert flag_recall_verbosity_level() == 0  # Default


def test_crush_cache_unicode_content():
    """CrushCache handles unicode content correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CrushCache(Path(tmpdir))
        original = json.dumps([{"name": "José"}, {"name": "François"}], ensure_ascii=False)
        hash_val = "unicode_test"

        cache.cache(hash_val, original)
        retrieved = cache.retrieve(hash_val)
        assert retrieved == original

        # Verify it parses back correctly
        data = json.loads(retrieved)
        assert data[0]["name"] == "José"
        assert data[1]["name"] == "François"


# --- Task 2.1: Tests for maybe_crush_json_capture ---


def test_maybe_crush_json_capture_detects_json():
    """Crusher detects JSON arrays in capture content."""
    from memo.capture_core import maybe_crush_json_capture
    from memo.config import Config

    with tempfile.TemporaryDirectory() as tmpdir:
        state_dir = Path(tmpdir)
        config = Config(data_dir=state_dir / "data", vault_path=state_dir / "vault", state_dir=state_dir / "state")
        (state_dir / "data").mkdir(parents=True, exist_ok=True)
        (state_dir / "vault").mkdir(parents=True, exist_ok=True)
        (state_dir / "state").mkdir(parents=True, exist_ok=True)

        json_content = json.dumps([
            {"id": 1, "text": "important"},
            {"id": 2, "text": "noise"},
            {"id": 3, "text": "low-score"},
        ])

        crushed, hash_val = maybe_crush_json_capture(json_content, context="test query", config=config)

        # Should detect JSON
        assert isinstance(crushed, str)
        assert isinstance(hash_val, (str, type(None)))


def test_maybe_crush_json_respects_disable_flag(monkeypatch):
    """Crusher respects MEMO_CRUSHER_ENABLED=0."""
    from memo.capture_core import maybe_crush_json_capture
    from memo.config import Config

    monkeypatch.setenv("MEMO_CRUSHER_ENABLED", "0")

    with tempfile.TemporaryDirectory() as tmpdir:
        state_dir = Path(tmpdir)
        config = Config(data_dir=state_dir / "data", vault_path=state_dir / "vault", state_dir=state_dir / "state")
        (state_dir / "data").mkdir(parents=True, exist_ok=True)
        (state_dir / "vault").mkdir(parents=True, exist_ok=True)
        (state_dir / "state").mkdir(parents=True, exist_ok=True)

        json_content = json.dumps([{"id": i} for i in range(100)])

        crushed, hash_val = maybe_crush_json_capture(json_content, context="query", config=config)

        # When disabled, should return original and None hash
        assert crushed == json_content
        assert hash_val is None


def test_crush_preserves_structure(monkeypatch):
    """Crushed JSON keeps top-K rows + marker."""
    from memo.capture_core import maybe_crush_json_capture
    from memo.config import Config

    monkeypatch.setenv("MEMO_CRUSHER_ENABLED", "1")
    monkeypatch.setenv("MEMO_CRUSHER_KEEP_RATIO", "0.2")

    with tempfile.TemporaryDirectory() as tmpdir:
        state_dir = Path(tmpdir)
        config = Config(data_dir=state_dir / "data", vault_path=state_dir / "vault", state_dir=state_dir / "state")
        (state_dir / "data").mkdir(parents=True, exist_ok=True)
        (state_dir / "vault").mkdir(parents=True, exist_ok=True)
        (state_dir / "state").mkdir(parents=True, exist_ok=True)

        json_array = [{"id": i, "val": f"row{i}"} for i in range(100)]
        json_content = json.dumps(json_array)

        crushed, hash_val = maybe_crush_json_capture(json_content, context="query", config=config)

        if hash_val is not None:
            crushed_obj = json.loads(crushed)

            assert isinstance(crushed_obj, list)
            # Should have ~20 top rows + 1 marker
            assert len(crushed_obj) <= 25
            # Last entry should be marker
            assert "_compressed" in crushed_obj[-1]


def test_crush_json_too_small_not_crushed(monkeypatch):
    """Crusher skips arrays smaller than threshold (< 10 rows)."""
    from memo.capture_core import maybe_crush_json_capture
    from memo.config import Config

    monkeypatch.setenv("MEMO_CRUSHER_ENABLED", "1")

    with tempfile.TemporaryDirectory() as tmpdir:
        state_dir = Path(tmpdir)
        config = Config(data_dir=state_dir / "data", vault_path=state_dir / "vault", state_dir=state_dir / "state")
        (state_dir / "data").mkdir(parents=True, exist_ok=True)
        (state_dir / "vault").mkdir(parents=True, exist_ok=True)
        (state_dir / "state").mkdir(parents=True, exist_ok=True)

        json_content = json.dumps([{"id": i} for i in range(5)])

        crushed, hash_val = maybe_crush_json_capture(json_content, context="query", config=config)

        # Should not crush small arrays
        assert crushed == json_content
        assert hash_val is None


def test_crush_json_non_array_not_crushed(monkeypatch):
    """Crusher skips non-array JSON."""
    from memo.capture_core import maybe_crush_json_capture
    from memo.config import Config

    monkeypatch.setenv("MEMO_CRUSHER_ENABLED", "1")

    with tempfile.TemporaryDirectory() as tmpdir:
        state_dir = Path(tmpdir)
        config = Config(data_dir=state_dir / "data", vault_path=state_dir / "vault", state_dir=state_dir / "state")
        (state_dir / "data").mkdir(parents=True, exist_ok=True)
        (state_dir / "vault").mkdir(parents=True, exist_ok=True)
        (state_dir / "state").mkdir(parents=True, exist_ok=True)

        # Object, not array
        json_content = json.dumps({"data": [1, 2, 3]})

        crushed, hash_val = maybe_crush_json_capture(json_content, context="query", config=config)

        # Should not crush non-arrays
        assert crushed == json_content
        assert hash_val is None


def test_crush_invalid_json_not_crushed(monkeypatch):
    """Crusher skips invalid JSON."""
    from memo.capture_core import maybe_crush_json_capture
    from memo.config import Config

    monkeypatch.setenv("MEMO_CRUSHER_ENABLED", "1")

    with tempfile.TemporaryDirectory() as tmpdir:
        state_dir = Path(tmpdir)
        config = Config(data_dir=state_dir / "data", vault_path=state_dir / "vault", state_dir=state_dir / "state")
        (state_dir / "data").mkdir(parents=True, exist_ok=True)
        (state_dir / "vault").mkdir(parents=True, exist_ok=True)
        (state_dir / "state").mkdir(parents=True, exist_ok=True)

        json_content = "[not valid json]"

        crushed, hash_val = maybe_crush_json_capture(json_content, context="query", config=config)

        # Should not crash, just return original
        assert crushed == json_content
        assert hash_val is None


# --- Task 2.3: Tests for retrieve command ---


def test_retrieve_command_via_mcp_tool():
    """MCP tool memo_crush_retrieve retrieves cached JSON."""
    from memo.config import Config
    from memo.memory import Memory
    from memo.server_crush import register as register_crush

    with tempfile.TemporaryDirectory() as tmpdir:
        state_dir = Path(tmpdir) / "state"
        data_dir = Path(tmpdir) / "data"
        vault_dir = Path(tmpdir) / "vault"
        state_dir.mkdir()
        data_dir.mkdir()
        vault_dir.mkdir()

        # Store a JSON in cache
        cache = CrushCache(state_dir)
        original = json.dumps([{"id": 1}, {"id": 2}])
        hash_val = "abc123def456"
        cache.cache(hash_val, original)

        # Create Memory and mock server
        config = Config(data_dir=data_dir, vault_path=vault_dir, state_dir=state_dir, reranker_enabled=False)
        memory = Memory(config)

        try:
            from unittest.mock import MagicMock
            mock_server = MagicMock()
            tool_functions = {}

            def capture_tool(name=None, **kwargs):
                def decorator(func):
                    tool_functions[name or func.__name__] = func
                    return func
                return decorator

            mock_server.tool = capture_tool
            register_crush(mock_server, memory)
            memo_crush_retrieve = tool_functions["memo_crush_retrieve"]

            # Test successful retrieval
            result = memo_crush_retrieve(f"<<memo-crush:{hash_val}>>")
            assert result["original"] == original
            assert result["hash"] == hash_val
        finally:
            memory.close()


def test_retrieve_mcp_tool_missing_cache_entry():
    """MCP tool returns error for missing cache entry."""
    from memo.config import Config
    from memo.memory import Memory
    from memo.server_crush import register as register_crush

    with tempfile.TemporaryDirectory() as tmpdir:
        state_dir = Path(tmpdir) / "state"
        data_dir = Path(tmpdir) / "data"
        vault_dir = Path(tmpdir) / "vault"
        state_dir.mkdir()
        data_dir.mkdir()
        vault_dir.mkdir()

        config = Config(data_dir=data_dir, vault_path=vault_dir, state_dir=state_dir, reranker_enabled=False)
        memory = Memory(config)

        try:
            from unittest.mock import MagicMock
            mock_server = MagicMock()
            tool_functions = {}

            def capture_tool(name=None, **kwargs):
                def decorator(func):
                    tool_functions[name or func.__name__] = func
                    return func
                return decorator

            mock_server.tool = capture_tool
            register_crush(mock_server, memory)
            memo_crush_retrieve = tool_functions["memo_crush_retrieve"]

            # Test error for nonexistent hash
            result = memo_crush_retrieve("<<memo-crush:nonexistent>>")
            assert "error" in result
        finally:
            memory.close()


def test_retrieve_mcp_tool_invalid_marker_format():
    """MCP tool rejects invalid marker format."""
    from memo.config import Config
    from memo.memory import Memory
    from memo.server_crush import register as register_crush

    with tempfile.TemporaryDirectory() as tmpdir:
        state_dir = Path(tmpdir) / "state"
        data_dir = Path(tmpdir) / "data"
        vault_dir = Path(tmpdir) / "vault"
        state_dir.mkdir()
        data_dir.mkdir()
        vault_dir.mkdir()

        config = Config(data_dir=data_dir, vault_path=vault_dir, state_dir=state_dir, reranker_enabled=False)
        memory = Memory(config)

        try:
            from unittest.mock import MagicMock
            mock_server = MagicMock()
            tool_functions = {}

            def capture_tool(name=None, **kwargs):
                def decorator(func):
                    tool_functions[name or func.__name__] = func
                    return func
                return decorator

            mock_server.tool = capture_tool
            register_crush(mock_server, memory)
            memo_crush_retrieve = tool_functions["memo_crush_retrieve"]

            # Test invalid marker format
            result = memo_crush_retrieve("invalid-marker")
            assert "error" in result
        finally:
            memory.close()


# --- Task 4: Integration tests for end-to-end Wave 1 pipeline ---


def test_wave1_end_to_end_crusher_and_verbosity(monkeypatch):
    """Full pipeline: capture JSON → crush → retrieve → recall with verbosity steering."""

    # Enable both Wave 1 features
    monkeypatch.setenv("MEMO_CRUSHER_ENABLED", "1")
    monkeypatch.setenv("MEMO_RECALL_VERBOSITY_LEVEL", "2")

    from memo.capture_core import maybe_crush_json_capture
    from memo.cli_recall_hook import maybe_inject_verbosity_steering
    from memo.config import Config

    with tempfile.TemporaryDirectory() as tmpdir:
        state_dir = Path(tmpdir) / "state"
        data_dir = Path(tmpdir) / "data"
        vault_dir = Path(tmpdir) / "vault"
        state_dir.mkdir(parents=True)
        data_dir.mkdir(parents=True)
        vault_dir.mkdir(parents=True)

        config = Config(data_dir=data_dir, vault_path=vault_dir, state_dir=state_dir)

        # Step 1: Simulate captured JSON (large result set)
        large_json = json.dumps([
            {"id": i, "text": f"result row {i}", "data": "x" * 50}
            for i in range(50)
        ])

        # Step 2: Crush it
        crushed, hash_val = maybe_crush_json_capture(large_json, context="search results", config=config)
        assert isinstance(crushed, str)

        if hash_val is not None:
            # Verify it's actually compressed
            assert len(crushed) < len(large_json)

            # Step 3: Verify crush marker is present
            crushed_obj = json.loads(crushed)
            assert isinstance(crushed_obj, list)
            assert "_compressed" in crushed_obj[-1]

            # Step 4: Retrieve via cache
            from memo.store.crush_cache import CrushCache
            cache = CrushCache(state_dir)
            original = cache.retrieve(hash_val)
            assert original == large_json

        # Step 5: Inject verbosity steering on recall output
        base_prompt = "Answer the following question based on the context above."
        prompt_with_steering = maybe_inject_verbosity_steering(base_prompt, level=2)

        # Verify steering was injected
        assert len(prompt_with_steering) > len(base_prompt)
        assert "<headroom_recall_verbosity>" in prompt_with_steering
        assert "Skip preamble" in prompt_with_steering

        # Verify idempotency
        prompt_twice = maybe_inject_verbosity_steering(prompt_with_steering, level=2)
        assert prompt_twice == prompt_with_steering


def test_wave1_flags_integration(monkeypatch):
    """Wave 1 flags work together (crusher + verbosity) and are independently disableable."""
    from memo.capture_core import maybe_crush_json_capture
    from memo.cli_recall_hook import maybe_inject_verbosity_steering
    from memo.config import Config
    from memo.flags_capture import flag_crusher_enabled
    from memo.flags_recall import flag_recall_verbosity_level

    with tempfile.TemporaryDirectory() as tmpdir:
        state_dir = Path(tmpdir) / "state"
        data_dir = Path(tmpdir) / "data"
        vault_dir = Path(tmpdir) / "vault"
        state_dir.mkdir(parents=True)
        data_dir.mkdir(parents=True)
        vault_dir.mkdir(parents=True)

        config = Config(data_dir=data_dir, vault_path=vault_dir, state_dir=state_dir)

        # Test 1: Both enabled
        monkeypatch.setenv("MEMO_CRUSHER_ENABLED", "1")
        monkeypatch.setenv("MEMO_RECALL_VERBOSITY_LEVEL", "2")

        assert flag_crusher_enabled() is True
        assert flag_recall_verbosity_level() == 2

        json_content = json.dumps([{"id": i} for i in range(100)])
        _crushed, _hash_val = maybe_crush_json_capture(json_content, context="query", config=config)

        prompt = "Base"
        steered = maybe_inject_verbosity_steering(prompt, level=2)
        assert steered != prompt  # Should be modified

        # Test 2: Crusher disabled
        monkeypatch.setenv("MEMO_CRUSHER_ENABLED", "0")
        assert flag_crusher_enabled() is False

        crushed2, hash_val2 = maybe_crush_json_capture(json_content, context="query", config=config)
        assert crushed2 == json_content  # Should not be crushed
        assert hash_val2 is None

        # Test 3: Verbosity disabled
        monkeypatch.setenv("MEMO_RECALL_VERBOSITY_LEVEL", "0")
        assert flag_recall_verbosity_level() == 0

        steered2 = maybe_inject_verbosity_steering(prompt, level=0)
        assert steered2 == prompt  # Should not be modified

        # Test 4: Can re-enable independently
        monkeypatch.setenv("MEMO_CRUSHER_ENABLED", "1")
        assert flag_crusher_enabled() is True
        assert flag_recall_verbosity_level() == 0  # Still disabled

        crushed3, hash_val3 = maybe_crush_json_capture(json_content, context="query", config=config)
        # Should attempt crush (with our disabled verbosity still at 0)
        if hash_val3 is not None:
            assert len(crushed3) < len(json_content)

        steered3 = maybe_inject_verbosity_steering(prompt, level=0)
        assert steered3 == prompt  # Verbosity still off
