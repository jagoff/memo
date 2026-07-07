"""Test suite for Wave 1 token economy overhaul (L1 + L4).

L1 = JSON array crushing on ingest (60-92% reduction possible)
L4 = Verbosity steering on recall output (5-10% reduction possible)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from freezegun import freeze_time

from src.memo.store.crush_cache import CrushCache, crush_marker


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
        cache = CrushCache(state_dir)
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

        # Evict with 30-day TTL
        evicted = cache.evict_expired(ttl_days=30)
        assert evicted == 2

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
