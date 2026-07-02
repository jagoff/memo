"""Tests for CachePolicy + CacheManager no-op behavior (memo/cache.py).

CachePolicy.from_env takes an explicit `env` dict, so these are hermetic — no
monkeypatch / real environment needed.
"""

from __future__ import annotations

from memo.cache import CacheManager, CachePolicy


def test_default_policy_is_off() -> None:
    p = CachePolicy.from_env(env={})
    assert p.mode == "off"
    assert p.enabled is False
    assert p.read_through is False
    assert p.write_through is False
    assert p.write_back is False
    assert p.max_entries == 0


def test_write_through_mode() -> None:
    p = CachePolicy.from_env(env={"MEMO_CACHE_MODE": "write_through"})
    assert p.enabled is True
    assert p.write_through is True
    assert p.write_back is False
    assert p.read_through is True  # every enabled mode reads through


def test_write_back_mode() -> None:
    p = CachePolicy.from_env(env={"MEMO_CACHE_MODE": "write_back"})
    assert p.write_back is True
    assert p.write_through is False
    assert p.enabled is True


def test_invalid_mode_falls_back_to_off() -> None:
    p = CachePolicy.from_env(env={"MEMO_CACHE_MODE": "bogus"})
    assert p.mode == "off"


def test_invalid_eviction_falls_back_to_lru() -> None:
    p = CachePolicy.from_env(
        env={"MEMO_CACHE_MODE": "read_through", "MEMO_CACHE_EVICTION": "bogus"}
    )
    assert p.eviction == "lru"


def test_numeric_knobs_parsed_and_floored() -> None:
    p = CachePolicy.from_env(
        env={
            "MEMO_CACHE_MODE": "read_through",
            "MEMO_CACHE_MAX_ENTRIES": "500",
            "MEMO_CACHE_TTL_DAYS": "30",
            "MEMO_CACHE_EVICTION": "ttl",
            "MEMO_CACHE_BACKEND": "vault",
        }
    )
    assert p.max_entries == 500
    assert p.ttl_days == 30
    assert p.eviction == "ttl"
    assert p.backend == "vault"


def test_manager_noops_when_disabled() -> None:
    mgr = CacheManager(None, policy=CachePolicy.from_env(env={}))
    # Disabled + unbounded: eviction short-circuits before touching memory.
    assert mgr.evict_if_needed() == []
    assert mgr.policy.enabled is False
