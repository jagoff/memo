"""Tests for server_cache MCP tool registration.

Mirrors the pattern from test_server_idle_capture.py: a fake server captures
registered callables by __name__; assertions run directly on the returned
Python objects without going through FastMCP/JSON-RPC transport.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from memo.memory import Memory


def _make_server_and_tools() -> tuple[MagicMock, dict]:
    """Return a (server_mock, tools_dict) pair.

    `server.tool()` is wired so each `@server.tool()` decorated function is
    captured in `tools` by its `__name__`, without going through FastMCP.
    """
    server = MagicMock()
    tools: dict = {}

    def tool_decorator():
        def wrapper(fn):
            tools[fn.__name__] = fn
            return fn

        return wrapper

    server.tool = tool_decorator
    return server, tools


def test_register_exposes_all_three_tools(tmp_cfg) -> None:
    """register() must expose exactly the three expected cache MCP tools."""
    from memo.server_cache import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    expected = {"memo_cache_stats", "memo_cache_evict", "memo_cache_flush"}
    assert expected == set(tools), f"Tool mismatch: {set(tools)}"


def test_memo_cache_stats_delegates_to_cache_stats(tmp_cfg) -> None:
    """memo_cache_stats must call memory.cache.stats() and return its result."""
    from memo.server_cache import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    fake_stats = {
        "mode": "off",
        "enabled": False,
        "backend": None,
        "eviction": "lru",
        "ttl_days": 0,
        "entries": 5,
        "max_entries": 0,
        "over_capacity": 0,
    }
    mem.cache.stats.return_value = fake_stats

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_cache_stats"]()

    mem.cache.stats.assert_called_once_with()
    assert result == fake_stats
    assert result["mode"] == "off"
    assert result["enabled"] is False
    assert result["entries"] == 5


def test_memo_cache_stats_enabled_mode(tmp_cfg) -> None:
    """memo_cache_stats returns stats correctly when cache is enabled."""
    from memo.server_cache import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    fake_stats = {
        "mode": "write_through",
        "enabled": True,
        "backend": "memflow",
        "eviction": "lru",
        "ttl_days": 0,
        "entries": 42,
        "max_entries": 100,
        "over_capacity": 0,
    }
    mem.cache.stats.return_value = fake_stats

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_cache_stats"]()

    assert result["enabled"] is True
    assert result["mode"] == "write_through"
    assert result["backend"] == "memflow"
    assert result["entries"] == 42


def test_memo_cache_evict_wraps_evicted_ids(tmp_cfg) -> None:
    """memo_cache_evict must call evict_if_needed() and return {evicted, count}."""
    from memo.server_cache import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    evicted_ids = ["abc123def", "xyz789uvw"]
    mem.cache.evict_if_needed.return_value = evicted_ids
    mem.cache.stats.return_value = {"enabled": True, "over_capacity": 0}

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = asyncio.run(tools["memo_cache_evict"]())

    mem.cache.evict_if_needed.assert_called_once_with()
    assert result["evicted"] == evicted_ids
    assert result["count"] == 2


def test_memo_cache_evict_empty_when_no_overflow(tmp_cfg) -> None:
    """memo_cache_evict returns count=0 and empty list when nothing was evicted."""
    from memo.server_cache import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    mem.cache.evict_if_needed.return_value = []
    mem.cache.stats.return_value = {"enabled": False, "over_capacity": 0}

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = asyncio.run(tools["memo_cache_evict"]())

    assert result["evicted"] == []
    assert result["count"] == 0


def test_memo_cache_flush_delegates_to_flush_all(tmp_cfg) -> None:
    """memo_cache_flush must call memory.cache.flush_all() and return its result."""
    from memo.server_cache import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    flush_result = {"flushed": 3, "failed": 0, "dirty_remaining": 0}
    mem.cache.flush_all.return_value = flush_result

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_cache_flush"]()

    mem.cache.flush_all.assert_called_once_with()
    assert result == flush_result
    assert result["flushed"] == 3
    assert result["failed"] == 0
    assert result["dirty_remaining"] == 0


def test_memo_cache_flush_noop_when_cache_disabled(tmp_cfg) -> None:
    """memo_cache_flush passes through the no-op result when cache mode is off."""
    from memo.server_cache import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    noop_result = {"flushed": 0, "failed": 0, "dirty_remaining": 0}
    mem.cache.flush_all.return_value = noop_result

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_cache_flush"]()

    mem.cache.flush_all.assert_called_once_with()
    assert result["flushed"] == 0
    assert result["failed"] == 0
    assert result["dirty_remaining"] == 0


def test_memo_cache_stats_omits_emit_ledger_with_the_flag_off(tmp_cfg, monkeypatch) -> None:
    """With MEMO_EMITTED_LEDGER unset (the default), memo_cache_stats' output
    must be byte-identical to before this feature existed -- no `emit_ledger`
    key, nothing written to disk."""
    from memo.server_cache import register

    monkeypatch.setenv("MEMO_EMITTED_LEDGER", "0")
    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    fake_stats = {"mode": "off", "enabled": False, "entries": 0}
    mem.cache.stats.return_value = fake_stats

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_cache_stats"]()

    assert result == fake_stats
    assert "emit_ledger" not in result
    assert not (tmp_cfg.state_dir / "emitted").exists()


def test_memo_cache_stats_surfaces_emit_ledger_when_flag_on(tmp_cfg, monkeypatch) -> None:
    from memo import emitted_ledger as el
    from memo.server_cache import register

    monkeypatch.setenv("MEMO_EMITTED_LEDGER", "1")
    monkeypatch.setenv("MEMO_SESSION_ID", "sess-cache-stats")
    el.bump(tmp_cfg.state_dir, "sess-cache-stats", digests_served=2, tokens_suppressed=40)

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    fake_stats = {"mode": "off", "enabled": False, "entries": 0}
    mem.cache.stats.return_value = fake_stats

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_cache_stats"]()

    # The base cache-tier keys are untouched, `emit_ledger` is additive.
    assert result["mode"] == "off"
    assert result["enabled"] is False
    assert result["emit_ledger"] == el.stats(tmp_cfg.state_dir, "sess-cache-stats")
    assert result["emit_ledger"]["digests_served"] == 2
    assert result["emit_ledger"]["tokens_suppressed"] == 40


def test_memo_cache_stats_emit_ledger_failure_falls_back_to_base_stats(
    tmp_cfg, monkeypatch
) -> None:
    """Fail-open: a broken emit_ledger read must not break memo_cache_stats
    itself -- the tool still returns the cache-tier stats it always did."""
    from memo.server_cache import register

    monkeypatch.setenv("MEMO_EMITTED_LEDGER", "1")
    monkeypatch.setenv("MEMO_SESSION_ID", "sess-cache-boom")
    monkeypatch.setattr(
        "memo.emitted_ledger.stats",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    fake_stats = {"mode": "off", "enabled": False, "entries": 0}
    mem.cache.stats.return_value = fake_stats

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_cache_stats"]()

    assert result == fake_stats
    assert "emit_ledger" not in result


def test_memo_cache_flush_partial_failure(tmp_cfg) -> None:
    """memo_cache_flush returns accurate failure counts when some pushes fail."""
    from memo.server_cache import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    partial_result = {"flushed": 2, "failed": 1, "dirty_remaining": 1}
    mem.cache.flush_all.return_value = partial_result

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_cache_flush"]()

    assert result["flushed"] == 2
    assert result["failed"] == 1
    assert result["dirty_remaining"] == 1
