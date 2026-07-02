"""Tests for server_analytics MCP tool registration."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from memo.analytics import CorpusMetrics, GrowthData


def _make_server_and_tools():
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


def _stub_corpus_metrics() -> CorpusMetrics:
    return CorpusMetrics(
        total_memories=42,
        total_entities=10,
        type_distribution={"fact": 20, "decision": 22},
        tag_frequency={"python": 5, "memo": 3},
        entity_frequency={"memo": 10, "mlx": 4},
        growth_rate=1.4,
        average_access_count=2.5,
    )


def _stub_growth_data() -> GrowthData:
    return GrowthData(
        dates=["2026-06-29", "2026-06-30", "2026-07-01"],
        counts=[3, 5, 2],
    )


def test_register_exposes_exactly_two_tools(tmp_cfg) -> None:
    """register() must expose exactly memo_analytics_summary and memo_analytics_growth."""
    from memo.memory import Memory
    from memo.server_analytics import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    assert {"memo_analytics_summary", "memo_analytics_growth"} == set(tools), (
        f"Unexpected tools registered: {set(tools)}"
    )


def test_memo_analytics_summary_calls_compute_corpus_metrics(tmp_cfg) -> None:
    """memo_analytics_summary must delegate to memory.analytics.compute_corpus_metrics."""
    from memo.memory import Memory
    from memo.server_analytics import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.analytics.compute_corpus_metrics.return_value = _stub_corpus_metrics()

    server, tools = _make_server_and_tools()
    register(server, mem)

    assert "memo_analytics_summary" in tools

    result = tools["memo_analytics_summary"]()

    mem.analytics.compute_corpus_metrics.assert_called_once_with()
    assert isinstance(result, dict)
    assert result["total_memories"] == 42
    assert result["total_entities"] == 10
    assert result["growth_rate"] == pytest.approx(1.4)
    assert result["average_access_count"] == pytest.approx(2.5)
    assert result["type_distribution"] == {"fact": 20, "decision": 22}


def test_memo_analytics_summary_returns_full_dict_envelope(tmp_cfg) -> None:
    """The returned dict must include all CorpusMetrics fields."""
    from memo.memory import Memory
    from memo.server_analytics import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.analytics.compute_corpus_metrics.return_value = _stub_corpus_metrics()

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_analytics_summary"]()

    expected_keys = {
        "total_memories",
        "total_entities",
        "type_distribution",
        "tag_frequency",
        "entity_frequency",
        "growth_rate",
        "average_access_count",
    }
    assert expected_keys == set(result.keys()), f"Missing/extra keys: {set(result.keys())}"


def test_memo_analytics_growth_default_days(tmp_cfg) -> None:
    """memo_analytics_growth with no args must call compute_growth_data(days=30)."""
    from memo.memory import Memory
    from memo.server_analytics import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.analytics.compute_growth_data.return_value = _stub_growth_data()

    server, tools = _make_server_and_tools()
    register(server, mem)

    assert "memo_analytics_growth" in tools

    result = tools["memo_analytics_growth"]()

    mem.analytics.compute_growth_data.assert_called_once_with(days=30)
    assert isinstance(result, dict)
    assert result["dates"] == ["2026-06-29", "2026-06-30", "2026-07-01"]
    assert result["counts"] == [3, 5, 2]


def test_memo_analytics_growth_custom_days(tmp_cfg) -> None:
    """memo_analytics_growth must forward the days argument to compute_growth_data."""
    from memo.memory import Memory
    from memo.server_analytics import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.analytics.compute_growth_data.return_value = GrowthData(
        dates=["2026-01-01"],
        counts=[1],
    )

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_analytics_growth"](days=7)

    mem.analytics.compute_growth_data.assert_called_once_with(days=7)
    assert isinstance(result, dict)
    assert "dates" in result
    assert "counts" in result
    assert len(result["dates"]) == len(result["counts"])


def test_no_module_level_mlx_imports() -> None:
    """server_analytics must not have module-level MLX imports (deferred-import invariant)."""
    import ast
    import pathlib

    src = pathlib.Path(__file__).parent.parent / "src" / "memo" / "server_analytics.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))

    violations: list[str] = []
    for node in tree.body:  # top-level only
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("mlx"):
                    violations.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("mlx"):
            violations.append(f"line {node.lineno}: from {node.module} import ...")

    assert not violations, f"Module-level MLX imports found: {violations}"
