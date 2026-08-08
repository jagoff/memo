"""The budget module: estimator, opt-in trimming, the error payload, and the
enforcement middleware."""

from __future__ import annotations

import pytest

from memo import mcp_budget
from memo.flags import REGISTRY


def test_est_tokens_matches_the_house_estimator() -> None:
    assert mcp_budget.est_tokens("") == 0
    assert mcp_budget.est_tokens("abcd") == 1
    assert mcp_budget.est_tokens("a" * 4000) == 1000


def test_default_cap_tokens_matches_the_registered_flag_default() -> None:
    # DEFAULT_CAP_TOKENS is derived from REGISTRY, not a second hard-coded
    # literal -- this pins that relationship so a future re-hardcoding drifts
    # loudly (a failing test) instead of silently.
    assert REGISTRY["MEMO_MCP_RESPONSE_BUDGET_TOKENS"].default == mcp_budget.DEFAULT_CAP_TOKENS


def test_bounded_list_passes_a_short_list_through_untouched() -> None:
    shown, meta = mcp_budget.bounded_list([1, 2, 3], cap=10)
    assert shown == [1, 2, 3]
    assert meta == {"shown": 3, "total": 3, "truncated": False}


def test_bounded_list_trims_and_reports_the_real_total() -> None:
    shown, meta = mcp_budget.bounded_list(list(range(100)), cap=5)
    assert shown == [0, 1, 2, 3, 4]
    assert meta == {"shown": 5, "total": 100, "truncated": True}


def test_bounded_list_keeps_the_best_by_key() -> None:
    items = [{"d": 9}, {"d": 1}, {"d": 5}]
    shown, meta = mcp_budget.bounded_list(items, cap=2, key=lambda x: x["d"])
    assert [x["d"] for x in shown] == [1, 5]
    assert meta["total"] == 3


def test_bounded_list_handles_empty_input() -> None:
    shown, meta = mcp_budget.bounded_list([], cap=5)
    assert shown == []
    assert meta == {"shown": 0, "total": 0, "truncated": False}


def test_bounded_list_cap_zero_keeps_nothing_but_reports_the_total() -> None:
    shown, meta = mcp_budget.bounded_list([1, 2, 3], cap=0)
    assert shown == []
    assert meta == {"shown": 0, "total": 3, "truncated": True}


def test_cap_for_prefers_the_per_tool_override(monkeypatch) -> None:
    monkeypatch.setitem(mcp_budget.CAPS, "memo_export_json", 500_000)
    assert mcp_budget.cap_for("memo_export_json") == 500_000
    assert mcp_budget.cap_for("memo_search") == mcp_budget.DEFAULT_CAP_TOKENS


def test_cap_for_honours_the_flag(monkeypatch) -> None:
    monkeypatch.setenv("MEMO_MCP_RESPONSE_BUDGET_TOKENS", "42")
    assert mcp_budget.cap_for("memo_search") == 42


def test_zero_cap_means_unlimited(monkeypatch) -> None:
    monkeypatch.setenv("MEMO_MCP_RESPONSE_BUDGET_TOKENS", "0")
    assert mcp_budget.cap_for("memo_search") == 0


def test_budget_exceeded_payload_names_the_tool_and_both_numbers() -> None:
    payload = mcp_budget.budget_exceeded_payload("memo_graph", 27500, 10000, hint="pass limit=")
    assert payload["error"] == "response_budget_exceeded"
    assert payload["tool"] == "memo_graph"
    assert payload["tokens"] == 27500
    assert payload["cap"] == 10000
    assert "limit=" in payload["hint"]


class _Block:
    def __init__(self, text: str) -> None:
        self.text = text


class _Result:
    def __init__(self, text: str) -> None:
        self.content = [_Block(text)]


def test_result_text_reads_content_blocks() -> None:
    assert mcp_budget.result_text(_Result("hello")) == "hello"


def test_result_text_falls_back_to_str() -> None:
    assert mcp_budget.result_text(1234) == "1234"


def test_result_text_sums_content_and_structured_content() -> None:
    """`ToolResult.to_mcp_result()` (fastmcp 3.4.5) puts BOTH fields on the
    wire -- `CallToolResult(content=..., structuredContent=...)`, or the
    `(content, structured_content)` tuple -- and for a dict-returning tool
    that is the same JSON twice. Measuring `content` alone, as this did
    originally, reported roughly half of what the caller pays for
    (`memo_list` on the 10k corpus: 4,586 content + 4,868 structured), so
    the effective ceiling was ~2x the configured cap."""
    from fastmcp.tools.base import ToolResult

    result = ToolResult(structured_content={"hits": ["x" * 400]})
    assert result.content and result.structured_content is not None
    content_only = "".join(str(getattr(b, "text", "") or "") for b in result.content)

    text = mcp_budget.result_text(result)

    assert text.startswith(content_only)
    assert text.endswith(str(result.structured_content))
    assert len(text) > len(content_only)


def test_result_text_never_raises_on_a_hostile_shape() -> None:
    # A shape where reading .content raises mid-projection must still
    # produce *something* -- a budget layer that can throw converts a
    # working tool into a broken one.
    class _Hostile:
        @property
        def content(self):
            raise RuntimeError("boom")

        def __str__(self) -> str:
            return "hostile-str"

    assert mcp_budget.result_text(_Hostile()) == "hostile-str"


@pytest.mark.asyncio
async def test_middleware_passes_a_small_result_through(monkeypatch) -> None:
    monkeypatch.setenv("MEMO_MCP_RESPONSE_BUDGET_TOKENS", "1000")
    mw = mcp_budget.make_response_budget_middleware()
    assert mw is not None

    small = _Result("ok")

    class _Ctx:
        message = type("M", (), {"name": "memo_search"})()

    async def _call_next(_ctx):
        return small

    assert await mw.on_call_tool(_Ctx(), _call_next) is small


@pytest.mark.asyncio
async def test_middleware_replaces_an_over_cap_result(monkeypatch) -> None:
    monkeypatch.setenv("MEMO_MCP_RESPONSE_BUDGET_TOKENS", "10")
    mw = mcp_budget.make_response_budget_middleware()
    assert mw is not None

    class _Ctx:
        message = type("M", (), {"name": "memo_graph"})()

    async def _call_next(_ctx):
        return _Result("x" * 4000)

    out = await mw.on_call_tool(_Ctx(), _call_next)
    # A bare dict is not a legal `on_call_tool` return -- FastMCP requires a
    # `ToolResult` (fastmcp.tools.base.ToolResult). is_error MUST stay False:
    # every other refusal on this MCP surface (server_graph_tool.py,
    # server_core_records.py, server_multimodal.py, server_sync.py, ...)
    # returns a plain {"error": ...} dict as a NORMAL successful call: a
    # caller inspects `result["error"]`, it never raises. `meta={}` (not
    # is_error) is what bypasses the original tool's output_schema
    # validation here -- ToolResult.to_mcp_result()'s gate is `self.meta is
    # not None or self.is_error`, and FastMCP's own ResponseLimitingMiddleware
    # sets `meta={}` for this exact reason, leaving is_error at its default.
    assert out.is_error is False
    assert out.structured_content["error"] == "response_budget_exceeded"
    assert out.structured_content["tool"] == "memo_graph"
    assert out.structured_content["tokens"] == 1000
    assert out.structured_content["cap"] == 10


@pytest.mark.asyncio
async def test_over_cap_result_reaches_a_real_client_as_a_normal_return(
    monkeypatch,
) -> None:
    # Regression proof at the actual client boundary, not just the ToolResult
    # fields: FastMCP's Client.call_tool() defaults raise_on_error=True and
    # raises ToolError whenever isError is set on the wire result. If the
    # substitute ever regresses back to is_error=True, this call raises and
    # the test fails loudly -- inspecting `.is_error` alone would not catch
    # a regression that only breaks the client-visible contract.
    monkeypatch.setenv("MEMO_MCP_RESPONSE_BUDGET_TOKENS", "10")
    from fastmcp import Client, FastMCP

    server = FastMCP("budget-test")

    @server.tool()
    def oversized() -> dict:
        return {"payload": "x" * 4000}

    mw = mcp_budget.make_response_budget_middleware()
    assert mw is not None
    server.add_middleware(mw)

    async with Client(server) as client:
        result = await client.call_tool("oversized", {})  # raise_on_error=True (default)

    assert result.is_error is False
    assert result.structured_content["error"] == "response_budget_exceeded"
    assert result.structured_content["tool"] == "oversized"


@pytest.mark.asyncio
async def test_zero_cap_disables_enforcement(monkeypatch) -> None:
    monkeypatch.setenv("MEMO_MCP_RESPONSE_BUDGET_TOKENS", "0")
    mw = mcp_budget.make_response_budget_middleware()
    assert mw is not None
    huge = _Result("x" * 40000)

    class _Ctx:
        message = type("M", (), {"name": "memo_graph"})()

    async def _call_next(_ctx):
        return huge

    assert await mw.on_call_tool(_Ctx(), _call_next) is huge
