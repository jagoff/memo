"""The budget module: estimator, opt-in trimming, and the error payload.

The middleware itself is covered in test 2; these are the pure pieces."""

from __future__ import annotations

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
