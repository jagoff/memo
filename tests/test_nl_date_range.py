"""parse_date_range: ES/EN relative expressions -> inclusive ISO [start, end]."""
from __future__ import annotations

import datetime as dt

REF = dt.date(2026, 7, 3)  # a Friday


def test_yesterday_es_and_en():
    from memo.nl_dates import parse_date_range

    assert parse_date_range("qué decidimos ayer", REF) == ("2026-07-02", "2026-07-02")
    assert parse_date_range("what did we decide yesterday", REF) == ("2026-07-02", "2026-07-02")


def test_last_week_is_prior_monday_to_sunday():
    from memo.nl_dates import parse_date_range

    assert parse_date_range("la semana pasada", REF) == ("2026-06-22", "2026-06-28")
    assert parse_date_range("last week", REF) == ("2026-06-22", "2026-06-28")


def test_n_days_ago_and_last_month():
    from memo.nl_dates import parse_date_range

    assert parse_date_range("hace 10 días", REF) == ("2026-06-23", "2026-06-23")
    assert parse_date_range("el mes pasado", REF) == ("2026-06-01", "2026-06-30")


def test_no_expression_returns_none_pair():
    from memo.nl_dates import parse_date_range

    assert parse_date_range("cómo funciona el reranker", REF) == (None, None)


def test_memo_search_when_param_fills_date_filters(tmp_cfg):
    from unittest.mock import MagicMock

    from memo.memory import Memory
    from memo.server_core_search import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.search.return_value = []
    server, tools = MagicMock(), {}

    def tool_decorator():
        def wrapper(fn):
            tools[fn.__name__] = fn
            return fn
        return wrapper

    server.tool = tool_decorator
    register(server, mem)
    tools["memo_search"](query="decisiones", when="ayer")
    kwargs = mem.search.call_args.kwargs
    assert kwargs["date_from"] is not None and kwargs["date_from"] == kwargs["date_to"][:10]
