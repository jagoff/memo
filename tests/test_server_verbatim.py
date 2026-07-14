from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_server_and_tools():
    server, tools = MagicMock(), {}

    def tool_decorator():
        def wrapper(fn):
            tools[fn.__name__] = fn
            return fn

        return wrapper

    server.tool = tool_decorator
    return server, tools


def test_register_exposes_bounded_verbatim_search(tmp_cfg):
    from memo.memory import Memory
    from memo.server_verbatim import register

    memory = MagicMock(spec=Memory)
    memory.cfg = tmp_cfg
    tmp_cfg.state_dir.mkdir(parents=True, exist_ok=True)
    tmp_cfg.verbatim_db.touch()
    server, tools = _make_server_and_tools()
    register(server, memory)

    fake_store = MagicMock()
    fake_store.search.return_value = []
    with (
        patch("memo.store.turn_store.TurnStore", return_value=fake_store),
        patch("memo.server_verbatim.log_consult") as log_consult,
    ):
        result = tools["memo_verbatim_search"](query="exact decision", limit=10_000, source="codex")

    assert result == {"hits": []}
    fake_store.search.assert_called_once_with(
        "exact decision", limit=100, session_id=None, since=None
    )
    fake_store.close.assert_called_once_with()
    log_consult.assert_called_once()
    assert log_consult.call_args.kwargs["source"] == "codex"


def test_build_server_wires_verbatim_registration():
    import memo.server as srv

    source = Path(srv.__file__).read_text(encoding="utf-8")
    assert re.search(r"^from memo import server_verbatim as _srv_verbatim$", source, re.M)
    assert "_srv_verbatim.register(server, memory)" in source
