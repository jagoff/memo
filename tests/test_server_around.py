"""memo_around MCP tool registration + pass-through."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock


def _make_server_and_tools():
    server, tools = MagicMock(), {}

    def tool_decorator():
        def wrapper(fn):
            tools[fn.__name__] = fn
            return fn

        return wrapper

    server.tool = tool_decorator
    return server, tools


def test_register_exposes_memo_around(tmp_cfg):
    from memo.memory import Memory
    from memo.server_around import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    server, tools = _make_server_and_tools()
    register(server, mem)
    assert set(tools) == {"memo_around"}


def test_memo_around_clamps_and_delegates(tmp_cfg):
    from memo.memory import Memory
    from memo.server_around import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.around.return_value = {"anchor": None, "mode": None, "neighbors": []}
    server, tools = _make_server_and_tools()
    register(server, mem)
    out = tools["memo_around"](id="a1b2c3d4", before=99, after=-1)
    mem.around.assert_called_once_with("a1b2c3d4", before=10, after=0)
    assert out["neighbors"] == []


def test_build_server_wires_memo_around_registration():
    """build_server (server.py) must import server_around and call its
    register — asserted against the source, same style as the repo's
    architecture-boundary source scans. This is the ONLY test that covers
    the Step 4 wiring (the inventory test never proves registration)."""
    import memo.server as srv

    source = Path(srv.__file__).read_text(encoding="utf-8")
    assert re.search(r"^from memo import server_around as _srv_around$", source, re.M), (
        "server.py must import server_around as _srv_around"
    )
    assert "_srv_around.register(server, memory)" in source, (
        "build_server must call _srv_around.register(server, memory)"
    )
