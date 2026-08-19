"""On-disk cache of tool schemas the proxy has seen and pruned.

`ToolSchemas` (`memo.proxy.transforms.toolschemas`) prunes tool definitions
out of the cached prefix to save tokens. Pruning only removes a schema from
what the model sees in that one request — for memo's own tools, the schema
is still recoverable via `server.get_tool(name)` on the live FastMCP server
(`memo_tool_docs`, `src/memo/server_tool_docs.py`). For a tool memo does not
own (another MCP server's tool, or a Claude Code built-in), there is no
FastMCP registration to ask — the proxy is the only thing that ever saw that
tool's real schema on the wire. This module is where it remembers it, so
`memo_tool_docs` — running in a different process, the MCP server, not the
proxy — can still serve it back by name.

Same locked-read-then-atomic-replace shape as `record_tool_usage` in
`memo.proxy.server`, and for the same reason: multiple concurrent requests
(different sessions) can prune different tools on the same tick, and a
plain read-modify-write would drop whichever write lost the race.

The proxy is the writer, `memo_tool_docs` is the reader — two different
processes agreeing only through this file, so both sides must tolerate the
other having never run yet (a fresh install, or a docs lookup before the
first proxied request): `lookup` on a missing/corrupt file returns `None`,
never raises.
"""

from __future__ import annotations

import fcntl
import json
import logging
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

SCHEMA = "memo.proxy.tool_schema_cache.v1"


def cache_path(state_dir: Path) -> Path:
    return Path(state_dir) / "proxy" / "tool_schema_cache.json"


def _read(path: Path) -> dict[str, Any]:
    tools: dict[str, Any] = {}
    if path.is_file():
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            data = json.loads(text) if text.strip() else {}
            if isinstance(data, dict) and isinstance(data.get("tools"), dict):
                tools = data["tools"]
        except Exception:
            tools = {}
    return {"schema": SCHEMA, "tools": tools}


def remember(state_dir: Path, tools: list[Any]) -> None:
    """Merge each tool's name/description/input_schema into the cache.

    `tools` is exactly the shape of an Anthropic Messages API `tools` array
    entry (what `ToolSchemas` sees on the wire before pruning): a dict with a
    `name`, an optional `description`, and an `input_schema`. An entry
    missing a usable string `name` is skipped rather than raising. Never
    raises: an unwritable state dir or a corrupt existing file both fall
    through to a no-op, matching every other function this file's writer
    counterpart (`record_tool_usage` in `memo.proxy.server`) uses this
    pattern for.
    """
    try:
        entries: dict[str, Any] = {}
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            name = tool.get("name")
            if not isinstance(name, str) or not name:
                continue
            entries[name] = {
                "description": tool.get("description") or "",
                "input_schema": tool.get("input_schema") or {"type": "object", "properties": {}},
            }
        if not entries:
            return
        path = cache_path(state_dir)
        lock_path = path.with_suffix(".json.lock")
        path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as lockfile:
            fcntl.flock(lockfile.fileno(), fcntl.LOCK_EX)
            try:
                data = _read(path)
                data["tools"].update(entries)
                tmp_path = path.with_suffix(".json.tmp")
                tmp_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                tmp_path.replace(path)
            finally:
                fcntl.flock(lockfile.fileno(), fcntl.LOCK_UN)
    except Exception:
        _log.warning("proxy: could not cache pruned tool schemas")


def lookup(state_dir: Path, name: str) -> dict[str, Any] | None:
    """The cached `{"description": ..., "input_schema": ...}` for `name`, or
    `None` if it was never cached (or the cache can't be read). Never
    raises.
    """
    try:
        entry = _read(cache_path(state_dir))["tools"].get(name)
        return entry if isinstance(entry, dict) else None
    except Exception:
        return None
