"""Prune memo's own MCP tool schemas to the ones this project actually uses.

Measured 2026-08-18: 41 memo tools cost 46,562 B ~= 11,640 tokens in *every*
request, paid whether or not a tool is called. Only memo's own tools are
pruned — pruning another server's schema would break a tool memo does not
own, so anything not named `memo_*` passes through untouched.

The retained set is derived from usage history and is stable for a whole
session, so the cached prefix changes once rather than every turn. This
transform runs in the CACHE-STABLE PREFIX zone: a transform that reshuffles
the prefix per turn pays for a fresh cache write every time and costs more
than it saves (see `memo.proxy.zones.prefix_fingerprint`).

Usage history comes from `<state_dir>/proxy/tool_usage.json`, written by
`record_tool_usage` in `memo.proxy.server` from `tool_use` blocks the proxy
observes on the wire — NOT from `recall.log` (that log has no per-tool-call
field; its rows carry hits/latency_ms/prompt/source/ts/via). The path is
duplicated here rather than imported from `memo.proxy.server` to avoid a
cycle: that module imports `memo.proxy.registry`, which registers this
transform.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from memo.flags import flag_bool, flag_int
from memo.mcp_budget import est_tokens
from memo.proxy.plan import ZONE_PREFIX, Context
from memo.proxy.zones import Zones

DOCS_TOOL_NAME = "memo_tool_docs"
_OWNED_PREFIX = "memo_"
# Kept regardless of usage: without these the model cannot reach memo at all.
_ALWAYS_KEEP = frozenset({DOCS_TOOL_NAME, "memo_search", "memo_save"})
_DEFAULT_WINDOW = 20


def recent_tool_names(state_dir: Path, window: int) -> set[str]:
    """memo tool names used in the last `window` sessions.

    Reads `<state_dir>/proxy/tool_usage.json` (see module docstring for why
    not `recall.log`). A cold start — no file yet, an empty/corrupt file, or
    no matching sessions — resolves to "no history" rather than raising; the
    caller then falls back to the always-keep set.
    """
    try:
        path = Path(state_dir) / "proxy" / "tool_usage.json"
        if not path.is_file():
            return set()
        text = path.read_text(encoding="utf-8", errors="replace")
        data = json.loads(text) if text.strip() else {}
        if not isinstance(data, dict):
            return set()
        sessions = data.get("sessions")
        if not isinstance(sessions, dict):
            return set()

        def _ts(session: Any) -> float:
            value = session.get("ts") if isinstance(session, dict) else None
            return value if isinstance(value, (int, float)) else 0.0

        ordered = sorted(
            (s for s in sessions.values() if isinstance(s, dict)),
            key=_ts,
            reverse=True,
        )
        names: set[str] = set()
        for session in ordered[: max(0, window)]:
            tools = session.get("tools")
            if not isinstance(tools, list):
                continue
            names.update(t for t in tools if isinstance(t, str) and t.startswith(_OWNED_PREFIX))
        return names
    except Exception:
        return set()


class ToolSchemas:
    name = "toolschemas"
    zone = ZONE_PREFIX

    def enabled(self) -> bool:
        try:
            return bool(flag_bool("MEMO_PROXY_TOOL_SCHEMAS"))
        except Exception:
            return False

    def apply(self, zones: Zones, ctx: Context) -> int:
        try:
            if not zones.tools:
                return 0
            window = flag_int("MEMO_PROXY_TOOL_WINDOW_SESSIONS") or _DEFAULT_WINDOW
            keep = recent_tool_names(ctx.state_dir, window) | _ALWAYS_KEEP

            def _keeps(tool: Any) -> bool:
                if not isinstance(tool, dict):
                    return True
                tool_name = tool.get("name")
                if not isinstance(tool_name, str) or not tool_name.startswith(_OWNED_PREFIX):
                    return True
                return tool_name in keep

            # List-comprehension over the ORIGINAL order — never a set, never
            # sorted by anything derived from `keep` — so the surviving tools
            # come out in the same relative order every time the same input
            # is pruned against the same keep-set: required for session
            # stability of the cached prefix.
            kept = [tool for tool in zones.tools if _keeps(tool)]
            if len(kept) == len(zones.tools):
                return 0

            before = est_tokens(json.dumps(zones.tools, separators=(",", ":"), ensure_ascii=False))
            zones.tools[:] = kept
            after = est_tokens(json.dumps(zones.tools, separators=(",", ":"), ensure_ascii=False))
            return max(0, before - after)
        except Exception:
            return 0
