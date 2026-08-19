"""Prune tool schemas to the ones this project actually uses.

Measured 2026-08-18 against real Claude Code traffic on this machine: a
captured request carried 154 tools / 264,277 B (~66,070 tokens) of schemas —
60.6% of the entire payload — and exactly 0 of them were memo_*. The
original version of this transform only pruned `memo_*` tools (43 tools,
~9.8k tokens), which is safe but touches none of that 66k: an 8-request live
A/B against real traffic reported `est_saved_tokens: 0` on every treated row.
Scope is now controlled by `MEMO_PROXY_TOOL_SCHEMAS_SCOPE` (`flags_proxy.py`):
`"all"` (default — every tool on the wire is a pruning candidate, not just
memo's) or `"memo"` (the original conservative scope, one flag away).

Widening past memo's own tools changes the risk: pruning a `memo_*` tool the
model doesn't call just means an extra round trip through `memo_tool_docs`.
Pruning `Read` or `Bash` on a cold-start session — no usage history yet, and
the keep-set freeze below means whatever survives turn 1 is what the WHOLE
session gets — can leave the agent unable to read a file or run a command for
the rest of the conversation. `_BUILTIN_NEVER_PRUNE` is the guard: Claude
Code's core built-ins (file I/O, shell, sub-agent delegation) plus
`ToolSearch` (Claude Code's own discovery-then-hydrate primitive for every
other deferred tool — losing it breaks recovery for everything else, not
just memo's) are kept unconditionally, the same way `memo_tool_docs` itself
already was. These names were taken from this exact machine's real tool
surface, not guessed.

Hydration for anything else pruned under scope `"all"` — a non-memo MCP
tool like `mcp__octocode__localSearchCode` — cannot go through
`server.get_tool()` the way a memo_* tool's docs do, because that only
resolves tools registered on memo's own FastMCP server. `ToolSchemas` sees
every tool definition in the payload before it prunes any of them, so it
caches what it prunes (`memo.proxy.tool_schema_cache`) and `memo_tool_docs`
falls back to that cache for a name it doesn't own. Without that cache write,
widening this transform past memo's own tools would be data loss, not
compression — the whole point of `memo_tool_docs` existing is that pruning a
schema must never be a one-way door.

The retained set is derived from usage history and is FROZEN at the first
request of a session, then reused byte-identically for the rest of it — the
design spec's binding requirement (docs/SPECS/2026-08-18-token-savings-proxy-
context-compression-design.md, Section 2 / Section 4 item 1). Recomputing
`recent_tool_names` fresh on every request would defeat this: `tool_usage.json`
is updated BEFORE this transform runs on the very same request that reports a
newly-called tool (`record_tool_usage` in `memo.proxy.server` runs first), so
a pruned tool getting called mid-session — the exact discover-then-hydrate
flow `memo_tool_docs` exists to enable — would otherwise grow the keep-set on
the very next turn and reshuffle the cached prefix. `_session_keep_cache`
below is the freeze: a module-level cache (a fresh `ToolSchemas` instance is
constructed per request by `registry.build_registry()`, so the cache cannot
live on `self`), keyed by `(state_dir, session_key)` and computed exactly
once per session.

Usage history comes from `<state_dir>/proxy/tool_usage.json`, written by
`record_tool_usage` in `memo.proxy.server` from `tool_use` blocks the proxy
observes on the wire — NOT from `recall.log` (that log has no per-tool-call
field; its rows carry hits/latency_ms/prompt/source/ts/via). The path is
duplicated here rather than imported from `memo.proxy.server` to avoid a
cycle: that module imports `memo.proxy.registry`, which registers this
transform. `record_tool_usage` already records every tool name it sees, not
just memo_*'s — `recent_tool_names` below used to be the thing that filtered
that down to memo_* only; it no longer filters at all (scope filtering moved
up into `ToolSchemas.apply`, the one place that actually knows the active
scope), so it keeps returning the exact set `record_tool_usage` wrote.
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

from memo.flags import flag_bool, flag_int, flag_str
from memo.mcp_budget import est_tokens
from memo.proxy import tool_schema_cache
from memo.proxy.plan import ZONE_PREFIX, Context
from memo.proxy.zones import Zones

_log = logging.getLogger(__name__)

DOCS_TOOL_NAME = "memo_tool_docs"
_OWNED_PREFIX = "memo_"

# Claude Code's core built-ins: real names taken from this machine's own tool
# surface, not guessed. No fallback path exists for these the way there does
# for a memo_* tool — a model that loses Read/Write/Edit/Bash/Glob/Grep or
# the sub-agent delegation tool (exposed as "Task" in vanilla Claude Code,
# "Agent" in this harness — both kept since the wire name varies by harness)
# has no way to recover file or shell access mid-session. ToolSearch is
# included for the same reason `memo_tool_docs` is: it is Claude Code's OWN
# discovery-then-hydrate primitive for every other deferred/pruned tool, so
# losing it breaks the recovery path for everything else, not just memo's.
_BUILTIN_NEVER_PRUNE = frozenset(
    {"Read", "Write", "Edit", "Bash", "Glob", "Grep", "Task", "Agent", "ToolSearch"}
)
# Kept regardless of usage or scope: without these the model cannot reach
# memo, or the agent's own built-in tools, at all.
_ALWAYS_KEEP = frozenset({DOCS_TOOL_NAME, "memo_search", "memo_save"}) | _BUILTIN_NEVER_PRUNE
_DEFAULT_WINDOW = 20
_DEFAULT_SCOPE = "all"


def _scope() -> str:
    """`MEMO_PROXY_TOOL_SCHEMAS_SCOPE`, defaulting (and falling back on any
    read failure) to the aggressive `"all"` setting. Never raises."""
    try:
        value = flag_str("MEMO_PROXY_TOOL_SCHEMAS_SCOPE")
        return value if value in ("all", "memo") else _DEFAULT_SCOPE
    except Exception:
        return _DEFAULT_SCOPE


# Session-frozen keep-set cache. Bounded so a long-lived proxy process can't
# grow this without limit; approximate LRU via OrderedDict (oldest-touched
# entry evicted first). Keyed on (str(state_dir), session_key) rather than
# session_key alone so distinct proxy deployments/tests sharing a session_key
# by coincidence never collide.
_MAX_CACHED_SESSIONS = 1000
# Freezes (scope, keep-set) TOGETHER, not just the keep-set — `scope` also
# decides the "pass every non-memo tool through" branch in
# `ToolSchemas.apply`'s `_keeps`, so if only the keep-set were frozen, an
# operator flipping MEMO_PROXY_TOOL_SCHEMAS_SCOPE mid-session (env change or
# `memo config set` on a live proxy process) could still change what gets
# pruned on turn 2 of an existing session even though the keep-set itself
# didn't move — exactly the prefix-rewrite this freeze exists to prevent.
_session_keep_cache: OrderedDict[tuple[str, str], tuple[str, frozenset[str]]] = OrderedDict()


def _frozen_keep_set(ctx: Context, window: int, scope: str) -> tuple[str, frozenset[str]]:
    """The (scope, keep-set) for this session, computed once and reused for
    every later request in it — see the module docstring for why
    recomputing per request would defeat session-stability, and the
    `_session_keep_cache` comment above for why `scope` is part of what's
    frozen, not just the keep-set it produced.

    `scope` decides how `recent_tool_names`'s result (every tool name ever
    seen called, regardless of owner) narrows down to a keep-set: `"memo"`
    filters it to `memo_*` names only (matching the pre-widening behavior);
    `"all"` uses it unfiltered. Like `window`, `scope` is only consulted on
    a cache MISS — a session's first-request scope is what it's frozen
    under for the rest of the session, same as its first-request window.

    Fail-open: any cache read/write failure falls back to computing fresh
    (never raises, never blocks a request on a caching problem).
    """
    key = (str(ctx.state_dir), ctx.session_key)
    try:
        cached = _session_keep_cache.get(key)
        if cached is not None:
            _session_keep_cache.move_to_end(key)
            return cached
    except Exception:
        _log.debug("proxy: session keep-set cache read failed; computing fresh", exc_info=True)

    names = recent_tool_names(ctx.state_dir, window)
    if scope == "memo":
        names = {n for n in names if n.startswith(_OWNED_PREFIX)}
    result = (scope, frozenset(names) | _ALWAYS_KEEP)

    try:
        _session_keep_cache[key] = result
        _session_keep_cache.move_to_end(key)
        while len(_session_keep_cache) > _MAX_CACHED_SESSIONS:
            _session_keep_cache.popitem(last=False)
    except Exception:
        _log.debug(
            "proxy: session keep-set cache write failed; continuing unaffected", exc_info=True
        )

    return result


def recent_tool_names(state_dir: Path, window: int) -> set[str]:
    """Every tool name actually called in the last `window` sessions,
    regardless of which server owns it.

    Reads `<state_dir>/proxy/tool_usage.json` (see module docstring for why
    not `recall.log`). A cold start — no file yet, an empty/corrupt file, or
    no matching sessions — resolves to "no history" rather than raising; the
    caller then falls back to the always-keep set. Scope-agnostic on purpose:
    `ToolSchemas.apply` (via `_frozen_keep_set`) is the one place that knows
    the active `MEMO_PROXY_TOOL_SCHEMAS_SCOPE`, so filtering to `memo_*` only
    happens there, not here — see the module docstring.
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
            names.update(t for t in tools if isinstance(t, str) and t)
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
            # `_scope()` reads the live flag, but only to seed a cache MISS —
            # `frozen_scope` (not this) is what `_keeps` below actually uses,
            # so a flag flip mid-session can't reshuffle an already-frozen
            # session (see `_session_keep_cache`'s comment).
            frozen_scope, keep = _frozen_keep_set(ctx, window, _scope())

            def _keeps(tool: Any) -> bool:
                if not isinstance(tool, dict):
                    return True
                tool_name = tool.get("name")
                if not isinstance(tool_name, str) or not tool_name:
                    return True
                if frozen_scope == "memo" and not tool_name.startswith(_OWNED_PREFIX):
                    # Conservative scope: memo never touches a schema it
                    # doesn't own.
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

            # Identity-based, not name-based: computed BEFORE zones.tools is
            # mutated below, so this is the exact set of dicts about to be
            # dropped. Cached to disk (memo.proxy.tool_schema_cache) so
            # memo_tool_docs — a different process — can still hydrate any
            # of these by name; pruning a schema must never be a one-way
            # door (see module docstring).
            kept_ids = {id(tool) for tool in kept}
            pruned = [tool for tool in zones.tools if id(tool) not in kept_ids]

            before = est_tokens(json.dumps(zones.tools, separators=(",", ":"), ensure_ascii=False))
            zones.tools[:] = kept
            after = est_tokens(json.dumps(zones.tools, separators=(",", ":"), ensure_ascii=False))

            try:
                tool_schema_cache.remember(ctx.state_dir, pruned)
            except Exception:
                _log.debug("proxy: could not cache pruned tool schemas", exc_info=True)

            return max(0, before - after)
        except Exception:
            return 0
