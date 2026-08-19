"""Collapse a RE-read of an already-seen file to just what changed.

`StructMap` (`structmap.py`) handles the FIRST time a file is read in a
conversation; this module handles every read after that. The two are
mutually exclusive by construction — both consult `seen_files()` below to
decide whether a given `tool_result` block is a first read or a re-read, and
each acts only on its own case, so registering both never lets one starve
the other of the blocks it wants.

State — no cross-request memory, and deliberately so
------------------------------------------------------
"Seen" is read straight out of `zones.frozen_messages` on every request, not
out of any cache this module keeps between requests. Two things make that the
right (and simplest) choice rather than a shortcut:

* **It is always correct.** Rewriting only ever happens on the way OUT to the
  provider (`memo.proxy.server.rewrite_body` mutates a *copy* of the request
  before forwarding it) — the caller's own conversation, and everything it
  resends on the next turn, is never touched. So `frozen_messages` on every
  request already carries the real, uncompressed `tool_result` content of
  every earlier `Read`, verbatim. A side cache keyed by `session_key` would
  just be a second, redundant copy of exactly that.
* **It cannot go stale.** If the client's own context ever drops an old
  `Read` result (compaction, a manual `/clear`, a fresh session reusing an
  id), that content simply stops appearing in `frozen_messages`, and the next
  read of that path is correctly treated as a first read again — StructMap's
  case, not this module's. A side cache has no way to learn that the model no
  longer has the content it remembers; it would keep emitting diffs against
  material the model can no longer see, which is actively worse than doing
  nothing.

It is therefore bounded for free (nothing outlives the one request that built
it, so a long-lived proxy process accumulates no state at all here — unlike
`toolschemas.py`'s session-keyed keep-set cache, which genuinely needs its own
LRU eviction because IT choses to remember across requests) and immune to
cross-session collision for free (each request's `zones` already belongs to
exactly one session; there is nothing to key by session for, and so nothing
to get wrong when two sessions happen to read the same path).
"""

from __future__ import annotations

import difflib
import logging

from memo.flags import flag_bool
from memo.mcp_budget import est_tokens
from memo.proxy import ccr
from memo.proxy.plan import ZONE_LIVE, Context
from memo.proxy.zones import Zones

_log = logging.getLogger(__name__)

_READ_TOOL_NAME = "Read"
# Kept intentionally short: this replaces a full re-read, and needs to stay
# cheaper than the file even when the file itself is tiny (see
# test_an_unchanged_reread_collapses_to_a_notice in test_proxy_delta.py).
_UNCHANGED_NOTICE = "[memo: unchanged]"


def _block_text(block: dict) -> str:
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            c["text"]
            for c in content
            if isinstance(c, dict) and c.get("type") == "text" and isinstance(c.get("text"), str)
        ]
        return "\n".join(parts)
    return ""


def _set_block_text(block: dict, text: str) -> None:
    if isinstance(block.get("content"), list):
        block["content"] = [{"type": "text", "text": text}]
    else:
        block["content"] = text


def _read_tool_paths(messages: list) -> dict[str, str]:
    """Map `tool_use_id -> file_path` for every `Read` tool_use block."""
    out: dict[str, str] = {}
    try:
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                if block.get("name") != _READ_TOOL_NAME:
                    continue
                tool_id = block.get("id")
                block_input = block.get("input")
                if not isinstance(tool_id, str) or not isinstance(block_input, dict):
                    continue
                path = block_input.get("file_path")
                if isinstance(path, str) and path:
                    out[tool_id] = path
    except Exception:
        return {}
    return out


def seen_files(zones: Zones) -> dict[str, str]:
    """`file_path -> content` for every file the model has already been sent
    a `Read` of, in this conversation. Reads only `zones.frozen_messages` —
    see the module docstring for why `live_messages` (the turn currently in
    flight, not yet part of any response) must not count as "seen"."""
    seen: dict[str, str] = {}
    try:
        paths = _read_tool_paths(zones.frozen_messages)
        for message in zones.frozen_messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                path = paths.get(block.get("tool_use_id", ""))
                if not path:
                    continue
                text = _block_text(block)
                if text:
                    seen[path] = text
    except Exception:
        return {}
    return seen


def diff_against(previous: str, current: str) -> str:
    """`current` collapsed to just what changed since `previous`.

    Returns `current` unchanged when there is no previous copy to diff
    against (`previous` empty — a first read; `StructMap`'s case, not ours).
    Returns a short fixed notice on an exact match. Uses `n=0` — no
    surrounding context lines — rather than `difflib`'s usual default:
    a context line repeats content the model already has from the earlier
    read, so for a change as small as one line out of many the context alone
    can cost more than the plain re-read would have. `apply()` below only
    ever writes this output back when it measurably beats the original
    anyway, so a diff that isn't actually smaller is simply never used.
    """
    try:
        if not previous:
            return current
        if previous == current:
            return _UNCHANGED_NOTICE
        out = "".join(
            difflib.unified_diff(
                previous.splitlines(keepends=True),
                current.splitlines(keepends=True),
                n=0,
            )
        )
        return out if out else current
    except Exception:
        return current


class Delta:
    name = "delta"
    zone = ZONE_LIVE

    def enabled(self) -> bool:
        try:
            return bool(flag_bool("MEMO_PROXY_STRUCTMAP"))
        except Exception:
            return False

    def apply(self, zones: Zones, ctx: Context) -> int:
        try:
            if not zones.live_messages:
                return 0
            seen = seen_files(zones)
            live_paths = _read_tool_paths(zones.live_messages)
            saved = 0
            for message in zones.live_messages:
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                for block in content:
                    saved += self._rewrite_block(block, live_paths, seen, ctx)
            return saved
        except Exception:
            return 0

    def _rewrite_block(
        self,
        block: object,
        live_paths: dict[str, str],
        seen: dict[str, str],
        ctx: Context,
    ) -> int:
        try:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                return 0
            path = live_paths.get(block.get("tool_use_id", ""))
            if not path:
                return 0
            previous = seen.get(path)
            if not previous:
                return 0  # first read of this path -- StructMap's case, not ours

            text = _block_text(block)
            if not text:
                return 0

            new_text = diff_against(previous, text)
            if not isinstance(new_text, str) or len(new_text) >= len(text):
                return 0

            key = ccr.stash(ctx.state_dir, text)
            if not key:
                # No recovery path -- leave the block completely untouched.
                return 0

            marked = new_text + ccr.marker(
                key,
                kept_chars=len(new_text),
                dropped_chars=len(text) - len(new_text),
                stashed=text,
            )
            # Only the FINAL, marker-included text counts as the saving --
            # a diff smaller than the marker's own overhead must leave the
            # block untouched rather than write back something net larger
            # than the original while still reporting a "cut".
            if len(marked) >= len(text):
                return 0

            _set_block_text(block, marked)
            return max(0, est_tokens(text) - est_tokens(marked))
        except Exception:
            return 0
