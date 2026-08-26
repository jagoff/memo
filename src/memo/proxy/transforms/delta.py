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
import re
import shlex

from memo.flags import flag_bool
from memo.mcp_budget import est_tokens
from memo.proxy import ccr
from memo.proxy.plan import ZONE_LIVE, Context

# `_set_block_text` is imported, not re-implemented: three identical copies
# existed and all three deleted any image or document block travelling with
# the text. One canonical version lives in the module that owns tool_result
# rewriting.
from memo.proxy.transforms.toolresults import _set_block_text
from memo.proxy.zones import Zones, whole_history_scope

_log = logging.getLogger(__name__)

_READ_TOOL_NAME = "Read"
_BASH_TOOL_NAME = "Bash"
# Kept intentionally short: this replaces a full re-read, and needs to stay
# cheaper than the file even when the file itself is tiny (see
# test_an_unchanged_reread_collapses_to_a_notice in test_proxy_delta.py).
_UNCHANGED_NOTICE = "[memo: unchanged]"

# Anywhere in the raw command text, any of these means "not an unambiguous
# single-file read" -- a pipe, redirect, glob, substitution, or subshell.
# `_extract_bash_read_path` skips rather than guesses: extracting the WRONG
# path would hand Delta a false identity, diffing unrelated content against
# each other under one path.
_UNSAFE_COMMAND_CHARS = frozenset("|&;<>*?$`(){}[]~\n")

_LINE_COUNT_RE = re.compile(r"^-?\d+$")


def _extract_bash_read_path(command: str) -> str | None:
    """The single file argument of an unambiguous, single-file Bash read --
    `cat X`, `cat -n X`, `head [-n N] X`, `tail [-n N] X`, `sed -n SCRIPT X`.

    None for anything else: a pipe, a redirect, a glob, more than one path,
    an unrecognized command, or a command `shlex` cannot even tokenize. This
    is not a shell interpreter and never guesses -- ground truth for why it
    exists: a real captured payload showed the model reading a source file
    via `cat -n <path>`, never `Read`, so `_read_tool_paths` (below) was
    blind to it and neither Delta nor StructMap ever saw a path for that
    block.
    """
    try:
        if not isinstance(command, str) or not command.strip():
            return None
        if any(ch in command for ch in _UNSAFE_COMMAND_CHARS):
            return None
        tokens = shlex.split(command)
    except Exception:
        return None
    if len(tokens) < 2:
        return None
    prog = tokens[0].rsplit("/", 1)[-1]
    rest = tokens[1:]

    if prog == "cat":
        if len(rest) == 1:
            return rest[0]
        if len(rest) == 2 and rest[0] == "-n":
            return rest[1]
        return None

    if prog in ("head", "tail"):
        if len(rest) == 1:
            return rest[0]
        if len(rest) == 3 and rest[0] == "-n" and _LINE_COUNT_RE.match(rest[1]):
            return rest[2]
        return None

    if prog == "sed":
        if len(rest) == 3 and rest[0] == "-n":
            return rest[2]
        return None

    return None


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


def _read_tool_paths(messages: list) -> dict[str, str]:
    """Map `tool_use_id -> file_path` for every `Read` tool_use block, plus
    every `Bash` tool_use block whose command is an unambiguous single-file
    read (`_extract_bash_read_path`).

    The tool's name only ever supplies IDENTITY here -- whether a block's
    CONTENT is actually worth treating as source is a separate, content-shape
    question (`structmap.sniff_signatures`), independent of how (or whether)
    a path was found. A path found here is strictly better evidence than a
    sniff when both are available: `_language_for` on a real extension, and
    Delta's diff identity, both need it.
    """
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
                tool_id = block.get("id")
                block_input = block.get("input")
                if not isinstance(tool_id, str) or not isinstance(block_input, dict):
                    continue
                name = block.get("name")
                path: str | None = None
                if name == _READ_TOOL_NAME:
                    candidate = block_input.get("file_path")
                    if isinstance(candidate, str) and candidate:
                        path = candidate
                elif name == _BASH_TOOL_NAME:
                    command = block_input.get("command")
                    if isinstance(command, str):
                        path = _extract_bash_read_path(command)
                if path:
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


def read_occurrences(messages: list) -> list[tuple[dict, str, str, str | None]]:
    """`(block, path, text, previous_or_none)` for every Read `tool_result`
    block across `messages`, IN ORDER. `previous_or_none` is the most recent
    PRIOR occurrence's raw text for the same path within THIS SAME list
    (`None` on a path's first occurrence).

    This is what makes StructMap/Delta safe to widen past the live zone
    (`MEMO_PROXY_CONTENT_SCOPE=all`, `zones.scan_scope`): classifying
    occurrence i (first read vs. re-read, and what to diff a re-read
    against) depends only on messages 0..i, a pure function of blocks that
    PRECEDE it in `messages` -- never on anything that comes after. Since the
    client resends its own conversation raw and unmodified (see the module
    docstring above), messages 0..i-1 are byte-identical on every later turn,
    so occurrence i's classification and diff target can never change once
    an earlier turn has already fixed them on the wire -- exactly the
    determinism the cache rule in `zones.py`'s module docstring requires of
    anything that touches the frozen zone.

    StructMap's case is `previous is None`; Delta's is `previous is not
    None` -- see each transform's `apply()`, which filters this shared walk
    down to its own half rather than duplicating the ordering logic."""
    seen: dict[str, str] = {}
    out: list[tuple[dict, str, str, str | None]] = []
    try:
        paths = _read_tool_paths(messages)
        for message in messages:
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
                if not text:
                    continue
                out.append((block, path, text, seen.get(path)))
                seen[path] = text
    except Exception:
        return []
    return out


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
            if whole_history_scope():
                if not zones.frozen_messages and not zones.live_messages:
                    return 0
                return self._apply_whole_history(zones, ctx)
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

    def _apply_whole_history(self, zones: Zones, ctx: Context) -> int:
        """`MEMO_PROXY_CONTENT_SCOPE=all` (default): a single ordered pass
        over the WHOLE conversation via `read_occurrences` -- see that
        function's docstring for why this stays deterministic and therefore
        safe to run over the frozen zone too."""
        saved = 0
        for block, _path, text, previous in read_occurrences(
            [*zones.frozen_messages, *zones.live_messages]
        ):
            if previous is None:
                continue  # first occurrence -- StructMap's case, not ours
            saved += self._rewrite(block, text, previous, ctx)
        return saved

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

            return self._rewrite(block, text, previous, ctx)
        except Exception:
            return 0

    def _rewrite(self, block: dict, text: str, previous: str, ctx: Context) -> int:
        """Shared final step once (block, its raw text, the prior version to
        diff against) are known -- identical for the tail-only and
        whole-history scopes; only how `previous` is discovered differs."""
        try:
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
