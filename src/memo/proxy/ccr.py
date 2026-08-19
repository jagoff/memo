"""Content-addressed recovery: nothing is cut without being recoverable.

Reuses the crush cache the capture crusher already writes to, so `memo
retrieve` keeps working unchanged and there is exactly one recovery path.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

_log = logging.getLogger(__name__)


def stash(state_dir: Path, content: str) -> str:
    """Store `content` and return its key. Returns "" if it could not be stored.

    An empty key is the caller's signal to skip the lossy edit entirely: cutting
    without a recovery path is not a trade this package makes.
    """
    try:
        key = hashlib.sha256(content.encode("utf-8")).hexdigest()
        from memo.store.crush_cache import CrushCache

        CrushCache(Path(state_dir)).cache(key, content)
    except Exception:
        _log.warning("proxy: crush cache unwritable; skipping lossy edit")
        return ""
    return key


def recover(state_dir: Path, key: str) -> str | None:
    try:
        from memo.store.crush_cache import CrushCache

        return CrushCache(Path(state_dir)).retrieve(key)
    except Exception:
        return None


_NESTED_CRUSH_MARKER = "<<memo-crush:"

# A REAL prior `marker()` call always substitutes actual digit counts for
# `{dropped_chars}`/`{kept_chars}`. This module's OWN source code -- read and
# compressed like any other file by, say, `StructMap`/`Delta` -- only ever
# contains the UN-RENDERED template text below, with literal
# `{dropped_chars}`/`{kept_chars}` placeholders, never digits in their
# place. Anchoring on a digit sequence (not just the literal
# `memo_crush_retrieve(hash_marker="` substring, which this module's own
# source also contains as plain code) is what tells an ALREADY-cut block's
# own prior, rendered marker apart from this module's source text merely
# containing the same literal fragments as CODE, not as evidence of an
# earlier cut -- see test_marker_does_not_false_positive_on_its_own_template_source.
_NESTED_CCR_MARKER_RE = re.compile(r"\[memo: \d+ chars elided, \d+ kept\. ")


def marker(key: str, *, kept_chars: int, dropped_chars: int, stashed: str = "") -> str:
    """The text that replaces what was cut. Tells the model how to get it back.

    `stashed` is the exact content stored under `key` (the same string passed
    to `stash()`) — pass it so the wording can be checked against what `key`
    ACTUALLY recovers, not just what the caller hopes it recovers. When a
    transform (e.g. `ToolResults`) runs on a block an EARLIER transform
    already cut -- `JsonCrush`, or `StructMap`/`Delta` compressing a large
    file whose reduced output STILL clears the next transform's own
    threshold (measured: this repo's own `search_ops.py` reduces to an
    8855-char signature map, still over `ToolResults`' 4000-char fallback
    cap) -- `key` recovers that earlier transform's output: an intermediate
    that still carries its own recovery reference, either JsonCrush's inline
    `<<memo-crush:HASH>>` marker or this module's own rendered
    `[memo: N chars elided, M kept. ...]` marker. Saying "Full original" in
    either case is false and would make a model reading the marker stop one
    hop too early; omitting `stashed` (or passing content with no nested
    reference) keeps the stronger, still-true "Full original" claim for the
    common case where this really is the first and only cut.

    The recovery call is `memo_crush_retrieve(hash_marker=...)` -- the SAME
    MCP tool `server_crush.py` registers for the ingest-time SmartCrusher,
    reused rather than duplicated (see this module's own docstring). It is
    registered unconditionally on every MCP surface profile (`server.py`),
    not just the advanced one, and accepts a bare hex `key` exactly like
    this in addition to the SmartCrusher's own `<<memo-crush:HASH>>`-wrapped
    form -- both read the one shared `CrushCache`.
    """
    if _NESTED_CRUSH_MARKER in stashed:
        return (
            f"\n[memo: {dropped_chars} chars elided, {kept_chars} kept. "
            f'Not the full original -- memo_crush_retrieve(hash_marker="{key}") recovers '
            f"this filter's input, which itself still contains a further "
            f"memo-crush reference; retrieve that one too for the true original.]"
        )
    if _NESTED_CCR_MARKER_RE.search(stashed):
        return (
            f"\n[memo: {dropped_chars} chars elided, {kept_chars} kept. "
            f'Not the full original -- memo_crush_retrieve(hash_marker="{key}") recovers '
            f"this filter's input, which itself still contains a further "
            f"recovery marker from an earlier cut; retrieve that one too "
            f"for the true original.]"
        )
    return (
        f"\n[memo: {dropped_chars} chars elided, {kept_chars} kept. "
        f'Full original: memo_crush_retrieve(hash_marker="{key}")]'
    )
