"""Content-addressed recovery: nothing is cut without being recoverable.

Reuses the crush cache the capture crusher already writes to, so `memo
retrieve` keeps working unchanged and there is exactly one recovery path.
"""

from __future__ import annotations

import hashlib
import logging
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


def marker(key: str, *, kept_chars: int, dropped_chars: int, stashed: str = "") -> str:
    """The text that replaces what was cut. Tells the model how to get it back.

    `stashed` is the exact content stored under `key` (the same string passed
    to `stash()`) — pass it so the wording can be checked against what `key`
    ACTUALLY recovers, not just what the caller hopes it recovers. When a
    transform (e.g. `ToolResults`) runs on a block an EARLIER transform (e.g.
    `JsonCrush`) already crushed, `key` recovers that earlier transform's
    output — an intermediate that still carries its own
    `<<memo-crush:HASH>>` reference — not the true original. Saying "Full
    original" in that case is false and would make a model reading the
    marker stop one hop too early; omitting `stashed` (or passing content
    with no nested reference) keeps the stronger, still-true "Full original"
    claim for the common case where this really is the first and only cut.
    """
    if _NESTED_CRUSH_MARKER in stashed:
        return (
            f"\n[memo: {dropped_chars} chars elided, {kept_chars} kept. "
            f'Not the full original -- memo_retrieve(key="{key}") recovers this '
            f"filter's input, which itself still contains a further "
            f"memo-crush reference; retrieve that one too for the true original.]"
        )
    return (
        f"\n[memo: {dropped_chars} chars elided, {kept_chars} kept. "
        f'Full original: memo_retrieve(key="{key}")]'
    )
