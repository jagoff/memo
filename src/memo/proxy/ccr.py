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
    key = hashlib.sha256(content.encode("utf-8")).hexdigest()
    try:
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


def marker(key: str, *, kept_chars: int, dropped_chars: int) -> str:
    """The text that replaces what was cut. Tells the model how to get it back."""
    return (
        f"\n[memo: {dropped_chars} chars elided, {kept_chars} kept. "
        f"Full original: memo_retrieve(key=\"{key}\")]"
    )
