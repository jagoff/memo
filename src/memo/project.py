"""Project-scoped recall — auto-derive a `project:<slug>` tag from the
caller's working directory so saves and recalls can be biased toward the
repo the user is actually in.

The detection is intentionally cheap and offline:

1. `MEMO_PROJECT_TAG` env var wins (lets hooks pin a tag explicitly).
2. Walk up from `cwd` looking for a `.git` directory; the basename of
   that directory becomes the slug.
3. Fall back to `None` — caller decides what to do.

`project:` prefix is reserved so legit user-supplied tags (`project-alpha`,
`projects`) don't accidentally collide with auto-tags.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_PROJECT_PREFIX = "project:"
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify_project(name: str) -> str:
    """Lower-case, ascii-collapse, dash-separate. Empty input → empty string."""
    s = _SLUG_RE.sub("-", name.lower()).strip("-")
    return s


GLOBAL_BUCKET = "_global"


def project_bucket(tags: list[str]) -> str:
    """On-disk folder bucket for a memory: the project slug, or `_global`.

    Derived from the first `project:` tag. The value is re-slugified here
    (idempotent for already-clean slugs) because user-supplied tags reach
    this point verbatim — a tag like `project:../../evil` must never become
    a path component. Memories with no project tag share the `_global`
    bucket. This is the one mapping used by both the save path and
    `memo migrate --bucket-by-project`, so on-disk layout never diverges
    from the tag.
    """
    for tag in tags:
        if tag.startswith(_PROJECT_PREFIX):
            slug = slugify_project(tag[len(_PROJECT_PREFIX) :])
            return slug or GLOBAL_BUCKET
    return GLOBAL_BUCKET


def _git_toplevel(start: Path) -> Path | None:
    cur = start.resolve()
    for parent in (cur, *cur.parents):
        if (parent / ".git").exists():
            return parent
    return None


def current_project_tag(cwd: str | os.PathLike[str] | None = None) -> str | None:
    """Return `project:<slug>` for the current working directory, or None.

    Resolution order:
        1. `MEMO_PROJECT_TAG` env var (treated as the full tag; the
           `project:` prefix is added if missing).
        2. Git toplevel under `cwd` (defaults to `os.getcwd()`).
        3. None.
    """
    from memo.flags import flag_str

    pinned = flag_str("MEMO_PROJECT_TAG").strip()
    if pinned:
        slug = pinned.split(":", 1)[1] if pinned.startswith(_PROJECT_PREFIX) else pinned
        slug = slugify_project(slug)
        return f"{_PROJECT_PREFIX}{slug}" if slug else None

    base = Path(cwd) if cwd else Path.cwd()
    try:
        top = _git_toplevel(base)
    except OSError:
        return None
    if top is None:
        return None
    slug = slugify_project(top.name)
    return f"{_PROJECT_PREFIX}{slug}" if slug else None


def is_project_tag(tag: str) -> bool:
    return tag.startswith(_PROJECT_PREFIX)


def has_project_tag(tags: list[str]) -> bool:
    return any(is_project_tag(t) for t in tags)
