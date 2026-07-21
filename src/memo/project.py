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

# Directory names owned by the lifecycle archive (lifecycle.archive_memory /
# consolidation write `<id>.md` under these). A project slug must NEVER land in
# one of them: reindex/gc deliberately SKIP these dirs (they hold de-indexed
# archives), so a project-bucket memory written there would become invisible to
# search and unrecoverable by `reindex` — breaking markdown-is-source-of-truth.
# `slugify_project` can never emit a leading underscore (it collapses every
# non-`[a-z0-9]` run to `-`), so the `_`-prefixed remap below is provably
# collision-free with real slugs — the same property that makes `_global` safe.
LIFECYCLE_ARCHIVE_DIRS: frozenset[str] = frozenset({"inactive", "archived"})


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
            if not slug:
                return GLOBAL_BUCKET
            # Keep project buckets out of the lifecycle-archive keyspace.
            return f"_{slug}" if slug in LIFECYCLE_ARCHIVE_DIRS else slug
    return GLOBAL_BUCKET


def _worktree_main_toplevel(dotgit_file: Path) -> Path | None:
    """Resolve a linked-worktree `.git` FILE to the MAIN repo's toplevel.

    Layout: the file is one line ``gitdir: <path>`` pointing at
    ``<main>/.git/worktrees/<name>``; that dir carries a ``commondir`` file
    whose (usually relative ``../..``) path leads back to the shared ``.git``
    directory — the main toplevel is its parent. Pure file reads, no ``git``
    subprocess: this sits on the recall-hook path (5s budget). Returns None
    for non-worktree ``.git`` files (submodules have no ``commondir``) so the
    caller keeps the historical basename behavior.
    """
    try:
        lines = dotgit_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return None
    if not lines or not lines[0].startswith("gitdir:"):
        return None
    gitdir = Path(lines[0][len("gitdir:") :].strip())
    if not gitdir.is_absolute():
        gitdir = (dotgit_file.parent / gitdir).resolve()
    try:
        common_raw = (gitdir / "commondir").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    common = Path(common_raw)
    if not common.is_absolute():
        common = (gitdir / common).resolve()
    if common.name != ".git" or not common.is_dir():
        return None
    return common.parent


def _git_toplevel(start: Path) -> Path | None:
    cur = start.resolve()
    for parent in (cur, *cur.parents):
        dotgit = parent / ".git"
        if dotgit.is_dir():
            return parent
        if dotgit.is_file():
            # Linked worktree: canonicalize to the MAIN repo so memories
            # saved from e.g. `/tmp/rel` (release flow) tag as the real
            # project instead of minting `project:rel` forever.
            return _worktree_main_toplevel(dotgit) or parent
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
