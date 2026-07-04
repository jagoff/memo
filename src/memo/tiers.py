"""Recall tiering — durable knowledge vs bulk reference material.

memo's corpus mixes two very different things:

  * **Durable knowledge** — decisions, facts, preferences, bugs, feedback,
    hand-saved notes. This is the source-of-truth memo exists to surface.
  * **Reference material** — bulk-ingested vault chunks (Obsidian notes, CVs,
    course quotes, lyrics). Useful to search on demand, but it must NOT drown
    durable knowledge in the auto-recall hook + briefing.

This module is the single source of truth for that split (imported by the
recall hook, the briefing, the `memo retier` migration, ingest, and the eval
harness) so the tier boundary is defined once. It deliberately has no memo
imports — it's a leaf so anything can depend on it without cycles.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# Reference tier: searchable on demand, excluded from auto-recall + briefing.
REFERENCE_TYPES: frozenset[str] = frozenset({"reference"})

# Durable tiers: the source-of-truth surfaced automatically. Mirrors
# `memory._VALID_TYPES` minus the reference tier. `procedure` (how-to
# workflows: "to do X, run Y") and `failure_pattern` (structured mistake
# notes: Pattern/Context/Wrong/Right) are the procedural-knowledge kinds
# mined from execution (2026-07-03 ecosystem survey, Tier2 #7).
DURABLE_TYPES: frozenset[str] = frozenset(
    {
        "decision",
        "fact",
        "bug",
        "feedback",
        "preference",
        "note",
        "manual",
        "synthesis",
        "procedure",
        "failure_pattern",
    }
)

# A chunk marker like "§54/130" in a title — the signature of a bulk vault
# ingest (one source doc split across many memories).
_CHUNK_TITLE_RE = re.compile(r"§\s*\d+\s*/\s*\d+")

# Path prefixes that mark vault-sourced material. memo's own saved memories
# live under the memory_dir with date-shaped relative paths (e.g.
# "2026/05/foo.md"), never under these.
_VAULT_PREFIXES: tuple[str, ...] = ("notes/", "work/")


def is_reference_candidate(
    path: str | None,
    tags: Iterable[str] | None,
    title: str | None,
) -> bool:
    """True if a `note` looks like bulk-ingested vault reference material.

    Hand-saved notes (no vault path, no chunk marker, no `chunk` tag) return
    False and stay in the durable tier. Used by `memo retier` to migrate the
    existing corpus and by ingest to tag new bulk material correctly.
    """
    p = (path or "").lower()
    if p.startswith(_VAULT_PREFIXES):
        return True
    if "chunk" in {str(t).lower() for t in (tags or [])}:
        return True
    return bool(title and _CHUNK_TITLE_RE.search(title))


# Types the dream prune-floor / LFU-eviction passes must never auto-archive:
# hard-won failure + how-to knowledge stays valuable even when rarely
# accessed. synthesis/reference are excluded separately at the query sites.
# (2026-07-03 ecosystem survey, Tier2 #7 — mcp-memory-service decay-protects
# structured mistake notes.)
EVICTION_PROTECTED_TYPES: frozenset[str] = frozenset({"bug", "failure_pattern", "procedure"})

__all__ = [
    "DURABLE_TYPES",
    "EVICTION_PROTECTED_TYPES",
    "REFERENCE_TYPES",
    "is_reference_candidate",
]
