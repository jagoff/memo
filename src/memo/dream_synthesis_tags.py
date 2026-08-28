"""Tags for dream-generated `synthesis` memories.

Every synthesis producer used to call ``mem.save(...)`` with no ``tags=``, so
the record kept only whatever ``save`` could derive on its own — usually one
tag. ``memo lint`` flags anything under three as ``few_tags``, citing the
CLAUDE.md convention, and on the live index `synthesis` violated it 107 times
out of 109 over eight days. That is a nightly producer writing below the
standard its own linter checks, not a legacy backlog.

Tags are derived, never generated: no embedder call, no LLM, and the same kind
always yields the same sorted set, so re-saving a memory never churns it.
"""

from __future__ import annotations

SYNTHESIS_TAG = "synthesis"

# What each producer is *about*, so a synthesis is findable by the question it
# answers and not only by the machinery that wrote it.
_KIND_TOPICS: dict[str, tuple[str, ...]] = {
    "distillation": ("distillation", "summary"),
    "cross_session": ("cross-session", "continuity"),
    "community": ("community", "clustering"),
    "bridge": ("bridge", "connection"),
    "folder_abstract": ("folder-abstract", "structure"),
}

# Applied to any kind, known or not, so a producer added later still clears the
# convention instead of silently reopening this defect.
_FALLBACK_TOPICS: tuple[str, ...] = ("dream", "generated")


def synthesis_tags(kind: str) -> list[str]:
    """The tag set for a synthesis memory of ``kind``.

    Always at least three tags, always containing ``synthesis`` so the whole
    class stays retrievable, and always containing ``kind`` so provenance
    survives into the record. Sorted and de-duplicated.
    """
    slug = (kind or "").strip() or "unknown"
    tags = {SYNTHESIS_TAG, slug, *_KIND_TOPICS.get(slug, _FALLBACK_TOPICS)}
    return sorted(tags)
