"""Dream-generated `synthesis` memories must meet memo's own tag convention.

`memo lint` flags any memory with fewer than 3 tags as `few_tags`, citing the
CLAUDE.md convention of "project + domain + topic". Measured on the live index
over the eight days to 2026-08-28, `synthesis` violated it **107 times out of
109** — because all five producers (distill, consolidate, communities,
bridges, folder abstracts) called `mem.save(...)` with no `tags=` at all.

That is why `few_tags` sat at ~3,900 and grew: it is not a legacy backlog to
sweep, it is a nightly producer writing below the standard the linter checks.
Fixing the producer costs no embedder calls — `save` derives tags without one
— and stops the debt accruing.
"""

from __future__ import annotations

from memo.dream_synthesis_tags import SYNTHESIS_TAG, synthesis_tags


def test_every_synthesis_kind_yields_at_least_three_tags() -> None:
    """One tag per kind is not enough to clear the linter's own bar."""
    for kind in ("distillation", "cross_session", "community", "bridge", "folder_abstract"):
        tags = synthesis_tags(kind)
        assert len(tags) >= 3, f"{kind} produced {tags}"
        assert SYNTHESIS_TAG in tags, f"{kind} is not findable as a synthesis: {tags}"


def test_tags_identify_the_producer_that_wrote_the_memory() -> None:
    """The kind must survive into the tags, or provenance is lost."""
    assert "distillation" in synthesis_tags("distillation")
    assert "community" in synthesis_tags("community")


def test_tags_are_deduplicated_and_ordered() -> None:
    """A kind that collides with a constant must not produce a repeat."""
    tags = synthesis_tags("synthesis")
    assert len(tags) == len(set(tags)), tags
    assert tags == sorted(tags), "unstable order would churn the record on rewrite"


def test_an_unknown_kind_still_clears_the_convention() -> None:
    """A new producer that forgets to register still writes a valid memory."""
    tags = synthesis_tags("some-new-kind")
    assert len(tags) >= 3
    assert "some-new-kind" in tags


def test_failure_pattern_defaults_meet_the_convention() -> None:
    """The ⛔ AVOID producers wrote exactly two tags, one short of the bar.

    Measured over the same eight days, `failure_pattern` violated `few_tags`
    43 times out of 52 — for the same reason as `synthesis`, one level down:
    both default tag tuples carried exactly two entries, and the git miner
    built exactly two. A third tag naming the class keeps every anti-memory
    retrievable as one, the way `synthesis` now is.
    """
    from memo.negative_recall import (
        DEFAULT_AVOID_VERDICT_TAGS,
        DEFAULT_SUPERSEDE_TAGS,
        FAILURE_PATTERN_TAG,
    )

    for tags in (DEFAULT_SUPERSEDE_TAGS, DEFAULT_AVOID_VERDICT_TAGS):
        assert len(tags) >= 3, tags
        assert FAILURE_PATTERN_TAG in tags, tags


def test_git_mined_anti_memories_meet_the_convention() -> None:
    """The git miner builds its own tag list and must clear the bar too."""
    from memo.git_miner import _mined_tags
    from memo.negative_recall import FAILURE_PATTERN_TAG

    tags = _mined_tags("memo")

    assert len(tags) >= 3, tags
    assert FAILURE_PATTERN_TAG in tags
    assert "project:memo" in tags
