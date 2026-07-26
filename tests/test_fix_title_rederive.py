"""Regression: `memo fix --body` must re-derive an auto-derived title so search
and recall don't keep showing the stale label after a body-only correction —
while never clobbering a title the user set explicitly.
"""

from __future__ import annotations

from memo.cli_memory import _rederived_title


def test_auto_derived_title_tracks_new_body():
    # title == first line of old body → auto-derived → re-derive from new body.
    new = _rederived_title(
        old_title="The Tessellate CI build timeout is 30 seconds",
        old_body="The Tessellate CI build timeout is 30 seconds.",
        new_body="The Tessellate CI build timeout is 90 seconds.",
    )
    assert new == "The Tessellate CI build timeout is 90 seconds"


def test_user_set_title_is_preserved():
    # title differs from the old body's first line → user-set → leave it.
    new = _rederived_title(
        old_title="CI timeout config",
        old_body="The Tessellate CI build timeout is 30 seconds.",
        new_body="The Tessellate CI build timeout is 90 seconds.",
    )
    assert new is None


def test_markdown_marker_title_still_matches():
    # _derive_title strips leading markdown markers; an auto-title from a
    # "# Heading" body still counts as auto-derived.
    new = _rederived_title(
        old_title="Heading one",
        old_body="# Heading one\n\nbody",
        new_body="# Heading two\n\nbody",
    )
    assert new == "Heading two"
