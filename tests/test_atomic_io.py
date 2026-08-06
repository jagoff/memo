"""Tests for the durable file-write primitives in ``memo.atomic_io``."""

from __future__ import annotations

import pytest

from memo.atomic_io import atomic_write_text


def test_write_through_a_symlinked_parent_directory(tmp_path) -> None:
    """A symlinked *directory* is a normal platform layout, not an attack.

    Regression: macOS ships ``/tmp`` as a symlink to ``/private/tmp``, so the
    parent-symlink guard rejected every user-chosen ``-o /tmp/...`` output path
    (``memo graph mindmap``, ``memo web build``, ``memo federation export``)
    with a raw ``ValueError`` traceback.
    """
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)

    atomic_write_text(link / "out.html", "<h1>ok</h1>")

    # The bytes land in the real directory the link points at.
    assert (real / "out.html").read_text(encoding="utf-8") == "<h1>ok</h1>"


def test_a_symlinked_destination_file_is_still_refused(tmp_path) -> None:
    """The guard that matters — never write *through* a symlinked file."""
    target = tmp_path / "secret.txt"
    target.write_text("original", encoding="utf-8")
    link = tmp_path / "innocent.txt"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="unsafe atomic-write destination"):
        atomic_write_text(link, "attacker")

    assert target.read_text(encoding="utf-8") == "original"


def test_dangling_parent_symlink_is_refused(tmp_path) -> None:
    """A parent link pointing nowhere is not a directory memo may create."""
    link = tmp_path / "dangling"
    link.symlink_to(tmp_path / "missing", target_is_directory=True)

    with pytest.raises(ValueError, match="unsafe atomic-write destination"):
        atomic_write_text(link / "out.txt", "nope")
