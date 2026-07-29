from __future__ import annotations

from pathlib import Path

import pytest

import memo.atomic_io as atomic_io


def test_secure_directory_rejects_symlinked_ancestor(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)

    opener = getattr(atomic_io, "open_secure_directory", None)
    assert opener is not None, "descriptor-relative authority I/O is required"
    with pytest.raises((OSError, ValueError)):
        with opener(linked / "authority", create=True):
            pass

    assert list(outside.iterdir()) == []


def test_secure_directory_survives_directory_to_symlink_swap(tmp_path: Path) -> None:
    root = tmp_path / "authority"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    retained = tmp_path / "retained"

    opener = getattr(atomic_io, "open_secure_directory", None)
    assert opener is not None, "descriptor-relative authority I/O is required"
    with opener(root) as authority:
        root.rename(retained)
        root.symlink_to(outside, target_is_directory=True)
        authority.atomic_write_bytes(Path("nested") / "value.bin", b"retained")

    assert (retained / "nested" / "value.bin").read_bytes() == b"retained"
    assert list(outside.iterdir()) == []
