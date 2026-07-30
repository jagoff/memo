from __future__ import annotations

import os
import threading
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


@pytest.mark.parametrize("marker_name", ("COMMITTED.json", "APPLIED.json"))
def test_create_bytes_exclusive_never_exposes_partial_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    marker_name: str,
) -> None:
    root = tmp_path / "authority"
    root.mkdir()
    destination = root / marker_name
    real_write_all = atomic_io._write_all

    def fail_after_partial_write(descriptor: int, data: bytes) -> None:
        os.write(descriptor, data[: max(1, len(data) // 2)])
        raise OSError("simulated process loss during marker write")

    monkeypatch.setattr(atomic_io, "_write_all", fail_after_partial_write)
    with atomic_io.open_secure_directory(root) as directory:
        with pytest.raises(OSError, match="simulated process loss"):
            directory.create_bytes_exclusive(destination.name, b'{"phase":"committed"}')

    assert destination.exists() is False

    monkeypatch.setattr(atomic_io, "_write_all", real_write_all)
    with atomic_io.open_secure_directory(root) as directory:
        directory.create_bytes_exclusive(destination.name, b'{"phase":"committed"}')
    assert destination.read_bytes() == b'{"phase":"committed"}'


def test_authority_admission_lock_retains_opened_root_across_path_swap(
    tmp_path: Path,
) -> None:
    root = tmp_path / "authority"
    root.mkdir()
    retained_path = tmp_path / "retained"
    outside = tmp_path / "outside"
    outside.mkdir()

    with atomic_io.authority_admission_lock(root) as authority:
        root.rename(retained_path)
        root.symlink_to(outside, target_is_directory=True)
        authority.atomic_write_bytes("marker.json", b"retained")

    assert (retained_path / "marker.json").read_bytes() == b"retained"
    assert list(outside.iterdir()) == []


def test_authority_write_lock_identity_is_stable_when_nested_root_is_created(
    tmp_path: Path,
) -> None:
    root = tmp_path / "missing-parent" / "journal"
    attempted = threading.Event()
    entered = threading.Event()

    def create_then_lock() -> None:
        root.mkdir(mode=0o700, parents=True)
        attempted.set()
        with atomic_io.authority_write_lock(root):
            entered.set()

    writer = threading.Thread(target=create_then_lock)
    with atomic_io.authority_write_lock(root):
        writer.start()
        assert attempted.wait(timeout=2)
        assert not entered.wait(timeout=0.25)

    assert entered.wait(timeout=2)
    writer.join(timeout=2)
    assert not writer.is_alive()


def test_authority_admission_lock_identity_is_stable_after_it_creates_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "missing-parent" / "authority"
    attempted = threading.Event()
    entered = threading.Event()

    def contend() -> None:
        attempted.set()
        with atomic_io.authority_admission_lock(root):
            entered.set()

    contender = threading.Thread(target=contend)
    with atomic_io.authority_admission_lock(root):
        contender.start()
        assert attempted.wait(timeout=2)
        assert not entered.wait(timeout=0.25)

    assert entered.wait(timeout=2)
    contender.join(timeout=2)
    assert not contender.is_alive()
