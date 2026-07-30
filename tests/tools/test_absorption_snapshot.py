from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from memo.operational_event import canonical_json_bytes
from tools.memflow_absorption.snapshot import SnapshotError, create_readonly_snapshot


def test_snapshot_copies_explicit_regular_file_and_publishes_receipt(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_bytes(b'{"value":1}\n')
    source.chmod(0o640)
    observed_mtime = 1_700_000_000_123_456_789
    os.utime(source, ns=(observed_mtime, observed_mtime))
    target = tmp_path / "attempt" / "snapshots" / "source.json"

    receipt = create_readonly_snapshot(source, target)

    assert target.read_bytes() == source.read_bytes()
    assert receipt.source_size == len(source.read_bytes())
    assert receipt.source_mtime_ns == observed_mtime
    assert receipt.source_mode == 0o640
    assert receipt.sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert stat.S_IMODE(target.stat().st_mode) == 0o440
    receipt_path = target.with_name(f"{target.name}.receipt.json")
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o444
    assert json.loads(receipt_path.read_bytes()) == receipt.to_dict()
    assert canonical_json_bytes(receipt.to_dict()) == receipt_path.read_bytes()


def test_snapshot_refuses_symlink_source_and_existing_destination(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("source", encoding="utf-8")
    linked = tmp_path / "linked.json"
    linked.symlink_to(source)
    target = tmp_path / "snapshots" / "source.json"

    with pytest.raises(SnapshotError, match="symlink"):
        create_readonly_snapshot(linked, target)

    create_readonly_snapshot(source, target)
    with pytest.raises(SnapshotError, match="already exists"):
        create_readonly_snapshot(source, target)


def test_snapshot_rejects_directory_and_symlinked_target_ancestor(tmp_path: Path) -> None:
    source_dir = tmp_path / "source-dir"
    source_dir.mkdir()
    with pytest.raises(SnapshotError, match="regular file"):
        create_readonly_snapshot(source_dir, tmp_path / "target")

    source = tmp_path / "source.json"
    source.write_text("safe", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)
    with pytest.raises(SnapshotError, match=r"symlink|authority"):
        create_readonly_snapshot(source, linked / "target.json")
    assert list(outside.iterdir()) == []
