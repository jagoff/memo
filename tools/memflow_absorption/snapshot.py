"""Descriptor-relative immutable snapshots for cutover evidence."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

from memo.atomic_io import open_secure_directory
from memo.operational_event import canonical_json_bytes
from tools.memflow_absorption.schemas import SnapshotReceipt

_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


class SnapshotError(RuntimeError):
    """A snapshot input or destination is unsafe."""


def _read_regular_file(path: Path) -> tuple[bytes, os.stat_result]:
    descriptor = -1
    try:
        with open_secure_directory(path.parent) as directory:
            observed = directory.stat(path.name)
            if not stat.S_ISREG(observed.st_mode):
                raise SnapshotError("snapshot source must be a regular file")
            descriptor = os.open(
                path.name,
                os.O_RDONLY | _NOFOLLOW,
                dir_fd=directory.descriptor,
            )
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise SnapshotError("snapshot source must be a regular file")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            final = os.fstat(descriptor)
            data = b"".join(chunks)
            if ((opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
                    != (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns, final.st_ctime_ns)
                    or final.st_size != len(data)):
                raise SnapshotError("snapshot source changed while reading")
            return data, final
    except SnapshotError:
        raise
    except (FileNotFoundError, OSError, ValueError) as exc:
        message = "snapshot source is a symlink" if path.is_symlink() else "unsafe snapshot source"
        raise SnapshotError(message) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def create_readonly_snapshot(source: Path, target: Path) -> SnapshotReceipt:
    """Copy one explicit regular file and publish an adjacent canonical receipt."""

    source_path = Path(os.path.abspath(os.fspath(source)))
    target_path = Path(os.path.abspath(os.fspath(target)))
    data, observed = _read_regular_file(source_path)
    target_mode = stat.S_IMODE(observed.st_mode) & 0o444
    if target_mode == 0:
        target_mode = 0o400
    receipt = SnapshotReceipt(
        schema="memo.cutover_snapshot_receipt.v2",
        source=str(source_path),
        target=str(target_path),
        source_size=observed.st_size,
        source_mtime_ns=observed.st_mtime_ns,
        source_mode=stat.S_IMODE(observed.st_mode),
        source_device=observed.st_dev,
        source_inode=observed.st_ino,
        target_size=len(data),
        target_mtime_ns=observed.st_mtime_ns,
        target_mode=target_mode,
        target_device=0,
        target_inode=0,
        sha256=hashlib.sha256(data).hexdigest(),
    )
    receipt_name = f"{target_path.name}.receipt.json"
    try:
        with open_secure_directory(target_path.parent, create=True) as directory:
            if directory.exists(target_path.name) or directory.exists(receipt_name):
                raise SnapshotError("snapshot destination already exists")
            directory.create_bytes_exclusive(target_path.name, data, mode=target_mode)
            target_stat = directory.stat(target_path.name)
            receipt = SnapshotReceipt(
                **{**receipt.to_dict(), "target_size": target_stat.st_size,
                   "target_mtime_ns": target_stat.st_mtime_ns,
                   "target_mode": stat.S_IMODE(target_stat.st_mode),
                   "target_device": target_stat.st_dev, "target_inode": target_stat.st_ino}
            )
            try:
                directory.create_bytes_exclusive(
                    receipt_name,
                    canonical_json_bytes(receipt.to_dict()),
                    mode=0o444,
                )
            except BaseException:
                directory.unlink(target_path.name, missing_ok=True)
                raise
    except SnapshotError:
        raise
    except (FileExistsError, OSError, ValueError) as exc:
        if target_path.exists() or target_path.with_name(receipt_name).exists():
            raise SnapshotError("snapshot destination already exists") from exc
        raise SnapshotError("snapshot target crosses symlink or authority boundary") from exc
    return receipt
