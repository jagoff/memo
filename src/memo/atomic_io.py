"""Small durable file-write primitives for runtime sidecar state."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def authority_write_lock(root: Path) -> Iterator[None]:
    """Cross-process lock keyed by the canonical filesystem authority root."""
    authority = str(Path(root).resolve()).encode("utf-8")
    lock_key = hashlib.sha256(authority).hexdigest()
    lock_root = Path(tempfile.gettempdir()) / f"memo-{os.getuid()}-locks"
    lock_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        lock_root.chmod(0o700)
    lock_path = lock_root / f"{lock_key}.lock"
    open_flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, open_flags, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def atomic_write_text(
    destination: Path,
    text: str,
    *,
    encoding: str = "utf-8",
    mode: int = 0o600,
    durable: bool = True,
) -> None:
    """Atomically replace ``destination`` without following a temp symlink.

    ``durable=False`` keeps the atomic replace but skips the data ``fsync``.
    Use it only for rebuildable caches whose authoritative source is already
    durable.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink() or destination.is_symlink():
        raise ValueError(f"unsafe atomic-write destination: {destination}")

    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding=encoding) as handle:
            descriptor = -1
            handle.write(text)
            handle.flush()
            if durable:
                os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, destination)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


__all__ = ["atomic_write_text", "authority_write_lock"]
