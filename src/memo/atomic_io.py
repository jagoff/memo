"""Small durable file-write primitives for runtime sidecar state."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import os
import secrets
import stat
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_LOCAL_LOCKS: dict[str, threading.RLock] = {}
_LOCAL_LOCKS_GUARD = threading.Lock()
_HELD_LOCKS = threading.local()


def _relative_parts(relative: Path | str) -> tuple[str, ...]:
    value = Path(relative)
    if value.is_absolute() or any(part in {"", ".", ".."} for part in value.parts):
        raise ValueError(f"unsafe descriptor-relative path: {relative}")
    return value.parts


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    written = 0
    while written < len(view):
        count = os.write(descriptor, view[written:])
        if count <= 0:
            raise OSError("short descriptor-relative write")
        written += count


def _read_all(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _require_directory(descriptor: int, description: str) -> os.stat_result:
    observed = os.fstat(descriptor)
    if not stat.S_ISDIR(observed.st_mode):
        raise ValueError(f"{description} is not a directory")
    return observed


def _require_regular_file(descriptor: int, description: str) -> os.stat_result:
    observed = os.fstat(descriptor)
    if not stat.S_ISREG(observed.st_mode):
        raise ValueError(f"{description} is not a regular file")
    return observed


def _open_absolute_directory(root: Path, *, create: bool) -> tuple[Path, int]:
    absolute = Path(os.path.abspath(os.fspath(root)))
    descriptor = os.open(os.sep, _DIRECTORY_FLAGS)
    try:
        for part in absolute.parts[1:]:
            try:
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                with contextlib.suppress(FileExistsError):
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
                os.fsync(descriptor)
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
            _require_directory(child, f"authority path component {part!r}")
            os.close(descriptor)
            descriptor = child
        return absolute, descriptor
    except BaseException:
        os.close(descriptor)
        raise


@dataclass
class SecureDirectory:
    """A trusted directory descriptor retained for one authority operation."""

    path: Path
    descriptor: int

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1

    def __enter__(self) -> SecureDirectory:
        return self

    def __exit__(
        self,
        _exc_type: object,
        _exc: object,
        _traceback: object,
    ) -> None:
        self.close()

    @property
    def identity(self) -> tuple[int, int]:
        observed = _require_directory(self.descriptor, str(self.path))
        return observed.st_dev, observed.st_ino

    def _open_directory(
        self,
        parts: tuple[str, ...],
        *,
        create: bool,
    ) -> int:
        descriptor = os.dup(self.descriptor)
        try:
            for part in parts:
                try:
                    child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
                except FileNotFoundError:
                    if not create:
                        raise
                    with contextlib.suppress(FileExistsError):
                        os.mkdir(part, mode=0o700, dir_fd=descriptor)
                    os.fsync(descriptor)
                    child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
                _require_directory(child, f"authority directory {part!r}")
                os.close(descriptor)
                descriptor = child
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def _parent(self, relative: Path | str, *, create: bool) -> tuple[int, str]:
        parts = _relative_parts(relative)
        parent = self._open_directory(parts[:-1], create=create)
        return parent, parts[-1]

    def ensure_directory(self, relative: Path | str) -> None:
        parts = _relative_parts(relative)
        descriptor = self._open_directory(parts, create=True)
        os.close(descriptor)

    def exists(self, relative: Path | str) -> bool:
        parent, name = self._parent(relative, create=False)
        try:
            try:
                os.stat(name, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                return False
            return True
        finally:
            os.close(parent)

    def stat(self, relative: Path | str) -> os.stat_result:
        parent, name = self._parent(relative, create=False)
        try:
            observed = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if stat.S_ISLNK(observed.st_mode):
                raise ValueError(f"unsafe authority symlink: {relative}")
            return observed
        finally:
            os.close(parent)

    def read_bytes(self, relative: Path | str) -> bytes:
        parent, name = self._parent(relative, create=False)
        descriptor = -1
        try:
            descriptor = os.open(name, os.O_RDONLY | _NOFOLLOW, dir_fd=parent)
            _require_regular_file(descriptor, f"authority file {relative}")
            return _read_all(descriptor)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            os.close(parent)

    def atomic_write_bytes(
        self,
        relative: Path | str,
        data: bytes,
        *,
        mode: int = 0o600,
        durable: bool = True,
        fsync_parent: bool = True,
    ) -> None:
        parent, name = self._parent(relative, create=True)
        temporary = f".{name}.{secrets.token_hex(16)}.tmp"
        descriptor = -1
        try:
            try:
                existing = os.stat(name, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                existing = None
            if existing is not None and not stat.S_ISREG(existing.st_mode):
                raise ValueError(f"unsafe authority destination: {relative}")
            descriptor = os.open(
                temporary,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY | _NOFOLLOW,
                mode,
                dir_fd=parent,
            )
            _require_regular_file(descriptor, f"temporary authority file {temporary}")
            _write_all(descriptor, bytes(data))
            os.fchmod(descriptor, mode)
            if durable:
                os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            os.rename(
                temporary,
                name,
                src_dir_fd=parent,
                dst_dir_fd=parent,
            )
            final = os.open(name, os.O_RDONLY | _NOFOLLOW, dir_fd=parent)
            try:
                _require_regular_file(final, f"authority file {relative}")
            finally:
                os.close(final)
            if fsync_parent:
                os.fsync(parent)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary, dir_fd=parent)
            os.close(parent)

    def create_bytes_exclusive(
        self,
        relative: Path | str,
        data: bytes,
        *,
        mode: int = 0o600,
    ) -> None:
        parent, name = self._parent(relative, create=True)
        descriptor = -1
        try:
            descriptor = os.open(
                name,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY | _NOFOLLOW,
                mode,
                dir_fd=parent,
            )
            _require_regular_file(descriptor, f"authority file {relative}")
            _write_all(descriptor, bytes(data))
            os.fchmod(descriptor, mode)
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            os.fsync(parent)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            os.close(parent)

    def append_bytes(
        self,
        relative: Path | str,
        data: bytes,
        *,
        mode: int = 0o600,
    ) -> None:
        parent, name = self._parent(relative, create=True)
        descriptor = -1
        created = False
        try:
            try:
                observed = os.stat(name, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                observed = None
                created = True
            if observed is not None and not stat.S_ISREG(observed.st_mode):
                raise ValueError(f"unsafe authority append destination: {relative}")
            descriptor = os.open(
                name,
                os.O_CREAT | os.O_APPEND | os.O_WRONLY | _NOFOLLOW,
                mode,
                dir_fd=parent,
            )
            _require_regular_file(descriptor, f"authority file {relative}")
            _write_all(descriptor, bytes(data))
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            if created:
                os.fsync(parent)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            os.close(parent)

    def truncate(self, relative: Path | str, size: int) -> None:
        parent, name = self._parent(relative, create=False)
        descriptor = -1
        try:
            descriptor = os.open(name, os.O_WRONLY | _NOFOLLOW, dir_fd=parent)
            _require_regular_file(descriptor, f"authority file {relative}")
            os.ftruncate(descriptor, size)
            os.fsync(descriptor)
            os.fsync(parent)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            os.close(parent)

    def list_names(self, relative: Path | str | None = None) -> tuple[str, ...]:
        descriptor = (
            os.dup(self.descriptor)
            if relative is None
            else self._open_directory(_relative_parts(relative), create=False)
        )
        try:
            return tuple(sorted(os.listdir(descriptor)))
        finally:
            os.close(descriptor)

    def unlink(self, relative: Path | str, *, missing_ok: bool = False) -> None:
        parent, name = self._parent(relative, create=False)
        try:
            try:
                observed = os.stat(name, dir_fd=parent, follow_symlinks=False)
                if not stat.S_ISREG(observed.st_mode):
                    raise ValueError(f"unsafe authority unlink target: {relative}")
                os.unlink(name, dir_fd=parent)
                os.fsync(parent)
            except FileNotFoundError:
                if not missing_ok:
                    raise
        finally:
            os.close(parent)

    def remove_tree(self, relative: Path | str, *, missing_ok: bool = False) -> None:
        parent, name = self._parent(relative, create=False)
        try:
            try:
                child = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent)
            except FileNotFoundError:
                if missing_ok:
                    return
                raise
            try:
                self._remove_tree_descriptor(child)
            finally:
                os.close(child)
            os.rmdir(name, dir_fd=parent)
            os.fsync(parent)
        finally:
            os.close(parent)

    def _remove_tree_descriptor(self, descriptor: int) -> None:
        for name in os.listdir(descriptor):
            observed = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if stat.S_ISREG(observed.st_mode):
                os.unlink(name, dir_fd=descriptor)
            elif stat.S_ISDIR(observed.st_mode):
                child = os.open(name, _DIRECTORY_FLAGS, dir_fd=descriptor)
                try:
                    self._remove_tree_descriptor(child)
                finally:
                    os.close(child)
                os.rmdir(name, dir_fd=descriptor)
            else:
                raise ValueError(f"unsafe authority tree entry: {name}")
        os.fsync(descriptor)


def open_secure_directory(root: Path, *, create: bool = False) -> SecureDirectory:
    """Open an absolute directory without following any path component."""
    absolute, descriptor = _open_absolute_directory(Path(root), create=create)
    return SecureDirectory(path=absolute, descriptor=descriptor)


def _open_compatible_directory(root: Path, *, create: bool) -> SecureDirectory:
    """Retain a directory fd while preserving legacy ancestor-symlink behavior."""
    absolute = Path(os.path.abspath(os.fspath(root)))
    if create:
        os.makedirs(absolute, mode=0o700, exist_ok=True)
    descriptor = os.open(
        absolute,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        _require_directory(descriptor, str(absolute))
    except BaseException:
        os.close(descriptor)
        raise
    return SecureDirectory(path=absolute, descriptor=descriptor)


def _local_lock(key: str) -> threading.RLock:
    with _LOCAL_LOCKS_GUARD:
        return _LOCAL_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def _directory_lock(
    root: Path,
    *,
    namespace: str,
    reject_ancestor_symlinks: bool,
) -> Iterator[None]:
    target = Path(os.path.abspath(os.fspath(root)))
    if target == Path(os.sep):
        parent_path = target
        target_parts: tuple[str, ...] = ()
    else:
        parent_path = target.parent
        target_parts = (target.name,)
    # Key locks to a retained no-follow parent identity plus the exact child
    # name. This supports both directory authorities and file sidecars without
    # creating the protected target merely to acquire its lock.
    opener = open_secure_directory if reject_ancestor_symlinks else _open_compatible_directory
    while True:
        try:
            directory = opener(parent_path, create=False)
            break
        except FileNotFoundError:
            ancestor = parent_path.parent
            if ancestor == parent_path:
                raise
            target_parts = (parent_path.name, *target_parts)
            parent_path = ancestor
    with directory:
        device, inode = directory.identity
        if len(target_parts) == 1:
            try:
                observed = os.stat(
                    target_parts[0],
                    dir_fd=directory.descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                observed = None
            if observed is not None and stat.S_ISLNK(observed.st_mode):
                raise ValueError(f"unsafe authority lock target: {target}")
        relative_target = "/".join(target_parts) or "."
        identity = f"{namespace}:{device}:{inode}:{relative_target}"
        lock_key = hashlib.sha256(identity.encode("ascii")).hexdigest()
        local = _local_lock(lock_key)
        with local:
            held = getattr(_HELD_LOCKS, "locks", None)
            if held is None:
                held = {}
                _HELD_LOCKS.locks = held
            if lock_key in held:
                held[lock_key] += 1
                try:
                    yield
                finally:
                    held[lock_key] -= 1
                return
            lock_root = Path(tempfile.gettempdir()) / f"memo-{os.getuid()}-locks"
            lock_root.mkdir(mode=0o700, parents=True, exist_ok=True)
            with contextlib.suppress(OSError):
                lock_root.chmod(0o700)
            lock_path = lock_root / f"{lock_key}.lock"
            descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_RDWR | _NOFOLLOW,
                0o600,
            )
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                held[lock_key] = 1
                yield
            finally:
                held.pop(lock_key, None)
                with contextlib.suppress(OSError):
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)


@contextmanager
def authority_admission_lock(root: Path) -> Iterator[None]:
    """Serialize admission against roster and epoch changes for one authority."""
    with _directory_lock(
        root,
        namespace="authority-admission-v1",
        reject_ancestor_symlinks=True,
    ):
        yield


@contextmanager
def authority_write_lock(root: Path) -> Iterator[None]:
    """Cross-process journal lock keyed by a trusted directory inode."""
    with _directory_lock(
        root,
        namespace="authority-write-v2",
        reject_ancestor_symlinks=False,
    ):
        yield


def atomic_write_text(
    destination: Path,
    text: str,
    *,
    encoding: str = "utf-8",
    mode: int = 0o600,
    durable: bool = True,
) -> None:
    """Atomically replace ``destination`` relative to a retained parent fd."""
    destination = Path(destination)
    with _open_compatible_directory(destination.parent, create=True) as directory:
        directory.atomic_write_bytes(
            destination.name,
            text.encode(encoding),
            mode=mode,
            durable=durable,
            fsync_parent=False,
        )


__all__ = [
    "SecureDirectory",
    "atomic_write_text",
    "authority_admission_lock",
    "authority_write_lock",
    "open_secure_directory",
]
