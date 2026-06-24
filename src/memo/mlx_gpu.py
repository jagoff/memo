"""Process-global GPU serialization for MLX forward passes.

MLX runs every array op on a single process-global Metal command stream
(`mlx.core`'s default stream). Two threads that force materialization
(`mx.eval`, `.tolist()`, `float(mx_scalar)`) concurrently race that one
command queue. When the race corrupts a command buffer, Metal reports the
error from inside its **async** completion handler:
`mlx::core::gpu::check_error` throws there, the exception can't unwind back
into Python, the C++ runtime calls `std::terminate` -> `abort()`. The whole
interpreter dies with SIGABRT — not a catchable Python exception. (Observed
crash: thread `com.Metal.CompletionQueueDispatch` aborting while a worker was
inside `mlx::core::eval`.)

There are **two** ways memo provokes this, and they need two locks:

1. *Intra-process* — the FastMCP HTTP server runs sync tool handlers on a
   worker threadpool, so a `memo_search` (embed) and a `memo_ask` (chat),
   or two concurrent searches, enter MLX at once. A process-global
   `threading.RLock` serializes those.

2. *Inter-process* — memo runs several independent MLX processes against the
   one physical GPU: the warm `memo-mcp` HTTP server, the recall daemon, and
   short-lived `memo` CLI / recall-hook invocations. They submit to the same
   Metal device concurrently and corrupt each other's command buffers exactly
   the same way. A `threading.Lock` cannot span processes; an advisory
   **file lock** (`flock`) can. We layer one over the RLock so the GPU has a
   single serialization point across every memo process on the machine.

Reentrant: `embed_query`->`embed` and `rerank`->`score` re-enter on the same
thread. The RLock allows that; the `flock` (which is *not* reentrant at the fd
level) is taken only on the outermost entry per process and released on the
outermost exit, tracked by a depth counter under the RLock.

This module is a pure-stdlib leaf (`threading`, `os`, `fcntl`) — it imports
nothing from `memo`, so every MLX caller can depend on it without a cycle, and
it stays safe under the deferred-MLX-import invariant.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl  # POSIX only; macOS (the only MLX platform) always has it.

    _HAVE_FLOCK = True
except ImportError:  # pragma: no cover - non-POSIX has no MLX/GPU to guard
    _HAVE_FLOCK = False

_GPU_LOCK = threading.RLock()

# Thread-local storage for the current GPU lock timeout.
# Set by `chat_with_timeout` in the submitted worker thread before calling
# `chat.chat()` so that `gpu_guard()` uses a matching deadline without
# requiring signature changes to `MLXChat.chat()`.
_gpu_tl: threading.local = threading.local()

# Cross-process flock state, only ever touched while holding `_GPU_LOCK`.
_depth = 0
_lock_fd: int | None = None

_LOCK_FILENAME = "memo-mlx-gpu.lock"
_FALSE = {"0", "false", "no", "off"}


def _lock_path() -> Path:
    """Filesystem path of the cross-process GPU lock.

    The GPU is machine-global, so the lock must be too: every memo process for
    a user — any runtime, any data dir, shell or launchd — has to resolve the
    **same** path or they stop coordinating. So it is a fixed user-global path
    (`~/.cache/memo/`), deliberately NOT tied to `MEMO_STATE_DIR`/`TMPDIR`
    (which can differ between a shell and a launchd daemon and silently split
    the lock). `MEMO_GPU_LOCK_PATH` overrides it (test isolation / edge setups).
    Read at call time so the override takes effect per call.
    """
    override = os.environ.get("MEMO_GPU_LOCK_PATH", "").strip()
    if override:
        return Path(override)
    return Path.home() / ".cache" / "memo" / _LOCK_FILENAME


def _xproc_lock_enabled() -> bool:
    """Whether the cross-process file lock is active (MEMO_GPU_XPROC_LOCK)."""
    if not _HAVE_FLOCK:
        return False
    return os.environ.get("MEMO_GPU_XPROC_LOCK", "").strip().lower() not in _FALSE


def _acquire_flock() -> None:
    """Take the exclusive cross-process lock (outermost entry only)."""
    global _lock_fd
    path = _lock_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
    except OSError:
        # If we can't even open the lock file, degrade to intra-process only
        # rather than block every MLX call — correctness-best-effort.
        _lock_fd = None
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)  # blocks until other processes release
    except OSError:
        os.close(fd)
        _lock_fd = None
        return
    _lock_fd = fd


def _release_flock() -> None:
    """Release the cross-process lock (outermost exit only)."""
    global _lock_fd
    fd = _lock_fd
    _lock_fd = None
    if fd is None:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


@contextmanager
def gpu_guard(timeout: float | None = None) -> Iterator[None]:
    """Serialize an MLX forward pass against all other GPU work on the machine.

    Holds a process-global reentrant lock (threads) and, on the outermost
    entry, an advisory file lock (other memo processes). Wrap the smallest
    region that builds and materializes MLX arrays (from `mx.array(...)`
    through the `.tolist()` / `float(...)` that forces evaluation). CPU-only
    tokenization/pooling setup can stay outside to keep the held window short.

    ``timeout`` may be passed explicitly or inherited from ``_gpu_tl.timeout``
    (a thread-local set by ``chat_with_timeout`` in the submitted worker).
    When set, a finite deadline is imposed on the lock acquire: if an abandoned
    thread from a prior timed-out call still holds the lock, subsequent calls
    raise ``TimeoutError`` after this many seconds rather than blocking forever.
    """
    global _depth
    effective_timeout: float | None = timeout if timeout is not None else getattr(_gpu_tl, "timeout", None)
    acquired = _GPU_LOCK.acquire(timeout=effective_timeout if effective_timeout is not None else -1.0)
    if not acquired:
        raise TimeoutError(
            f"GPU lock not acquired within {effective_timeout}s — an abandoned MLX thread "
            "may still hold it; this cluster will be skipped"
        )
    try:
        if _depth == 0 and _xproc_lock_enabled():
            _acquire_flock()
        _depth += 1
        try:
            yield
        finally:
            _depth -= 1
            if _depth == 0:
                _release_flock()
    finally:
        _GPU_LOCK.release()
