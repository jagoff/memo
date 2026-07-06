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

import errno
import os
import threading
import time
import warnings
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


def suppress_swig_deprecation_warnings() -> None:
    """Silence Python 3.14 metadata noise from generated SWIG helpers.

    sqlite-vec and tokenizer dependencies emit these warnings during import
    and again at interpreter shutdown; their generated types are outside
    memo's control.
    """
    warnings.filterwarnings(
        "ignore",
        message=(
            r"builtin type (?:SwigPyPacked|SwigPyObject|swigvarlink) "
            r"has no __module__ attribute"
        ),
        category=DeprecationWarning,
    )


# Cross-process flock state, only ever touched while holding `_GPU_LOCK`.
_depth = 0
_lock_fd: int | None = None
_prio_fd: int | None = None

# Set once by the resident recall daemon (`set_process_gpu_priority`): its
# tiny query embeds must not starve behind batch MLX jobs (capture-stop,
# refresh-summary, idle-daemon, test suites) each holding the GPU flock for
# long stretches — observed 2026-07-05 as minutes of `recall_lock_bail` while
# the daemon's embed thread sat in flock() behind the batch queue.
_process_priority = False

_LOCK_FILENAME = "memo-mlx-gpu.lock"
_PRIO_FILENAME = "memo-mlx-gpu.prio.lock"
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
    # Raw os.environ reads: mlx_gpu is a pure-stdlib leaf module that must not
    # import memo.flags (would create a cycle — memo.flags depends on config
    # which indirectly imports store which would import memo.flags...). The
    # flags registry in flags_ingest.py documents both vars for auditability.
    override = os.environ.get("MEMO_GPU_LOCK_PATH", "").strip()
    if override:
        return Path(override)
    return Path.home() / ".cache" / "memo" / _LOCK_FILENAME


def _xproc_lock_enabled() -> bool:
    """Whether the cross-process file lock is active (MEMO_GPU_XPROC_LOCK)."""
    if not _HAVE_FLOCK:
        return False
    return os.environ.get("MEMO_GPU_XPROC_LOCK", "").strip().lower() not in _FALSE


_FLOCK_POLL_INTERVAL_S = 0.025


def set_process_gpu_priority(enabled: bool) -> None:
    """Mark this process as a priority GPU user (the resident recall daemon).

    Priority processes hold the priority flock while waiting for / holding the
    main GPU flock; non-priority processes probe it before each outermost
    acquire and back off while it is held. Net effect: batch jobs yield the
    GPU to the daemon at their next chunk boundary instead of starving it
    through a whole queue of long holds. Liveness is kernel-managed (flock
    drops on process death), so a crashed priority process cannot wedge
    anyone.
    """
    global _process_priority
    with _GPU_LOCK:
        _process_priority = enabled


def _prio_path() -> Path:
    return _lock_path().with_name(_PRIO_FILENAME)


def _acquire_prio_flock() -> None:
    """Take the priority flock (priority process, outermost entry only).

    Contention here is priority-vs-priority only (one recall daemon per
    machine), so a plain blocking acquire is fine. Failure degrades to
    running without the fast lane — never blocks the caller's real work.
    """
    global _prio_fd
    try:
        path = _prio_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
    except OSError:
        _prio_fd = None
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
    except OSError:
        os.close(fd)
        _prio_fd = None
        return
    _prio_fd = fd


def _release_prio_flock() -> None:
    global _prio_fd
    fd = _prio_fd
    _prio_fd = None
    if fd is None:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _priority_waiting() -> bool:
    """True when a live priority process holds the priority flock.

    A non-blocking shared probe: if the shared lock is refused, a priority
    process holds it exclusively right now. Any OSError (missing file,
    perms) reads as "no priority waiter" — degrade open, never block.
    """
    try:
        fd = os.open(str(_prio_path()), os.O_RDONLY)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        return exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK)
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
    return False


def _acquire_flock(timeout: float | None = None) -> bool:
    """Take the exclusive cross-process lock (outermost entry only).

    With no ``timeout``, waits until other processes release (historical
    behavior). With a ``timeout`` (seconds remaining on the caller's deadline),
    returns ``False`` if the deadline passes — so one stuck MLX process (e.g.
    an abandoned `chat_with_timeout` worker still generating) cannot wedge
    every other memo process on the machine. Open or unexpected flock failures
    degrade to intra-process-only (return ``True`` without the lock) rather
    than block every MLX call — correctness-best-effort.

    Priority lane: a priority process (the recall daemon) takes the priority
    flock before waiting, so non-priority waiters back off; a non-priority
    process polls with ``LOCK_NB`` and skips acquisition attempts while a
    live priority process holds the priority flock — yielding the GPU to the
    daemon at the next release instead of racing it for the grant.
    """
    global _lock_fd
    if _process_priority:
        _acquire_prio_flock()
    path = _lock_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
    except OSError:
        # If we can't even open the lock file, degrade to intra-process only.
        _lock_fd = None
        _release_prio_flock()
        return True
    if timeout is None and _process_priority:
        # Priority + no deadline: a blocking acquire is the fastest grant
        # (no poll latency), and nobody outranks us.
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)  # blocks until other processes release
        except OSError:
            os.close(fd)
            _lock_fd = None
            _release_prio_flock()
            return True
        _lock_fd = fd
        return True
    deadline = time.monotonic() + max(timeout, 0.0) if timeout is not None else None
    while True:
        if not _process_priority and _priority_waiting():
            # A live priority process wants (or holds) the GPU — don't
            # compete for the grant, just wait it out.
            if deadline is not None and deadline - time.monotonic() <= 0:
                os.close(fd)
                _lock_fd = None
                return False
            time.sleep(_FLOCK_POLL_INTERVAL_S)
            continue
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                os.close(fd)
                _lock_fd = None
                _release_prio_flock()
                return True  # unexpected failure — degrade, don't block
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                os.close(fd)
                _lock_fd = None
                _release_prio_flock()
                return False  # deadline exceeded — caller raises TimeoutError
            time.sleep(
                _FLOCK_POLL_INTERVAL_S
                if remaining is None
                else min(_FLOCK_POLL_INTERVAL_S, remaining)
            )
        else:
            _lock_fd = fd
            return True


def _release_flock() -> None:
    """Release the cross-process lock (outermost exit only)."""
    global _lock_fd
    fd = _lock_fd
    _lock_fd = None
    try:
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
    finally:
        _release_prio_flock()


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
    When set, a finite deadline is imposed on the lock acquire — covering BOTH
    the intra-process RLock and the cross-process file lock: if an abandoned
    thread (or another memo process stuck in an MLX pass) still holds a lock,
    subsequent calls raise ``TimeoutError`` after this many seconds rather than
    blocking forever.
    """
    global _depth
    effective_timeout: float | None = (
        timeout if timeout is not None else getattr(_gpu_tl, "timeout", None)
    )
    start = time.monotonic()
    acquired = _GPU_LOCK.acquire(
        timeout=effective_timeout if effective_timeout is not None else -1.0
    )
    if not acquired:
        raise TimeoutError(
            f"GPU lock not acquired within {effective_timeout}s — an abandoned MLX thread "
            "may still hold it; this cluster will be skipped"
        )
    try:
        if _depth == 0 and _xproc_lock_enabled():
            remaining: float | None = None
            if effective_timeout is not None:
                remaining = effective_timeout - (time.monotonic() - start)
            if not _acquire_flock(timeout=remaining):
                raise TimeoutError(
                    f"GPU cross-process lock not acquired within {effective_timeout}s — "
                    "another memo process (possibly a stuck MLX pass) still holds it; "
                    "this cluster will be skipped"
                )
        _depth += 1
        try:
            yield
        except BaseException:
            _depth -= 1
            if _depth == 0:
                _release_flock()
            raise
        else:
            _depth -= 1
            if _depth == 0:
                _release_flock()
    finally:
        _GPU_LOCK.release()


@contextmanager
def gpu_deadline(seconds: float | None) -> Iterator[None]:
    """Impose a lock-acquisition deadline on every `gpu_guard()` in this thread.

    Sets the thread-local that `gpu_guard` reads when called without an
    explicit ``timeout`` (the same channel `chat_with_timeout` uses), and
    restores the previous value on exit. Callers that hold their own
    serialization lock while entering MLX (the recall daemon's PriorityLock)
    wrap the MLX work in this so a busy GPU flock raises ``TimeoutError``
    after ``seconds`` instead of blocking forever while poisoning their lock.

    ``None`` restores unbounded waiting for the wrapped block.
    """
    prev = getattr(_gpu_tl, "timeout", None)
    _gpu_tl.timeout = seconds
    try:
        yield
    finally:
        _gpu_tl.timeout = prev
