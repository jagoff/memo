"""Tests for the process-global MLX GPU serialization guard.

`gpu_guard()` exists to prevent two threads from entering MLX's single
process-global Metal command stream at once — the race that aborts the
interpreter via `mlx::core::gpu::check_error` throwing inside Metal's
async completion handler. These tests assert the two properties the
call sites rely on: mutual exclusion across threads, and reentrancy on
the same thread.
"""

from __future__ import annotations

import fcntl
import os
import threading
import time

import pytest

from memo.mlx_gpu import (
    _gpu_tl,
    _lock_path,
    _prio_path,
    gpu_deadline,
    gpu_guard,
    set_process_gpu_priority,
)


def test_gpu_guard_is_reentrant_on_same_thread(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MEMO_GPU_LOCK_PATH", str(tmp_path / "gpu.lock"))
    # embed_query()->embed() and rerank()->score() re-enter on one thread.
    # The nesting is the assertion (a second acquire must not block), so keep
    # the two `with` blocks separate rather than collapsing them.
    with gpu_guard():
        with gpu_guard():
            entered = True
    assert entered


def test_gpu_guard_serializes_across_threads(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MEMO_GPU_LOCK_PATH", str(tmp_path / "gpu.lock"))
    # Two threads doing "GPU work" must never overlap inside the guard.
    concurrency = 0
    max_concurrency = 0
    lock = threading.Lock()
    barrier = threading.Barrier(4)

    def worker() -> None:
        nonlocal concurrency, max_concurrency
        barrier.wait()  # maximise contention: all start together
        with gpu_guard():
            with lock:
                concurrency += 1
                max_concurrency = max(max_concurrency, concurrency)
            time.sleep(0.02)  # hold the "GPU" long enough to expose overlap
            with lock:
                concurrency -= 1

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert max_concurrency == 1


def _other_process_can_lock(tmp_path: object) -> bool:
    """Simulate another memo process taking the cross-process GPU lock:
    open the same lock file on a fresh fd and try a non-blocking exclusive
    flock. Returns True if it acquired (no other holder), False if blocked."""
    fd = os.open(str(_lock_path()), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True
    except BlockingIOError:
        return False
    finally:
        os.close(fd)


def test_gpu_guard_holds_cross_process_flock(tmp_path, monkeypatch) -> None:
    # Inside the guard, a *separate* process (fresh fd) must NOT be able to
    # take the lock — this is what the threading.RLock alone could not do.
    monkeypatch.setenv("MEMO_GPU_LOCK_PATH", str(tmp_path / "gpu.lock"))
    with gpu_guard():
        assert _other_process_can_lock(tmp_path) is False
    # released on exit
    assert _other_process_can_lock(tmp_path) is True


def test_gpu_guard_reentrant_does_not_deadlock_on_flock(tmp_path, monkeypatch) -> None:
    # flock is not reentrant at the fd level; nested guards on one process must
    # not try to re-acquire it (that would deadlock under LOCK_EX).
    monkeypatch.setenv("MEMO_GPU_LOCK_PATH", str(tmp_path / "gpu.lock"))
    with gpu_guard():
        with gpu_guard():
            assert _other_process_can_lock(tmp_path) is False


def test_gpu_guard_xproc_lock_can_be_disabled(tmp_path, monkeypatch) -> None:
    # With MEMO_GPU_XPROC_LOCK=0 the file lock is skipped (intra-process RLock
    # still applies). Opt-out escape hatch for latency-sensitive setups.
    monkeypatch.setenv("MEMO_GPU_LOCK_PATH", str(tmp_path / "gpu.lock"))
    monkeypatch.setenv("MEMO_GPU_XPROC_LOCK", "0")
    with gpu_guard():
        assert _other_process_can_lock(tmp_path) is True


def _hold_lock_as_other_process(path: str) -> int:
    """Hold an exclusive flock on `path` via a fresh fd (a separate open file
    description conflicts with the guard's fd even within one process —
    exactly how another memo process looks to flock). Caller closes the fd."""
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def test_gpu_deadline_sets_and_restores_thread_local() -> None:
    assert getattr(_gpu_tl, "timeout", None) is None
    with gpu_deadline(1.5):
        assert _gpu_tl.timeout == 1.5
        with gpu_deadline(0.2):
            assert _gpu_tl.timeout == 0.2
        assert _gpu_tl.timeout == 1.5
    assert _gpu_tl.timeout is None


def test_gpu_deadline_bounds_flock_wait(tmp_path, monkeypatch) -> None:
    # With another process holding the GPU flock, a gpu_guard under
    # gpu_deadline must raise TimeoutError at the deadline instead of
    # blocking forever (the recall-daemon PriorityLock poisoning).
    lock = tmp_path / "gpu.lock"
    monkeypatch.setenv("MEMO_GPU_LOCK_PATH", str(lock))
    holder = _hold_lock_as_other_process(str(lock))
    try:
        t0 = time.monotonic()
        with pytest.raises(TimeoutError):
            with gpu_deadline(0.2):
                with gpu_guard():
                    pass  # pragma: no cover - must not enter
        assert time.monotonic() - t0 < 2.0
    finally:
        fcntl.flock(holder, fcntl.LOCK_UN)
        os.close(holder)


def test_priority_process_holds_prio_flock_while_guarded(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MEMO_GPU_LOCK_PATH", str(tmp_path / "gpu.lock"))
    set_process_gpu_priority(True)
    try:
        with gpu_guard():
            # Priority flock must be EX-held for the duration of the guard —
            # that is the signal non-priority processes back off on.
            fd = os.open(str(_prio_path()), os.O_RDONLY)
            try:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            finally:
                os.close(fd)
        # Released on exit.
        fd = os.open(str(_prio_path()), os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
    finally:
        set_process_gpu_priority(False)


def test_nonpriority_backs_off_while_priority_flock_held(tmp_path, monkeypatch) -> None:
    # A live priority process holding the prio flock means non-priority
    # acquirers must NOT take the main lock even when it is free — they
    # yield the grant to the daemon and time out on their own deadline.
    monkeypatch.setenv("MEMO_GPU_LOCK_PATH", str(tmp_path / "gpu.lock"))
    set_process_gpu_priority(False)
    prio_holder = _hold_lock_as_other_process(str(_prio_path()))
    try:
        with pytest.raises(TimeoutError):
            with gpu_guard(timeout=0.3):
                pass  # pragma: no cover - must not enter
    finally:
        fcntl.flock(prio_holder, fcntl.LOCK_UN)
        os.close(prio_holder)
    # Once the priority holder is gone, acquisition proceeds normally.
    with gpu_guard(timeout=1.0):
        entered = True
    assert entered


def test_priority_flock_wait_honors_gpu_guard_timeout(tmp_path, monkeypatch) -> None:
    # A duplicate daemon or stale in-process priority state must not turn a
    # finite gpu_guard deadline into a blocking priority-flock acquisition.
    monkeypatch.setenv("MEMO_GPU_LOCK_PATH", str(tmp_path / "gpu.lock"))
    prio_holder = _hold_lock_as_other_process(str(_prio_path()))
    set_process_gpu_priority(True)
    try:
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            with gpu_guard(timeout=0.2):
                pass  # pragma: no cover - must not enter
        assert time.monotonic() - started < 2.0
    finally:
        set_process_gpu_priority(False)
        fcntl.flock(prio_holder, fcntl.LOCK_UN)
        os.close(prio_holder)


def test_priority_releases_prio_flock_on_main_lock_timeout(tmp_path, monkeypatch) -> None:
    # If the priority process times out waiting for the main lock, it must
    # not leave the prio flock held (that would freeze every batch job).
    lock = tmp_path / "gpu.lock"
    monkeypatch.setenv("MEMO_GPU_LOCK_PATH", str(lock))
    holder = _hold_lock_as_other_process(str(lock))
    set_process_gpu_priority(True)
    try:
        with pytest.raises(TimeoutError):
            with gpu_guard(timeout=0.2):
                pass  # pragma: no cover - must not enter
        fd = os.open(str(_prio_path()), os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
    finally:
        set_process_gpu_priority(False)
        fcntl.flock(holder, fcntl.LOCK_UN)
        os.close(holder)


@pytest.mark.skip(reason="run manually: real two-process serialization smoke")
def test_two_processes_serialize_for_real() -> None:  # pragma: no cover
    pass
