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

from memo.mlx_gpu import _lock_path, gpu_guard


def test_gpu_guard_is_reentrant_on_same_thread() -> None:
    # embed_query()->embed() and rerank()->score() re-enter on one thread.
    # The nesting is the assertion (a second acquire must not block), so keep
    # the two `with` blocks separate rather than collapsing them.
    with gpu_guard():
        with gpu_guard():
            entered = True
    assert entered


def test_gpu_guard_serializes_across_threads() -> None:
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


@pytest.mark.skip(reason="run manually: real two-process serialization smoke")
def test_two_processes_serialize_for_real() -> None:  # pragma: no cover
    pass
