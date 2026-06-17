"""Tests for the process-global MLX GPU serialization guard.

`gpu_guard()` exists to prevent two threads from entering MLX's single
process-global Metal command stream at once — the race that aborts the
interpreter via `mlx::core::gpu::check_error` throwing inside Metal's
async completion handler. These tests assert the two properties the
call sites rely on: mutual exclusion across threads, and reentrancy on
the same thread.
"""

from __future__ import annotations

import threading
import time

from memo.mlx_gpu import gpu_guard


def test_gpu_guard_is_reentrant_on_same_thread() -> None:
    # embed_query()->embed() and rerank()->score() re-enter on one thread.
    # The nesting is the assertion (a second acquire must not block), so keep
    # the two `with` blocks separate rather than collapsing them.
    with gpu_guard():  # noqa: SIM117
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
