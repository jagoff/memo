"""Tests for the GPU lock deadlock fix in chat_with_timeout / gpu_guard."""
from __future__ import annotations

import threading


class TestGpuTlTimeout:
    """gpu_guard() uses _gpu_tl.timeout to impose a deadline on the lock acquire."""

    def test_gpu_guard_raises_when_other_thread_holds_lock(self):
        from memo.mlx_gpu import _GPU_LOCK, _gpu_tl, gpu_guard

        lock_held = threading.Event()
        release_lock = threading.Event()

        def hold_lock():
            _GPU_LOCK.acquire()
            lock_held.set()
            release_lock.wait(timeout=5.0)
            _GPU_LOCK.release()

        t = threading.Thread(target=hold_lock, daemon=True)
        t.start()
        lock_held.wait(timeout=2.0)

        try:
            _gpu_tl.timeout = 0.1  # 100 ms deadline
            import pytest

            with pytest.raises(TimeoutError), gpu_guard():
                pass
        finally:
            _gpu_tl.timeout = None
            release_lock.set()
            t.join(timeout=2.0)

    def test_gpu_guard_succeeds_when_lock_free(self, monkeypatch):
        monkeypatch.setenv("MEMO_GPU_XPROC_LOCK", "0")
        from memo.mlx_gpu import _gpu_tl, gpu_guard

        _gpu_tl.timeout = 0.5
        try:
            entered = False
            with gpu_guard():
                entered = True
            assert entered
        finally:
            _gpu_tl.timeout = None

    def test_gpu_guard_no_timeout_blocks_normally(self, monkeypatch):
        """With no timeout, gpu_guard should acquire successfully (lock is free)."""
        monkeypatch.setenv("MEMO_GPU_XPROC_LOCK", "0")
        from memo.mlx_gpu import gpu_guard

        entered = False
        with gpu_guard():
            entered = True
        assert entered


class TestFlockTimeout:
    """gpu_guard()'s timeout also bounds the cross-process flock acquire."""

    def test_gpu_guard_raises_when_other_process_holds_flock(self, tmp_path, monkeypatch):
        # A fresh fd on the lock file has the same flock semantics as a foreign
        # process (per open-file-description) — simulates a stuck MLX process.
        import fcntl
        import os
        import time

        import pytest

        monkeypatch.setenv("MEMO_GPU_LOCK_PATH", str(tmp_path / "gpu.lock"))
        import memo.mlx_gpu as mlx_gpu
        from memo.mlx_gpu import gpu_guard

        fd = os.open(str(tmp_path / "gpu.lock"), os.O_CREAT | os.O_RDWR, 0o644)
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            start = time.monotonic()
            with pytest.raises(TimeoutError), gpu_guard(timeout=0.5):
                pass
            assert time.monotonic() - start < 2.0  # bounded, not the holder's 6s
            # Lock state fully restored: no leaked depth/fd...
            assert mlx_gpu._depth == 0
            assert mlx_gpu._lock_fd is None
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        # ...and a subsequent uncontended guard succeeds (RLock not wedged).
        entered = False
        with gpu_guard(timeout=0.5):
            entered = True
        assert entered

    def test_gpu_guard_no_timeout_blocks_until_flock_released(self, tmp_path, monkeypatch):
        # With NO timeout, the historical behavior is preserved: the waiter
        # blocks on the flock and proceeds once the holder releases.
        import fcntl
        import os

        monkeypatch.setenv("MEMO_GPU_LOCK_PATH", str(tmp_path / "gpu.lock"))
        from memo.mlx_gpu import gpu_guard

        fd = os.open(str(tmp_path / "gpu.lock"), os.O_CREAT | os.O_RDWR, 0o644)
        fcntl.flock(fd, fcntl.LOCK_EX)
        entered = threading.Event()

        def waiter():
            with gpu_guard():
                entered.set()

        t = threading.Thread(target=waiter, daemon=True)
        t.start()
        try:
            assert not entered.wait(timeout=0.3)  # still blocked on the flock
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        assert entered.wait(timeout=2.0)  # unblocks after release
        t.join(timeout=2.0)


class TestChatWithTimeoutSetsThreadLocal:
    """chat_with_timeout sets _gpu_tl.timeout in the worker thread."""

    def test_timeout_propagated_to_worker(self):
        from memo.memory.record import chat_with_timeout
        from memo.mlx_gpu import _gpu_tl

        observed: list[float | None] = []

        class FakeChat:
            _gen_lock = threading.Lock()

            def chat(self, model, messages, options=None):
                observed.append(getattr(_gpu_tl, "timeout", None))
                return {"message": {"content": "ok"}}

        result = chat_with_timeout(FakeChat(), timeout=42.0, model="m", messages=[])
        assert result == {"message": {"content": "ok"}}
        assert observed == [42.0]

    def test_returns_none_on_timeout(self):
        from memo.memory.record import chat_with_timeout

        _started = threading.Event()
        _release = threading.Event()

        class SlowChat:
            _gen_lock = threading.Lock()

            def chat(self, model, messages, options=None):
                _started.set()
                _release.wait(timeout=5.0)
                return {"message": {"content": "done"}}

        result = chat_with_timeout(SlowChat(), timeout=0.1, model="m", messages=[])
        assert result is None
        _release.set()
